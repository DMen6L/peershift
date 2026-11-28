"""IID MNIST training with one client per MPI rank and dense discovery."""
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .detector import collaboration_metrics
from .learning import (make_model, snapshot, local_train, evaluate, similarities,
                       aggregate_peers, aggregation_weights)
from .model_buffer import pack_state
from .provenance import code_provenance
from .mpi_transport import MPITransport
from .selection import closest_select
from .training import (TrainingConfig, load_datasets, training_partitions,
                       make_client, _benefit_summary)


def run_mpi_training(config_path, output, datasets=None, download=False):
    """Collective entrypoint. Unexpected local failures abort the whole MPI job.

    datasets is an optional rank-local pair for offline verification. Real CLI
    runs open MNIST on each rank; only rank zero may download missing files.
    """
    from mpi4py import MPI
    world = MPI.COMM_WORLD
    rank, size = world.Get_rank(), world.Get_size()
    started = MPI.Wtime()
    try:
        setup = None
        if rank == 0:
            try:
                config = TrainingConfig.load(config_path)
                if config.policy != 'closest':
                    raise ValueError('MPI training currently supports only closest selection')
                if config.num_clients != size:
                    raise ValueError('MPI rank count must equal num_clients')
                output = Path(output).resolve()
                if output.exists() and any(output.iterdir()):
                    raise ValueError('Output directory must be empty')
                # Serialize dataset downloads, and announce root failures to all ranks.
                if datasets is None:
                    datasets = load_datasets(config, download=download)
                partitions = training_partitions(config, len(datasets[0]), len(datasets[1]))
                setup = dict(config=asdict(config), output=str(output), partitions=partitions,
                             dataset_sizes=[len(data) for data in datasets], error=None)
            except Exception as error:
                setup = dict(error=str(error))
        setup = world.bcast(setup, root=0)
        if setup['error']:
            raise ValueError(setup['error'])
        config, output = TrainingConfig(**setup['config']), Path(setup['output'])
        local_error = None
        try:
            if datasets is None:
                datasets = load_datasets(config, download=False)
            if [len(data) for data in datasets] != setup['dataset_sizes']:
                raise ValueError('Dataset sizes differ across MPI ranks')
        except Exception as error:
            local_error = f'rank {rank}: {error}'
        failures = [error for error in world.allgather(local_error) if error]
        if failures:
            raise ValueError('MPI dataset setup failed: ' + '; '.join(failures))
        train, test = datasets
        partitions = setup['partitions']
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(config.seed)
        initial = snapshot(make_model(config.hidden_size))
        fingerprint = hashlib.sha256(b''.join(value.numpy().tobytes() for value in initial.values())).hexdigest()
        if len(set(world.allgather(fingerprint))) != 1:
            raise ValueError('Initial model weights differ across ranks')
        client = make_client(config, train, partitions['clients'][rank], initial, rank)
        test_loader = DataLoader(Subset(test, partitions['test']), batch_size=config.batch_size)
        peers = [peer for peer in range(size) if peer != rank]
        transport = MPITransport(world)
        if rank == 0:
            provenance = code_provenance()
            output.mkdir(parents=True, exist_ok=True)
            (output/'config.toml').write_text(''.join(f'{key} = {json.dumps(value)}\n'
                                                    for key, value in asdict(config).items()))
            (output/'partitions.json').write_text(json.dumps(partitions))
            (output/'manifest.json').write_text(json.dumps(dict(
                **provenance, python=platform.python_version(), torch=torch.__version__,
                torchvision=version('torchvision'), numpy=np.__version__, mpi4py=version('mpi4py'),
                mpi_library=MPI.Get_library_version(), initial_model=fingerprint,
                similarity_metric=config.similarity_metric, aggregation_rule=config.aggregation_rule,
                num_collaborators=config.num_collaborators,
                candidate_discovery='dense frozen post-training snapshots; received selected snapshots reused',
                initialization='identical seeded weights; persistent seed+client+1 minibatch generators',
                dataset=config.dataset, dataset_class=f'{type(train).__module__}.{type(train).__name__}',
                execution='CPU IID deterministic MLP; one client per rank; single-host shared output',
                communication_accounting='exact discovery snapshot payload; logical JSON metadata/log sizes; setup collectives excluded; wire bytes unmeasured')))
        # Logs only are gathered. Each rank evaluates its own model, selects its
        # own peers and merges locally; no model averaging takes place at root.
        with ExitStack() as stack:
            streams = {}
            if rank == 0:
                streams = {name: stack.enter_context((output/f'{name}.jsonl').open('w'))
                           for name in ('evaluations', 'selections', 'aggregation', 'exchanges')}
            def evaluation(round_index):
                val, _ = evaluate(client.model, client.validation)
                loss, accuracy = evaluate(client.model, test_loader)
                return dict(round=round_index, client=rank, validation_loss=val,
                            test_loss=loss, test_accuracy=accuracy)

            initial_evaluations = world.gather(evaluation(-1), root=0)
            if rank == 0:
                for row in initial_evaluations:
                    streams['evaluations'].write(json.dumps(row)+'\n')
            benefits, exchanges, final_evaluations, round_timings = [], [], [], []
            for round_index in range(config.rounds):
                round_started = MPI.Wtime()
                pre_training_hash = pack_state(snapshot(client.model)).digest
                training_started = MPI.Wtime()
                local_train(client.model, client.optimizer, client.batches, config.local_steps)
                training_seconds = MPI.Wtime()-training_started
                own = snapshot(client.model)
                validation_started = MPI.Wtime()
                before = evaluate(client.model, client.validation)[0]
                validation_seconds = MPI.Wtime()-validation_started
                # Collective validation waits for every rank's training and
                # snapshot before any payload is posted. Sources stay frozen.
                alignment_started = MPI.Wtime()
                world.Barrier()
                training_alignment_seconds = MPI.Wtime()-alignment_started
                acquired, metrics = transport.exchange_many(own, peers, round_index)
                collaboration_started = MPI.Wtime()
                states = dict(zip(peers, acquired))
                states[rank] = own
                # Transport packs alphabetical keys; restore serial model order
                # for both distance formulas to reproduce the reference exactly.
                states = [{name: states[peer][name] for name in own} for peer in range(size)]
                scores = similarities(states, config.similarity_metric)[rank]
                selected = closest_select(rank, scores, config.num_collaborators)
                prior = {key+'_before_selection': value for key, value in
                         collaboration_metrics(client.history, size-1).items()}
                aggregate_peers(client.model, own, [states[peer] for peer in selected],
                                rule=config.aggregation_rule, own_weight=config.own_weight)
                client.history.append(tuple(sorted(selected)))
                self_weight, peer_weights = aggregation_weights(len(selected), config.aggregation_rule, config.own_weight)
                peer_details = dict(selected_peers=list(selected),
                                    selected_peer_distances=[-scores[peer] for peer in selected],
                                    similarity_metric=config.similarity_metric,
                                    aggregation_rule=config.aggregation_rule, self_weight=self_weight,
                                    selected_peer_weights=list(peer_weights))
                selection = dict(round=round_index, client=rank, policy=config.policy,
                                 selection_branch='closest', fallback_reason=None, **peer_details,
                                 **collaboration_metrics(client.history, size-1))
                selection_aggregation_seconds = MPI.Wtime()-collaboration_started
                evaluation_started = MPI.Wtime()
                after = evaluation(round_index)
                post_aggregation_evaluation_seconds = MPI.Wtime()-evaluation_started
                benefit = before-after['validation_loss']
                diagnostic = dict(round=round_index, client=rank, **peer_details,
                                  benefit_scope='joint_aggregation', policy=config.policy,
                                  validation_loss_before_aggregation=before,
                                  validation_loss_after_aggregation=after['validation_loss'],
                                  aggregation_benefit=benefit, **prior)
                if pack_state(own).digest != metrics['snapshot_sha256']:
                    raise RuntimeError('Post-training snapshot changed during aggregation')
                acquired_hashes = dict(zip(peers, metrics.pop('received_sha256s')))
                metrics['discovery_peers'] = metrics.pop('selected_peers')
                for key in ('sent_model_messages', 'received_model_messages', 'sent_model_bytes', 'received_model_bytes'):
                    metrics['discovery_'+key] = metrics.pop(key)
                metrics.update(client=rank, selected_peers=list(selected),
                               discovery_sha256s=[acquired_hashes[peer] for peer in peers],
                               selected_snapshot_sha256s=[acquired_hashes[peer] for peer in selected],
                               selected_snapshot_reuse=True, selected_model_contributions=len(selected),
                               selected_payload_retransmission_messages=0, selected_payload_retransmission_bytes=0,
                               pre_training_sha256=pre_training_hash,
                               averaged_sha256=pack_state(snapshot(client.model)).digest,
                               source_snapshot_unchanged=True, training_seconds=training_seconds,
                               additional_validation_seconds=validation_seconds,
                               training_alignment_seconds=training_alignment_seconds,
                               selection_aggregation_seconds=selection_aggregation_seconds,
                               post_aggregation_evaluation_seconds=post_aggregation_evaluation_seconds,
                               training_alignment_calls=1, round_completion_calls=1,
                               round_timing_collective_calls=1)
                encoded = json.dumps(dict(selection=selection, evaluation=after, aggregation=diagnostic, exchange=metrics), sort_keys=True)
                logging_started = MPI.Wtime()
                gathered = world.gather(encoded, root=0)
                logging_gather_seconds = MPI.Wtime()-logging_started
                logging_write_seconds = 0.0
                if rank == 0:
                    writing_started = MPI.Wtime()
                    round_records = []
                    final_evaluations = []
                    for raw in gathered:
                        record = json.loads(raw)
                        record['exchange'].update(logging_collective_calls=1, logging_local_encoded_bytes=len(raw.encode()))
                        for name, key in (('selections','selection'), ('evaluations','evaluation'),
                                          ('aggregation','aggregation')):
                            streams[name].write(json.dumps(record[key])+'\n')
                        benefits.append(record['aggregation']['aggregation_benefit'])
                        round_records.append(record['exchange'])
                        final_evaluations.append(record['evaluation'])
                    for name in ('selections', 'evaluations', 'aggregation'):
                        streams[name].flush()
                    logging_write_seconds = MPI.Wtime()-writing_started
                completion_started = MPI.Wtime()
                world.Barrier()
                completion_seconds = MPI.Wtime()-completion_started
                # Full algorithm round includes gathered diagnostic log writes.
                # Its own timing telemetry is necessarily emitted afterward.
                timing = dict(round_seconds=MPI.Wtime()-round_started,
                              logging_gather_seconds=logging_gather_seconds,
                              logging_write_seconds=logging_write_seconds,
                              round_completion_wait_seconds=completion_seconds)
                timings = world.gather(timing, root=0)
                if rank == 0:
                    maximum = max(row['round_seconds'] for row in timings)
                    round_timings.append(dict(round=round_index, max_round_seconds=maximum))
                    for record, timing in zip(round_records, timings):
                        record.update(timing, max_round_seconds=maximum)
                        streams['exchanges'].write(json.dumps(record)+'\n')
                        exchanges.append(record)
        transport.close()
        rank_output = output/f'rank-{rank}'
        rank_output.mkdir()
        torch.save(snapshot(client.model), rank_output/'final_model.pt')
        # Wait for all checkpoints before root announces successful completion.
        world.Barrier()
        if rank == 0:
            count_keys = ('discovery_sent_model_messages', 'discovery_received_model_messages',
                          'discovery_sent_model_bytes', 'discovery_received_model_bytes',
                          'selected_model_contributions', 'selected_payload_retransmission_messages',
                          'selected_payload_retransmission_bytes', 'metadata_local_encoded_bytes',
                          'logging_local_encoded_bytes', 'metadata_collective_calls',
                          'synchronization_calls', 'logging_collective_calls',
                          'training_alignment_calls', 'round_completion_calls', 'round_timing_collective_calls')
            totals = {key: sum(row[key] for row in exchanges) for key in count_keys}
            summary = dict(training=True, policy=config.policy, num_clients=size, rounds=config.rounds,
                           aggregation_events=size*config.rounds,
                           local_steps=size*config.rounds*config.local_steps,
                           collaborators_per_event=config.num_collaborators,
                           similarity_metric=config.similarity_metric, aggregation_rule=config.aggregation_rule,
                           discovery='dense full-snapshot; selected contributions are reuse, not extra transfers',
                           wire_bytes='not measured', runtime_seconds=MPI.Wtime()-started,
                           round_timings=round_timings,
                           total_max_round_seconds=sum(row['max_round_seconds'] for row in round_timings),
                           additional_validation_seconds=sum(row['additional_validation_seconds'] for row in exchanges),
                           final_mean_test_accuracy=sum(row['test_accuracy'] for row in final_evaluations)/size,
                           final_mean_test_loss=sum(row['test_loss'] for row in final_evaluations)/size,
                           **totals, **_benefit_summary(benefits))
            (output/'summary.json').write_text(json.dumps(summary))
            (output/'summary.txt').write_text(
                f'MPI IID training: {size} clients, {config.rounds} rounds, {summary["local_steps"]} total local steps\n'
                f'Final mean held-out test accuracy: {summary["final_mean_test_accuracy"]:.4f}\n'
                f'Sum of slowest-rank round durations: {summary["total_max_round_seconds"]:.4f} seconds\n'
                f'Discovery transfers: {totals["discovery_sent_model_messages"]}; sent payload: {totals["discovery_sent_model_bytes"]} bytes\n'
                f'Selected model contributions: {totals["selected_model_contributions"]}; reused with zero retransmission\n'
                f'Mean immediate validation benefit: {summary["mean_aggregation_benefit"]:.8f}\n'
                'Benefit describes a joint merge, not long-term improvement or proof of collapse.\n'
                'Dense discovery traffic is included; wire bytes and multi-host performance are unmeasured.\n')
            print(f'MPI IID training complete. Results: {output}', flush=True)
        world.Barrier()
    except Exception as error:
        print(f'Rank {rank}: {error}', file=sys.stderr, flush=True)
        world.Abort(1)
