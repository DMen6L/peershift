"""Dense discovery and SONAR-style collaboration across ranks, without training."""
from collections import deque
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import tomllib

import numpy as np
import torch

from .detector import collaboration_metrics
from .learning import aggregate_peers, aggregation_weights, make_model, similarities, snapshot
from .model_buffer import pack_state
from .mpi_transport import MPITransport
from .selection import closest_select


@dataclass(frozen=True)
class CollaborationConfig:
    num_clients: int = 4
    rounds: int = 3
    num_collaborators: int = 2
    hidden_size: int = 8
    seed: int = 42
    history_window: int = 20
    policy: str = 'closest'
    similarity_metric: str = 'sum_tensor_l2'
    aggregation_rule: str = 'uniform_self_and_peers'
    own_weight: float = .5

    def __post_init__(self):
        for name in ('num_clients', 'rounds', 'num_collaborators', 'hidden_size', 'history_window'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.num_clients < 2 or self.num_collaborators >= self.num_clients:
            raise ValueError('Require at least two clients and fewer collaborators than clients')
        if type(self.seed) is not int:
            raise ValueError('seed must be an integer')
        if self.policy != 'closest':
            raise ValueError('MPI collaboration currently supports only closest selection')
        if self.similarity_metric not in ('flat_l2', 'sum_tensor_l2'):
            raise ValueError('Unknown similarity metric')
        aggregation_weights(self.num_collaborators, self.aggregation_rule, self.own_weight)


def fixture_model(config, rank):
    """Distinct reproducible fixtures, not the identical initialization for learning."""
    torch.manual_seed((config.seed + rank) % (2**63))
    return make_model(config.hidden_size)


def _revision():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return 'unavailable'


def run_collaboration(config_path, output):
    from mpi4py import MPI
    world = MPI.COMM_WORLD
    rank = world.Get_rank()
    started = MPI.Wtime()
    transport = None
    try:
        setup = None
        if rank == 0:
            try:
                with open(config_path, 'rb') as stream:
                    config = CollaborationConfig(**tomllib.load(stream))
                if config.num_clients != world.Get_size():
                    raise ValueError('MPI rank count must equal num_clients')
                output = Path(output).resolve()
                if output.exists() and any(output.iterdir()):
                    raise ValueError('Output directory must be empty')
                output.mkdir(parents=True, exist_ok=True)
                setup = dict(config=asdict(config), output=str(output), error=None)
            except Exception as error:
                setup = dict(error=str(error))
        setup = world.bcast(setup, root=0)
        if setup['error']:
            raise ValueError(setup['error'])
        config, output = CollaborationConfig(**setup['config']), Path(setup['output'])
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        model = fixture_model(config, rank)
        history = deque(maxlen=config.history_window)
        discovery_peers = [peer for peer in range(world.Get_size()) if peer != rank]
        if rank == 0:
            (output/'config.toml').write_text(''.join(f'{key} = {json.dumps(value)}\n'
                                                    for key, value in asdict(config).items()))
            (output/'manifest.json').write_text(json.dumps(dict(
                code_revision=_revision(), python=platform.python_version(),
                numpy=np.__version__, torch=torch.__version__, mpi4py=version('mpi4py'),
                mpi_library=MPI.Get_library_version(),
                discovery='dense full-snapshot; every rank receives every other rank',
                selected_snapshot_reuse=True,
                communication_accounting='exact application snapshot payload; logical JSON control/log sizes; wire bytes unmeasured',
                initialization='distinct seeded fixture MLPs; no training or dataset',
                output_filesystem='shared path on one host; multi-host execution unvalidated')))
        transport = MPITransport(world)
        all_records = []
        for round_index in range(config.rounds):
            own = snapshot(model)
            acquired, metrics = transport.exchange_many(own, discovery_peers, round_index)
            # Canonical tensor order avoids rank-dependent summation order in distances.
            states = dict(zip(discovery_peers, acquired))
            states[rank] = own
            states = [{name: states[peer][name] for name in sorted(own)}
                      for peer in range(world.Get_size())]
            selection_started = MPI.Wtime()
            scores = similarities(states, config.similarity_metric)[rank]
            selected = closest_select(rank, scores, config.num_collaborators)
            before = collaboration_metrics(history, config.num_clients-1)
            aggregate_peers(model, own, [states[peer] for peer in selected],
                            rule=config.aggregation_rule, own_weight=config.own_weight)
            history.append(tuple(sorted(selected)))
            if pack_state(own).digest != metrics['snapshot_sha256']:
                raise RuntimeError('Source snapshot changed during collaboration')
            received_hashes = dict(zip(discovery_peers, metrics.pop('received_sha256s')))
            metrics['discovery_peers'] = metrics.pop('selected_peers')
            for key in ('sent_model_messages', 'received_model_messages', 'sent_model_bytes', 'received_model_bytes'):
                metrics['discovery_' + key] = metrics.pop(key)
            self_weight, peer_weights = aggregation_weights(
                config.num_collaborators, config.aggregation_rule, config.own_weight)
            metrics.update(
                discovery_sha256s=[received_hashes[peer] for peer in discovery_peers],
                selected_peers=list(selected), selected_peer_distances=[-scores[peer] for peer in selected],
                selected_snapshot_sha256s=[received_hashes[peer] for peer in selected],
                policy=config.policy, similarity_metric=config.similarity_metric,
                aggregation_rule=config.aggregation_rule, self_weight=self_weight,
                selected_peer_weights=list(peer_weights), selected_snapshot_reuse=True,
                selected_model_contributions=len(selected),
                selected_payload_retransmission_messages=0, selected_payload_retransmission_bytes=0,
                averaged_sha256=pack_state(snapshot(model)).digest,
                source_snapshot_unchanged=True, collaboration_seconds=MPI.Wtime()-selection_started,
                **{key+'_before_selection': value for key, value in before.items()},
                **collaboration_metrics(history, config.num_clients-1))
            encoded = json.dumps(metrics, sort_keys=True)
            gathered = world.gather(encoded, root=0)
            if rank == 0:
                rows = [json.loads(row) for row in gathered]
                for row, raw in zip(rows, gathered):
                    row.update(logging_collective_calls=1, logging_local_encoded_bytes=len(raw.encode()))
                all_records.extend(rows)
        transport.close()
        transport = None
        # Each rank writes its own final state; models are never gathered at root.
        rank_output = output/f'rank-{rank}'
        rank_output.mkdir()
        torch.save(snapshot(model), rank_output/'final_model.pt')
        if rank == 0:
            with (output/'collaborations.jsonl').open('w') as stream:
                for record in all_records:
                    stream.write(json.dumps(record, sort_keys=True)+'\n')
            keys = ('discovery_sent_model_messages', 'discovery_received_model_messages',
                    'discovery_sent_model_bytes', 'discovery_received_model_bytes',
                    'metadata_local_encoded_bytes', 'logging_local_encoded_bytes',
                    'selected_model_contributions', 'selected_payload_retransmission_messages',
                    'selected_payload_retransmission_bytes')
            totals = {key: sum(row[key] for row in all_records) for key in keys}
            (output/'summary.json').write_text(json.dumps(dict(
                num_clients=config.num_clients, rounds=config.rounds,
                aggregation_events=config.num_clients*config.rounds,
                discovery='dense full-snapshot', training=False, wire_bytes='not measured',
                runtime_seconds=MPI.Wtime()-started,
                **totals)))
            (output/'summary.txt').write_text(
                f'MPI core collaboration: {config.num_clients} clients, {config.rounds} rounds; no training\n'
                f'Discovery transfers: {totals["discovery_sent_model_messages"]}; sent payload: {totals["discovery_sent_model_bytes"]} bytes\n'
                f'Selected model contributions: {totals["selected_model_contributions"]}; snapshots reused, no retransmission\n'
                'Each rank selects and aggregates locally. Dense discovery is not sparse gossip traffic. Wire bytes unmeasured.\n')
            print(f'MPI core collaboration complete (no training). Results: {output}', flush=True)
        world.Barrier()
    except Exception as error:
        print(f'Rank {rank}: {error}', file=sys.stderr, flush=True)
        world.Abort(1)
