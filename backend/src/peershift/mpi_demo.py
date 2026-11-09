"""Known-model MPI exchange and local averaging, without training or MNIST."""
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import sys
import tomllib

import numpy as np
import torch

from .learning import aggregate, make_model, snapshot
from .model_buffer import pack_state
from .mpi_transport import MPITransport


@dataclass(frozen=True)
class DemoConfig:
    num_clients: int = 3
    rounds: int = 3
    hidden_size: int = 8
    pattern: str = 'ring'
    own_weight: float = .5

    def __post_init__(self):
        for name in ('num_clients', 'rounds', 'hidden_size'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.num_clients < 2:
            raise ValueError('At least two clients are required')
        if self.pattern not in ('ring', 'many_to_one', 'rotating'):
            raise ValueError('Unknown MPI demo pattern')
        if (type(self.own_weight) not in (int, float) or not math.isfinite(self.own_weight)
                or not 0 <= self.own_weight <= 1):
            raise ValueError('own_weight must be finite and between zero and one')


def demo_peer(rank, size, round_index, pattern):
    if pattern == 'many_to_one':
        return 1 if rank == 0 else 0
    offset = 1 if pattern == 'ring' else 1 + round_index % (size-1)
    return (rank+offset) % size


def run_demo(config_path, output):
    from mpi4py import MPI
    world = MPI.COMM_WORLD
    rank = world.Get_rank()
    transport = None
    try:
        setup = None
        if rank == 0:
            try:
                with open(config_path, 'rb') as stream:
                    config = DemoConfig(**tomllib.load(stream))
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
        config, output = DemoConfig(**setup['config']), Path(setup['output'])
        torch.set_num_threads(1)
        model = make_model(config.hidden_size)
        # Deliberately distinct fixture models, not a training initialization rule.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(rank+1)
        if rank == 0:
            (output/'config.toml').write_text(''.join(f'{key} = {json.dumps(value)}\n'
                                                    for key, value in asdict(config).items()))
            (output/'manifest.json').write_text(json.dumps(dict(
                python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                mpi4py=version('mpi4py'), mpi_library=MPI.Get_library_version(),
                discovery='not implemented; fixture-supplied peers',
                communication_accounting='application model payload; JSON metadata/log records exclude MPI framing',
                initialization='all parameters filled with rank + 1; no training or dataset')))
        transport = MPITransport(world)
        all_records = []
        for round_index in range(config.rounds):
            own = snapshot(model)
            received, metrics = transport.exchange(
                own, demo_peer(rank, world.Get_size(), round_index, config.pattern), round_index)
            if pack_state(own).digest != metrics['snapshot_sha256']:
                raise RuntimeError('Source snapshot changed during exchange')
            aggregate(model, own, received, config.own_weight)
            metrics.update(averaged_sha256=pack_state(snapshot(model)).digest,
                           first_parameter_value=next(model.parameters()).detach().flatten()[0].item(),
                           source_snapshot_unchanged=True)
            # Gather logs only; rank zero never gathers/averages model tensors.
            encoded = json.dumps(metrics, sort_keys=True)
            gathered = world.gather(encoded, root=0)
            if rank == 0:
                rows = [json.loads(row) for row in gathered]
                for row, raw in zip(rows, gathered):
                    row.update(logging_collective_calls=1, logging_local_encoded_bytes=len(raw.encode()))
                all_records.extend(rows)
        transport.close()
        transport = None
        if rank == 0:
            with (output/'exchanges.jsonl').open('w') as stream:
                for record in all_records:
                    stream.write(json.dumps(record, sort_keys=True)+'\n')
            totals = {key: sum(row[key] for row in all_records) for key in
                      ('sent_model_messages', 'received_model_messages', 'sent_model_bytes',
                       'received_model_bytes', 'metadata_local_encoded_bytes', 'logging_local_encoded_bytes')}
            (output/'summary.json').write_text(json.dumps(dict(
                num_clients=config.num_clients, rounds=config.rounds, pattern=config.pattern,
                discovery='not implemented', wire_bytes='not measured', **totals)))
            (output/'summary.txt').write_text(
                f'MPI model transport: {config.num_clients} clients, {config.rounds} rounds, {config.pattern}\n'
                f'Model transfers: {totals["sent_model_messages"]}; sent payload: {totals["sent_model_bytes"]} bytes\n'
                'All averaging is receiver-local. No training, similarity discovery, or wire-byte measurement.\n')
            print(f'MPI model exchange complete. Results: {output}', flush=True)
        world.Barrier()
    except Exception as error:
        print(f'Rank {rank}: {error}', file=sys.stderr, flush=True)
        world.Abort(1)
