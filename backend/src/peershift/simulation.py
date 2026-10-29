"""Selection-only simulation; no model training or network traffic."""
from collections import deque
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import random
import tomllib

from .selection import top_k_select
from .detector import peer_entropy, repetition_rate


@dataclass(frozen=True)
class Config:
    seed: int = 42
    num_clients: int = 10
    rounds: int = 100
    k: int = 2
    history_window: int = 20

    def __post_init__(self):
        if any(type(getattr(self, f.name)) is not int for f in fields(self)):
            raise ValueError('All configuration values must be integers')
        if self.num_clients < 2 or self.rounds < 1 or self.history_window < 1:
            raise ValueError('Need at least two clients and positive rounds/history window')
        if not 1 <= self.k < self.num_clients:
            raise ValueError('K must be between 1 and the eligible peer count')

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as stream:
            return cls(**tomllib.load(stream))


@dataclass
class ClientState:
    client_id: int
    history: deque


def simulate(config):
    scores = [[0.0] * config.num_clients for _ in range(config.num_clients)]
    source = random.Random(f'{config.seed}:scores')
    for i in range(config.num_clients):
        for j in range(i + 1, config.num_clients):
            scores[i][j] = scores[j][i] = source.random()
    clients = [ClientState(i, deque(maxlen=config.history_window)) for i in range(config.num_clients)]
    streams = [random.Random(f'{config.seed}:client:{i}') for i in range(config.num_clients)]
    for round_index in range(config.rounds):
        for client in clients:
            peer = top_k_select(client.client_id, scores[client.client_id], config.k,
                                rng=streams[client.client_id])
            client.history.append(peer)
            yield dict(round=round_index, client=client.client_id, selected_peer=peer,
                       entropy=peer_entropy(client.history, config.num_clients - 1),
                       repetition_rate=repetition_rate(client.history),
                       unique_peers=len(set(client.history)))


def run(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    (output / 'config.toml').write_text(''.join(f'{key} = {value}\n' for key, value in asdict(config).items()))
    with (output / 'selections.jsonl').open('w') as stream:
        for record in simulate(config):
            stream.write(json.dumps(record, sort_keys=True) + '\n')
