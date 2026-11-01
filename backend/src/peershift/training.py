"""Synchronous receiver-side gossip runner; no network transport."""
from dataclasses import dataclass, asdict
from collections import deque
import hashlib
import json
import random
import time
from pathlib import Path
import tomllib
import platform
from importlib.metadata import version
import torch
from torch.utils.data import DataLoader, Subset
from .learning import make_model, snapshot, aggregate, similarities, local_train, evaluate
from .selection import select_peer
from .detector import peer_entropy, repetition_rate


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    num_clients: int = 10
    rounds: int = 10
    local_steps: int = 10
    batch_size: int = 32
    learning_rate: float = .05
    own_weight: float = .5
    k: int = 2
    history_window: int = 20
    train_per_client: int = 500
    validation_per_client: int = 100
    test_samples: int = 1000
    hidden_size: int = 128
    policy: str = 'top_k'
    dataset: str = 'mnist'
    model: str = 'mlp'
    partition: str = 'iid'
    device: str = 'cpu'
    data_dir: str = 'backend/data'

    def __post_init__(self):
        for key in ('num_clients','rounds','local_steps','batch_size','k','history_window','train_per_client','validation_per_client','test_samples','hidden_size'):
            if type(getattr(self,key)) is not int or getattr(self,key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        if type(self.seed) is not int or self.num_clients < 2 or self.k >= self.num_clients:
            raise ValueError('Invalid seed, client count or K')
        import math
        for key in ('learning_rate','own_weight'):
            value = getattr(self,key)
            if type(value) not in (int,float) or not math.isfinite(value):
                raise ValueError(f'Invalid {key}')
        if self.learning_rate <= 0 or not 0 <= self.own_weight <= 1:
            raise ValueError('Invalid learning rate or averaging weight')
        if self.policy not in ('top_k','random') or (self.dataset,self.model,self.partition,self.device) != ('mnist','mlp','iid','cpu'):
            raise ValueError('Unsupported training configuration')

    @classmethod
    def load(cls,path):
        with open(path,'rb') as stream:
            return cls(**tomllib.load(stream))


@dataclass
class LearningClient:
    model: object
    optimizer: object
    batches: object
    validation: object
    history: deque


def cycle(loader):
    while True:
        yield from loader


def run_training(config, output, datasets=None, download=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    if datasets is None:
        from torchvision.datasets import MNIST
        from torchvision.transforms import ToTensor
        datasets = (MNIST(config.data_dir, train=True, transform=ToTensor(), download=download),
                    MNIST(config.data_dir, train=False, transform=ToTensor(), download=download))
    train, test = datasets
    required = config.num_clients*(config.train_per_client+config.validation_per_client)
    if len(train)<required or len(test)<config.test_samples:
        raise ValueError('Insufficient dataset samples')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    initial = snapshot(make_model(config.hidden_size))
    indices=list(range(len(train))); random.Random(f'{config.seed}:partition').shuffle(indices)
    test_indices=list(range(len(test))); random.Random(f'{config.seed}:test').shuffle(test_indices)
    partitions=[]; clients=[]
    for i in range(config.num_clients):
        start=i*(config.train_per_client+config.validation_per_client)
        training=indices[start:start+config.train_per_client]
        validation=indices[start+config.train_per_client:start+config.train_per_client+config.validation_per_client]
        partitions.append(dict(train=training,validation=validation))
        model=make_model(config.hidden_size); model.load_state_dict(initial)
        generator=torch.Generator().manual_seed(config.seed+i+1)
        loader=DataLoader(Subset(train,training),batch_size=config.batch_size,shuffle=True,generator=generator)
        clients.append(LearningClient(model,torch.optim.SGD(model.parameters(),lr=config.learning_rate),cycle(loader),DataLoader(Subset(train,validation),batch_size=config.batch_size),deque(maxlen=config.history_window)))
    test_loader=DataLoader(Subset(test,test_indices[:config.test_samples]),batch_size=config.batch_size)
    streams=[random.Random(f'{config.seed}:client:{i}') for i in range(config.num_clients)]
    output.mkdir(parents=True,exist_ok=True)
    (output/'config.toml').write_text(''.join(f'{k} = {json.dumps(v)}\n' for k,v in asdict(config).items()))
    (output/'partitions.json').write_text(json.dumps(dict(clients=partitions,test=test_indices[:config.test_samples])))
    fingerprint=hashlib.sha256(b''.join(v.numpy().tobytes() for v in initial.values())).hexdigest()
    (output/'manifest.json').write_text(json.dumps(dict(python=platform.python_version(),torch=torch.__version__,torchvision=version("torchvision"),initial_model=fingerprint)))
    started=time.perf_counter()
    with (output/'evaluations.jsonl').open('w') as evaluations, (output/'selections.jsonl').open('w') as selections:
        def record(round_index):
            for i,c in enumerate(clients):
                val,_=evaluate(c.model,c.validation); loss,accuracy=evaluate(c.model,test_loader)
                evaluations.write(json.dumps(dict(round=round_index,client=i,validation_loss=val,test_loss=loss,test_accuracy=accuracy))+'\n')
        record(-1)
        for round_index in range(config.rounds):
            for c in clients:
                local_train(c.model,c.optimizer,c.batches,config.local_steps)
            states=[snapshot(c.model) for c in clients]; scores=similarities(states)
            decisions=[select_peer(i,scores[i],config.k,policy=config.policy,rng=streams[i]) for i in range(config.num_clients)]
            for i,(c,d) in enumerate(zip(clients,decisions)):
                aggregate(c.model,states[i],states[d.selected_peer],config.own_weight)
                c.history.append(d.selected_peer)
                selections.write(json.dumps(dict(round=round_index,client=i,policy=config.policy,selected_peer=d.selected_peer,selection_branch=d.selection_branch,fallback_reason=d.fallback_reason,entropy=peer_entropy(c.history,config.num_clients-1),repetition_rate=repetition_rate(c.history),unique_peers=len(set(c.history))))+'\n')
            record(round_index)
    exchanges=config.rounds*config.num_clients
    (output/'summary.json').write_text(json.dumps(dict(local_steps=exchanges*config.local_steps,logical_model_exchanges=exchanges,estimated_parameter_payload_bytes=exchanges*sum(v.numel()*v.element_size() for v in initial.values()),runtime_seconds=time.perf_counter()-started)))
