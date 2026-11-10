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
    exploration_probability: float = 0.1
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
        probability = self.exploration_probability
        if type(probability) not in (int, float) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Exploration probability must be between zero and one')
        if self.policy not in ('top_k','random','fixed_exploration') or (self.dataset,self.model,self.partition,self.device) != ('mnist','mlp','iid','cpu'):
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


def _benefit_summary(benefits):
    """Classify immediate benefits using the documented numerical tolerance."""
    return dict(
        mean_aggregation_benefit=sum(benefits)/len(benefits),
        improved=sum(b > 1e-8 for b in benefits),
        worsened=sum(b < -1e-8 for b in benefits),
        unchanged=sum(abs(b) <= 1e-8 for b in benefits),
    )


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
    exploration_streams=[random.Random(f'{config.seed}:exploration:{i}') for i in range(config.num_clients)]
    output.mkdir(parents=True,exist_ok=True)
    (output/'config.toml').write_text(''.join(f'{k} = {json.dumps(v)}\n' for k,v in asdict(config).items()))
    (output/'partitions.json').write_text(json.dumps(dict(clients=partitions,test=test_indices[:config.test_samples])))
    fingerprint=hashlib.sha256(b''.join(v.numpy().tobytes() for v in initial.values())).hexdigest()
    (output/'manifest.json').write_text(json.dumps(dict(python=platform.python_version(),torch=torch.__version__,torchvision=version("torchvision"),initial_model=fingerprint)))
    benefits=[]
    additional_validation_seconds=0.0
    started=time.perf_counter()
    with (output/'evaluations.jsonl').open('w') as evaluations, (output/'selections.jsonl').open('w') as selections, (output/'aggregation.jsonl').open('w') as diagnostics:
        def record(round_index):
            validation_losses=[]
            for i,c in enumerate(clients):
                val,_=evaluate(c.model,c.validation); loss,accuracy=evaluate(c.model,test_loader)
                validation_losses.append(val)
                evaluations.write(json.dumps(dict(round=round_index,client=i,validation_loss=val,test_loss=loss,test_accuracy=accuracy))+'\n')
            return validation_losses
        record(-1)
        for round_index in range(config.rounds):
            for c in clients:
                local_train(c.model,c.optimizer,c.batches,config.local_steps)
            states=[snapshot(c.model) for c in clients]; scores=similarities(states)
            evaluation_started=time.perf_counter()
            before=[evaluate(c.model,c.validation)[0] for c in clients]
            additional_validation_seconds += time.perf_counter()-evaluation_started
            prior=[dict(history_length_before_selection=len(c.history),
                        entropy_before_selection=peer_entropy(c.history,config.num_clients-1),
                        repetition_rate_before_selection=repetition_rate(c.history),
                        unique_peers_before_selection=len(set(c.history))) for c in clients]
            decisions=[select_peer(i,scores[i],config.k,policy=config.policy,rng=streams[i],exploration_probability=config.exploration_probability,exploration_rng=exploration_streams[i]) for i in range(config.num_clients)]
            for i,(c,d) in enumerate(zip(clients,decisions)):
                aggregate(c.model,states[i],states[d.selected_peer],config.own_weight)
                c.history.append(d.selected_peer)
                selections.write(json.dumps(dict(round=round_index,client=i,policy=config.policy,selected_peer=d.selected_peer,selection_branch=d.selection_branch,fallback_reason=d.fallback_reason,entropy=peer_entropy(c.history,config.num_clients-1),repetition_rate=repetition_rate(c.history),unique_peers=len(set(c.history))))+'\n')
            after=record(round_index)
            for i,d in enumerate(decisions):
                benefit=before[i]-after[i]
                benefits.append(benefit)
                diagnostics.write(json.dumps(dict(round=round_index,client=i,selected_peer=d.selected_peer,
                    policy=config.policy,validation_loss_before_aggregation=before[i],
                    validation_loss_after_aggregation=after[i],aggregation_benefit=benefit,
                    selected_peer_distance=-scores[i][d.selected_peer],**prior[i]))+'\n')
    exchanges=config.rounds*config.num_clients
    benefit_summary = _benefit_summary(benefits)
    counts = {name: benefit_summary[name] for name in ('improved', 'worsened', 'unchanged')}
    summary=dict(local_steps=exchanges*config.local_steps,logical_model_exchanges=exchanges,
                 estimated_parameter_payload_bytes=exchanges*sum(v.numel()*v.element_size() for v in initial.values()),
                 runtime_seconds=time.perf_counter()-started,
                 additional_validation_seconds=additional_validation_seconds,
                 **benefit_summary)
    (output/'summary.json').write_text(json.dumps(summary))
    lines=[f'Policy: {config.policy}; clients: {config.num_clients}; rounds: {config.rounds}; exchanges: {exchanges}',
           f'Mean immediate validation benefit: {summary["mean_aggregation_benefit"]:.8f}']
    lines.extend(f'{name.capitalize()}: {count} ({100*count/exchanges:.1f}%)' for name,count in counts.items())
    lines.extend([f'Runtime: {summary["runtime_seconds"]:.4f} seconds',
                  f'Additional pre-aggregation validation time: {additional_validation_seconds:.4f} seconds',
                  'Benefit = validation loss before minus after averaging. Positive means immediate local improvement.',
                  'Unchanged uses absolute tolerance 1e-8. These observations do not establish collapse or long-term benefit.'])
    (output/'summary.txt').write_text('\n'.join(lines)+'\n')
