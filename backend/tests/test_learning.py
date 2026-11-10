import json
import math
import pytest

torch = pytest.importorskip('torch')
from torch.utils.data import TensorDataset, DataLoader
from peershift.learning import make_model, snapshot, aggregate, similarities, local_train, evaluate
from peershift.training import TrainingConfig, run_training, cycle


def fixture_data(seed=123):
    generator = torch.Generator().manual_seed(seed)
    return TensorDataset(torch.rand(30, 1, 28, 28, generator=generator), torch.arange(30) % 10)


def test_snapshot_and_receiver_aggregation():
    a, b = make_model(4), make_model(4)
    own, peer = snapshot(a), snapshot(b)
    expected = {k: .25*own[k]+.75*peer[k] for k in own}
    aggregate(a, own, peer, .25)
    for key, value in a.state_dict().items():
        assert torch.equal(value, expected[key])
        assert torch.equal(b.state_dict()[key], peer[key])
        assert value.data_ptr() != own[key].data_ptr()
    models = [make_model(4) for _ in range(3)]
    states = [snapshot(m) for m in models]
    first = []
    for i in range(3):
        aggregate(models[i], states[i], states[(i+1)%3])
        first.append(snapshot(models[i]))
    for i in reversed(range(3)):
        aggregate(models[i], states[i], states[(i+1)%3])
        assert all(torch.equal(v, first[i][k]) for k,v in models[i].state_dict().items())


def test_training_changes_parameters_and_learns():
    torch.manual_seed(1)
    model = make_model(4)
    data = TensorDataset(torch.ones(8,1,28,28), torch.zeros(8,dtype=torch.long))
    loader = DataLoader(data,batch_size=4)
    before = snapshot(model)
    initial,_ = evaluate(model,loader)
    local_train(model,torch.optim.SGD(model.parameters(),lr=.1),cycle(loader),10)
    final,_ = evaluate(model,loader)
    assert final < initial
    assert any(not torch.equal(before[k],v) for k,v in model.state_dict().items())
    scores = similarities([{'x':torch.tensor([0.])},{'x':torch.tensor([1.])},{'x':torch.tensor([4.])}])
    assert scores[0] == [0., -1., -4.]


@pytest.mark.parametrize('policy',['top_k','random','fixed_exploration'])
def test_training_outputs(tmp_path,policy):
    config = TrainingConfig(num_clients=3,k=1,rounds=2,local_steps=1,batch_size=2,
                            train_per_client=4,validation_per_client=2,test_samples=5,hidden_size=4,policy=policy)
    data, test_data = fixture_data(), fixture_data(456)
    for name in ('a','b'):
        run_training(config,tmp_path/name,datasets=(data,test_data))
    for artifact in ('config.toml','partitions.json','selections.jsonl','evaluations.jsonl','manifest.json'):
        assert (tmp_path/'a'/artifact).read_bytes() == (tmp_path/'b'/artifact).read_bytes()
    assert TrainingConfig.load(tmp_path/'a/config.toml') == config
    partitions=json.loads((tmp_path/'a/partitions.json').read_text())
    indices=[index for client in partitions['clients'] for split in ('train','validation') for index in client[split]]
    assert len(indices)==len(set(indices))==18
    records=[json.loads(line) for line in (tmp_path/'a/selections.jsonl').read_text().splitlines()]
    assert len(records)==6 and all(r['client']!=r['selected_peer'] for r in records)
    evaluations=[json.loads(line) for line in (tmp_path/'a/evaluations.jsonl').read_text().splitlines()]
    assert len(evaluations)==9
    assert all(math.isfinite(r['validation_loss']) and 0<=r['test_accuracy']<=1 for r in evaluations)
    assert len({r['test_loss'] for r in evaluations if r['round']==-1})==1
    summary=json.loads((tmp_path/'a/summary.json').read_text())
    assert summary['logical_model_exchanges']==6 and summary['local_steps']==6
    with pytest.raises(ValueError,match='empty'):
        run_training(config,tmp_path/'a',datasets=(data,test_data))


def test_fixed_exploration_training_matches_its_probability(tmp_path):
    common = dict(num_clients=3,k=1,rounds=2,local_steps=1,batch_size=2,train_per_client=4,
                  validation_per_client=2,test_samples=5,hidden_size=4)
    data = fixture_data(), fixture_data(456)
    run_training(TrainingConfig(policy='top_k',**common), tmp_path/'top', datasets=data)
    run_training(TrainingConfig(policy='fixed_exploration',exploration_probability=0,**common), tmp_path/'zero', datasets=data)
    run_training(TrainingConfig(policy='fixed_exploration',exploration_probability=1,**common), tmp_path/'always', datasets=data)

    def peers(name):
        rows = [json.loads(line) for line in (tmp_path/name/'selections.jsonl').read_text().splitlines()]
        return rows

    assert [row['selected_peer'] for row in peers('top')] == [row['selected_peer'] for row in peers('zero')]
    assert all(row['selection_branch']=='top_k' and row['fallback_reason'] is None for row in peers('zero'))
    explored = peers('always')
    assert explored and all(row['selection_branch']=='exploration' and row['client']!=row['selected_peer'] for row in explored)


@pytest.mark.parametrize('values',[{'num_clients':1},{'k':10},{'learning_rate':float('nan')}, {'own_weight':True},{'rounds':0},{'exploration_probability':True},{'exploration_probability':-0.1},{'exploration_probability':1.1},{'partition':'non_iid'},{'device':'cuda'}])
def test_invalid_training_config(values):
    with pytest.raises(ValueError):
        TrainingConfig(**values)


def test_separate_process_cli_fixture(tmp_path):
    import subprocess
    import sys
    config = tmp_path/'fixture.toml'
    config.write_text('num_clients=2\nk=1\nrounds=1\nlocal_steps=1\nbatch_size=2\ntrain_per_client=4\nvalidation_per_client=2\ntest_samples=4\nhidden_size=4\n')
    script = '''
import sys, torch
from torch.utils.data import TensorDataset
import torchvision.datasets
from peershift.cli import main
x=torch.rand(20,1,28,28,generator=torch.Generator().manual_seed(123))
data=TensorDataset(x,torch.arange(20)%10)
test_x=torch.rand(20,1,28,28,generator=torch.Generator().manual_seed(456))
test_data=TensorDataset(test_x,torch.arange(20)%10)
torchvision.datasets.MNIST=lambda *args,**kwargs:data if kwargs["train"] else test_data
sys.argv=['peershift','train','--config',sys.argv[1],'--output',sys.argv[2]]
main()
'''
    for name in ('one','two'):
        subprocess.run([sys.executable,'-c',script,str(config),str(tmp_path/name)],check=True,capture_output=True)
    for artifact in ('selections.jsonl','evaluations.jsonl','partitions.json','aggregation.jsonl'):
        assert (tmp_path/'one'/artifact).read_bytes()==(tmp_path/'two'/artifact).read_bytes()


def test_training_validation_and_test_access_stay_separate(tmp_path, monkeypatch):
    """Audit real sample reads, not just the partition manifest."""
    import peershift.training as training

    phase = [None]

    class RecordingData(TensorDataset):
        def __init__(self, seed):
            source = fixture_data(seed)
            super().__init__(*source.tensors)
            self.reads = []

        def __getitem__(self, index):
            self.reads.append((phase[0], index))
            return super().__getitem__(index)

    train_data, test_data = RecordingData(123), RecordingData(456)
    clients = []
    original_client = training.LearningClient
    original_train, original_evaluate = training.local_train, training.evaluate
    evaluations = []

    def capture_client(*args, **kwargs):
        client = original_client(*args, **kwargs)
        clients.append(client)
        return client

    def train(model, optimizer, batches, steps):
        phase[0] = ('train', id(model))
        try:
            return original_train(model, optimizer, batches, steps)
        finally:
            phase[0] = None

    def evaluate(model, loader):
        phase[0] = ('evaluate', id(model))
        before = snapshot(model)
        try:
            result = original_evaluate(model, loader)
        finally:
            phase[0] = None
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
        evaluations.append((id(model), loader.dataset.dataset, list(loader.dataset.indices)))
        return result

    monkeypatch.setattr(training, 'LearningClient', capture_client)
    monkeypatch.setattr(training, 'local_train', train)
    monkeypatch.setattr(training, 'evaluate', evaluate)
    cfg = TrainingConfig(num_clients=3, k=1, rounds=2, local_steps=3, batch_size=2,
                         train_per_client=4, validation_per_client=2, test_samples=5, hidden_size=4)
    training.run_training(cfg, tmp_path, datasets=(train_data, test_data))
    partitions = json.loads((tmp_path/'partitions.json').read_text())
    assigned = {id(client.model): part for client, part in zip(clients, partitions['clients'])}
    all_training = {index for part in assigned.values() for index in part['train']}
    all_validation = {index for part in assigned.values() for index in part['validation']}
    assert all_training.isdisjoint(all_validation)
    for model_id, source, indices in evaluations:
        expected = assigned[model_id]['validation'] if source is train_data else partitions['test']
        assert source is train_data or source is test_data
        assert indices == expected
    for context, index in train_data.reads:
        role, model_id = context
        expected = assigned[model_id]['train' if role == 'train' else 'validation']
        assert index in expected
    assert test_data.reads and all(context[0] == 'evaluate' for context, _ in test_data.reads)
    assert {index for _, index in test_data.reads} == set(partitions['test'])
    for client in clients:
        model_id = id(client.model)
        training_reads = [index for context, index in train_data.reads if context == ('train', model_id)]
        assert len(training_reads) == cfg.rounds*cfg.local_steps*cfg.batch_size
        validation_calls = [indices for mid, source, indices in evaluations
                            if mid == model_id and source is train_data]
        assert len(validation_calls) == 1 + 2*cfg.rounds
