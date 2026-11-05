import json
import math
import pytest

torch = pytest.importorskip('torch')
from torch.utils.data import TensorDataset, DataLoader
from peershift.learning import make_model, snapshot, aggregate, similarities, local_train, evaluate
from peershift.training import TrainingConfig, run_training, cycle


def fixture_data():
    generator = torch.Generator().manual_seed(123)
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


@pytest.mark.parametrize('policy',['top_k','random'])
def test_training_outputs(tmp_path,policy):
    config = TrainingConfig(num_clients=3,k=1,rounds=2,local_steps=1,batch_size=2,
                            train_per_client=4,validation_per_client=2,test_samples=5,hidden_size=4,policy=policy)
    data = fixture_data()
    for name in ('a','b'):
        run_training(config,tmp_path/name,datasets=(data,data))
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
        run_training(config,tmp_path/'a',datasets=(data,data))


@pytest.mark.parametrize('values',[{'num_clients':1},{'k':10},{'learning_rate':float('nan')}, {'own_weight':True},{'rounds':0},{'policy':'fixed_exploration'},{'partition':'non_iid'},{'device':'cuda'}])
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
torchvision.datasets.MNIST=lambda *args,**kwargs:data
sys.argv=['peershift','train','--config',sys.argv[1],'--output',sys.argv[2]]
main()
'''
    for name in ('one','two'):
        subprocess.run([sys.executable,'-c',script,str(config),str(tmp_path/name)],check=True,capture_output=True)
    for artifact in ('selections.jsonl','evaluations.jsonl','partitions.json','aggregation.jsonl'):
        assert (tmp_path/'one'/artifact).read_bytes()==(tmp_path/'two'/artifact).read_bytes()
