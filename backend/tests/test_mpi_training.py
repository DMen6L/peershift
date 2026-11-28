"""Small real-rank training runs matched against the serial learner."""
import json

import pytest

pytest.importorskip('torch')
pytest.importorskip('torchvision')
pytest.importorskip('mpi4py')
from test_mpi_transport import launch

pytestmark = pytest.mark.mpi

FIXTURE = '''
import torch
from torch.utils.data import TensorDataset
def data(seed):
    return TensorDataset(torch.rand(60,1,28,28,generator=torch.Generator().manual_seed(seed)),torch.arange(60)%10)
datasets=(data(123),data(456))
'''
DIRECT = FIXTURE + '''
import sys
from peershift.mpi_training import run_mpi_training
run_mpi_training(sys.argv[1],sys.argv[2],datasets=datasets)
'''


def parse(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize('metric,rule,clients,count,delayed', [
    ('flat_l2','fixed_self',3,1,False),
    ('flat_l2','uniform_self_and_peers',4,2,False),
    ('sum_tensor_l2','fixed_self',4,2,False),
    ('sum_tensor_l2','uniform_self_and_peers',3,1,False),
    ('sum_tensor_l2','uniform_self_and_peers',3,2,False),
    ('sum_tensor_l2','uniform_self_and_peers',3,1,True),
])
def test_mpi_training_matches_serial_tensors_and_observations(tmp_path, monkeypatch, metric, rule, clients, count, delayed):
    import torch
    from torch.utils.data import TensorDataset
    import peershift.training as training
    cfg = training.TrainingConfig(num_clients=clients, rounds=3, local_steps=4,
        batch_size=2, train_per_client=5, validation_per_client=3, test_samples=7,
        hidden_size=4, history_window=2, policy='closest', num_collaborators=count,
        similarity_metric=metric, aggregation_rule=rule, own_weight=.7)
    config = tmp_path/'config.toml'
    from dataclasses import asdict
    config.write_text(''.join(f'{key}={json.dumps(value)}\n' for key,value in asdict(cfg).items()))
    make_data = lambda seed: TensorDataset(torch.rand(60,1,28,28,generator=torch.Generator().manual_seed(seed)),torch.arange(60)%10)
    captured=[]
    original=training.LearningClient
    def capture(*args,**kwargs):
        client=original(*args,**kwargs)
        captured.append(client)
        return client
    monkeypatch.setattr(training,'LearningClient',capture)
    training.run_training(cfg,tmp_path/'serial',datasets=(make_data(123),make_data(456)))
    script = DIRECT
    if delayed:
        script = FIXTURE + """
import sys,time
from mpi4py import MPI
import peershift.mpi_training as training
original=training.local_train
def slow_train(*args,**kwargs):
    if MPI.COMM_WORLD.rank==1: time.sleep(.2)
    return original(*args,**kwargs)
training.local_train=slow_train
training.run_mpi_training(sys.argv[1],sys.argv[2],datasets=datasets)
"""
    result=launch(script,config,tmp_path/'mpi',ranks=clients,timeout=60)
    assert result.returncode==0,result.stderr
    assert training.TrainingConfig.load(tmp_path/'mpi/config.toml')==cfg
    for name in ('partitions.json','manifest.json'):
        serial=json.loads((tmp_path/'serial'/name).read_text())
        distributed=json.loads((tmp_path/'mpi'/name).read_text())
        if name=='partitions.json':
            assert distributed==serial
        else:
            assert distributed['initial_model']==serial['initial_model']
    for name in ('selections','evaluations','aggregation'):
        assert parse(tmp_path/f'mpi/{name}.jsonl')==parse(tmp_path/f'serial/{name}.jsonl')
    for rank,client in enumerate(captured):
        actual=torch.load(tmp_path/f'mpi/rank-{rank}/final_model.pt',weights_only=True)
        for key,value in client.model.state_dict().items():
            torch.testing.assert_close(actual[key],value,rtol=1e-6,atol=1e-6)
    rows=parse(tmp_path/'mpi/exchanges.jsonl')
    assert len(rows)==clients*cfg.rounds
    payload=sum(value.numel()*value.element_size() for value in captured[0].model.state_dict().values())
    for step in range(cfg.rounds):
        group=rows[step*clients:(step+1)*clients]
        maximum=max(row['round_seconds'] for row in group)
        assert all(row['max_round_seconds']==maximum for row in group)
        for row in group:
            components=('training_seconds','additional_validation_seconds','training_alignment_seconds',
                        'exchange_seconds','selection_aggregation_seconds','post_aggregation_evaluation_seconds',
                        'logging_gather_seconds','logging_write_seconds','round_completion_wait_seconds')
            assert all(row[key]>=0 for key in components)
            assert row['round_seconds']>=sum(row[key] for key in components)
            assert row['exchange_seconds']>=row['metadata_collective_seconds']+row['model_payload_seconds']+row['synchronization_seconds']
    for row in rows:
        assert row['snapshot_sha256']!=row['pre_training_sha256']
        assert row['source_snapshot_unchanged'] and row['selected_snapshot_reuse']
        assert row['discovery_received_model_bytes']==(clients-1)*payload
        assert row['discovery_sent_model_messages']==row['discovery_received_model_messages']==clients-1
        for peer,digest in zip(row['selected_peers'],row['selected_snapshot_sha256s']):
            source=rows[row['round']*clients+peer]
            assert digest==source['snapshot_sha256']
    summary=json.loads((tmp_path/'mpi/summary.json').read_text())
    assert summary['training'] is True
    assert summary['total_max_round_seconds']==sum(row['max_round_seconds'] for row in summary['round_timings'])
    assert summary['training_alignment_calls']==summary['round_completion_calls']==summary['round_timing_collective_calls']==clients*cfg.rounds
    assert summary['local_steps']==clients*cfg.rounds*cfg.local_steps
    assert summary['aggregation_events']==clients*cfg.rounds
    assert summary['selected_model_contributions']==clients*cfg.rounds*count
    assert summary['discovery_sent_model_messages']==summary['discovery_received_model_messages']==clients*(clients-1)*cfg.rounds
    assert summary['discovery_sent_model_bytes']==summary['discovery_received_model_bytes']==clients*(clients-1)*cfg.rounds*payload
    assert summary['selected_payload_retransmission_messages']==summary['selected_payload_retransmission_bytes']==0
    initial=[row for row in parse(tmp_path/'mpi/evaluations.jsonl') if row['round']==-1]
    assert len({row['test_loss'] for row in initial})==1


def test_mpi_training_cli_reproducible_and_root_only_download(tmp_path):
    import torch
    config=tmp_path/'config.toml'
    config.write_text('num_clients=3\npolicy="closest"\nrounds=2\nlocal_steps=2\nbatch_size=2\ntrain_per_client=5\nvalidation_per_client=3\ntest_samples=7\nhidden_size=4\n')
    cli=FIXTURE+'''
import sys,json
from pathlib import Path
from mpi4py import MPI
import torchvision.datasets
from peershift.cli import main
calls=[]
def mnist(*args,**kwargs):
    calls.append(kwargs['download'])
    return datasets[0 if kwargs['train'] else 1]
torchvision.datasets.MNIST=mnist
sys.argv=['peershift','mpi-train','--config',sys.argv[1],'--output',sys.argv[2],'--download']
main()
Path(sys.argv[sys.argv.index('--output')+1],f'downloads-{MPI.COMM_WORLD.rank}.json').write_text(json.dumps(calls))
'''
    for name in ('a','b'):
        result=launch(cli,config,tmp_path/name,ranks=3)
        assert result.returncode==0,result.stderr
    for name in ('selections','evaluations','aggregation'):
        assert parse(tmp_path/f'a/{name}.jsonl')==parse(tmp_path/f'b/{name}.jsonl')
    stable = lambda rows: [{key: value for key, value in row.items()
                             if not key.endswith('_seconds') and key != 'logging_local_encoded_bytes'} for row in rows]
    assert stable(parse(tmp_path/'a/exchanges.jsonl')) == stable(parse(tmp_path/'b/exchanges.jsonl'))
    for rank in range(3):
        assert json.loads((tmp_path/f'a/downloads-{rank}.json').read_text())==[rank==0]*2
        a=torch.load(tmp_path/f'a/rank-{rank}/final_model.pt',weights_only=True)
        b=torch.load(tmp_path/f'b/rank-{rank}/final_model.pt',weights_only=True)
        assert all(torch.equal(value,b[key]) for key,value in a.items())
    before={path.relative_to(tmp_path/'a'):path.read_bytes() for path in (tmp_path/'a').rglob('*') if path.is_file()}
    result=launch(cli,config,tmp_path/'a',ranks=3)
    assert result.returncode!=0 and 'Output directory must be empty' in result.stderr
    after={path.relative_to(tmp_path/'a'):path.read_bytes() for path in (tmp_path/'a').rglob('*') if path.is_file()}
    assert before==after


@pytest.mark.parametrize('fault', ['rank_count','policy','dataset','training'])
def test_mpi_training_failures_terminate_without_hang(tmp_path,fault):
    config=tmp_path/'config.toml'
    config.write_text('num_clients=3\nrounds=1\nlocal_steps=1\ntrain_per_client=5\nvalidation_per_client=3\ntest_samples=7\nhidden_size=4\n'+ ('policy="random"\n' if fault=='policy' else 'policy="closest"\n'))
    script=FIXTURE+'''
import sys
from mpi4py import MPI
import peershift.mpi_training as training
fault=sys.argv[3]
if fault=='dataset' and MPI.COMM_WORLD.rank==1:
    datasets=(data(123),torch.utils.data.TensorDataset(torch.ones(1,1,28,28),torch.zeros(1,dtype=torch.long)))
if fault=='training' and MPI.COMM_WORLD.rank==1:
    def fail(*args): raise RuntimeError('injected local training failure')
    training.local_train=fail
training.run_mpi_training(sys.argv[1],sys.argv[2],datasets=datasets)
'''
    result=launch(script,config,tmp_path/'out',fault,ranks=2 if fault=='rank_count' else 3,timeout=30)
    assert result.returncode!=0
    expected={'rank_count':'rank count','policy':'only closest','dataset':'Dataset sizes differ','training':'injected local training failure'}[fault]
    assert expected in result.stderr


def test_rank_sample_access_persistent_loader_and_real_toy_learning(tmp_path):
    config=tmp_path/'config.toml'
    config.write_text('num_clients=3\npolicy="closest"\nrounds=3\nlocal_steps=4\nbatch_size=2\ntrain_per_client=5\nvalidation_per_client=3\ntest_samples=7\nhidden_size=4\n')
    script='''
import json,sys,torch
from pathlib import Path
from torch.utils.data import TensorDataset
from mpi4py import MPI
import peershift.mpi_training as training
phase=None
class RecordingData(TensorDataset):
    def __init__(self,seed):
        super().__init__(torch.rand(60,1,28,28,generator=torch.Generator().manual_seed(seed)),torch.zeros(60,dtype=torch.long))
        self.reads=[]
    def __getitem__(self,index):
        self.reads.append((phase,index))
        return super().__getitem__(index)
train,test=RecordingData(123),RecordingData(456)
original_train,original_evaluate=training.local_train,training.evaluate
def train_model(*args,**kwargs):
    global phase
    phase='train'
    try:return original_train(*args,**kwargs)
    finally:phase=None
def evaluate_model(*args,**kwargs):
    global phase
    phase='evaluate'
    try:return original_evaluate(*args,**kwargs)
    finally:phase=None
training.local_train,training.evaluate=train_model,evaluate_model
training.run_mpi_training(sys.argv[1],sys.argv[2],datasets=(train,test))
Path(sys.argv[2],f'rank-{MPI.COMM_WORLD.rank}','access.json').write_text(json.dumps(dict(train=train.reads,test=test.reads)))
'''
    result=launch(script,config,tmp_path/'output',ranks=3,timeout=60)
    assert result.returncode==0,result.stderr
    import torch
    from torch.utils.data import DataLoader,TensorDataset
    from peershift.learning import make_model
    output=tmp_path/'output'
    partitions=json.loads((output/'partitions.json').read_text())
    all_training={index for part in partitions['clients'] for index in part['train']}
    all_validation={index for part in partitions['clients'] for index in part['validation']}
    assert all_training.isdisjoint(all_validation)
    evaluations=parse(output/'evaluations.jsonl')
    torch.manual_seed(42)
    initial=make_model(4).state_dict()
    for rank,part in enumerate(partitions['clients']):
        access=json.loads((output/f'rank-{rank}/access.json').read_text())
        actual_training=[index for phase,index in access['train'] if phase=='train']
        assert actual_training
        assert all(index in part['train'] for index in actual_training)
        assert all(phase in ('train','evaluate') for phase,index in access['train'])
        validation=[index for phase,index in access['train'] if phase=='evaluate']
        assert set(validation)==set(part['validation'])
        assert len(validation)==(1+2*3)*3
        assert all(phase=='evaluate' and index in partitions['test'] for phase,index in access['test'])
        assert len(access['test'])==(1+3)*7
        loader=DataLoader(TensorDataset(torch.tensor(part['train'])),batch_size=2,shuffle=True,
                          generator=torch.Generator().manual_seed(42+rank+1))
        expected=[]
        iterator=iter(loader)
        for _ in range(12):
            try:batch=next(iterator)
            except StopIteration:
                iterator=iter(loader)
                batch=next(iterator)
            expected.extend(batch[0].tolist())
        assert actual_training==expected
        before=next(row for row in evaluations if row['client']==rank and row['round']==-1)
        after=next(row for row in evaluations if row['client']==rank and row['round']==2)
        assert after['test_loss']<before['test_loss']
        state=torch.load(output/f'rank-{rank}/final_model.pt',weights_only=True)
        assert any(not torch.equal(value,initial[key]) for key,value in state.items())


@pytest.mark.parametrize('failure', ['missing_cache','download_failure'])
def test_root_dataset_failures_exit_without_partial_outputs(tmp_path,failure):
    config=tmp_path/'config.toml'
    config.write_text('num_clients=3\npolicy="closest"\nrounds=1\nlocal_steps=1\n'
                      f'data_dir={json.dumps(str(tmp_path/"missing"))}\n')
    script='''
import sys
from mpi4py import MPI
import peershift.mpi_training as training
failure=sys.argv[3]
if failure=='download_failure':
    def fail(*args,**kwargs):
        assert MPI.COMM_WORLD.rank==0
        assert kwargs['download'] is True
        raise RuntimeError('injected root download failure')
    training.load_datasets=fail
training.run_mpi_training(sys.argv[1],sys.argv[2],download=failure=='download_failure')
'''
    result=launch(script,config,tmp_path/'output',failure,ranks=3,timeout=30)
    assert result.returncode!=0
    expected='Dataset not found' if failure=='missing_cache' else 'injected root download failure'
    assert expected in result.stderr
    assert not (tmp_path/'output').exists()
