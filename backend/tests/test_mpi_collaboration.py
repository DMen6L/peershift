"""Real multi-process collaboration and independent serial-reference checks."""
import json
import math

import pytest

pytest.importorskip('torch')
pytest.importorskip('mpi4py')
from test_mpi_transport import launch

pytestmark = pytest.mark.mpi
CLI = '''
from peershift.cli import main
import sys
sys.argv=['peershift','mpi-collaborate','--config',sys.argv[1],'--output',sys.argv[2]]
main()
'''


@pytest.mark.parametrize('metric', ['flat_l2', 'sum_tensor_l2'])
@pytest.mark.parametrize('rule', ['uniform_self_and_peers', 'fixed_self'])
def test_collaboration_matches_independent_serial_reference(tmp_path, metric, rule):
    import torch
    from peershift.learning import make_model
    config = tmp_path/'config.toml'
    config.write_text(f'num_clients=4\nrounds=2\nhidden_size=4\nhistory_window=1\n'
                      f'similarity_metric="{metric}"\naggregation_rule="{rule}"\nown_weight=0.7\n')
    result = launch(CLI, config, tmp_path/'output', ranks=4)
    assert result.returncode == 0, result.stderr
    output = tmp_path/'output'
    records = [json.loads(row) for row in (output/'collaborations.jsonl').read_text().splitlines()]
    assert len(records) == 8
    states = []
    for rank in range(4):
        torch.manual_seed(42+rank)
        states.append({key: value.detach().clone() for key, value in make_model(4).state_dict().items()})
    payload_bytes = sum(value.numel()*value.element_size() for value in states[0].values())
    for round_index in range(2):
        updated = []
        for rank, own in enumerate(states):
            def distance(peer):
                diffs = [own[key]-states[peer][key] for key in sorted(own)]
                if metric == 'flat_l2':
                    return torch.linalg.vector_norm(torch.cat([value.flatten() for value in diffs])).item()
                return sum(torch.linalg.vector_norm(value).item() for value in diffs)
            selected = sorted((peer for peer in range(4) if peer != rank), key=lambda peer: (distance(peer), peer))[:2]
            row = records[4*round_index+rank]
            assert row['rank'] == rank and row['round'] == round_index
            assert row['selected_peers'] == selected
            assert row['selected_peer_distances'] == pytest.approx([distance(peer) for peer in selected], abs=1e-6)
            own_weight = 1/3 if rule == 'uniform_self_and_peers' else .7
            peer_weight = 1/3 if rule == 'uniform_self_and_peers' else .15
            merged = {key: own_weight*own[key]+peer_weight*states[selected[0]][key]+peer_weight*states[selected[1]][key]
                      for key in own}
            updated.append(merged)
            assert row['self_weight'] == own_weight
            assert row['selected_peer_weights'] == pytest.approx([peer_weight]*2)
            assert row['discovery_sent_model_messages'] == row['discovery_received_model_messages'] == 3
            assert row['discovery_sent_model_bytes'] == row['discovery_received_model_bytes'] == 3*payload_bytes
            assert row['selected_payload_retransmission_bytes'] == row['selected_payload_retransmission_messages'] == 0
            assert row['source_snapshot_unchanged'] and row['selected_snapshot_reuse']
            assert row['history_rounds'] == 1 and row['unique_collaborators'] == 2
            assert row['history_rounds_before_selection'] == min(round_index, 1)
            assert row['peer_selection_entropy'] == pytest.approx(math.log(2)/math.log(3))
        states = updated
    for rank, expected in enumerate(states):
        actual = torch.load(output/f'rank-{rank}/final_model.pt', weights_only=True)
        assert actual.keys() == expected.keys()
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=1e-6, atol=1e-6)
    summary = json.loads((output/'summary.json').read_text())
    assert summary['aggregation_events'] == 8
    assert summary['selected_model_contributions'] == 16
    assert summary['discovery_sent_model_messages'] == summary['discovery_received_model_messages'] == 24
    assert summary['discovery_sent_model_bytes'] == summary['discovery_received_model_bytes'] == 24*payload_bytes
    assert summary['training'] is False
    rejected = launch(CLI, config, output, ranks=4)
    assert rejected.returncode != 0 and 'Output directory must be empty' in rejected.stderr


def test_multi_peer_transport_preserves_order_and_large_frozen_buffers():
    script = '''
import torch
from peershift.mpi_transport import MPITransport
transport=MPITransport()
rank=transport.rank
choices=[[2,1],[0,2],[1,0],[0,1]]
for step in range(2):
    state={'x':torch.full((600000,),float(rank+step*10))}
    received,metrics=transport.exchange_many(state,choices[rank],step)
    assert len(received)==2
    for peer,snapshot in zip(choices[rank],received):
        assert torch.all(snapshot['x']==peer+step*10)
    assert torch.all(state['x']==rank+step*10)
    assert metrics['received_model_bytes']==4800000
    assert metrics['sent_model_messages']==sum(rank in peers for peers in choices)
    assert metrics['selected_peers']==choices[rank]
    assert transport.next_round==step+1
transport.close()
'''
    result = launch(script, ranks=4)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('bad', ['[1,1]', '[]', '[True]', '[4]', '[0]'])
def test_invalid_multi_peer_inputs_fail_collectively(bad):
    script = '''
import sys,torch
from peershift.mpi_transport import MPITransport
transport=MPITransport()
peers=[(transport.rank+1)%transport.size]
if transport.rank==0: peers=eval(sys.argv[1],{'__builtins__':{}},{})
try:
    transport.exchange_many({'x':torch.ones(2)},peers,0)
except ValueError:
    print('REJECTED',flush=True)
else:
    transport.comm.Abort(2)
transport.close()
'''
    result = launch(script, bad, ranks=4)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count('REJECTED') == 4


@pytest.mark.parametrize('count', [1, 3])
def test_tied_scores_choose_lowest_ids_at_collaborator_boundaries(tmp_path, count):
    import torch
    from peershift.learning import make_model
    config=tmp_path/'config.toml'
    config.write_text(f'num_clients=4\nrounds=2\nhidden_size=4\nnum_collaborators={count}\n')
    script = '''
import sys
import peershift.mpi_collaboration as collaboration
original=collaboration.fixture_model
collaboration.fixture_model=lambda config,rank:original(config,0)
collaboration.run_collaboration(sys.argv[1],sys.argv[2])
'''
    result=launch(script,config,tmp_path/'output',ranks=4)
    assert result.returncode==0,result.stderr
    rows=[json.loads(line) for line in (tmp_path/'output/collaborations.jsonl').read_text().splitlines()]
    torch.manual_seed(42)
    initial=make_model(4).state_dict()
    for row in rows:
        assert row['selected_peers']==[peer for peer in range(4) if peer!=row['rank']][:count]
        # Uniform addition may round identical parameters at m=3, so later
        # distances are checked separately from first-round exact ties.
        if row['round']==0:
            assert row['selected_peer_distances']==[0.]*count
        assert row['self_weight']==1/(count+1)
        assert row['selected_peer_weights']==[1/(count+1)]*count
    for rank in range(4):
        state=torch.load(tmp_path/f'output/rank-{rank}/final_model.pt',weights_only=True)
        for key in initial:
            torch.testing.assert_close(state[key],initial[key],rtol=1e-6,atol=1e-7)
    summary=json.loads((tmp_path/'output/summary.json').read_text())
    assert summary['selected_model_contributions']==8*count
    assert summary['discovery_sent_model_messages']==24


def test_collaboration_reproducibility_and_output_protection(tmp_path):
    import torch
    from peershift.model_buffer import pack_state
    config=tmp_path/'config.toml'
    config.write_text('num_clients=4\nrounds=2\nhidden_size=4\nhistory_window=1\n')
    for name in ('a','b'):
        result=launch(CLI,config,tmp_path/name,ranks=4)
        assert result.returncode==0,result.stderr
    parse=lambda name:[json.loads(line) for line in (tmp_path/name/'collaborations.jsonl').read_text().splitlines()]
    # Sizes of logged timing strings vary along with the timings themselves.
    stable=lambda rows:[{key:value for key,value in row.items()
                        if not key.endswith('_seconds') and key!='logging_local_encoded_bytes'} for row in rows]
    assert stable(parse('a'))==stable(parse('b'))
    for rank in range(4):
        first=torch.load(tmp_path/f'a/rank-{rank}/final_model.pt',weights_only=True)
        second=torch.load(tmp_path/f'b/rank-{rank}/final_model.pt',weights_only=True)
        assert all(torch.equal(value,second[key]) for key,value in first.items())
        assert pack_state(first).digest==parse('a')[-4+rank]['averaged_sha256']
    before={path.relative_to(tmp_path/'a'):path.read_bytes() for path in (tmp_path/'a').rglob('*') if path.is_file()}
    result=launch(CLI,config,tmp_path/'a',ranks=4)
    assert result.returncode!=0 and 'Output directory must be empty' in result.stderr
    after={path.relative_to(tmp_path/'a'):path.read_bytes() for path in (tmp_path/'a').rglob('*') if path.is_file()}
    assert before==after


@pytest.mark.parametrize('settings,ranks,error', [
    ('num_clients=4\n',3,'MPI rank count must equal num_clients'),
    ('num_clients=4\nnum_collaborators=4\n',4,'fewer collaborators than clients'),
    ('num_clients=4\npolicy="random"\n',4,'supports only closest'),
])
def test_collaboration_setup_errors_exit_without_hanging(tmp_path,settings,ranks,error):
    config=tmp_path/'config.toml'
    config.write_text(settings)
    result=launch(CLI,config,tmp_path/'output',ranks=ranks)
    assert result.returncode!=0
    assert error in result.stderr
    assert not (tmp_path/'output').exists()


def test_variable_peer_counts_tuple_inputs_independent_receives_and_wrapper():
    script='''
import torch
from peershift.mpi_transport import MPITransport
transport=MPITransport()
rank=transport.rank
choices=((2,1),(0,),(3,0,1),(1,0))
state={'x':torch.arange(4,dtype=torch.float32)+rank*10}
received,metrics=transport.exchange_many(state,choices[rank],0)
assert isinstance(received,tuple)
assert len(received)==len(choices[rank])
assert metrics['received_model_messages']==len(choices[rank])
assert metrics['received_model_bytes']==16*len(choices[rank])
assert metrics['sent_model_messages']==sum(rank in peers for peers in choices)
for peer,model in zip(choices[rank],received):
    assert torch.equal(model['x'],torch.arange(4,dtype=torch.float32)+peer*10)
    assert model['x'].data_ptr()!=state['x'].data_ptr()
assert len({model['x'].data_ptr() for model in received})==len(received)
if len(received)>1:
    other=received[1]['x'].clone()
    received[0]['x'].zero_()
    assert torch.equal(received[1]['x'],other)
assert torch.equal(state['x'],torch.arange(4,dtype=torch.float32)+rank*10)
peer=(rank+1)%4
single,metrics=transport.exchange(state,peer,1)
assert torch.equal(single['x'],torch.arange(4,dtype=torch.float32)+peer*10)
assert metrics['selected_peer']==peer and 'selected_peers' not in metrics
assert 'received_sha256' in metrics and 'received_sha256s' not in metrics
transport.close()
'''
    result=launch(script,ranks=4)
    assert result.returncode==0,result.stderr
