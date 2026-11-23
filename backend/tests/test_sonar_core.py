"""Independent contracts for the SONAR-style closest/merge baseline."""
import json
import math

import pytest

torch = pytest.importorskip('torch')
from torch.utils.data import TensorDataset
from peershift.detector import collaboration_metrics
from peershift.learning import aggregate_peers, similarities, snapshot
from peershift.selection import closest_select
from peershift.training import TrainingConfig, run_training


def test_closest_select_returns_actual_collaborators_with_stable_ties():
    assert closest_select(0, [100., .9, .8, .1], 2) == (1, 2)
    assert closest_select(1, [1., 100., 1., 1.], 2) == (0, 2)
    assert closest_select(0, [100., .9, .8, .1], 3) == (1, 2, 3)


@pytest.mark.parametrize('count', [0, True, 3])
def test_closest_rejects_invalid_counts(count):
    with pytest.raises(ValueError):
        closest_select(0, [1., .5, .2], count)


def test_metric_formulas_can_produce_different_nearest_peers():
    states = [dict(a=torch.tensor([a]), b=torch.tensor([b]))
              for a, b in [(0., 0.), (3., 0.), (2., 2.)]]
    flat = similarities(states, 'flat_l2')
    summed = similarities(states, 'sum_tensor_l2')
    assert flat[0][1] == -3
    assert flat[0][2] == pytest.approx(-math.sqrt(8))
    assert summed[0] == [0., -3., -4.]
    assert closest_select(0, flat[0], 1) == (2,)
    assert closest_select(0, summed[0], 1) == (1,)
    for i, state in enumerate(states):
        state['num_batches_tracked'] = torch.tensor(i*100000)
    assert similarities(states, 'sum_tensor_l2') == summed


@pytest.mark.parametrize('rule,expected', [('uniform_self_and_peers', (4., 6.)),
                                          ('fixed_self', (3.25, 5.25))])
def test_multi_peer_average_is_analytic_and_preserves_sources(rule, expected):
    model = torch.nn.Linear(1, 1)
    own = dict(weight=torch.tensor([[1.]]), bias=torch.tensor([3.]))
    peers = [dict(weight=torch.tensor([[3.]]), bias=torch.tensor([9.])),
             dict(weight=torch.tensor([[8.]]), bias=torch.tensor([6.]))]
    copies = [{k: v.clone() for k, v in state.items()} for state in (own, *peers)]
    aggregate_peers(model, own, peers, rule=rule)
    assert model.weight.item() == pytest.approx(expected[0])
    assert model.bias.item() == pytest.approx(expected[1])
    for state, original in zip((own, *peers), copies):
        for name in state:
            assert torch.equal(state[name], original[name])
            assert model.state_dict()[name].data_ptr() != state[name].data_ptr()


def test_multi_peer_history_counts_rounds_and_canonical_sets():
    assert collaboration_metrics([], 3) == dict(history_rounds=0, peer_selection_entropy=0.,
        adjacent_peer_set_repetition_rate=0., unique_collaborators=0)
    result = collaboration_metrics([(1, 2), (2, 1)], 3)
    assert result['history_rounds'] == 2
    assert result['peer_selection_entropy'] == pytest.approx(math.log(2)/math.log(3))
    assert result['adjacent_peer_set_repetition_rate'] == 1
    assert result['unique_collaborators'] == 2


@pytest.mark.parametrize('values', [dict(num_collaborators=0), dict(num_collaborators=True),
    dict(num_collaborators=10), dict(policy='random', num_collaborators=2),
    dict(similarity_metric='unknown'), dict(aggregation_rule='unknown')])
def test_training_rejects_ambiguous_collaboration_settings(values):
    with pytest.raises(ValueError):
        TrainingConfig(**values)


def test_closest_training_logs_joint_events_and_correct_budgets(tmp_path):
    train = TensorDataset(torch.rand(40, 1, 28, 28, generator=torch.Generator().manual_seed(123)),
                          torch.arange(40)%10)
    test = TensorDataset(torch.rand(10, 1, 28, 28, generator=torch.Generator().manual_seed(456)),
                         torch.arange(10)%10)
    config = TrainingConfig(num_clients=3, rounds=3, local_steps=2, batch_size=2,
        train_per_client=4, validation_per_client=2, test_samples=5, hidden_size=4,
        policy='closest', num_collaborators=2, similarity_metric='sum_tensor_l2',
        aggregation_rule='uniform_self_and_peers', history_window=2)
    for name in ('a', 'b'):
        run_training(config, tmp_path/name, datasets=(train,test))
    assert TrainingConfig.load(tmp_path/'a/config.toml') == config
    for artifact in ('selections.jsonl', 'aggregation.jsonl', 'evaluations.jsonl', 'partitions.json'):
        assert (tmp_path/'a'/artifact).read_bytes() == (tmp_path/'b'/artifact).read_bytes()
    selected = [json.loads(line) for line in (tmp_path/'a/selections.jsonl').read_text().splitlines()]
    observations = [json.loads(line) for line in (tmp_path/'a/aggregation.jsonl').read_text().splitlines()]
    assert len(selected) == len(observations) == 9
    for choice, observation in zip(selected, observations):
        assert 'selected_peer' not in choice and 'selected_peer_distance' not in observation
        assert set(choice['selected_peers']) == set(range(3)) - {choice['client']}
        assert len(choice['selected_peer_distances']) == 2
        assert choice['selected_peers'] == observation['selected_peers']
        assert choice['self_weight'] == pytest.approx(1/3)
        assert choice['selected_peer_weights'] == pytest.approx([1/3,1/3])
        assert observation['benefit_scope'] == 'joint_aggregation'
        assert observation['history_rounds_before_selection'] == min(choice['round'],2)
        assert choice['history_rounds'] == min(choice['round']+1,2)
    summary = json.loads((tmp_path/'a/summary.json').read_text())
    assert summary['aggregation_events'] == 9
    assert summary['logical_model_exchanges'] == 18
    assert summary['local_steps'] == 18
    assert sum(summary[key] for key in ('improved','worsened','unchanged')) == 9
    reference_model = torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(784,4),
                                          torch.nn.ReLU(),torch.nn.Linear(4,10))
    model_bytes = sum(t.numel()*t.element_size() for t in snapshot(reference_model).values())
    assert summary['estimated_parameter_payload_bytes'] == 18*model_bytes
    text = (tmp_path/'a/summary.txt').read_text()
    for label in ('improved','worsened','unchanged'):
        count = summary[label]
        assert f'{label.capitalize()}: {count} ({100*count/9:.1f}%)' in text


@pytest.mark.parametrize('rule', ['uniform_self_and_peers', 'fixed_self'])
def test_four_client_final_tensors_match_independent_multi_peer_rounds(tmp_path, monkeypatch, rule):
    """Four clients/two peers avoids a trivial all-neighbor merging reference."""
    from copy import deepcopy
    from itertools import cycle
    import random
    from torch.utils.data import DataLoader, Subset
    import peershift.training as training

    cfg = TrainingConfig(num_clients=4, rounds=3, local_steps=2, batch_size=2,
        train_per_client=4, validation_per_client=2, test_samples=5, hidden_size=4,
        policy='closest', num_collaborators=2, similarity_metric='sum_tensor_l2',
        aggregation_rule=rule, own_weight=.25, history_window=2)
    train = TensorDataset(torch.rand(40,1,28,28,generator=torch.Generator().manual_seed(123)), torch.arange(40)%10)
    test = TensorDataset(torch.rand(10,1,28,28,generator=torch.Generator().manual_seed(456)), torch.arange(10)%10)
    actual, expected = [], []
    original = training.LearningClient
    def capture(*args, **kwargs):
        client = original(*args, **kwargs)
        actual.append(client)
        expected.append(deepcopy(client.model))
        return client
    monkeypatch.setattr(training, 'LearningClient', capture)
    training.run_training(cfg, tmp_path, datasets=(train,test))
    partitions = json.loads((tmp_path/'partitions.json').read_text())['clients']
    # Recreate the actual seeded shuffled minibatch streams, including reshuffles.
    def batches(loader):
        while True:
            yield from loader
    streams = [batches(DataLoader(Subset(train, part['train']), batch_size=cfg.batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(cfg.seed+i+1)))
        for i, part in enumerate(partitions)]
    optimizers = [torch.optim.SGD(model.parameters(), lr=cfg.learning_rate) for model in expected]
    expected_choices = []
    for _ in range(cfg.rounds):
        for model, optimizer, stream in zip(expected,optimizers,streams):
            model.train()
            for _ in range(cfg.local_steps):
                x,y = next(stream)
                optimizer.zero_grad()
                torch.nn.functional.cross_entropy(model(x),y).backward()
                optimizer.step()
        states = [{k:v.detach().clone() for k,v in model.state_dict().items()} for model in expected]
        choices = [sorted((j for j in range(4) if j!=i), key=lambda j:(
            sum(torch.linalg.vector_norm(states[i][k]-states[j][k]).item() for k in states[i]),j))[:2]
            for i in range(4)]
        expected_choices.extend(choices)
        own_weight = 1/3 if rule=='uniform_self_and_peers' else .25
        peer_weight = 1/3 if rule=='uniform_self_and_peers' else .375
        for i,model in enumerate(expected):
            model.load_state_dict({k:states[i][k]*own_weight + states[choices[i][0]][k]*peer_weight
                + states[choices[i][1]][k]*peer_weight for k in states[i]})
    selected = [json.loads(line) for line in (tmp_path/'selections.jsonl').read_text().splitlines()]
    assert [row['selected_peers'] for row in selected] == expected_choices
    for client, reference in zip(actual,expected):
        for key,value in client.model.state_dict().items():
            assert torch.equal(value,reference.state_dict()[key]), key
    # Check history from independently sliced round groups, never flatten then evict.
    for i in range(4):
        by_client = selected[i::4]
        for round_index,row in enumerate(by_client):
            history = [tuple(sorted(r['selected_peers'])) for r in by_client[max(0,round_index-1):round_index+1]]
            peers = [p for group in history for p in group]
            entropy = -sum((peers.count(p)/len(peers))*math.log(peers.count(p)/len(peers)) for p in set(peers))/math.log(3)
            assert row['history_rounds']==len(history)
            assert row['peer_selection_entropy']==pytest.approx(entropy)
            assert row['unique_collaborators']==len(set(peers))
            assert row['adjacent_peer_set_repetition_rate']==(float(history[0]==history[1]) if len(history)==2 else 0.)
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert summary['aggregation_events']==12
    assert summary['logical_model_exchanges']==24
    assert summary['local_steps']==24
    assert sum(summary[key] for key in ('improved','worsened','unchanged'))==12


@pytest.mark.parametrize('metric', ['flat_l2', 'sum_tensor_l2'])
def test_one_collaborator_uniform_half_matches_top_k_one(tmp_path, monkeypatch, metric):
    import peershift.training as training
    captured=[]
    original=training.LearningClient
    def capture(*args,**kwargs):
        client=original(*args,**kwargs)
        captured.append(client)
        return client
    monkeypatch.setattr(training,'LearningClient',capture)
    data=tuple(TensorDataset(torch.rand(30,1,28,28,generator=torch.Generator().manual_seed(seed)),torch.arange(30)%10)
               for seed in (123,456))
    common=dict(num_clients=4,k=1,rounds=3,local_steps=2,batch_size=2,train_per_client=4,
        validation_per_client=2,test_samples=5,hidden_size=4,similarity_metric=metric)
    training.run_training(TrainingConfig(**common),tmp_path/'top',datasets=data)
    training.run_training(TrainingConfig(**common,policy='closest',aggregation_rule='uniform_self_and_peers'),tmp_path/'closest',datasets=data)
    top=[json.loads(line) for line in (tmp_path/'top/selections.jsonl').read_text().splitlines()]
    closest=[json.loads(line) for line in (tmp_path/'closest/selections.jsonl').read_text().splitlines()]
    assert [[r['selected_peer']] for r in top]==[r['selected_peers'] for r in closest]
    assert (tmp_path/'top/evaluations.jsonl').read_bytes()==(tmp_path/'closest/evaluations.jsonl').read_bytes()
    for first,second in zip(captured[:4],captured[4:]):
        assert all(torch.equal(value,second.model.state_dict()[key]) for key,value in first.model.state_dict().items())


@pytest.mark.parametrize('fault', ['missing_key','shape','dtype','nonfinite','non_tensor'])
def test_merge_and_summed_metric_reject_bad_snapshot_without_mutation(fault):
    model=torch.nn.Linear(2,1)
    own=snapshot(model)
    bad={k:v.clone() for k,v in own.items()}
    if fault=='missing_key': del bad['weight']
    if fault=='shape': bad['weight']=torch.ones(1,3)
    if fault=='dtype': bad['weight']=bad['weight'].double()
    if fault=='nonfinite': bad['weight'].fill_(float('nan'))
    if fault=='non_tensor': bad['weight']=[1,2]
    with pytest.raises(ValueError):
        aggregate_peers(model,own,[bad])
    with pytest.raises(ValueError):
        similarities([own,bad],metric='sum_tensor_l2')
    assert all(torch.equal(value,own[key]) for key,value in model.state_dict().items())


def test_closest_public_cli_is_reproducible_across_processes(tmp_path):
    import subprocess
    import sys
    config=tmp_path/'config.toml'
    config.write_text('num_clients=4\nrounds=2\nlocal_steps=1\nbatch_size=2\ntrain_per_client=4\nvalidation_per_client=2\ntest_samples=4\nhidden_size=4\npolicy="closest"\nnum_collaborators=2\nsimilarity_metric="sum_tensor_l2"\naggregation_rule="uniform_self_and_peers"\n')
    script='''
import sys,torch
from torch.utils.data import TensorDataset
import torchvision.datasets
from peershift.cli import main
sources=[TensorDataset(torch.rand(32,1,28,28,generator=torch.Generator().manual_seed(seed)),torch.arange(32)%10) for seed in (123,456)]
torchvision.datasets.MNIST=lambda *args,**kwargs:sources[0 if kwargs['train'] else 1]
sys.argv=['peershift','train','--config',sys.argv[1],'--output',sys.argv[2]]
main()
'''
    for name in ('one','two'):
        subprocess.run([sys.executable,'-c',script,str(config),str(tmp_path/name)],check=True,capture_output=True,timeout=30)
    for artifact in ('selections.jsonl','aggregation.jsonl','evaluations.jsonl','partitions.json'):
        assert (tmp_path/'one'/artifact).read_bytes()==(tmp_path/'two'/artifact).read_bytes()
