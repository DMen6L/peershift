"""Check diagnostic meaning independently of Git history and evaluation order."""
from copy import deepcopy
from itertools import cycle
import json
import math
import random

import pytest

torch = pytest.importorskip('torch')
from torch.utils.data import TensorDataset, DataLoader, Subset
import peershift.training as training
from peershift.selection import select_peer


def datasets():
    return tuple(TensorDataset(
        torch.rand(30, 1, 28, 28, generator=torch.Generator().manual_seed(seed)),
        torch.arange(30) % 10,
    ) for seed in (123, 456))


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def config(**extra):
    values = dict(num_clients=3, k=1, rounds=2, local_steps=1, batch_size=2,
                  train_per_client=4, validation_per_client=2, test_samples=5, hidden_size=4)
    return training.TrainingConfig(**(values | extra))


@pytest.mark.parametrize('peer_margin,expected_sign', [(2., 1), (-2., -1), (0., 0)])
def test_real_validation_benefit(peer_margin, expected_sign):
    # Label zero: raising its logit helps, lowering it hurts, equal weights do nothing.
    own = torch.nn.Linear(1, 2, bias=False, dtype=torch.float64)
    peer = deepcopy(own)
    with torch.no_grad():
        own.weight.zero_()
        peer.weight.copy_(torch.tensor([[peer_margin], [0.]], dtype=torch.float64))
    validation = DataLoader(TensorDataset(torch.ones(4, 1, dtype=torch.float64),
                                         torch.zeros(4, dtype=torch.long)), batch_size=2)
    before, _ = training.evaluate(own, validation)
    training.aggregate(own, training.snapshot(own), training.snapshot(peer), .5)
    after, _ = training.evaluate(own, validation)
    expected = math.log(2) - math.log1p(math.exp(-peer_margin/2))
    assert before - after == pytest.approx(expected, abs=1e-12)
    assert ((before - after > 0) - (before - after < 0)) == expected_sign


@pytest.mark.parametrize('benefit,classification', [
    (0., 'unchanged'),
    (math.nextafter(1e-8, 0.), 'unchanged'),
    (-math.nextafter(1e-8, 0.), 'unchanged'),
    (1e-8, 'unchanged'),
    (-1e-8, 'unchanged'),
    (math.nextafter(1e-8, math.inf), 'improved'),
    (-math.nextafter(1e-8, math.inf), 'worsened'),
])
def test_summary_tolerance_boundaries(benefit, classification):
    result = training._benefit_summary([benefit])
    assert result['mean_aggregation_benefit'] == benefit
    assert {name: result[name] for name in ('improved', 'worsened', 'unchanged')} == {
        name: int(name == classification) for name in ('improved', 'worsened', 'unchanged')
    }


def test_diagnostic_arithmetic_and_readable_summary(tmp_path):
    cfg = config()
    training.run_training(cfg, tmp_path, datasets=datasets())
    rows = records(tmp_path/'aggregation.jsonl')
    evaluations = {(r['round'], r['client']): r for r in records(tmp_path/'evaluations.jsonl')}
    benefits = []
    for row in rows:
        assert row['validation_loss_after_aggregation'] == evaluations[row['round'], row['client']]['validation_loss']
        benefit = row['validation_loss_before_aggregation'] - row['validation_loss_after_aggregation']
        assert row['aggregation_benefit'] == benefit
        assert math.isfinite(benefit)
        benefits.append(benefit)
    counts = dict(improved=sum(b > 1e-8 for b in benefits),
                  worsened=sum(b < -1e-8 for b in benefits),
                  unchanged=sum(abs(b) <= 1e-8 for b in benefits))
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert summary['mean_aggregation_benefit'] == pytest.approx(sum(benefits)/len(benefits))
    assert summary['logical_model_exchanges'] == len(rows) == cfg.num_clients*cfg.rounds
    text = (tmp_path/'summary.txt').read_text()
    for label, count in counts.items():
        assert summary[label] == count
        assert f'{label.capitalize()}: {count} ({100*count/len(rows):.1f}%)' in text


def test_own_weight_and_rolling_prior_history(tmp_path, monkeypatch):
    # Known choices exercise warmup, repetition, and eviction after a two-peer window.
    choices = (1, 1, 2, 1, 2)
    calls = {i: 0 for i in range(3)}

    # Only partner choices are controlled; real training, averaging and evaluation run.
    from peershift.selection import SelectionDecision
    def choose(client_id, *args, **kwargs):
        eligible = [i for i in range(3) if i != client_id]
        position = calls[client_id]
        calls[client_id] += 1
        return SelectionDecision(eligible[choices[position]-1], 'top_k')

    monkeypatch.setattr(training, 'select_peer', choose)
    cfg = config(own_weight=1, rounds=5, history_window=2)
    training.run_training(cfg, tmp_path, datasets=datasets())
    rows = records(tmp_path/'aggregation.jsonl')
    expected = [(0, 0, 0., 0.), (1, 1, 0., 0.), (2, 1, 0., 1.),
                (2, 2, 1., 0.), (2, 2, 1., 0.)]
    for row in rows:
        assert row['aggregation_benefit'] == pytest.approx(0., abs=1e-12)
        assert math.isfinite(row['selected_peer_distance']) and row['selected_peer_distance'] >= 0
        length, unique, entropy, repetition = expected[row['round']]
        assert row['history_length_before_selection'] == length
        assert row['unique_peers_before_selection'] == unique
        assert row['entropy_before_selection'] == pytest.approx(entropy)
        assert row['repetition_rate_before_selection'] == repetition
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert summary['unchanged'] == 15
    assert summary['additional_validation_seconds'] >= 0


@pytest.mark.parametrize('policy', ['top_k', 'random', 'fixed_exploration'])
def test_final_tensors_match_reference_gossip_without_evaluations(tmp_path, monkeypatch, policy):
    """Reference the mathematical round contract, never the current Git commit."""
    cfg = config(policy=policy, local_steps=2, rounds=3)
    train_data, test_data = datasets()
    actual_clients, reference_models = [], []
    client_type = training.LearningClient

    def capture_client(*args, **kwargs):
        client = client_type(*args, **kwargs)
        actual_clients.append(client)
        reference_models.append(deepcopy(client.model))
        return client

    # Fix minibatch order for this averaging test; CLI tests cover seeded shuffling.
    def fixed_batches(loader):
        return cycle(DataLoader(loader.dataset, batch_size=loader.batch_size, shuffle=False))

    monkeypatch.setattr(training, 'LearningClient', capture_client)
    monkeypatch.setattr(training, 'cycle', fixed_batches)
    training.run_training(cfg, tmp_path, datasets=(train_data, test_data))
    for model in reference_models[1:]:
        for key, value in reference_models[0].state_dict().items():
            assert torch.equal(value, model.state_dict()[key])
    assert len({client.model[1].weight.data_ptr() for client in actual_clients}) == cfg.num_clients
    partitions = json.loads((tmp_path/'partitions.json').read_text())['clients']
    batches = [cycle(DataLoader(Subset(train_data, part['train']), batch_size=cfg.batch_size))
               for part in partitions]
    optimizers = [torch.optim.SGD(model.parameters(), lr=cfg.learning_rate) for model in reference_models]
    streams = [random.Random(f'{cfg.seed}:client:{i}') for i in range(cfg.num_clients)]
    exploration_streams = [random.Random(f'{cfg.seed}:exploration:{i}') for i in range(cfg.num_clients)]
    expected_peers = []
    for _ in range(cfg.rounds):
        for model, optimizer, batch_stream in zip(reference_models, optimizers, batches):
            model.train()
            for _ in range(cfg.local_steps):
                x, y = next(batch_stream)
                optimizer.zero_grad()
                torch.nn.functional.cross_entropy(model(x), y).backward()
                optimizer.step()
        states = [{k: v.detach().clone() for k, v in model.state_dict().items()}
                  for model in reference_models]
        vectors = [torch.cat([v.flatten() for v in state.values()]) for state in states]
        peers = [select_peer(i, [-torch.linalg.vector_norm(vectors[i]-v).item() for v in vectors],
                             cfg.k, policy=policy, rng=streams[i],
                             exploration_probability=cfg.exploration_probability,
                             exploration_rng=exploration_streams[i]).selected_peer
                 for i in range(cfg.num_clients)]
        expected_peers.extend(peers)
        for i, model in enumerate(reference_models):
            model.load_state_dict({k: cfg.own_weight*v + (1-cfg.own_weight)*states[peers[i]][k]
                                   for k, v in states[i].items()})
    assert [r['selected_peer'] for r in records(tmp_path/'selections.jsonl')] == expected_peers
    for client, reference in zip(actual_clients, reference_models):
        for key, tensor in client.model.state_dict().items():
            assert torch.equal(tensor, reference.state_dict()[key]), key
