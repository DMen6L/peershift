import importlib.util
import json
import subprocess
import sys

import pytest

torch = pytest.importorskip('torch')
from torch.utils.data import TensorDataset
import peershift.training as training


def data():
    return TensorDataset(torch.rand(30, 1, 28, 28, generator=torch.Generator().manual_seed(123)), torch.arange(30) % 10)


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def config(**extra):
    return training.TrainingConfig(num_clients=3, k=1, rounds=2, local_steps=1,
        batch_size=2, train_per_client=4, validation_per_client=2, test_samples=5,
        hidden_size=4, **extra)


def test_benefit_signs_and_summary(tmp_path, monkeypatch):
    # Initial validation/test pairs, pre-aggregation validation, then post pairs.
    losses = iter([2]*6 + [3,1,2] + [2]*6)
    monkeypatch.setattr(training, 'evaluate', lambda *args: (next(losses), .5))
    cfg = training.TrainingConfig(num_clients=3,k=1,rounds=1,local_steps=1,
        batch_size=2,train_per_client=4,validation_per_client=2,test_samples=5,hidden_size=4)
    fixture = data()
    training.run_training(cfg, tmp_path, datasets=(fixture,fixture))
    rows = records(tmp_path/'aggregation.jsonl')
    assert [r['aggregation_benefit'] for r in rows] == [1,-1,0]
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert [summary[k] for k in ('improved','worsened','unchanged')] == [1,1,1]
    assert summary['mean_aggregation_benefit'] == 0
    assert summary['logical_model_exchanges'] == len(rows) == 3
    text = (tmp_path/'summary.txt').read_text()
    assert all(f'{label}: 1 (33.3%)' in text for label in ('Improved','Worsened','Unchanged'))


def test_own_weight_and_prior_history(tmp_path):
    fixture = data()
    training.run_training(config(own_weight=1), tmp_path, datasets=(fixture,fixture))
    rows = records(tmp_path/'aggregation.jsonl')
    selections = records(tmp_path/'selections.jsonl')
    assert len(rows) == 6
    assert all(r['aggregation_benefit'] == 0 for r in rows)
    for row in rows:
        assert row['selected_peer_distance'] >= 0
        assert row['history_length_before_selection'] == row['round']
        if row['round'] == 0:
            assert row['unique_peers_before_selection'] == 0
            assert row['entropy_before_selection'] == 0
        else:
            previous = selections[row['client']]
            assert row['unique_peers_before_selection'] == previous['unique_peers']
            assert row['entropy_before_selection'] == previous['entropy']
            assert row['repetition_rate_before_selection'] == previous['repetition_rate']
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert summary['unchanged'] == 6
    assert summary['additional_validation_seconds'] >= 0


def test_instrumentation_preserves_baseline(tmp_path):
    source = subprocess.check_output(['git','show','HEAD:backend/src/peershift/training.py'],text=True)
    baseline_file = tmp_path/'baseline.py'
    baseline_file.write_text(source)
    spec = importlib.util.spec_from_file_location('peershift._diagnostic_baseline', baseline_file)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    fixture = data()
    for policy in ('top_k','random'):
        values = config(policy=policy).__dict__
        baseline.run_training(baseline.TrainingConfig(**values),tmp_path/(policy+'old'),datasets=(fixture,fixture))
        training.run_training(training.TrainingConfig(**values),tmp_path/(policy+'new'),datasets=(fixture,fixture))
        for artifact in ('selections.jsonl','evaluations.jsonl','partitions.json','manifest.json'):
            assert (tmp_path/(policy+'old')/artifact).read_bytes() == (tmp_path/(policy+'new')/artifact).read_bytes()
