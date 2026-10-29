import random
import pytest
from peershift.selection import top_k_select, explore_select
from peershift.detector import peer_entropy, repetition_rate
from peershift.simulation import Config, simulate, run


def test_selection():
    assert {top_k_select(0, [1, .9, .8, .1], rng=random.Random(i)) for i in range(50)} == {1, 2}
    assert explore_select(0, [1, .9, .8], [1], rng=random.Random(1)) == 2
    assert explore_select(0, [1, .9, .8], [1, 2], k=1) == 1
    assert top_k_select(0, [1, 1, 1], k=1) == 1


@pytest.mark.parametrize('client,scores,k', [(0,[1],1), (3,[1,2],1), (0,[1,2],0), (0,[1,float('nan')],1)])
def test_invalid_selection(client, scores, k):
    with pytest.raises(ValueError):
        top_k_select(client, scores, k)


def test_metrics():
    assert peer_entropy([], 3) == 0
    assert peer_entropy([1,2], 2) == pytest.approx(1)
    assert repetition_rate([1,1,2]) == .5


def test_simulation(tmp_path):
    config = Config(rounds=5, history_window=1)
    records = list(simulate(config))
    assert records == list(simulate(config))
    assert len(records) == 50
    assert all(r['client'] != r['selected_peer'] and r['unique_peers'] == 1 and r['entropy'] == 0 for r in records)
    run(config, tmp_path)
    assert len((tmp_path/'selections.jsonl').read_text().splitlines()) == 50
    assert Config.load(tmp_path/'config.toml') == config
    with pytest.raises(ValueError):
        run(config, tmp_path)


@pytest.mark.parametrize('kwargs', [dict(num_clients=1), dict(k=10), dict(rounds=0), dict(seed=True)])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)
