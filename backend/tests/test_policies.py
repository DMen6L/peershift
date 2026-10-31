import json
from pathlib import Path
import sys
import subprocess

import pytest

from peershift.selection import random_select, select_peer
from peershift.simulation import Config, simulate, run


class Probe:
    def __init__(self, coin=0):
        self.coin = coin
        self.pool = None

    def random(self):
        return self.coin

    def choice(self, pool):
        self.pool = list(pool)
        return pool[0]


def test_random_eligible_pool():
    rng = Probe()
    assert random_select(1, [1, 1, 1, 1], rng=rng) == 0
    assert rng.pool == [0, 2, 3]


@pytest.mark.parametrize('coin,branch,pool', [(0.1, 'exploration', [3, 4]), (0.8, 'top_k', [1, 2])])
def test_fixed_branches(coin, branch, pool):
    rng = Probe()
    decision = select_peer(0, [1, .9, .8, .2, .1], policy='fixed_exploration',
                           exploration_probability=.5, rng=rng, exploration_rng=Probe(coin))
    assert decision.selection_branch == branch
    assert rng.pool == pool
    assert decision.fallback_reason is None


def test_probability_boundaries_and_unchanged_top_k():
    top = list(simulate(Config(rounds=10)))
    fixed = list(simulate(Config(rounds=10, policy='fixed_exploration', exploration_probability=0)))
    assert [r['selected_peer'] for r in top] == [r['selected_peer'] for r in fixed]
    scores = [1, 1, 1, 1]
    assert select_peer(0, scores, k=1, rng=Probe()).selected_peer == 1
    result = select_peer(0, scores, k=1, policy='fixed_exploration', exploration_probability=1,
                         rng=Probe(), exploration_rng=Probe())
    assert result.selected_peer == 2 and result.selection_branch == 'exploration'


@pytest.mark.parametrize('scores,k', [([1, 1], 1), ([1, 1, 1], 2)])
def test_empty_outside_fallback(scores, k):
    result = select_peer(0, scores, k, policy='fixed_exploration', exploration_probability=1,
                         rng=Probe(), exploration_rng=Probe())
    assert result.selection_branch == 'top_k'
    assert result.fallback_reason == 'no_outside_candidates'
    assert result.selected_peer == 1


@pytest.mark.parametrize('p', [True, False, -.1, 1.1, float('nan'), float('inf'), '0.1'])
def test_invalid_probability(p):
    with pytest.raises(ValueError):
        Config(exploration_probability=p)
    with pytest.raises(ValueError):
        select_peer(0, [1, 1], k=1, exploration_probability=p)


@pytest.mark.parametrize('policy', ['top_k', 'random', 'fixed_exploration'])
def test_config_round_trip_and_cli(tmp_path, policy):
    config = Config(policy=policy)
    run(config, tmp_path/'written')
    assert Config.load(tmp_path/'written/config.toml') == config
    cli = str(Path(sys.executable).with_name('peershift'))
    project = Path(__file__).resolve().parents[1]
    args = [cli, 'simulate', '--config', str(project / f'configs/{policy}.toml'), '--output']
    for name in ('a', 'b'):
        subprocess.run(args + [str(tmp_path/name)], check=True, capture_output=True)
    a = (tmp_path/'a/selections.jsonl').read_bytes()
    assert a == (tmp_path/'b/selections.jsonl').read_bytes()
    rows = [json.loads(line) for line in a.splitlines()]
    assert len(rows) == 1000
    assert all(r['policy'] == policy and r['client'] != r['selected_peer'] and 0 <= r['selected_peer'] < 10 for r in rows)
    assert all(0 <= r['entropy'] <= 1 and 0 <= r['repetition_rate'] <= 1 for r in rows)
    rejected = subprocess.run(args + [str(tmp_path/'a')], capture_output=True)
    assert rejected.returncode != 0
    assert (tmp_path/'a/selections.jsonl').read_bytes() == a
