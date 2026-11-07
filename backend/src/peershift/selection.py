from dataclasses import dataclass
import random
from math import isfinite


def top_k_select(client_id, similarities, k: int = 2, *, rng=None) -> int:
    if not 0 <= client_id < len(similarities):
        raise ValueError("Invalid client ID")
    if not 1 <= k < len(similarities):
        raise ValueError("K must be between 1 and the eligible peer count")
    if not all(isfinite(score) for score in similarities):
        raise ValueError("Similarity scores must be finite")
    candidates = [
        (peer, score) for peer, score in enumerate(similarities) if peer != client_id
    ]

    candidates.sort(key=lambda item: item[1], reverse=True)

    pool = [peer for peer, _ in candidates[:k]]
    return (rng or random).choice(pool)


def explore_select(client_id, similarities, history, k=2, *, rng=None) -> int:
    candidates = [
        peer
        for peer in range(len(similarities))
        if peer != client_id and peer not in history
    ]

    if candidates:
        return (rng or random).choice(candidates)

    return top_k_select(client_id, similarities, k, rng=rng)



@dataclass(frozen=True)
class SelectionDecision:
    selected_peer: int
    selection_branch: str
    fallback_reason: str | None = None


def random_select(client_id, similarities, *, rng=None):
    if not 0 <= client_id < len(similarities) or len(similarities) < 2:
        raise ValueError('Need a valid client and at least one eligible peer')
    return (rng or random).choice([i for i in range(len(similarities)) if i != client_id])


def select_peer(client_id, similarities, k=2, *, policy='top_k',
                exploration_probability=0.1, rng=None, exploration_rng=None):
    if policy not in ('top_k', 'random', 'fixed_exploration'):
        raise ValueError('Unknown selection policy')
    if (isinstance(exploration_probability, bool)
            or not isinstance(exploration_probability, (int, float))
            or not isfinite(exploration_probability)
            or not 0 <= exploration_probability <= 1):
        raise ValueError('Exploration probability must be finite and between zero and one')
    if not 0 <= client_id < len(similarities) or not 1 <= k < len(similarities):
        raise ValueError('Invalid client ID or K')
    if not all(isfinite(score) for score in similarities):
        raise ValueError('Similarity scores must be finite')
    if policy == 'random':
        return SelectionDecision(random_select(client_id, similarities, rng=rng), 'random')
    if policy == 'fixed_exploration' and (exploration_rng or random).random() < exploration_probability:
        ranked = sorted((i for i in range(len(similarities)) if i != client_id),
                        key=lambda i: (-similarities[i], i))
        outside = ranked[k:]
        if outside:
            return SelectionDecision((rng or random).choice(outside), 'exploration')
        return SelectionDecision(top_k_select(client_id, similarities, k, rng=rng),
                                 'top_k', 'no_outside_candidates')
    return SelectionDecision(top_k_select(client_id, similarities, k, rng=rng), 'top_k')
