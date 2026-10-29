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
