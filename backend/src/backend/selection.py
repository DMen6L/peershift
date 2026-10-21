import random


def top_k_select(client_id, similarities, k: int = 2) -> int:
    candidates = [
        (peer, score) for peer, score in enumerate(similarities) if peer != client_id
    ]

    candidates.sort(key=lambda item: item[1], reverse=True)

    pool = [peer for peer, _ in candidates[:k]]
    return random.choice(pool)


def explore_select(client_id, similarities, history, k=2) -> int:
    candidates = [
        peer
        for peer in range(len(similarities))
        if peer != client_id and peer not in history
    ]

    if candidates:
        return random.choice(candidates)

    return top_k_select(client_id, similarities, k)
