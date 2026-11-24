from collections import Counter
from math import log


def peer_entropy(history, total_peers):
    if not history or total_peers <= 1:
        return 0.0

    counts = Counter(history)
    total = len(history)

    entropy = -sum((count / total) * log(count / total) for count in counts.values())

    return entropy / log(total_peers)


def repetition_rate(history):
    if len(history) < 2:
        return 0.0

    repeated = sum(history[i] == history[i - 1] for i in range(1, len(history)))

    return repeated / (len(history) - 1)


def collaboration_metrics(history, total_peers):
    """Multi-peer histories contain one canonical peer set per round.

    Entropy counts individual selections; repetition compares entire peer sets.
    A repeated multi-peer set can have nonzero entropy without exploring.
    """
    rounds = [tuple(sorted(set(peers))) for peers in history]
    selections = [peer for peers in rounds for peer in peers]
    return dict(history_rounds=len(rounds),
                peer_selection_entropy=peer_entropy(selections, total_peers),
                adjacent_peer_set_repetition_rate=repetition_rate(rounds),
                unique_collaborators=len(set(selections)))
