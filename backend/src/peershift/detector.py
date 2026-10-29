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
