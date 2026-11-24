"""CPU learning primitives, independent of selection."""
import torch
from torch import nn


def make_model(hidden_size=128):
    return nn.Sequential(nn.Flatten(), nn.Linear(784, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 10))


def snapshot(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def aggregate(model, own, peer, own_weight=.5):
    model.load_state_dict({key: own_weight * own[key] + (1-own_weight) * peer[key] for key in own})


def aggregation_weights(peer_count, rule='uniform_self_and_peers', own_weight=.5):
    """Return explicit self and per-peer weights for one receiver-local merge."""
    import math
    if type(peer_count) is not int or peer_count < 1:
        raise ValueError('At least one peer is required')
    if type(own_weight) not in (int, float) or not math.isfinite(own_weight) or not 0 <= own_weight <= 1:
        raise ValueError('Invalid own weight')
    if rule == 'uniform_self_and_peers':
        weight = 1 / (peer_count + 1)
        return weight, (weight,) * peer_count
    if rule == 'fixed_self':
        return own_weight, ((1-own_weight) / peer_count,) * peer_count
    raise ValueError('Unknown aggregation rule')


def aggregate_peers(model, own, peers, *, rule='uniform_self_and_peers', own_weight=.5):
    """Merge self and peer snapshots without modifying any source tensors."""
    peers = tuple(peers)
    self_weight, peer_weights = aggregation_weights(len(peers), rule, own_weight)
    expected = model.state_dict()
    for state in (own, *peers):
        if state.keys() != expected.keys():
            raise ValueError('Model snapshot keys do not match')
        for name, template in expected.items():
            value = state[name]
            if (not isinstance(value, torch.Tensor)
                    or value.shape != template.shape or value.dtype != template.dtype
                    or value.device != template.device or not value.is_floating_point()
                    or not torch.isfinite(value).all().item()):
                raise ValueError(f'Unsupported or incompatible tensor: {name}')
    if len(peers) == 1 and rule == 'fixed_self':
        # Preserve the original one-peer arithmetic and seeded trajectories.
        aggregate(model, own, peers[0], own_weight)
        return
    with torch.no_grad():
        merged = {name: value * self_weight for name, value in own.items()}
        for state, weight in zip(peers, peer_weights):
            for name in merged:
                merged[name] = merged[name] + state[name] * weight
        model.load_state_dict(merged)


def similarities(states, metric='flat_l2'):
    """Return larger-is-closer scores; sum_tensor_l2 follows SONAR's formula."""
    if metric == 'sum_tensor_l2':
        if not states:
            raise ValueError('At least one snapshot is required')
        keys = [name for name in states[0] if not name.endswith('num_batches_tracked')]
        if not keys:
            raise ValueError('No comparable model tensors')
        for state in states:
            if state.keys() != states[0].keys():
                raise ValueError('Model snapshot keys do not match')
            for name in keys:
                value, template = state[name], states[0][name]
                if (not isinstance(value, torch.Tensor)
                        or value.shape != template.shape or value.dtype != template.dtype
                        or value.device != template.device or not value.is_floating_point()
                        or not torch.isfinite(value).all().item()):
                    raise ValueError(f'Unsupported or incompatible tensor: {name}')
        return [[-sum(torch.linalg.vector_norm(a[name]-b[name]).item() for name in keys)
                 for b in states] for a in states]
    if metric != 'flat_l2':
        raise ValueError('Unknown similarity metric')
    vectors = [torch.cat([v.flatten() for v in state.values()]) for state in states]
    return [[-torch.linalg.vector_norm(a-b).item() for b in vectors] for a in vectors]


def local_train(model, optimizer, batches, steps):
    model.train()
    for _ in range(steps):
        x, y = next(batches)
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        optimizer.step()


def evaluate(model, loader):
    model.eval()
    loss = correct = count = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss += nn.functional.cross_entropy(logits, y, reduction='sum').item()
            correct += (logits.argmax(1) == y).sum().item()
            count += len(y)
    return loss/count, correct/count
