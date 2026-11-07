"""CPU learning primitives, independent of selection."""
import torch
from torch import nn


def make_model(hidden_size=128):
    return nn.Sequential(nn.Flatten(), nn.Linear(784, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 10))


def snapshot(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def aggregate(model, own, peer, own_weight=.5):
    model.load_state_dict({key: own_weight * own[key] + (1-own_weight) * peer[key] for key in own})


def similarities(states):
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
