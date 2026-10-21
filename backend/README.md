# PeerShift Backend

The backend contains the core algorithms for peer selection and collaboration monitoring in PeerShift.

At the current development stage, these components operate independently of actual decentralized model training.

## Implemented Components

### 1. Peer Selection (`selection.py`)

Provides two peer-selection functions.

**`top_k_select(client_id, similarities, k=2)`**

Implements simplified similarity-based peer selection.

- Excludes the current client from the candidate list.
- Sorts other clients by their similarity scores.
- Constructs a pool containing the K most similar peers.
- Randomly selects one peer from that pool.

**`explore_select(client_id, similarities, history, k=2)`**

Implements basic exploration outside recent collaboration history.

- Identifies peers that do not appear in the supplied history.
- Randomly selects an available unexplored peer.
- Falls back to Top-K selection if no unexplored candidates remain.

This is an exploration primitive, not yet the complete adaptive PeerShift policy.

### 2. Collaboration Monitoring (`detector.py`)

Provides metrics used to identify potentially concentrated collaboration.

**`peer_entropy(history, total_peers)`**

Calculates normalized Shannon entropy over peer-selection history.

\[
H=-\frac{\sum_{i=1}^{N}p_i\ln p_i}{\ln N}
\]

Where:
- \(p_i\) is the proportion of selections assigned to peer \(i\).
- \(N\) is the number of available peer candidates.

Interpretation:
- Values near 0 indicate concentrated selection.
- Values near 1 indicate a relatively uniform distribution.

**`repetition_rate(history)`**

Measures the proportion of consecutive selections that choose the same peer.

\[
R=\frac{\sum_{t=2}^{T}\mathbf{1}(p_t=p_{t-1})}{T-1}
\]

Where \(p_t\) is the peer selected at round \(t\).

Higher values indicate more frequent consecutive repetition.

These metrics describe collaboration patterns but do not independently establish harmful collaborator collapse.

## Example Usage

```python
from peershift.selection import top_k_select, explore_select
from peershift.detector import peer_entropy, repetition_rate

similarities = [1.0, 0.9, 0.8, 0.3, 0.2, 0.1]
client_id = 0

peer = top_k_select(client_id, similarities, k=2)
print("Selected peer:", peer)

history = [1, 1, 1, 2, 1, 1]

print("Peer entropy:", peer_entropy(history, 5))
print("Repetition rate:", repetition_rate(history))

new_peer = explore_select(client_id, similarities, history, k=2)
print("Exploration peer:", new_peer)
```

The example assumes the `peershift` package is available on the Python import path.

## Current Limitations

- No model training or gradient exchange
- No actual peer-to-peer communication
- No validation-loss plateau detection
- No automated adaptive exploration controller
- No communication-cost measurements
- No distributed execution

## Next Development Steps

1. Create a multi-client simulation with configurable client count and rounds.
2. Record rolling peer-selection histories.
3. Add adaptive exploration triggers and cooldown mechanisms.
4. Introduce model training and validation tracking.
5. Integrate peer selection with decentralized model aggregation.
6. Support distributed execution and network-condition experiments.

## Development Goal

The backend will eventually allow each client to independently evaluate collaboration patterns and adjust peer selection without relying on a central model aggregator.
