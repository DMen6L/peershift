# MPI model transport prototype

This feature exchanges actual MLP parameter buffers between independent MPI ranks and averages only at each receiver. The original `mpi-demo` does not train models, load MNIST, or discover similar peers. The `mpi-collaborate` increment below adds dense discovery and core selection/merging without training. It is the transport prerequisite for the future MPI learning runner.

## Install and run

A compatible MPI runtime and `mpiexec` must already be installed. Initial verification uses Open MPI 5.0.11, Python 3.12 and mpi4py 4.1.2 on one host. Multi-host deployment is not yet validated.

From the repository root, install extras once before launching ranks:

```sh
uv sync --project backend --extra learning --extra mpi --locked
mpi_output=$(mktemp -d /tmp/peershift-mpi.XXXXXX)
mpiexec -n 3 backend/.venv/bin/python -m peershift mpi-demo \
  --config backend/configs/mpi_demo.toml --output "$mpi_output"
cat "$mpi_output/summary.txt"
head -n 3 "$mpi_output/exchanges.jsonl"
```

The launcher rank count must equal `num_clients`; the output directory must be empty. Run the installed Python executable under MPI, rather than launching several concurrent `uv sync` operations.

Configuration defaults are three clients, three rounds, hidden size eight, and own-model weight 0.5. All parameters of client r's initial model are filled with r+1 for this fixture; actual training continues to use identical initial weights. Change `pattern` in a copied TOML file to:

- `ring`: rank r requests rank (r+1) modulo the client count.
- `many_to_one`: rank zero requests rank one, and all other ranks request rank zero. This includes mutual selections, multiple requesters and ranks with no incoming requests.
- `rotating`: rotate the non-self peer offset each round.

Each rank owns one model. Every round freezes current parameters, exchanges snapshots, and averages the receiver's own and selected peer's snapshots. The averaged model becomes the next round's source. The root collects only control/log data, not models for centralized averaging.

## Transport contract

`MPITransport.exchange(state, selected_peer, round_index)` is collective over a private communicator. Every rank must participate in the same sequential rounds beginning at zero. The result is an independent received state dictionary plus rank-local metrics. Call `close()` collectively after the final exchange.

Only nonempty dense CPU float32 tensors are supported. Scalar and noncontiguous tensors can be packed; unsupported dtypes, empty tensors and nonfinite values fail validation instead of being silently converted. Parameter names are sorted; a layout hash covers both names and shapes. All ranks must share a layout, which remains fixed across rounds.

All ranks first share small validation records containing the round, requested peer and layout identity. Any announced input error rejects the round on all ranks before payload operations begin. Each receiver posts an explicit-source receive, each sender posts nonblocking sends to every requester, and all requests complete before the round-end barrier. Owned send buffers remain immutable throughout. Runtime MPI errors abort the job; this version does not recover failed ranks. Callers must abort if they fail outside collective validation, since a missing rank cannot be rescued by a barrier.

The demo exits nonzero for setup errors (including mismatched process counts and nonempty output directories). A launcher timeout is required for automated tests; arbitrary process crashes or omitted collective calls are not internally recoverable.

## Outputs and accounting

- `config.toml`: resolved demo settings.
- `manifest.json`: dependency/runtime versions and explicit fixture limitations.
- `exchanges.jsonl`: peer IDs, send/receive counts, model bytes, source/received/averaged hashes, first parameter value, snapshot-preservation check and timings.
- `summary.json` and `summary.txt`: aggregate application payload counts and human-readable status.

A receiver gets one model per round; a sender may serve zero or multiple receivers. Sent and received totals describe the same transfers and must not be added as though they were separate exchanges. Model bytes are exact application-buffer sizes. Metadata/log byte fields measure logical encoded JSON records, excluding Python pickling and MPI framing; collective algorithms may transmit them more than once. Timing-dependent log record sizes and timing fields are excluded from deterministic log comparisons. Setup broadcasts, final synchronization and MPI internal overhead are not included in these byte totals. Wire bytes are unmeasured.

No similarity discovery is performed: fixed peer choices are fixture inputs. A future Top-K MPI runner must design and account for discovery separately. Copying all models to all ranks is not a free sparse exchange.

## Verification

```sh
uv run --project backend --extra learning --extra mpi pytest backend/tests \
  --require-learning --require-mpi
```

The explicit flags fail if required dependencies/launcher are missing; MPI tests are optional in lightweight environments. A sandbox may need permission for MPI to open local sockets. Tests use real `mpiexec` processes with timeouts, known-model averaging references, multiple topologies, repeated rounds, large payloads and invalid input cases.

Protocol references: [mpi4py tutorial](https://mpi4py.readthedocs.io/en/stable/tutorial.html), [request completion API](https://mpi4py.readthedocs.io/en/stable/reference/mpi4py.MPI.Request.html), and [MPI buffer-completion rules](https://www.mpi-forum.org/docs/mpi-5.0/mpi50-report/node76.htm).

## Multi-peer core collaboration

`mpi-collaborate` connects real MPI transfers to the existing SONAR-style closest-peer selector, similarity metrics and receiver-local aggregation. It runs distinct seeded MLP fixtures without training or a dataset. The single-peer `mpi-demo` command and `MPITransport.exchange` remain supported.

After installing the extras above, run from the repository root:

```sh
collaboration_output=$(mktemp -d /tmp/peershift-collaboration.XXXXXX)
mpiexec -n 4 backend/.venv/bin/python -m peershift mpi-collaborate \
  --config backend/configs/mpi_collaboration.toml --output "$collaboration_output"
cat "$collaboration_output/summary.txt"
head -n 4 "$collaboration_output/collaborations.jsonl"
```

Defaults are four ranks/clients, three rounds, two collaborators, hidden size eight, seed 42 and a twenty-round history. Each fixture MLP is initialized with seed `(seed + rank) % 2**63`; this intentionally differs from identical initial weights for real learning. Supported settings are `policy="closest"`, `similarity_metric="sum_tensor_l2"` or `"flat_l2"`, and `aggregation_rule="uniform_self_and_peers"` or `"fixed_self"`. Uniform averaging gives self and each of m peers weight 1/(m+1). Fixed-self averaging uses `own_weight` for self and divides the remainder among peers. Rank count must equal `num_clients`, with 1 ≤ `num_collaborators` < `num_clients`.

Each round freezes the rank's current model, acquires all other ranks' frozen snapshots, scores them locally and selects the m nearest peers (ties by rank). Each receiver merges only its selected snapshots; it reuses exactly the snapshots acquired for discovery. The merged local model becomes the next round's source. No rank centrally selects peers or aggregates models. Tensor names are sorted for discovery scoring so ranks use the same distance summation order. Selected peers retain distance/rank order for aggregation; history stores canonical peer sets by round.

### Generalized transport interface

`MPITransport.exchange_many(state, peers, round_index)` returns `(received_states, metrics)`. `received_states` is a tuple aligned with the supplied nonempty list/tuple of distinct non-self rank IDs. Peers may differ in number and order across ranks. Duplicate, self, out-of-range or noninteger IDs are rejected collectively. Existing layout, dtype, immutable-buffer, sequential-round and abort guarantees also apply. One call advances the transport round once, regardless of peer count. `exchange` wraps a one-peer call and retains its scalar return and metric fields.

For discovery the runner requests every other rank through `exchange_many`. All nonblocking receives and sends finish before selection or aggregation. There is no second exchange, extra round advancement, or selected-payload retransmission.

### Discovery cost and artifacts

Dense exact discovery is a correctness reference, not sparse or communication-efficient gossip. Per round, N clients transfer N(N−1) snapshots. The Nm selected model contributions are uses of those received snapshots, not extra messages. The default run therefore has **36 discovery transfers, 24 selected contributions and 12 aggregation events**. Self weights are 1/3 and each selected peer has weight 1/3.

`collaborations.jsonl` contains one record per client-round, ordered by round then rank: discovery peers and hashes, selected peers and aligned distances/hashes, actual aggregation weights, source/merged hashes, rolling group metrics, and timings. `history_rounds`, `peer_selection_entropy`, `adjacent_peer_set_repetition_rate` and `unique_collaborators` have the same meanings as the serial closest-policy logs; preceding history uses `_before_selection` fields. Repeated m-peer groups can have nonzero entropy without exploration.

`discovery_sent_model_bytes`/`discovery_received_model_bytes` and corresponding message counters measure actual application-buffer transfers. Sent and received totals balance and describe the same traffic; do not add them together. `selected_model_contributions` counts model uses, while `selected_payload_retransmission_messages` and `selected_payload_retransmission_bytes` are zero. Control metadata, logging record sizes and collective-call counts remain separate. As in the original demo, encoded JSON sizes exclude pickle/MPI framing, setup broadcasts, final synchronization and physical collective overhead. Wire bytes are unmeasured.

Outputs also include resolved `config.toml`, `manifest.json`, human/machine summaries and each rank's `rank-<id>/final_model.pt` checkpoint. Models are written by their owning ranks, not gathered by rank zero. The manifest records the Git revision when available, dependencies, fixture initialization and discovery mode. `runtime_seconds` is rank zero's elapsed time through result collection/checkpoint writing, before final synchronization. Checkpoints can be loaded with `torch.load(path, weights_only=True)` for small serial-reference comparisons. This runner assumes a shared output path on one host; multi-host deployment remains unvalidated.

Separate-process reproducibility compares model tensors and deterministic record values, excluding timing fields and timing-dependent logging sizes. Verification uses the strict learning/MPI suite above, real launchers with process-group timeouts, full-tensor serial references, malformed inputs and large multi-request payloads. MPI training, non-IID data, adaptive exploration and network experiments remain future increments.
