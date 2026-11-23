# MPI model transport prototype

This feature exchanges actual MLP parameter buffers between independent MPI ranks and averages only at each receiver. It does not train models, load MNIST, or discover similar peers. It is the transport prerequisite for the future MPI learning runner.

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
