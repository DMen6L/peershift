# PeerShift simulator

Peer-selection simulation and minimal IID CPU gossip learning, using Python 3.12 or later. Training is available through the optional learning dependencies.

From the repository root:

```sh
uv sync --project backend --locked
uv run --project backend peershift simulate --config backend/configs/default.toml --output /tmp/peershift-run
uv run --project backend pytest backend/tests
```

The output directory must be empty. Configuration controls seed, client count, rounds, K, and rolling history size. K defines the pool of similar candidates; each client selects one peer per round.

The simulator generates a seeded symmetric similarity matrix, keeps it fixed, and uses independent per-client random streams. Equal-score ties use client ID ordering. Outputs are resolved `config.toml` and deterministic `selections.jsonl` records with round, client, peer, entropy, repetition rate, and unique peers. Metrics include the current selection and use bounded history; entropy is normalized over all eligible peers.

The `peershift.selection` module retains Top-K and history-based exploration primitives. `peershift.detector` provides entropy and repetition metrics. History-based exploration excludes recent peers; fixed exploration instead excludes the current Top-K pool.

See [the execution plan](../PROJECT_PLAN.md) for subsequent baseline, learning, adaptive-control, and distributed milestones.

The distributed milestone requires MPI via `mpi4py`: one process per client, real peer model messages, and local receiver-side averaging. The `mpi` extra and `mpi-demo` command now provide a fixed-peer model exchange prototype; learning commands remain single-process. The existing simulator will serve as the parity reference for that implementation.

### Selection baseline configurations

Use `backend/configs/top_k.toml`, `backend/configs/random.toml`, or `backend/configs/fixed_exploration.toml` with the existing simulate command, each into a separate empty output directory. All three use matching seeds and selection budgets.

`policy` defaults to `top_k`; `exploration_probability` defaults to 0.1 and applies only to `fixed_exploration`. Random selection samples any eligible peer. Fixed exploration samples outside Top-K with that probability, otherwise inside Top-K. If no outside candidates exist it falls back to Top-K. Logs include `policy`, `selection_branch`, and nullable `fallback_reason`. Diversity differences do not establish learning benefits; IID training supports all three policies, while adaptive control remains future work.

## Minimal IID gossip learning

Implemented CPU MNIST MLP training with Top-K, random, fixed-exploration, and SONAR-style closest-peer selection, separate local validation and held-out test data, and receiver-side averaging from immutable post-training snapshots. Existing selection simulation remains available without ML dependencies. Non-IID learning, adaptive control, MPI training integration, and network experiments remain future work.

```sh
uv sync --project backend --extra learning --locked
uv run --project backend --extra learning peershift train --config backend/configs/train_top_k.toml --output /tmp/peershift-learning --download
```

Use train_random.toml or train_fixed_exploration.toml for the paired baselines. Default settings: 10 clients, 10 rounds, 10 local steps, batch size 32, learning rate .05, own weight .5, hidden size 128, 500 training and 100 validation examples per client, 1000 test examples. All settings are configurable; Training supports mnist/mlp/iid/cpu with top_k, random, fixed_exploration, and closest policies. The closest policy also supports multiple collaborators; see backend/configs/train_sonar_closest.toml. `exploration_probability` defaults to 0.1 and applies only to fixed exploration. Data lives in backend/data by default. Omit --download once cached. Outputs include resolved configuration, exact partitions, environment/initial-model fingerprint, selection and evaluation JSONL, and summary counts/runtime. Round -1 denotes initial evaluation. Estimated parameter payload excludes similarity discovery and transport overhead; no network traffic occurs. Similarity uses full snapshot access as a simulation assumption. Same-environment reproducibility excludes runtime.

## Immediate exchange measurements

Training now writes aggregation.jsonl and summary.txt. Each client is evaluated on the same local validation examples immediately before and after averaging. Benefit is before-loss minus after-loss: positive indicates immediate improvement, negative worsening. Logs include selected-peer L2 distance and collaboration history metrics from preceding rounds. These diagnostics do not influence selection or establish harmful collapse. Summary counts use 1e-8 tolerance for unchanged results; additional validation time is recorded separately. Estimated model bytes and exchange counts retain their previous meaning.

After the documented training command, inspect `cat <output-directory>/summary.txt` and `head -n 3 <output-directory>/aggregation.jsonl`. Non-IID learning, adaptive control, MPI training integration, and network experiments remain future work.

## Test verification modes

The lightweight command above permits learning tests to skip when PyTorch is absent. For any learning or diagnostic change, require working learning dependencies explicitly:

```sh
uv sync --project backend --extra learning --locked
uv run --project backend --extra learning pytest backend/tests --require-learning
```

This command fails with an installation hint if torch or torchvision cannot import. A green learning verification must have no skipped learning modules. The suite checks actual model tensors against a small evaluation-free gossip reference, audits training/validation/test sample access using distinct offline datasets, and verifies benefit signs, tolerance boundaries and history eviction. These checks need no Git history or MNIST download. They establish mechanics, not research conclusions about exploration.

## MPI model transport

See [MPI transport instructions](../docs/MPI_TRANSPORT.md) for the separate-process exchange/averaging demo and mandatory MPI verification command. Ring, many-to-one and rotating fixture choices are supported. The demo does not train models or implement distributed similarity discovery.

## SONAR-style closest collaborators

The first independent SONAR-style baseline is available in the **training** command:

```sh
uv run --project backend --extra learning peershift train --config backend/configs/train_sonar_closest.toml --output /tmp/peershift-sonar-closest --download
```

Use a fresh output directory. The profile sets `policy="closest"`, `num_collaborators=2`, `similarity_metric="sum_tensor_l2"`, and `aggregation_rule="uniform_self_and_peers"`. Each receiver selects both nearest peers, then averages self plus both peers at weight 1/3 each. This independently implements the inspected SONAR dynamic-selection and uniform-aggregation formulas; it is not a SONAR runtime dependency or a reproduction of published results.

`num_collaborators` is the actual number of peers. Existing `k` retains its candidate-pool meaning for legacy `top_k`/`fixed_exploration` policies and is unused by `closest`. Existing policies still select one peer and reject a different collaborator count. The default metric `flat_l2` and rule `fixed_self` preserve existing behavior. `sum_tensor_l2` sums per-tensor Euclidean norms and ignores `num_batches_tracked`; it can rank peers differently from flattened L2. Both metrics return larger-is-closer scores. Equal scores for `closest` break by client ID.

`uniform_self_and_peers` assigns 1/(m+1) to each of self and m peers. `fixed_self` instead assigns `own_weight` to self and divides the remaining weight among peers; `own_weight` only controls this rule. Actual weights are logged for closest-policy runs. Serial multi-peer averaging supports matching floating tensor schemas; nonfloating/incompatible state is rejected. MPI transport supports multiple peers with dense CPU float32 snapshots; see the collaboration fixture below.

Closest-policy JSONL uses `selected_peers` and aligned `selected_peer_distances`, not a fabricated scalar peer. Diagnostics label benefit as `joint_aggregation`: it measures the entire merge, not a contribution attributable to each peer. Histories retain peer sets by round. `peer_selection_entropy` counts individual peer selections in those sets; `adjacent_peer_set_repetition_rate` compares canonical sets; `unique_collaborators` and `history_rounds` describe the same window. Prior metrics have a `_before_selection` suffix. A repeated multi-peer set can have nonzero entropy without any exploration; thresholds must account for peer count and history window. Legacy scalar-peer records keep their existing meanings.

Summary counts separate `aggregation_events` (clients × rounds), `logical_model_exchanges` (events × collaborators), and local training steps (events × local_steps). Benefit percentages use events. The two-collaborator example is not an equal-exchange-budget comparison with legacy one-peer runs; use `num_collaborators=1` for a matched peer count, or report the different budgets explicitly. Payload bytes remain estimates; all candidate models are available in memory and no distributed discovery or training is performed by this runner. The artificial-score `simulate` command retains its three existing policies.

See [the reimplementation guide](../docs/SONAR_REIMPLEMENTATION_GUIDE.md) for upstream references, intended architecture and remaining migration work.

## MPI core collaboration

The `mpi-collaborate` command runs independent ranks using multi-peer transfers, dense frozen-model discovery, closest selection and receiver-local aggregation. It reuses discovered snapshots and counts discovery traffic separately from selected model contributions. This is a seeded fixture simulation without training; see [MPI launch instructions](../docs/MPI_TRANSPORT.md#multi-peer-core-collaboration) for the four-client example, outputs and limitations. MPI training remains the next integration step.
