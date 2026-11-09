# PeerShift execution plan

## Purpose and outcome

PeerShift investigates whether adapting peer exploration when collaboration becomes concentrated and validation progress stalls improves decentralized learning under a consistent communication budget.

The final deliverable is a reproducible research prototype, experiment results, and a report answering the README's detection, mitigation, and systems-cost questions. A useful negative result is acceptable: success means credible evidence, not a guaranteed accuracy improvement.

## Current baseline

Implemented: runnable `peershift` package and CLI, seeded selection-only simulation, Top-K/random/fixed-exploration policies, bounded histories, entropy/repetition/unique-peer metrics, resolved TOML configuration and JSONL logs. History-based exploration remains a separate primitive. Python 3.12 is the development baseline with a dependency lockfile.

Implemented additionally: minimal IID MNIST training and snapshot aggregation for Top-K/random. Implemented additionally: fixed-peer MPI model transport and receiver-side averaging with real messages. Missing: non-IID learning, fixed-exploration learning comparisons, validation monitoring, adaptive control, MPI training integration, distributed discovery, and network experiments. Selection counts are not network transfers. K is the candidate pool size, not the number of peers selected.

## Goals and boundaries

1. Detect persistent concentration together with validation stagnation using local client observations.
2. Compare adaptive exploration against Top-K, uniform random, and fixed exploration under matched training and exchange budgets.
3. Measure how latency and bandwidth affect learning progress and communication cost.
4. Make each reported result reproducible from a configuration, seed, code revision, and saved metrics.

Start with a single dataset, small model, synchronous rounds, and static client membership. A single-process simulator may coordinate execution, but clients must select peers and aggregate locally without a central model aggregator. Asynchronous training, client churn, privacy mechanisms, production deployment, and a web interface are outside the initial scope.

**Mandatory distributed execution requirement:** implement the future distributed simulator using MPI through `mpi4py`, with one rank per client. Each rank owns its model, optimizer, data partition, peer history, and eventual controller state. Real MPI messages must carry peer model parameters; an in-process loop or containers alone do not satisfy this requirement. The current single-process implementation remains the reference for correctness. Optional MPI dependencies and a fixed-peer exchange demo are now implemented. MPI training and similarity-discovery integration remain future increments.

## Milestones and completion gates

### M0 — Establish a runnable foundation

Tasks:
- Choose one package/import name and replace the placeholder CLI with an experiment entry point.
- Set compatible Python and dependency versions, a lockfile, and documented setup commands.
- Define configuration fields, seeded random generators, client state, policy interface, and metrics output schema.
- Validate selection inputs and define behavior for invalid K, too few clients, and short histories.
- Add focused tests for selection and metrics, including entropy normalization over eligible peers.

Complete when a clean environment can install the project and run deterministic selection tests using documented commands.

### M1 — Build a reproducible peer-selection simulator

Tasks:
- Simulate configurable clients and rounds with controlled similarity scores and rolling histories.
- Implement Top-K, uniform random, and fixed-probability exploration through the same interface.
- Log client, round, selected peer, selection reason, entropy, repetition, unique peers, seed, and configuration.
- Add scenarios with persistent concentration, changing similarities, and already diverse collaboration.

Complete when identical configurations and seeds reproduce selection logs, self-selection is excluded, and all policies use the same number of peer exchanges. These scenarios validate policy behavior; they do not establish a learning benefit.

### M2 — Implement detection and adaptive recovery

Tasks:
- Specify plateau detection using a rolling validation-loss window, minimum improvement, warmup, and patience.
- Combine sustained concentration with sustained plateau as a warning condition. Treat this as a heuristic, not proof of harmful collapse.
- Implement a controller with normal, exploring, and cooldown states; bounded exploration probability; exploration duration; and explicit recovery rules.
- Log trigger evidence and every state transition.
- Test synthetic loss traces: improvement, plateau, noise, concentration without plateau, and plateau without concentration.

Complete when documented traces produce expected transitions, isolated warnings do not trigger intervention, and exploration ends predictably without rapid oscillation.

### M3 — Demonstrate actual decentralized learning

Tasks:
- Choose a small dataset/model pair and define IID and non-IID client partitions.
- Separate local training, validation used by the controller, and held-out test evaluation. Define handling for clients missing particular classes.
- Define the similarity signal, its update frequency, and any extra information exchanges it requires.
- Specify local training steps, model-exchange semantics, local aggregation weights, and synchronous update ordering so execution order does not change results.
- Run all four policies with identical initial models, partitions, local work, and exchanges for each paired seed.
- Keep detection and tuning independent of held-out test outcomes.

Complete when every client trains and aggregates peer models, a small end-to-end experiment is reproducible, and accuracy/loss can be plotted alongside collaboration metrics and controller events.

### M4 — Implement MPI distributed simulation

**Partial progress:** the transport-only `mpi-demo` exchanges and averages known models across real MPI ranks, without training. See [MPI transport instructions](docs/MPI_TRANSPORT.md). Full M4 completion still requires training/reference parity and the remaining tasks below.

Tasks:
- Add the required MPI execution backend using `mpi4py`, with dependencies in an optional install extra so the single-process reference remains lightweight. Document a supported MPI runtime and launcher. Require the rank count to equal the configured client count; use stable rank-to-client mapping.
- Reuse selection, detection, training, and averaging logic through a separate MPI transport/round coordinator. Each rank trains locally; all ranks finish local training before exchanges use immutable post-training snapshots from that round.
- Transfer selected peers' snapshot parameters through MPI and aggregate only at the receiver, preserving one peer selection per client per round. A rank must serve all incoming requests, including multiple clients selecting the same peer.
- Use a protocol that completes sends/receives without deadlock for cycles, mutual selections, and many-to-one requests; keep snapshot buffers immutable until communication completes. Define mismatched-round detection and job termination behavior.
- Keep peer selection and model aggregation local to each rank. Rank zero may distribute configuration/initial weights and collect logs, but must not average client models or choose all clients' peers. Collectives for synchronization and metadata are permitted.
- Specify how similarity information becomes available across ranks before implementation. Count any discovery messages or additional model transfers separately; the current free access to all models cannot be assumed in MPI or hidden from budget comparisons.
- Record model payload, discovery/control bytes and messages, runtime, and round duration separately. Distinguish application-level payload measurements from wire-level overhead.
- Verify a small multi-rank run on one host against the single-process reference using paired initialization, partitions, minibatches, and policy seeds. Document multi-host execution for later network experiments.
- Treat SONAR as optional: integrate only if it preserves MPI transport and the required learning semantics. Docker Compose may package an MPI deployment later, but does not replace MPI.

Complete when at least three MPI client ranks complete training with actual peer messages, including cyclic, many-to-one, zero-incoming, and large-payload checks with timeouts to detect hangs, auditable communication counts, and reproducible per-client outputs. Verify selections and model updates against the single-process reference with documented floating-point tolerances. Deliver MPI launch instructions; claim multi-host/network results only after separately validating them.

See the [MPI requirements review](docs/research/2026-11-06-mpi-requirements.md) for official communication references and protocol questions to resolve before implementation.

### M5 — Run controlled learning and network experiments

Tasks:
- Freeze baseline settings and a modest adaptive-parameter tuning budget before the final evaluation.
- Evaluate all four policies on IID and non-IID partitions under a homogeneous network, then selected heterogeneous latency/bandwidth profiles.
- Apply Linux NetEm in the Linux container/host environment; verify the achieved latency and throughput before collecting results.
- Use a proposed minimum of five paired seeds, adjusting the experiment size after pilot runtime measurements.
- Run ablations: concentration-only trigger, plateau-only trigger, and the combined trigger. Use these to assess whether combining signals adds value.
- Save raw per-client data and report aggregate results with uncertainty and client-level variation.

Complete when the planned matrix is collected or omissions are documented, comparisons use matched exchange budgets, and similarity/monitoring traffic is included in communication cost.

### M6 — Deliver the research artifact

Tasks:
- Produce learning curves, collaboration graphs, diversity traces, controller-event plots, and accuracy-versus-wall-time/bytes comparisons.
- Answer each research question with evidence, limitations, and failure cases.
- Document exact reproduction commands, configurations, environment, seed list, and code revision.
- Prepare the course report and a small demonstration configuration.

Complete when another team member can reproduce a representative experiment and regenerate the main figures from saved results.

## Experimental contract

| Area | Planned rule |
| --- | --- |
| Policies | Top-K pool selection; uniform random; fixed exploration; adaptive exploration |
| Fairness | Same initial weights, partitions, local steps, exchange count, and paired seeds |
| Learning outcomes | Held-out test accuracy/loss; validation trajectories; per-client distribution |
| Collaboration outcomes | Rolling entropy, repetition, unique peers, and controller events |
| Systems outcomes | Actual bytes/messages, runtime, round duration, and time to a predeclared target if reached |
| Detector evaluation | Trigger timing on controlled traces; downstream learning outcomes and ablations in real training |
| Tuning | Validation-only tuning with a declared budget; test results reserved for final evaluation |
| Interpretation | Increased diversity alone is not evidence of improved learning; report null and adverse results |

Equal exchanges do not imply equal bytes or wall-clock time. Report those costs separately, including policy-specific metadata. Real-training runs have no automatic ground-truth label for collapse, so do not claim detector precision/recall without an independently justified labeling procedure.

## Approved incremental execution order

1. **Current increment:** deterministic Top-K selection simulation only. Preserve existing selection and metrics; add seeded randomness, bounded histories, TOML configuration, JSONL logs, and focused tests. No policy registry or ML dependencies.
2. **Implemented:** uniform-random and fixed exploration outside the Top-K pool. The existing history-based exploration primitive remains distinct.
3. Add PyTorch MNIST gossip learning: configurable small MLP, ten clients by default, IID/non-IID partitions, identical initialization, and fixed local steps. Begin with Top-K and random, then fixed exploration.
4. Add synthetic detector tests and adaptive Normal/Exploring/Cooldown control, requiring concentration AND validation stagnation.
5. Implement the required MPI/mpi4py execution path with one rank per client, then network experiments and final evaluation. SONAR is optional and must preserve MPI execution.

For gossip learning, all clients train before selection/exchange. Aggregate receiver-side from immutable post-training snapshots, with default weights 0.5 own and 0.5 selected peer. The selected peer is not automatically updated. Use replaceable negative flattened-parameter distance initially. Keep training, controller-validation, and test data separate.

These increments supersede the earlier milestone numbering as the execution order. Future milestones are documentation only until separately requested. No calendar commitment is assumed.

## Current simulator usage

From the repository root:

```sh
uv sync --project backend --locked
uv run --project backend peershift simulate --config backend/configs/default.toml --output /tmp/peershift-run
uv run --project backend pytest backend/tests
```

The output directory must be empty. Logs represent selections, not network transfers. Identical seeds/configurations reproduce logs in the same supported environment.

## Minimal IID gossip learning

Implemented CPU MNIST MLP training with Top-K/random selection, separate local validation and held-out test data, and receiver-side averaging from immutable post-training snapshots. Existing selection simulation remains available without ML dependencies. Non-IID, adaptive control, and networking remain future work.

```sh
uv sync --project backend --extra learning --locked
uv run --project backend --extra learning peershift train --config backend/configs/train_top_k.toml --output /tmp/peershift-learning --download
```

Use train_random.toml for the paired random baseline. Default settings: 10 clients, 10 rounds, 10 local steps, batch size 32, learning rate .05, own weight .5, hidden size 128, 500 training and 100 validation examples per client, 1000 test examples. All settings are configurable; only mnist/mlp/iid/cpu and top_k/random are supported for training. Data lives in backend/data by default. Omit --download once cached. Outputs include resolved configuration, exact partitions, environment/initial-model fingerprint, selection and evaluation JSONL, and summary counts/runtime. Round -1 denotes initial evaluation. Estimated parameter payload excludes similarity discovery and transport overhead; no network traffic occurs. Similarity uses full snapshot access as a simulation assumption. Same-environment reproducibility excludes runtime.

## Immediate exchange measurements

Training now writes aggregation.jsonl and summary.txt. Each client is evaluated on the same local validation examples immediately before and after averaging. Benefit is before-loss minus after-loss: positive indicates immediate improvement, negative worsening. Logs include selected-peer L2 distance and collaboration history metrics from preceding rounds. These diagnostics do not influence selection or establish harmful collapse. Summary counts use 1e-8 tolerance for unchanged results; additional validation time is recorded separately. Estimated model bytes and exchange counts retain their previous meaning.

After the documented training command, inspect `cat <output-directory>/summary.txt` and `head -n 3 <output-directory>/aggregation.jsonl`. Non-IID learning, fixed-exploration learning comparisons, adaptive control and networking remain future work.
