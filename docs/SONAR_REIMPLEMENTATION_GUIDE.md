# Rebuilding SONAR core functionality in PeerShift

This is an engineering guide for implementing a new decentralized-learning testbed using SONAR's core ideas, without depending on its runtime or copying its implementation. The target is a small, reproducible system that can compare collaboration policies and later evaluate PeerShift's collapse detector and adaptive exploration.

The central idea is straightforward: clients train separate models, choose collaborators, exchange representations, update their own models, and record the learning and communication consequences. Network structure is an experimental variable, alongside the model and data distribution. SONAR presents this as a modular combination of orchestration, topology, communication, and telemetry. [Paper, sections 2 and 4][paper]

**Scope:** documentation and proposed implementation contracts. This file does not add runtime functionality or replace the current implementation plan. “Upstream behavior” refers to inspected source at `3200465cfaca2c8ce68b0dc811e4f2a1f317f09a`; “new implementation” describes the design to build here. Reproducing a particular published experiment additionally requires matching its configuration and measured outcomes.

## 1. What to preserve and what to leave optional

Preserve these core capabilities:

- Independent client state and local learning, without a global model aggregator.
- Configurable communication graphs and interchangeable peer-selection policies.
- Actual peer communication, with MPI required for PeerShift.
- Receiver-local model averaging with explicit weights and round semantics.
- Reproducible data partitions, initialization, random streams, and experiment records.
- Joint measurement of learning progress, collaboration patterns, and systems cost.

SONAR covers broader algorithms, datasets, and robustness/privacy studies. Knowledge distillation, split learning, learned collaboration weights, malicious clients, dropout, gRPC, and browser communication are separate extensions, not prerequisites for this core. The paper's breadth is not evidence that every backend and algorithm in a particular source revision is production-ready. [Paper, sections 4–5][paper]; [repository status][upstream]

For PeerShift, complete a small CPU model-averaging implementation first. Heterogeneous model architectures require a different exchange/aggregation contract; full-model averaging requires compatible named tensors.

## 2. Verified SONAR behavior that affects compatibility

| Component | Upstream idea and implementation | Consequence for a new implementation |
| --- | --- | --- |
| Experiment entry point | `main.py` delegates to `Scheduler`; configuration and communication determine algorithm/node creation. | Keep setup outside client learning logic. |
| Algorithm selection | The scheduler maps an algorithm name to server/client classes; rank zero has a distinct role. | Preserve configurable algorithms, but do not copy rank conventions implicitly. |
| Static topology | `BaseTopology` separates graph construction, neighbor access, and neighbor sampling. | Implement graph generation independently from scoring and sampling. |
| Dynamic topology | `DynamicTopology` gathers candidate models, computes distances, chooses collaborators, then calls client aggregation. | Separate discovery, scoring, selection, and aggregation into testable functions. |
| Closest selection | `select_smallest_k` returns all K candidates with the smallest distances. | K means actual collaborators in this upstream path. |
| Default averaging | `BaseFedAvgClient.aggregate` includes self and equally weights all supplied models unless coefficients are supplied. | With m peers, self receives weight `1/(m+1)`. |
| Communication | `MPICommUtils` uses request messages and a threaded listener to serve model requests. | Retain real message exchange; the threading design is not required. |
| Round helpers | `round_init` starts timing, `local_round_done` increments a counter, and `round_finalize` logs statistics. | These names do not establish synchronization or immutable snapshots. |
| Model representation | `get_model_weights` wraps `state_dict()` tensors moved to CPU without explicit cloning. | Moving an already-CPU tensor does not guarantee independent storage. |

Source mapping: [scheduler][scheduler], [topology base][topology], [dynamic collaboration][dynamic], [client primitives][base], and [MPI backend][mpi]. These are behavioral references, not code to import.

### Differences from the current PeerShift baseline

The legacy PeerShift `top_k` policy chooses **one random peer from a Top-K pool**, uses one L2 norm over flattened parameters, and averages self/peer with default weights 0.5/0.5. Those are valid experimental choices, but not the examined SONAR closest-K rule.

For example, with K=2, SONAR's closest-K path combines self and two peers at default weights 1/3 each. The legacy PeerShift policy combines self and one sampled peer at 1/2 each. The new explicit `closest`/`sum_tensor_l2`/`uniform_self_and_peers` training profile implements the upstream-style selection and averaging formulas independently. Using the same label “Top-K” does not make these equal-budget or equivalent algorithms.

The paper discusses gradient-alignment-based collapse experiments; the inspected dynamic source implements tensor-distance and output-distance options. Do not assume that the latest repository state is the exact code/configuration used for every published figure. [Paper, section 2.3][paper]; [dynamic source][dynamic]

## 3. Architecture for the new implementation

A client is a learning participant; an MPI rank is its process ID. A topology defines eligible communication links. Discovery obtains the information needed to evaluate candidates. Telemetry records what happened.

| Subsystem | Owned state | Function contracts to implement |
| --- | --- | --- |
| Experiment setup | Resolved configuration, versions, seeds, partition manifest | `validate_config`, `prepare_run`, `initialize_clients` |
| Local learner | One model, optimizer, batch stream, train/validation subsets | `train_local`, `evaluate`, `freeze_snapshot` |
| Topology | Eligible IDs or graph and topology seed | `eligible_peers`, `build_graph` |
| Discovery | Current candidate representations and acquisition cost | `discover_candidates` |
| Similarity | Metric name and any receiver-only probe data | `score_candidates` |
| Selection | Policy configuration and independent random streams | `select_peers` |
| Transport | Communicator, model schema, round/request identities | `validate_round`, `exchange_snapshots`, `close` |
| Aggregation | Weight rule and tensor/buffer handling rules | `aggregate_local` |
| Observation | Rolling histories, validation trajectory, event records | `observe_round`, `write_metrics` |
| Future controller | Normal/Exploring/Cooldown state and trigger counters | `update_controller` |

These are proposed responsibilities, not a requirement to create ten classes. Keep existing small functions where possible. Discovery must report candidates and costs; selection must return decisions rather than mutate models. Aggregation must have no hidden peer-selection behavior.

Minimum data contracts:

- **Snapshot:** owner ID, round ID, schema ID, ordered named tensors, immutable storage.
- **Candidate view:** eligible IDs, available representations, representation rounds, acquisition costs. Reject stale representations in the synchronous reference mode.
- **Selection decision:** selected IDs, policy/branch, scores used, and fallback reason.
- **Exchange result:** received snapshots by sender, application payload/control counters, timing, and completion status.
- **Client observation:** prior history, before/after validation loss, selected peers, and current policy/controller state.

For the first implementation, use one common model schema, fixed membership, CPU execution, and synchronous rounds. Rank zero remains a normal learning client; it may distribute setup and collect logs. This intentionally differs from upstream's additional server rank. Only the client updates its own model.

## 4. Round semantics and model ownership

The new synchronous reference contract is:

1. Each client trains for the configured amount of local work.
2. Every client creates an owned snapshot of its post-training parameters.
3. All ranks reach a phase alignment point before discovery/exchange proceeds.
4. Each client evaluates its local post-training model on its own validation set.
5. Discovery obtains this round's candidate information; scoring and selection use only permitted observations.
6. Transport obtains the selected snapshots, or explicitly reuses exact snapshots already obtained by discovery.
7. Each receiver averages locally from its own snapshot and the selected snapshots.
8. Evaluate after aggregation, record events, update history, and prepare the next round.

A request to a peer does not update that peer. A peer may serve several receivers while selecting a different peer itself. The implementation must never assume selections are mutual.

Snapshots must remain unchanged through scoring, sends, and aggregation. CPU tensors need cloning, not only `detach()` or a device conversion. Receive buffers must not share storage with live sender models. Complete requests before reusing send buffers.

These are deliberate PeerShift guarantees. The inspected SONAR round helpers and current-model-on-request MPI backend do not themselves guarantee this frozen, globally aligned behavior. [Client lifecycle and representation][base]; [MPI backend][mpi]

Upstream local work is configured in epochs in the examined loops. PeerShift uses local minibatch steps. Record which unit is used, including batch size, sample count, and examples processed; one epoch and one step are not interchangeable experiment budgets.

## 5. Topologies, similarity, and collaborator selection

### Static topologies

Start with isolated/local-only, ring, uniform random, and fully connected reference graphs. Later add grid/torus if required by an experiment. Graph generation returns allowed peer IDs; a separate sampler chooses actual collaborators. Never select self or duplicate a collaborator. Directed graphs need separate incoming/outgoing interpretations.

A local-only baseline performs zero model exchanges and must be reported as a different communication budget. For policies that should have matched budgets, require the same number of selected peers and local training steps.

### Similarity metrics

Keep metric names and formulas explicit. For matching tensor keys p:

- **Sum of tensor L2 distances:** `d(i,j) = sum_p norm(W_i[p] - W_j[p], 2)`. This matches the inspected upstream `weights_l2` calculation, excluding its `num_batches_tracked` keys.
- **Flattened parameter L2:** `d(i,j) = norm(concat_p W_i[p] - concat_p W_j[p], 2)`. This is current PeerShift behavior; it is generally not equal to the first metric.
- **Output L2 distance:** compare both models' predictions on a specified receiver-local probe set. Upstream's option named `loss` measures output distance on its local training loader, not validation-loss improvement. [Metric implementation][dynamic]

All distances above are smaller-is-closer. If the policy API instead accepts similarity scores, negate the distance explicitly. Validate finite scores, matching schemas, and stable candidate-ID ordering. Document normalization and tie-breaking. The new implementation should break equal scores by peer ID; upstream reproduction also requires aligning the candidate enumeration/order.

For the proposed output-distance metric, evaluate both snapshots on the **same ordered probe examples in evaluation mode**, with stochastic transforms disabled or identically realized. Use a separate probe model or restore all receiver parameters, buffers, and mode afterward. Repeated passes through a shuffled loader do not guarantee aligned outputs. This is a correctness requirement for the new implementation, not a guarantee of the inspected upstream code.

Gradient alignment and cosine similarity can be later metrics. Specify the exact vectors, probe data, and acquisition cost before adding them. Parameter similarity is a hypothesis about useful collaboration, not measured receiver utility.

### Two different meanings of K

Use separate settings:

| Setting | Meaning |
| --- | --- |
| `num_collaborators` | Number of distinct peers actually used in aggregation |
| `candidate_pool_size` | Size of a shortlist used by a sampling policy |
| `exploration_probability` | Probability of taking the exploration branch |

A `closest` rule selects the nearest `num_collaborators` peers, following upstream's closest-K idea. A `top_pool_random` rule samples from `candidate_pool_size`; current PeerShift samples one. These must be separately named in results.

For current PeerShift fixed exploration, the exploration branch samples outside the Top-K shortlist, not outside recent history. Preserve the existing primitive that excludes history as a separate policy. If no outside candidate remains, use the documented Top-K fallback and log it.

A future generalized multi-peer sampler must return distinct IDs and either satisfy its configured count or fail clearly; do not silently change the budget. Baselines should receive the same eligible graph and the same candidate information when comparing selection algorithms.

## 6. Aggregation and model compatibility

For selected peers S_i, the default upstream-style rule is:

`W_i_next = (W_i_local + sum_{j in S_i} W_j_local) / (1 + len(S_i))`

A separate fixed-self-weight rule is:

`W_i_next = alpha * W_i_local + (1-alpha) * mean_{j in S_i}(W_j_local)`

Both give 0.5 self/0.5 peer when there is one peer and alpha=0.5. For more peers, specify which rule applies. Explicit coefficients must be finite, nonnegative, and normalized over the actual included models. Local-only behavior leaves the trained model unchanged. [Upstream aggregation][base]

The initial transport supports dense CPU float32 tensors. Retain this constraint for the first integrated MPI learner. Reject unsupported dtypes/layouts instead of coercing them silently.

Before supporting models with integer buffers or BatchNorm counters, define separate buffer rules and tests; do not average integer state blindly. An ignore rule in a distance metric is not automatically an aggregation rule. Compatible architecture, parameter names/shapes, and serialization order are required for full-model averaging. Distillation is a separate protocol for genuinely different architectures.

Keep optimizer state local. With current momentum-free SGD there are no momentum buffers to reconcile. Adding momentum/Adam requires an explicit policy for optimizer state after model replacement; resetting it and retaining it are different algorithms.

## 7. MPI communication and discovery

Use MPI through `mpi4py`, one rank per client, with a private communicator for protocol messages. Our transport supports multiple requested peers per rank through `exchange_many`, with the one-peer `exchange` compatibility wrapper. The `mpi-collaborate` fixture implements dense full-snapshot discovery, local closest-peer selection and merging; training integration remains pending.

For a synchronous multi-peer design:

1. Announce selected IDs, round, schema, and local validation status collectively as small metadata.
2. Reject inconsistent schemas, stale rounds, invalid IDs, or input errors before payload traffic.
3. Derive incoming requesters from the announced directed selections.
4. Post an explicit-source receive for every selected peer and a nonblocking send to every requester.
5. Complete all operations, validate lengths/identities, and align the phase before buffer reuse.

Keep buffers alive until completion, and distinguish send completion from receive completion. Use tests for cycles, mutual selections, multiple requesters, zero incoming requests, and payloads beyond the eager-buffer threshold. A missing rank can still hang a collective; the first version uses job abort plus external launcher timeouts, not crash recovery. [MPI completion rules][mpi-completion]

### Discovery is a separate systems problem

Static and random policies can choose from known eligible IDs without full model discovery. Exact global parameter-distance selection needs candidate model information.

Provide an explicit **full-snapshot reference mode** when reproducing the inspected dynamic rule: collect current candidate snapshots, score them locally, then select. Count all acquisition traffic, even if only one selected peer contributes to averaging. If the selected snapshot was already received during discovery, reuse it explicitly and count that transfer once; do not pretend discovery was free or count the same bytes twice. Upstream's dynamic class gathers candidates before selection. [Discovery path][dynamic]

Later sketches, cached candidates, neighborhood-only discovery, or less-frequent refresh can reduce cost, but change information availability. Record their freshness/scope and compare them as separate methods, not automatically equivalent implementations.

Do not carry forward upstream's threaded listener or assume it is required for MPI. Our completed-request protocol avoids that design. The inspected upstream factory also has a commented MPI import and manager/backend interface inconsistencies; a new implementation should define one tested transport interface rather than inherit these details. [MPI source][mpi]; [communication factory][factory]

## 8. Data, configuration, and reproducibility

Separate the dataset from the partition strategy. Start with MNIST/MLP mechanics; add controlled label skew, then multiple domains if evaluating domain-dependent collaborator collapse. Label skew is not the same as domain shift.

The SONAR paper's benchmark uses ResNet-10 with CIFAR-10, DomainNet, Camelyon17, and Digit-Five. These supply a later experimental comparison target; matching a small MNIST run alone does not reproduce those results. [Paper, section 5.1][paper]

Upstream includes IID, support-class, and non-IID data paths, but not every named strategy is implemented; validation is optional in that source and support subsets may overlap. For PeerShift's default experiment contract, persist disjoint per-client training and validation indices and keep held-out test data separate. Explicitly label any experiment that intentionally permits overlap. [Client data setup][base]; [partition utilities][data]

Persist enough configuration to determine the entire algorithm:

- Dataset version, partition strategy, indices, domain/class distributions, and sample counts.
- Model schema, initialization seed, optimizer settings, batch size, and local-work unit/count.
- Eligible graph, metric, selection rule, candidate-pool size, collaborator count, aggregation rule/weights, and exploration/controller settings.
- Discovery mode, communication backend, client/rank mapping, round count, and protocol version.
- Independent seeds for partitions, initialization, minibatches, peer sampling, and exploration.

All clients begin from identical model weights in the reference experiment. Keep model storage independent. Per-client random streams must not depend on process scheduling, rank-zero execution order, wall time, or Python's process-randomized hash. Paired policy runs use the same initial models, partitions, batch schedules, and training budgets.

Record the code revision, dependency/runtime versions, resolved configuration, and seeds. Define same-environment numerical tolerances for MPI/reference comparisons; cross-platform bitwise equality is not the default promise. Stop or abort with a shared error when configuration/rank counts disagree.

## 9. Telemetry and experimental interpretation

Record enough information to separate local training from collaboration:

| Group | Minimum observations |
| --- | --- |
| Identity | Run, round, client/rank, model schema, seed and configuration references |
| Learning | Local training loss; validation before/after aggregation; held-out test loss/accuracy; exact local work |
| Selection | Eligible/available candidates, selected peers, scores, branch/fallback, representation age |
| Diversity | Prior-history length, entropy, consecutive repetition, unique collaborators, and window size |
| Transport | Payload bytes/messages sent and received; discovery/control/log traffic separately; timing and wait duration |
| Controller | State, trigger evidence, exploration probability, transitions and cooldown |

Compute immediate aggregation benefit as `validation_before - validation_after` on the same receiver validation examples. Positive means immediate improvement on those examples; it does not establish long-term generalization or causality.

For single-peer history, entropy uses frequencies normalized by log(number of eligible peers); consecutive repetition detects adjacent repeats. Cycling between two peers may have zero repetition and still be concentrated. For multiple peers per round, define the counting unit explicitly: neighbor-set events, individual peer selections, and model-weighted contributions are different metrics. Do not reuse the single-peer formula without defining the extension.

Upstream records training/testing, neighbors, timing, communication-counter returns, and memory estimates. The examined MPI counters are initialized without updates in that file, and a field called `peak_dram` sums CPU tensor storage rather than measuring OS peak RSS. The new system needs its own validated measurement definitions. Do not infer accurate wire bytes, peak process memory, or latency from field names. [Logging hooks][base]; [MPI counters][mpi]

Actual model-buffer bytes, logical encoded metadata bytes, and network wire traffic are different quantities. Record payload counters exactly; document collective accounting and avoid summing send and receive totals as distinct exchanges. Label memory-based estimates separately from real transport measurements. Report wall time and, when useful, a clearly defined area under an accuracy-versus-round curve.

## 10. PeerShift extensions and staged delivery

The research addition is an adaptive exploration controller, not merely another transport. It should observe local validation progress and prior collaboration history without using held-out test outcomes.

Keep these proposed states explicit:

- **Normal:** baseline selection and configured normal exploration.
- **Exploring:** temporarily increase exploration after persistent concentration AND validation stagnation, with adequate warmup/history.
- **Cooldown:** suppress new triggers for a configured period before returning to Normal.

Specify trigger windows, minimum improvement, patience, duration and reset rules in a separate implementation plan. Test concentration without plateau, plateau without concentration, noise, recovery, and adverse exploration. Upstream's examined dynamic sampler has closest selection and exploratory TODOs; do not describe the PeerShift controller as already supplied by SONAR. [Dynamic sampler][dynamic]

Implement in bounded feature increments:

| Increment | Deliverable and completion condition |
| --- | --- |
| Existing reference | Preserve the working single-process learner, diagnostics, and tests. |
| MPI learning | Connect local training to real exchange using random selection first; verify per-client models and observations against the reference. |
| SONAR-style baselines | Add explicit closest-collaborator semantics, matching distance/aggregation choices and accounted discovery; retain current pool-sampling baselines under distinct names. |
| Heterogeneous experiments | Add reproducible non-IID/domain partitions and paired baseline comparisons. |
| Adaptive controller | Add synthetic-trace verification, then integrate observation-driven exploration. |
| Systems evaluation | Validate multi-host operation, controlled latency/bandwidth profiles, and end-to-end communication/performance trade-offs. |

This sequence rebuilds the relevant concepts using our own implementation. It does not require reproducing every SONAR algorithm or adopting its original module layout.

## 11. Current components to reuse and remaining work

| PeerShift component | Current capability | Work remaining |
| --- | --- | --- |
| `learning.py` | MLP, local SGD, snapshots, flattened and summed tensor-L2 scores, fixed-self and uniform multi-peer averaging | Additional model/buffer types or metrics only when required |
| `selection.py` | Legacy pool/random/exploration primitives plus deterministic closest-m selection | Generalized sampling for non-closest multi-peer policies if required |
| `training.py` | IID training with three legacy policies and the closest multi-peer baseline, distinct pool/collaborator settings, joint benefit diagnostics | Reusable per-client runtime, MPI runner, non-IID partitions |
| `model_buffer.py` | Owned CPU float32 serialization and schema hashing | Additional tensor/buffer types only when a selected model requires them |
| `mpi_transport.py` | One selected snapshot per rank, real sends, collective validation and round completion | Learning integration; multiple selected peers if required by a baseline |
| `mpi_demo.py` | Deterministic exchange/averaging fixtures with known peer patterns | Remains a transport correctness fixture, not a training benchmark |
| `detector.py` | Entropy and repetition primitives | Plateau detection and controller state machine |

No current component needs to be discarded merely to adopt SONAR's architecture. The essential change is to make algorithm semantics, discovery costs, and process ownership explicit and independently testable.

## 12. Acceptance checks for a new implementation

Before claiming the core works:

- Check analytic local averaging, independent model storage, immutable snapshots, and processing-order independence.
- Distinguish closest-m selection from one-from-K sampling with fixtures whose rankings and expected recipients are known.
- Verify metric formulas on tensors where summed and flattened distances give different rankings.
- Audit actual train/validation/test sample access and paired minibatch schedules, not only written partition manifests.
- Test policy probability boundaries, exhausted exploration pools, prior-history warmup and rolling-window eviction.
- Match final tensors, peer choices, and deterministic records between serial and MPI modes under a declared tolerance.
- Run real MPI cycle, many-to-one, zero-incoming, repeated-round, malformed-input and large-payload cases with launcher timeouts.
- Check payload accounting against known buffer sizes, including discovery acquisitions.
- Test that telemetry/evaluation changes do not alter the learner's model trajectory.
- Use synthetic traces for controller behavior; use repeated controlled experiments for claims about learning benefit or collapse.

Reuse the project's tester for each feature. Required learning/MPI verification must not silently skip unavailable components. Tests establish implementation contracts; reproduction of SONAR's results requires a separately documented benchmark configuration and repeated measurements.

## Sources and related documentation

The source pin separates stable references from future upstream changes. The paper and current code may describe different algorithm variants; preserve that distinction when reporting results.

- [SONAR paper][paper] — framework objectives, collaboration findings, and published experimental context.
- [Official SONAR repository][upstream] — source and current project status.
- [Prior source comparison](research/2026-11-23-sonar-implementation-comparison.md) — detailed differences from PeerShift.
- [Core-contracts audit](research/2026-11-23-sonar-core-contracts-audit.md) — lifecycle, data, and telemetry caveats.
- [Current MPI transport guide](MPI_TRANSPORT.md) — implemented transport behavior and launch/verification commands.
- [Project execution plan](../PROJECT_PLAN.md) — currently authorized milestones.

[paper]: https://proceedings.mlsys.org/paper_files/paper/2026/file/4abf39685e89559523a4d644036fd2b3-Paper-Conference.pdf
[upstream]: https://github.com/aidecentralized/sonar
[scheduler]: https://github.com/aidecentralized/sonar/blob/3200465cfaca2c8ce68b0dc811e4f2a1f317f09a/src/scheduler.py
[topology]: https://github.com/aidecentralized/sonar/blob/3200465cfaca2c8ce68b0dc811e4f2a1f317f09a/src/algos/topologies/base.py
[dynamic]: https://github.com/aidecentralized/sonar/blob/3200465cfaca2c8ce68b0dc811e4f2a1f317f09a/src/algos/fl_dynamic.py
[base]: https://github.com/aidecentralized/sonar/blob/3200465cfaca2c8ce68b0dc811e4f2a1f317f09a/src/algos/base_class.py
[mpi]: https://github.com/aidecentralized/sonar/blob/3200465cfaca2c8ce68b0dc811e4f2a1f317f09a/src/utils/communication/mpi.py
[factory]: https://github.com/aidecentralized/sonar/blob/3200465cfaca2c8ce68b0dc811e4f2a1f317f09a/src/utils/communication/comm_utils.py
[data]: https://github.com/aidecentralized/sonar/blob/3200465cfaca2c8ce68b0dc811e4f2a1f317f09a/src/utils/data_utils.py
[mpi-completion]: https://www.mpi-forum.org/docs/mpi-5.0/mpi50-report/node76.htm
