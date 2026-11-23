# SONAR core contracts audit for an independent implementation guide

Date: 2026-11-23. Static source audit only; no upstream execution. Source revision: `3200465cfaca2c8ce68b0dc811e4f2a1f317f09a` from the official repository's parent-provided temporary checkout. This note clarifies implementation facts; it does not prescribe copying upstream code.

## Round lifecycle: names do not imply synchrony

[Base classes](https://github.com/aidecentralized/sonar/blob/3200465cfaca2c8ce68b0dc811e4f2a1f317f09a/src/algos/base_class.py), `BaseNode`:

- `round_init` (line 302) records the start timestamp only.
- `local_round_done` (266) increments `self.round` only. It is not a global training-completion barrier.
- `round_finalize` (308) calculates duration, adds cumulative elapsed time, reads communication and memory metrics, and logs the stats. No snapshot creation or synchronization appears here.
- `get_model_weights` (269) constructs sender/round/model metadata from `model.state_dict()` (or split-model state) and moves entries to CPU. It does not explicitly clone or deep-copy tensors. For a CPU model, `.to('cpu')` need not create independent storage; the returned mapping must not be assumed an immutable snapshot. GPU-to-CPU movement may copy, but that does not establish a cross-backend freeze contract.
- `set_model_weights` uses a shallow mapping copy, moves entries to the device and loads them into the model. A shallow mapping copy is not tensor snapshot isolation.

[Dynamic algorithm](https://github.com/aidecentralized/sonar/blob/3200465cfaca2c8ce68b0dc811e4f2a1f317f09a/src/algos/fl_dynamic.py), `FedDynamicNode.run_protocol`: round timer, local epochs, increment local round, gather/score/select/aggregate, local test, finalize logging. This ordering alone does not prove a synchronous global round; transport semantics matter.

**Guide requirement:** explicitly label frozen post-training snapshots, all-client phase alignment and immutable buffers as PeerShift's deliberate reference contract. Do not claim these base lifecycle methods already enforce it upstream.

## Training, data and heterogeneity

[BaseClient.set_parameters/set_data_parameters/local_train](https://github.com/aidecentralized/sonar/blob/3200465cfaca2c8ce68b0dc811e4f2a1f317f09a/src/algos/base_class.py): seeds are set before model/data creation; local training loops over epochs rather than fixed optimizer-step counts. Configuration supports per-client dataset assignments; IID slices come from a common seeded permutation within dataset groups. `support` filters classes and may overlap examples across clients, as documented in the source. Label-distribution branches ending `non_iid` call `non_iid_balanced` with `alpha_data` (default .4); an inter-domain branch supplies class priors. Actual partition properties require inspecting/testing that helper, not inferring them from its name. The `shard` branch raises `NotImplementedError`.

`validation_prop` optionally splits a client's selected training data and creates a validation loader; the default is zero. Validation separation is therefore supported but not universally guaranteed. Test distribution has IID, support and non-IID branches and optional subsampling. A rebuild should save exact partition indices and class counts and explicitly enforce disjoint training/validation/test roles.

[ModelUtils.get_model](https://github.com/aidecentralized/sonar/blob/3200465cfaca2c8ce68b0dc811e4f2a1f317f09a/src/utils/model_utils.py) has ResNet-6/10/18/34/50 and YOLO branches. Per-node architecture flexibility in a framework does not imply arbitrary architectures can be weight-averaged; PeerShift's full-model exchange requires matching layouts and semantics.

## Telemetry: available fields versus measurement guarantees

`BaseNode.log_metrics` writes scalar metrics to TensorBoard, NumPy arrays to NPY, all stats to CSV and optional images. Local train/test update loss, accuracy and duration; dynamic training records neighbors. `round_finalize` records elapsed time, communication-cost return values, and memory fields. These are concrete hooks, not proof that every promised statistic is measured correctly.

`get_memory_metrics` returns zero when memory logging is disabled. With logging enabled, its CPU calculation sums currently discoverable CPU tensor storage estimates, while the GPU field uses PyTorch peak allocated memory. The `peak_dram` name should not be repeated as a verified OS process peak-RSS measurement.

[Inspected MPI backend](https://github.com/aidecentralized/sonar/blob/3200465cfaca2c8ce68b0dc811e4f2a1f317f09a/src/utils/communication/mpi.py) initializes byte counters and exposes `get_comm_cost`, but no counter increments appear in that file. Its listener gets the current model on request; there is no local-round eligibility check in the inspected send/receive path. Its `all_gather` calls `receive(i)` although `receive` expects a one-element list; backend/manager signature mismatches need integration repair, not imitation. No actual runtime claim is made.

The inspected base/dynamic files do not establish automatic latency tracing, graph-entropy calculation, validation-plateau detection, or immutable-model persistence. Such features may exist elsewhere or in other revisions; the guide should specify independently verified output contracts instead of advertising paper-wide claims as source-complete behavior.

## Recommended guide boundaries

1. Rebuild ideas and contracts independently: topology as an experimental variable; local model ownership; explicit transport; receiver-side aggregation; auditable telemetry. Distinguish faithful SONAR-style algorithm semantics from intentional PeerShift simplifications.
2. Require snapshot/round/message invariants and accurate payload/discovery/control accounting through tests. A timer or round counter does not enforce a distributed phase boundary.
3. Keep the initial model family homogeneous, data partitioning explicit and local-step budgets matched. Add extra architectures, attacks, domains and dashboards only after the small reference/MPI comparison works.

No new web sources were needed. Confidence is high for the cited function bodies, limited for repository-wide feature absence and runtime behavior.
