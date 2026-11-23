# SONAR and PeerShift: source-level comparison

Review date: 2026-11-23. No application changes, installs, or experiments.

## Conclusion

PeerShift currently implements its own research prototype motivated by SONAR's collaborator-collapse question. It does not currently wrap or extend the SONAR runtime: the independent local audit found no SONAR dependency, import, submodule or vendored implementation. Similar concepts alone do not establish source reuse or a faithful reimplementation.

## Verified primary sources

- [MLSys paper](https://proceedings.mlsys.org/paper_files/paper/2026/file/4abf39685e89559523a4d644036fd2b3-Paper-Conference.pdf), especially sections 2.3, 4 and 5.1. The paper explicitly identifies [aidecentralized/sonar](https://github.com/aidecentralized/sonar) as its official open-source repository. It describes orchestration/topology/communication/telemetry layers; MPI, gRPC and WebRTC support; directed pull-based collaboration; ResNet-10 experiments on CIFAR-10, DomainNet, Camelyon17 and Digit-Five. Its collapse discussion uses gradient alignment, 39 nodes across three domains and long horizons. These are paper claims, not independently reproduced results or verified current source behavior.
- [Official repository](https://github.com/aidecentralized/sonar), main branch README as fetched on review date: explicitly states MPI and experimental gRPC communication, modular component replacement, and warns of a major revamp. README statements establish claimed support, not tested compatibility.

## Comparison with the local implementation

PeerShift has its own IID MNIST MLP runner, negative-L2 parameter distance, randomized selection from a Top-K candidate pool, one peer per receiver per round, and default equal-weight averaging from frozen post-training snapshots. Scores currently access all models in memory. Its MPI demo uses fixed peers without training/discovery integration. Consequently it neither reproduces the paper's heterogeneous benchmark nor establishes equivalence to SONAR's algorithm semantics.

SONAR's stated MPI support means the project's mandatory MPI requirement does not itself rule out SONAR integration. Actual interoperability and snapshot/selection semantics remain unverified.

## Verified implementation at a pinned revision

The official repository was shallow-cloned into a temporary review directory for read-only inspection. The reviewed revision is **3200465cfaca2c8ce68b0dc811e4f2a1f317f09a**. No upstream code was executed.

### Dynamic selection differs materially from PeerShift

[ src/algos/fl_dynamic.py ](https://github.com/aidecentralized/sonar/blob/3200465cfaca2c8ce68b0dc811e4f2a1f317f09a/src/algos/fl_dynamic.py):

- `select_smallest_k` and `DynamicTopology.sample_neighbours` with sampling `closest` choose all K smallest-distance candidates. PeerShift randomly chooses one peer from a Top-K pool; K therefore has a different meaning.
- `DynamicTopology.l2_distance` sums individual state-tensor L2 norms and skips `num_batches_tracked`. PeerShift computes one Euclidean norm over concatenated parameters. These can produce different rankings even for the same weights.
- The configured `loss` comparison actually computes L2 distance between model outputs on the receiver's local training loader via `similarity_loss`; it is not a validation-loss improvement metric.
- `get_neighbor_model_wts` requests `comm_utils.all_gather(ignore_super_node=True)` each time `recv_and_agg` runs. The class has no discovery cache. Selecting K after collecting all candidate models is not sparse discovery traffic.
- `recv_and_agg` scores gathered models, selects K, and calls the client's aggregate method. `FedDynamicNode.run_protocol` performs local training, `local_round_done`, then aggregation, local testing and finalization. `FedDynamicServer.run_protocol` is a no-op. This does not itself prove equivalence to PeerShift's explicit frozen-snapshot semantics across all backends.
- Epsilon-greedy sampling appears as a TODO. The only implemented dynamic sampling branch in this file is `closest`; it does not provide PeerShift's fixed or adaptive exploration.

### Local aggregation and topology hooks

[ src/algos/base_class.py ](https://github.com/aidecentralized/sonar/blob/3200465cfaca2c8ce68b0dc811e4f2a1f317f09a/src/algos/base_class.py): `BaseFedAvgClient.aggregate` inserts the receiver's current model into the selected-model list. Without explicit coefficients it averages all included models equally, removes empty models and renormalizes coefficients, then installs the result locally. With K peers the default receiver coefficient is 1/(K+1); it equals PeerShift's default .5 only for one selected peer. It accepts optional collaboration coefficients and ignored keys, supplying a possible adaptation point.

[ src/algos/topologies/base.py ](https://github.com/aidecentralized/sonar/blob/3200465cfaca2c8ce68b0dc811e4f2a1f317f09a/src/algos/topologies/base.py): `BaseTopology` exposes `generate_graph`, `initialize`, neighborhood access and `sample_neighbours`. Sampling uses a seeded per-rank NumPy generator and samples K neighbors without replacement when enough exist. Node labels start at one because zero is reserved for the super-node. These are concrete extension points, although `DynamicTopology` currently mixes scoring, fetching and aggregation rather than exposing a separate policy interface.

### MPI source audit and integration caveat

The MPI communication backend and scheduler were inspected at the same SHA: MPI code exists, uses a threaded listener and `MPI.THREAD_MULTIPLE`, with a rank-zero server and positive-rank clients. However, in `src/utils/communication/comm_utils.py`, the `MPICommUtils` import is commented while the MPI factory branch refers to it, and manager/backend send signatures differ. These are static integration warnings, not demonstrated runtime failures. MPI source presence is established; compatibility of the current entrypoint with PeerShift remains untested. Upstream requirements pin torch 2.3 and mpi4py 3.1.6, requiring a deliberate environment decision before reuse.

Relevant pinned source: [MPI backend](https://github.com/aidecentralized/sonar/blob/3200465cfaca2c8ce68b0dc811e4f2a1f317f09a/src/utils/communication/mpi.py), [communication factory](https://github.com/aidecentralized/sonar/blob/3200465cfaca2c8ce68b0dc811e4f2a1f317f09a/src/utils/communication/comm_utils.py), and [scheduler](https://github.com/aidecentralized/sonar/blob/3200465cfaca2c8ce68b0dc811e4f2a1f317f09a/src/scheduler.py).

Paper-level gradient-alignment experiments and the examined current dynamic code's tensor/output distances should not be conflated: a code snapshot is not automatic proof of the exact published experimental implementation.

## Recommended wording and next decision

Describe the current project as an independent prototype inspired by SONAR's research findings. To call it built on SONAR, pin an official source revision, inspect and execute its actual runner/client/communication interfaces, then implement PeerShift's detector/controller through verified extension points (or explicitly document an adapted fork). Validate matching selection, averaging, round semantics and benchmark controls. To claim a SONAR reproduction instead, define the precise paper experiment and reproduce its settings/results; sharing a Top-K idea is insufficient.

Paper PDF author order and publisher metadata differ among equal-contribution authors; do not treat the earlier shorthand-author correction as a scientific discrepancy.
