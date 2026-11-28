# IID MPI training parity review

Date: 2026-11-28. Scope: connect the serial IID learner to existing dense MPI discovery for closest selection, initially three ranks/one collaborator. Static advisory review only; no application changes, installs, experiments or downloads.

## Recommendation

Proceed with a bounded MPI learner using one model/SGD optimizer/local batch iterator per rank and the same existing learning, scoring, selection, aggregation and diagnostic functions as the serial reference. Do not reuse mpi-collaborate's distinct per-rank fixture initialization. All learning ranks must start from identical weights. Keep non-IID partitions, additional MPI policies, adaptive control and multi-host deployment outside this increment.

## Locally verified facts and parity hazards

Inspected current README, PROJECT_PLAN, training.py, mpi_transport.py, mpi_collaboration.py, learning.py, model_buffer.py, tests/conftest.py and the reimplementation guide.

- Serial initialization seeds torch once, snapshots one MLP, then loads that state into every independent model. Each training DataLoader has an explicit generator seeded `config.seed + client_id + 1`. Its iterator survives round boundaries and cycles over shuffled epochs. Preserve batch-order continuation, including partial final batches and shuffle boundaries; resetting the loader each round changes the algorithm.
- Partitioning uses independently seeded Python Random instances for training permutation and test subset permutation. Each client gets fixed disjoint train/validation slices; the official test split is independent. Shared helpers should return exactly the existing indices, not replace this with a new splitter. MPI rank count must equal client count.
- The serial scorer and averaging implementation iterate model mappings. Transport packing sorts keys and reconstruction retains sorted order, while a local MLP state_dict uses registration order. Rebuild every received mapping in the local model's state_dict key order before scoring/merge to match reference summation order. Keep candidate snapshots indexed by rank and selected peers in closest_select's distance/ID order; sorting selected peers before arithmetic could change floating-point results.
- Evaluation loaders currently omit explicit generators, so iterator creation may consume process-global torch RNG even when shuffle=False. Global RNG state will differ between a serial process and per-rank processes. Current MLP has no dropout and ToTensor has no augmentation, while training batch permutations have independent generators: these assumptions make that difference non-operative for the present model trajectory. Do not promise arbitrary stochastic-model parity. Future random transforms/dropout need explicit per-client RNG ownership.
- MPITransport.exchange_many performs collective validation, then explicit-source receives/sends and completion synchronization. Entering its validation after local training and snapshot creation ensures no payload traffic starts until every rank has reached that phase. It cannot handle a rank that raises and exits before entering the collective.
- The current closest serial diagnostics use peer groups and joint before-minus-after validation benefit. Preserve prior-history timing and evaluation subsets; benefit belongs to the group, not to individual peers.

## Fairness and information boundaries

Use the same initial fingerprint, indices, local-step count, examples/batches, CPU thread setting, deterministic algorithms, model schema, metric, collaborator count and averaging coefficients for paired serial/MPI runs. Evaluation must use inference mode/no gradients, keep training/test data out of policy tuning, and restore train mode before the next local step through local_train. Test evaluation is observational only. Reuse discovered snapshots directly; no selected-peer retransmission or counterfactual test-based selection.

Root may validate/broadcast resolved configuration, prepare the output directory and download MNIST once. Broadcast success/error before peers continue; all ranks then open the same cache with download disabled and validate availability. The first supported deployment should explicitly require a shared cache/output path on one host. Every rank may independently calculate partition indices, but only its owned train/validation loaders should be built; computing all index assignments is not centralized learning.

## Accounting and diagnostic outputs

For N ranks, m selected peers and R rounds:

- Physical application-level model discovery sends: `R*N*(N-1)` globally, with equal receive totals. Count transfers once using sent totals, not sends plus receives.
- Selected collaborator contributions: `R*N*m`; these are logical uses of acquired models, not additional network messages.
- Selected snapshot retransmission messages/bytes: zero.
- Aggregation events: `R*N`; optimizer steps: `R*N*local_steps`, not multiplied by m.
- Model payload bytes derive from actual transport buffers; metadata/control/log encoded sizes and collective calls remain separately labelled. They are not MPI wire bytes.

Retain rank-local losses and selection diagnostics, shared identities and source fingerprints. Keep evaluation cost, discovery/wait time and whole-job time distinct. Timing differences are expected and excluded from deterministic comparisons.

## Failure gates

Before startup/payload phases, share validation errors when all ranks can still participate. After an unexpected rank-local exception in loading, training, evaluation, scoring, model decoding or log writing, abort the communicator/job rather than allowing other ranks to hang. Root-only operations need propagated failures. Do not put a collective Free/finalization in an exception path where not all peers can reach it. External launcher timeouts remain mandatory in tests; MPI rank recovery is not part of this increment.

## Minimum acceptance checks

1. Existing serial fixtures produce unchanged deterministic outputs after helper extraction. Check legacy policies too, not only closest.
2. Paired three-client fixture runs compare exact partitions, initialization fingerprint, selected IDs, history metrics, and before/after diagnostic values; compare final tensors with a declared same-environment tolerance. Actual consumed minibatch IDs should be checked across more than one epoch boundary.
3. Verify transport reconstruction order using names whose alphabetical and registration orders differ; exercise both distance modes if both are supported. Confirm local/source snapshots remain unmodified.
4. Repeat MPI runs in separate jobs and compare deterministic records. Test m=1 first and at least one m>1 boundary case if configuration permits it; m=N-1 should use all acquired models without extra sends.
5. Force rank mismatch, missing cache, insufficient samples, occupied output, malformed snapshot, and a rank-local pre-collective failure; jobs must exit nonzero under launcher timeout, never present partial logs as success.
6. Require learning and MPI dependencies/tests explicitly so an absent component cannot turn verification into skips. Run a small real-MNIST serial/MPI smoke with the same configuration; smoke success establishes mechanics, not research superiority.

## Evidence basis and limitations

The existing [reimplementation guide](../SONAR_REIMPLEMENTATION_GUIDE.md) and [MPI requirements audit](2026-11-06-mpi-requirements.md) already establish the architectural and buffer-lifetime contracts. Primary documentation underlying that audit: [mpi4py tutorial](https://mpi4py.readthedocs.io/en/stable/tutorial.html) and [MPI communication completion](https://www.mpi-forum.org/docs/mpi-5.0/mpi50-report/node76.htm). No fresh external survey was needed. This review predicts necessary controls; it does not establish runtime parity, performance, or the scientific value of closest collaboration.
