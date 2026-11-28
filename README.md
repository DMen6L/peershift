# PeerShift

**Adaptive Peer Exploration for Decentralized Learning**

PeerShift is a research project developed for the Parallel and Distributed Computing course (Group 104).

The project investigates **collaborator collapse** in decentralized machine learning, where clients repeatedly select a limited group of peers for collaboration. While selecting similar peers can improve learning, excessive concentration may reduce collaboration diversity and potentially harm model generalization.

PeerShift aims to detect potentially harmful collaboration patterns and dynamically introduce alternative peers when needed.

## Objectives

The project focuses on three research questions:

1. **Detection:** Can peer-selection history and stagnating validation performance identify harmful collaboration concentration?
2. **Mitigation:** Can adaptive peer exploration improve model generalization compared with similarity-based selection and fixed exploration?
3. **Systems cost:** How do network latency and bandwidth differences affect the effectiveness and overhead of peer exploration?

## Proposed Architecture

The intended system consists of multiple decentralized clients. Each client maintains its own model and training data, selects collaborators, exchanges model updates, and performs local aggregation.

PeerShift introduces two additional components:

- **Collapse Detector:** Monitors collaboration diversity, peer repetition, and validation progress.
- **Exploration Controller:** Temporarily increases the probability of selecting new peers when sustained warning conditions are detected.

The final implementation will evaluate these components within a decentralized training environment.

**Required future execution model: MPI.** The distributed simulator must use `mpi4py`, with one MPI process (rank) per client and actual model-parameter messages between peers. Each rank owns its model, optimizer, data partition, and selection history, and performs receiver-side aggregation locally. The existing single-process simulator remains the correctness reference. Fixed-peer MPI transport and dense discovery with local closest-peer collaboration are implemented as fixtures; integration with actual training remains future work.

## Technologies

- Python — Core implementation
- PyTorch — Implemented CPU learning framework
- MPI / mpi4py — Implemented model-transport prototype; distributed training planned
- SONAR — Optional integration, only if compatible with the required MPI execution model
- Docker Compose — Optional later packaging for MPI clients
- Linux NetEm — Planned network latency and bandwidth simulation
- Matplotlib / NetworkX — Experimental visualization

## Current Progress

**Stage: IID Gossip Learning and MPI Model Transport Prototype**

Implemented:
- Configurable deterministic selection simulation with rolling histories and JSONL logs
- Basic similarity-based Top-K peer selection
- Selection of peers outside recent collaboration history
- Normalized peer entropy calculation
- Consecutive peer repetition measurement

These functions provide the foundation for future collapse detection and adaptive peer-selection experiments.

Single-process MNIST gossip learning is implemented for IID Top-K, random, and fixed-exploration experiments. MPI ranks can exchange models using fixed patterns or discover all frozen models and select closest collaborators locally. MPI training and adaptive exploration remain future work.

## Roadmap

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for execution milestones, completion gates, experiment rules, and a proposed schedule.

See [the SONAR reimplementation guide](docs/SONAR_REIMPLEMENTATION_GUIDE.md) for the core architecture, verified upstream semantics, and implementation contracts for rebuilding those ideas independently in PeerShift.

- [x] Define research questions and experimental methodology
- [x] Initialize project repository
- [x] Implement initial peer-selection functions
- [x] Implement basic collaboration metrics
- [x] Build a reproducible multi-client simulation
- [ ] Implement validation-plateau detection
- [ ] Implement adaptive exploration and recovery logic
- [x] Implement MPI model-transport demo with one process per client and real peer messages
- [ ] Integrate training and peer-selection policies with MPI
- [ ] Evaluate optional framework integration compatible with MPI
- [ ] Run multi-client training experiments
- [ ] Simulate heterogeneous network conditions
- [ ] Evaluate accuracy, collaboration diversity, and communication overhead

## Evaluation Plan

PeerShift will compare similarity-based Top-K selection, uniform random selection, fixed exploration, and adaptive exploration.

Evaluation will include:

- Model accuracy and validation loss
- Peer entropy and collaboration repetition
- Number of unique collaborators
- Communication overhead and runtime
- Performance under different network conditions

Experiments will use reproducible configurations and consistent communication budgets.

## Research References

1. Yuan et al. (2026). *SONAR: Benchmarking Topology and Collaboration in Decentralized Learning.*
2. Cox et al. (2026). *Dynamic Topology Optimization for Non-IID Data in Decentralized Learning (Morph).*
3. Masmoudi and Jaafar (2024). *OCD-FL: A Novel Communication-Efficient Peer Selection-based Decentralized Federated Learning.*

## Project Status

PeerShift is currently under active development. Initial implementations are prototypes intended to support subsequent distributed learning experiments.

Run instructions are in [backend/README.md](backend/README.md). Selection simulation and minimal IID gossip learning are implemented; later milestones remain future work.

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

## MPI model transport demo

The first MPI increment runs separate client processes exchanging frozen MLP parameters and averaging locally, without training or MNIST. See [MPI_TRANSPORT.md](docs/MPI_TRANSPORT.md) for install/run commands, fixture patterns, communication accounting, protocol limits and tests. This is a partial M4 implementation; distributed training remains pending; the collaboration fixture below adds accounted dense discovery.

## SONAR-style core baseline

The independent SONAR migration now includes `closest` selection of multiple actual collaborators, summed tensor-L2 distance, and equal self-plus-peers model averaging. Run the training command with `backend/configs/train_sonar_closest.toml`; see [backend instructions](backend/README.md#sonar-style-closest-collaborators) for settings and multi-peer diagnostics. Existing Top-K pool sampling, random and fixed exploration remain available with their original defaults. This serial baseline does not yet connect training to MPI or reproduce SONAR's heterogeneous experiments.

## MPI core collaboration

The `mpi-collaborate` command runs independent ranks using multi-peer transfers, dense frozen-model discovery, closest selection and receiver-local aggregation. It reuses discovered snapshots and counts discovery traffic separately from selected model contributions. This is a seeded fixture simulation without training; see [MPI launch instructions](docs/MPI_TRANSPORT.md#multi-peer-core-collaboration) for the four-client example, outputs and limitations. MPI training remains the next integration step.
