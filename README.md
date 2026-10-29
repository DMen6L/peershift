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

## Technologies

- Python — Core implementation
- PyTorch — Planned machine learning framework
- SONAR — Target framework for decentralized learning experiments, subject to integration verification
- Docker Compose — Planned multi-container deployment
- Linux NetEm — Planned network latency and bandwidth simulation
- Matplotlib / NetworkX — Experimental visualization

## Current Progress

**Stage: Reproducible Top-K Selection Simulator**

Implemented:
- Configurable deterministic selection simulation with rolling histories and JSONL logs
- Basic similarity-based Top-K peer selection
- Selection of peers outside recent collaboration history
- Normalized peer entropy calculation
- Consecutive peer repetition measurement

These functions provide the foundation for future collapse detection and adaptive peer-selection experiments.

The current implementation does not yet perform distributed model training or automatically adapt exploration based on validation performance.

## Roadmap

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for execution milestones, completion gates, experiment rules, and a proposed schedule.

- [x] Define research questions and experimental methodology
- [x] Initialize project repository
- [x] Implement initial peer-selection functions
- [x] Implement basic collaboration metrics
- [ ] Build a reproducible multi-client simulation
- [ ] Implement validation-plateau detection
- [ ] Implement adaptive exploration and recovery logic
- [ ] Integrate with a decentralized learning framework
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

1. Singh et al. (2026). *SONAR: Benchmarking Topology and Collaboration in Decentralized Learning.*
2. Cox et al. (2026). *Dynamic Topology Optimization for Non-IID Data in Decentralized Learning (Morph).*
3. Masmoudi and Jaafar (2024). *OCD-FL: A Novel Communication-Efficient Peer Selection-based Decentralized Federated Learning.*

## Project Status

PeerShift is currently under active development. Initial implementations are prototypes intended to support subsequent distributed learning experiments.

Run instructions are in [backend/README.md](backend/README.md). Only the simulator foundation is implemented; remaining milestones are future work.
