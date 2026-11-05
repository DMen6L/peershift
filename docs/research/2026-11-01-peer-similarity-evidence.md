# Peer similarity and collaboration concentration: evidence review

Search date: 2026-11-01. Scope: PeerShift's current IID MNIST, common-initialization MLP, single-receiver gossip simulator. No experiments were run or changed. No saved run artifacts were supplied for analysis; implementation observations below come from source inspection, not measured learning outcomes.

## Answer and confidence

Model distance and repeated selection are reasonable **signals to investigate**, not validated measures of useful collaboration or harmful collapse in this implementation. Published evidence supports investigating similarity-induced feedback loops, but also supports useful within-domain collaboration and dissimilarity-based exchange under different conditions. No inspected paper establishes PeerShift's proposed entropy-plus-plateau detector or its thresholds. Confidence is high in the bibliographic verification and code observations, moderate in transfer of the qualitative mechanisms, and low in any prediction of improvement on this small IID setup.

## Verified sources and reference audit

Four primary research works were inspected. Search hits from aggregators were used only to locate originals, not as supporting evidence.

1. **SONAR**, Joyce Yuan, Yichuan Shi, Abhishek Singh, Rishi Sharma, Ramesh Raskar, Jonas Blanc, Martin Jaggi, MLSys 2026. The README's title/year are verified, but its author shorthand should be **Yuan et al.**, not Singh et al. Inspected the publisher record and paper, especially sections 2.3, 5.1, and 6.2. Their experiments use ResNet-10 and heterogeneous image benchmarks including DomainNet and Digit-Five. They report a similarity/reselection feedback loop, sensitivity to Top-K pool size, and harmful cross-domain mixing when collaboration becomes too broad. They emphasize long horizons (200+ rounds) for detecting the reported instability. These observations motivate PeerShift but do not establish its detector or guarantee collapse in IID MNIST. The paper also reports benefits from domain-aligned collaboration, contradicting a universal preference for exploration. [Publisher record](https://proceedings.mlsys.org/paper_files/paper/2026/hash/4abf39685e89559523a4d644036fd2b3-Abstract-Conference.html), [paper](https://proceedings.mlsys.org/paper_files/paper/2026/file/4abf39685e89559523a4d644036fd2b3-Paper-Conference.pdf).

2. **Dynamic Topology Optimization for Non-IID Data in Decentralized Learning (Morph)**, Bart Cox, Antreas Ioannou, Jérémie Decouchant, arXiv:2602.03383v1, 3 February 2026. README title/authors/year verified as a preprint, not a confirmed venue publication. Inspected sections III and IV. Morph favors cosine-based model dissimilarity while managing incoming connections and discovery. Evaluation includes non-IID CIFAR-10 (Dirichlet alpha 0.1), FEMNIST, 50/100 nodes, and multi-seed comparisons. It supplies evidence that dissimilar models can be valuable under those conditions, not evidence that raw Euclidean distance predicts receiver benefit or that its algorithm transfers to one-peer MNIST. The inspected version contains differing parameter descriptions; reproduction would require checking a fixed code revision against the reported setup. [Paper](https://arxiv.org/html/2602.03383v1).

3. **OCD-FL: A Novel Communication-Efficient Peer Selection-based Decentralized Federated Learning**, Nizar Masmoudi and Wael Jaafar, arXiv:2403.04037, first submitted 6 March 2024; inspected v2 dated 22 November 2024. README's 2024 reference is verified as this preprint. Sections IV–V study loss-derived knowledge gain and wireless energy with 20 mobile nodes, MNIST/CIFAR-10 CNNs, and IID/non-IID partitions. Regularization discourages overly narrow peer selection, while receiving a poor model may harm a stronger receiver. Its norm-based argument assumes distance to an optimum reflects performance; it is not a general neural-network loss guarantee. Communication counts differ across its comparison schemes, so reported energy savings cannot be imported into PeerShift's matched-exchange design. [Record](https://arxiv.org/abs/2403.04037), [paper](https://arxiv.org/html/2403.04037v2).

4. **Git Re-Basin: Merging Models Modulo Permutation Symmetries**, Samuel K. Ainsworth, Jonathan Hayase, Siddhartha Srinivasa, arXiv:2209.04836, inspected v6 (1 March 2023; first submitted 2022). Inspected full-text discussion and matching methods. Hidden-unit permutations can preserve model function while changing coordinates; aligning models can improve weight merging. Consequently, raw parameter distance is not generally functional distance. PeerShift's identical initialization and frequent averaging make arbitrary independent-model permutations less immediate, but do not establish distance as a utility measure. [Paper](https://arxiv.org/html/2209.04836v6).

## Local implementation observations and inferences

Inspected `backend/src/peershift/learning.py`, `training.py`, README, and project plan.

- **Observed:** similarity is negative Euclidean distance over all flattened state values. It includes biases and is not layer-normalized. **Inference:** layer sizes/scales and movement from a common initialization can dominate ranking; a low distance can reflect past averaging rather than complementary information.
- **Observed:** each receiver selects once, averages its own and selected snapshots, and never directly updates its selected sender. **Inference:** this removes processing-order contamination but does not guarantee graph connectivity, balanced model influence, or network-wide information coverage.
- **Observed:** training currently supports IID partitions and Top-K/random only. Defaults are ten rounds, K=2, ten clients, with shared test evaluation and separate per-client validation. **Inference:** an absence of collapse in short IID runs would be inconclusive about heterogeneous, longer-horizon behavior.
- **Observed:** metrics log entropy, immediate repetition, and unique peers. **Inference:** these measure different phenomena. Cycling through two peers can have zero consecutive repetition and still be narrowly concentrated. Short windows also limit achievable entropy. Compare against matched-window baselines rather than interpreting a fixed low value as harm.
- **Observed:** evaluation occurs initially and after aggregation, not immediately before it. **Inference:** current logs cannot separate local-training progress from the immediate contribution of the selected peer. A plateau can also arise from optimizer settings or insufficient local work.
- **Observed:** all-pairs similarity accesses snapshots in memory; summary bytes estimate selected-model payloads only. **Inference:** these are not real distributed discovery costs. A future proxy comparison must account for additional information access or computation separately.

## Ranked recommendations (proposals, not executed)

### 1. Measure receiver utility before designing collapse thresholds

Benefit: directly test the proxy assumption. Effort: small instrumentation change.

At a few predetermined rounds, compare receiver validation loss immediately before and after its actual aggregation, and log selected distance and recent concentration. Define benefit as pre-aggregation minus post-aggregation validation loss. Use the same initial weights, partitions, batch schedules, local steps, and one exchange per client for Top-K/random, across five paired seeds. Fix observation rounds before seeing results. Examine per-client benefit distributions and distance association; do not label correlation causal. Optional in-memory candidate probes must be marked offline diagnostics with separate evaluation cost, not deployable free selection. Keep test data out of selection/tuning.

### 2. Introduce controlled heterogeneity plus fixed exploration

Benefit: test the hypothesized failure mechanism in a more relevant regime. Effort: moderate, consistent with the planned next learning increment.

Add one reproducible label-skew partition alongside IID, then compare Top-K, random, and fixed exploration with identical sample counts and exchange budgets. Start with a small pilot to estimate runtime, then prerecord the round horizon and validation-only settings for paired seeds. Include later-round validation behavior; a short smoke run only validates mechanics. Report client-level variation and adverse mixing, not just mean accuracy. Do not assume a particular horizon guarantees collapse.

### 3. Compare proxy reliability at matched budgets

Benefit: determine whether adaptation is reacting to a misleading similarity measure. Effort: moderate, after recommendations 1–2.

Compare raw L2 and cosine similarity under the same Top-K size and training/exchange budget, using identical paired inputs. Record ranking stability and validation benefit; optionally inspect per-layer distance contributions without changing policy. Keep additional scoring time visible. A functional-distance probe can be a later offline diagnostic using an explicitly documented validation/probe set; it must not silently use held-out test examples or extra free network access.

## Remaining questions

Does concentration precede diminished receiver utility, or merely follow convergence? Does any association survive IID/non-IID controls and matched local-work budgets? Which validation distribution should drive the controller when local specialization conflicts with global accuracy? What similarity discovery protocol is affordable in real transport? Can concentration plus plateau identify recoverable cases better than either signal alone? None is answered by current smoke-test success.
