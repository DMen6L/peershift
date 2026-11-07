# MPI requirements for future distributed PeerShift

Review date: 2026-11-06. Scope: advisory requirements for a future MPI milestone, based on the explicit user constraint, current README/PROJECT_PLAN and training.py. No implementation, installation, or experiments were performed.

## Decision and current observations

**MPI is a required part of future distributed execution.** Use one MPI rank per learning client; mpi4py is the natural Python binding. Containers may package an MPI deployment but do not replace MPI. SONAR is optional and must preserve the MPI requirement and PeerShift semantics if integrated.

Current code keeps every client in one process, trains all clients before exchange, clones post-training states, then performs receiver-side averaging. The future MPI path must preserve this behavior and diagnostic timing. Existing transport/framework language is broader than the new constraint and should be revised by the parent.

## Official documentation inspected

1. [mpi4py tutorial](https://mpi4py.readthedocs.io/en/stable/tutorial.html), documentation version 4.1.2. Confirms `mpiexec` launch, per-process ranks, Python-object messages and direct array buffers, plus request-based nonblocking calls. Uppercase methods support buffer communication, avoiding Python-object pickling. This supports proposing contiguous CPU arrays for model payloads and small metadata messages for control. Actual MPI runtime and mpi4py compatibility remain installation checks, not verified here.
2. [MPI Forum communication completion](https://www.mpi-forum.org/docs/mpi-5.0/mpi50-report/node76.htm). Wait/test completion determines buffer safety: do not change send buffers before send completion or use receive contents before receive completion. Send completion does not necessarily mean receiver completion. These are required lifetime constraints; initiating a nonblocking operation alone does not prove a complete protocol is deadlock-free.

## Recommended requirements (design proposals)

- Each rank owns its model, optimizer, training iterator, validation data and controller/history state. Rank identity maps stably to client ID. Rank zero may distribute configuration/initialization and collect logs; it must not average models globally or select peers for other ranks.
- Preserve synchronous logical rounds: local training completes, immutable snapshots become available, each client selects one eligible peer, then locally aggregates its own snapshot with its chosen peer's snapshot. A phase synchronization/control protocol must prevent stale or next-round reads. No model `Allreduce` or root averaging substitutes for receiver-side gossip.
- Selection is directed: a rank can have zero or many requesters while choosing exactly one peer itself. A pairwise mutual-exchange assumption is insufficient. One candidate protocol announces selections collectively as small metadata, derives incoming requesters at every rank, posts matching receives and nonblocking payload sends, then completes all requests before snapshot reuse. The implementation design must prove matching/counts for cycles, many-to-one selections, and zero-incoming ranks. Avoid blocking sends followed by receives or waiting for one rank before initiating necessary traffic.
- Freeze each snapshot and its contiguous send buffers until every outgoing request completes. A receiver may aggregate only after its receive completes; its training must not corrupt a still-active send buffer. Use deterministic parameter ordering, shapes/dtypes and round/message identity.
- Preserve initialization, exact partitions, per-client minibatch schedules and random streams across simulator/MPI paths. Compare selected peers exactly and model/validation results under a declared numerical tolerance in the same environment; exclude wall-clock timing from reproducibility claims.
- All-pairs parameter similarity currently has free in-memory access. MPI requires an explicit discovery design. Exchanging all snapshots for exact L2 could dominate communication and duplicate selected transfers. If used as a reference parity mode, label and count that discovery traffic separately; it is not a sparse one-model-transfer protocol. A sketch/neighbor discovery alternative changes information available to selection and requires its own validation.

## Outstanding decisions and acceptance gates

Before implementation, lock discovery versus payload protocol, request matching, buffer layout, phase/round identifiers, failure handling, launch environment and logging ownership. Count actual application payload and control/discovery bytes separately from MPI/network framing, which needs separate instrumentation.

First gate: small multi-rank fixture with ring, many-to-one and zero-incoming cases, large payloads and a timeout detecting hangs. Next gate: paired simulator/MPI training checks, diagnostic parity and auditable message counts. Network heterogeneity follows only after these pass. Confidence is high in documented buffer/rank mechanics; performance, platform support and eventual research outcomes remain untested.
