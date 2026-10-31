# PeerShift simulator

A selection-only research foundation, using Python 3.12 or later. No model training or networking is performed.

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

### Selection baseline configurations

Use `backend/configs/top_k.toml`, `backend/configs/random.toml`, or `backend/configs/fixed_exploration.toml` with the existing simulate command, each into a separate empty output directory. All three use matching seeds and selection budgets.

`policy` defaults to `top_k`; `exploration_probability` defaults to 0.1 and applies only to `fixed_exploration`. Random selection samples any eligible peer. Fixed exploration samples outside Top-K with that probability, otherwise inside Top-K. If no outside candidates exist it falls back to Top-K. Logs include `policy`, `selection_branch`, and nullable `fallback_reason`. Diversity differences do not establish learning benefits; training and adaptive control remain future work.
