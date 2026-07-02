# beater-memory

Agent-first memory for Beater. It is a local Rust app and library that turns
ledgered agent traces into typed, temporal memory and answers queries with
provenance, contradiction warnings, and a token budget.

The design follows [`research/agent-memory.md`](research/agent-memory.md):
memory is a projection over append-only traces, not a plain vector database.
See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the layer boundaries.

## System Shape

```text
ledgered agent traces
  -> offline distiller
  -> typed temporal graph and substores
  -> tiered read path
  -> compact answer with provenance, contradictions, and token budget
```

The current implementation includes:

- SQLite ledger/projection store
- deterministic `ADD / UPDATE / INVALIDATE / NOOP` distiller
- typed nodes and edges for facts, episodes, procedures, state, gotchas, and
  anti-memory
- lexical cue seeding plus graph activation ranking
- `beater.js` journal import from `.beater/journal.db`
- canonical span JSONL import aligned with `beater-agents` span kinds
- CLI commands for `init`, `remember`, `project`, `query`, and import flows
- production operations for schema/integrity health checks and SQLite
  maintenance
- answer-shaped `MemoryAnswer` with citations, stale assumptions,
  contradictions, suggested follow-up queries, and token estimates

## Quick Start

```bash
cargo run -p beater-memory -- init

cargo run -p beater-memory -- remember \
  --tenant local --project demo --kind gotcha \
  "Checkout fails when DATABASE_URL is missing. Fix by setting DATABASE_URL."

cargo run -p beater-memory -- query \
  --tenant local --project demo \
  "How do I fix checkout database failures?"

cargo run -p beater-memory -- health --json
```

Import a `beater.js` journal:

```bash
cargo run -p beater-memory -- import-beater-js \
  --journal ../beater.js/path/to/app/.beater/journal.db \
  --tenant local --project my-app
```

Import canonical span JSONL, useful for `beater-agents` exports:

```bash
cargo run -p beater-memory -- import-jsonl \
  --path ./spans.jsonl --tenant local --project observed-agent
```

The default DB path is `.beater-memory/memory.db`; override with `--db`.

## Operations

Projection is atomic per ledger event. The engine uses an immediate SQLite
transaction, rechecks that the event is still pending inside the transaction,
then commits the memory nodes, edges, cue index, citations, and projected marker
together.

```bash
cargo run -p beater-memory -- health
cargo run -p beater-memory -- maintenance
cargo run -p beater-memory -- maintenance --vacuum
```

## Crate API

The public API exports:

- `MemoryEngine`
- `SqliteMemoryStore`
- `StoreHealth`, `StoreStats`, and `MaintenanceReport`
- `LedgerEvent`
- `Distiller` and `HeuristicDistiller`
- `MemoryQuery` and `MemoryAnswer`
- `MemoryTier`, `MemoryNodeKind`, `MemoryEdgeKind`, `BeliefRevisionOp`
- import helpers for `beater.js` journals and canonical JSONL
- evidence token budgeting helpers

Run checks:

```bash
cargo fmt --all --check
cargo test
cargo clippy --workspace --all-targets -- -D warnings
```

## Design Constraints

- Memory is `write / manage / read`, not just `retrieve`.
- The write hot path should stay append-only and robust.
- LLM distillation, when added, must happen off-path and use constrained schemas
  before writing projections.
- Retrieval returns an answer-shaped evidence bundle, not raw chunks.
- Token cost, read latency, write amplification, and context pollution are
  first-class metrics.
- Contradictions are graph edges and stale assumptions, not silently overwritten
  summaries.

## Feature Workflow

For each coherent feature slice: implement, self-review the diff, run focused
tests plus the workspace checks, commit only intended files, open a PR, and
merge only after local checks and GitHub CI are clean.
