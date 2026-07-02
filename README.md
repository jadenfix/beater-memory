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
- authenticated HTTP API for service deployments
- optional idempotency keys for retry-safe direct `remember` writes
- production operations for schema/integrity health checks and SQLite
  maintenance, backup, and restore
- graph projection integrity checks and orphan repair for edges, citations, and
  cue index entries
- ledger validation that rejects malformed events before they enter the
  append-only log
- guarded projection rebuild from the append-only ledger
- explicit audit retention pruning by age or newest-row count
- database identity checks that reject unrelated SQLite files instead of
  silently migrating them
- service metrics, persisted audit events, fixed-window request limiting, and
  bounded DB work concurrency
- `x-request-id` propagation for HTTP responses and durable audit rows
- answer-shaped `MemoryAnswer` with citations, stale assumptions,
  contradictions, suggested follow-up queries, and token estimates

## Quick Start

```bash
cargo run -p beater-memory -- init

cargo run -p beater-memory -- remember \
  --idempotency-key demo-checkout-db-gotcha \
  --tenant local --project demo --kind gotcha \
  "Checkout fails when DATABASE_URL is missing. Fix by setting DATABASE_URL."

cargo run -p beater-memory -- query \
  --tenant local --project demo \
  "How do I fix checkout database failures?"

cargo run -p beater-memory -- health --json
```

Run the HTTP API:

```bash
export BEATER_MEMORY_TOKEN='dev-secret'
cargo run -p beater-memory -- serve --bind 127.0.0.1:8765
```

The server refuses to start without a bearer token unless `--allow-no-auth` is
passed, trims configured bearer tokens, and rejects blank tokens. Public request
limits for body size, projection batch size, query token budget, and audit page
size must be greater than zero. It drains with Axum graceful shutdown on Ctrl-C
and on SIGTERM for Unix process managers. All `/v1/*` routes require
`Authorization: Bearer <token>`; `/livez` is the unauthenticated liveness
endpoint, and `/readyz` is the unauthenticated readiness endpoint for DB-backed
traffic. The service defaults to 600 authenticated requests per actor per
minute; use
`--max-requests-per-minute 0` to disable the fixed-window limiter for a trusted
local deployment. Every response includes `x-request-id`; client-supplied valid
request IDs are echoed, and generated IDs are written into durable audit detail.
DB-backed HTTP requests are also capped at 32 concurrent blocking SQLite tasks
by default; use `--max-concurrent-db-tasks` to tune this. When saturated,
DB-backed routes return `503 service_busy` with `Retry-After`, while `/livez`,
JSON `/v1/metrics`, and Prometheus `/v1/metrics/prometheus` remain available;
`/readyz` returns `503` until the database work queue is available and health
checks pass. Each DB-backed task also has a 30s wall-time budget by default;
adjust it with `--db-task-timeout-ms`. Timed-out DB routes return
`504 service_timeout` and increment `db_timeout_requests`.

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

For retryable direct writes, pass an idempotency key. Reusing the same key in
the same tenant, project, environment, and memory kind maps the write to the same
ledger event; repeated submissions return `ingested: false` over HTTP instead of
appending duplicate ledger work.

## Operations

Projection is atomic per ledger event. The engine uses an immediate SQLite
transaction, rechecks that the event is still pending inside the transaction,
then commits the memory nodes, edges, cue index, citations, and projected marker
together. New databases are stamped with the Beater Memory SQLite
`application_id`; existing databases must already carry that identity before
schema migration runs.

```bash
cargo run -p beater-memory -- health
cargo run -p beater-memory -- maintenance
cargo run -p beater-memory -- maintenance --vacuum
cargo run -p beater-memory -- maintenance --repair-orphans
cargo run -p beater-memory -- maintenance --retain-audit-events 10000
cargo run -p beater-memory -- maintenance --prune-audit-before-unix-ms 1782864000000
cargo run -p beater-memory -- rebuild-projection --yes-clear-projections
cargo run -p beater-memory -- backup --path ./backups/memory.db
cargo run -p beater-memory -- restore --path ./backups/memory.db --yes-replace-current-db
```

Backups use SQLite's online backup API and refuse to overwrite an existing
backup path. Restore replaces the active database and requires the explicit
`--yes-replace-current-db` flag. Health reports graph projection orphan counts;
maintenance reports graph integrity before and after the pass, and only removes
orphan projection rows when `--repair-orphans` or HTTP `repair_orphans: true` is
set. Projection rebuild clears only derived memory nodes, edges, citations, cue
indexes, and projection markers, then replays the append-only ledger; audit rows
and ledger events remain intact. Audit retention is explicit: maintenance can
drop rows older than a Unix millisecond cutoff and/or keep only the newest N
audit rows. If both are set, age pruning runs first and newest-row retention is
applied to the remaining audit trail.

HTTP equivalents:

```bash
curl -H "Authorization: Bearer $BEATER_MEMORY_TOKEN" \
  http://127.0.0.1:8765/v1/health

curl http://127.0.0.1:8765/readyz

curl -H "Authorization: Bearer $BEATER_MEMORY_TOKEN" \
  http://127.0.0.1:8765/v1/metrics

curl -H "Authorization: Bearer $BEATER_MEMORY_TOKEN" \
  http://127.0.0.1:8765/v1/metrics/prometheus

curl -H "Authorization: Bearer $BEATER_MEMORY_TOKEN" \
  'http://127.0.0.1:8765/v1/audit?limit=50'

curl -H "Authorization: Bearer $BEATER_MEMORY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"repair_orphans":true,"retain_latest_audit_events":10000}' \
  http://127.0.0.1:8765/v1/maintenance

curl -H "Authorization: Bearer $BEATER_MEMORY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"local","project_id":"demo","kind":"gotcha","idempotency_key":"demo-checkout-db-gotcha","text":"Checkout fails when DATABASE_URL is missing."}' \
  http://127.0.0.1:8765/v1/remember

curl -H "Authorization: Bearer $BEATER_MEMORY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question":"checkout database failure","scope":{"tenant_id":"local","project_id":"demo","environment_id":null,"as_of_unix_ms":null}}' \
  http://127.0.0.1:8765/v1/query
```

## Crate API

The public API exports:

- `MemoryEngine`
- `ProjectReport` and `ProjectionRebuildReport`
- `SqliteMemoryStore`
- `MemoryServerConfig`, `try_memory_router`, `memory_router`, `serve`, and
  `serve_with_shutdown`
- `StoreHealth`, `StoreStats`, `MaintenanceOptions`, `MaintenanceReport`,
  `GraphIntegrityReport`, `GraphRepairReport`, `AuditPruneReport`,
  `ProjectionResetReport`, `BackupReport`, `RestoreReport`, `AuditRecord`, and
  `AuditEvent`
- `LiveResponse`, `ReadyResponse`, and `ServiceMetricsSnapshot`
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
