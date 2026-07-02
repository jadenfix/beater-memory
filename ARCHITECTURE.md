# beater-memory Architecture

`beater-memory` is an agent memory engine, not a vector database. The durable
source of truth is an append-only ledger. Memory nodes, cue indexes, and graph
edges are projections that can be rebuilt.

## Repo Alignment

- `beater.js` owns local agent execution and writes `.beater/journal.db`.
  `beater-memory import-beater-js` reads the `runs` and `steps` tables without
  changing that journal.
- `beater-agents` owns canonical trace semantics. This engine mirrors the
  canonical `memory.read` and `memory.write` span kinds and accepts canonical
  span JSONL through `import-jsonl`.
- `beater-memory` owns projection and query: ledger events become typed memory
  nodes, typed edges, cue seeds, citations, contradiction warnings, and compact
  answer-shaped context.

## Layers

1. **Ledger**
   - Table: `ledger_events`
   - Contract: append-only observations with tenant/project/environment scope,
     trace/span/seq provenance, payload JSON, observed time, and projection time.
   - Hot path: no model call is required to record an event.

2. **Distiller**
   - Trait: `Distiller`
   - Current implementation: `HeuristicDistiller`
   - Output schema: `DistilledMemory { ADD | UPDATE | INVALIDATE | NOOP }`
   - Future LLM distillers must still emit this constrained shape before
     touching projections.

3. **Typed Graph Projection**
   - Nodes: `Episode`, `Fact`, `EntityCue`, `Tag`, `Procedure`, `State`,
     `Gotcha`, `AntiMemory`, `Topic`
   - Edges: `mentions`, `derived_from`, `observed_in`, `supersedes`,
     `contradicts`, plus causal/procedural edge kinds reserved in the model.
   - Contradicted memory is invalidated with `valid_to_unix_ms`; it is not
     deleted.

4. **Read Path**
   - Tier 0: lexical cue seeding through `cue_index`
   - Tier 1: LLM-free graph activation using personalized PageRank-style
     propagation, ACT-R-like base-level activation, edge weights, and freshness
   - Tier 2: reserved API slot for budgeted active reconstruction
   - Return type: `MemoryAnswer`, not raw chunks.

5. **Service API**
   - CLI command: `beater-memory serve`
   - Public endpoints: `GET /livez`, `GET /readyz`
   - Authenticated endpoints: `/v1/health`, `/v1/stats`, `/v1/remember`,
     `/v1/project`, `/v1/query`, `/v1/maintenance`, `/v1/metrics`,
     `/v1/metrics/prometheus`, `/v1/audit`
   - Auth: bearer token from `--bearer-token` or `BEATER_MEMORY_TOKEN` by
     default; unauthenticated serving requires explicit `--allow-no-auth`
   - Limits: max request body bytes, max projection batch size, max query token
     budget, max concurrent blocking SQLite tasks, and DB task timeout are
     configurable at startup.
   - Controls: a fixed-window per-actor limiter protects `/v1/*`; in-memory
     JSON and Prometheus service metrics expose request totals, DB saturation,
     and DB timeouts; durable audit rows record successes, failures, denied
     auth, and throttled attempts.
   - Writes: direct `remember` calls can carry an idempotency key so client
     retries map to the same ledger event.
   - Shutdown: `serve` uses Axum graceful shutdown on Ctrl-C and SIGTERM on
     Unix; `serve_with_shutdown` exposes the same server loop with an injected
     shutdown future for embedding and tests.

## Why No Embeddings In The MVP

The first-principles read path needs typed structure, temporal validity, and
provenance before approximate nearest-neighbor search. Embeddings can become
another seed channel later, but the system should already know how to route,
invalidate, cite, and budget memory without them.

## Storage

The default database is SQLite. Tables are intentionally boring:

- `ledger_events`: imported or direct observations
- `memory_nodes`: bitemporal typed memories
- `memory_edges`: typed graph relationships
- `node_spans`: many-to-many provenance citations
- `cue_index`: deterministic lexical seed index
- `audit_events`: durable service audit records

Projection is idempotent for repeated imports because ledger events are keyed by
`tenant_id + project_id + trace_id + span_id + seq`.

Production safeguards:

- every connection enables `foreign_keys`, `busy_timeout`, WAL mode, and
  `synchronous=NORMAL`
- new databases are stamped with the Beater Memory SQLite `application_id`;
  existing SQLite files must already have that identity before migration runs
- `PRAGMA user_version` records the supported schema version
- each ledger event is projected inside `BEGIN IMMEDIATE ... COMMIT`
- projection rechecks `projected_at_unix_ms IS NULL` inside the transaction so
  concurrent workers cannot double-count a stale pending row
- direct writes can apply an idempotency key to stabilize the ledger
  `trace_id + span_id + seq` for retry-safe ingestion
- `health` runs schema, integrity, foreign-key, and count checks
- `health` also reports graph projection orphan counts for edges, citations,
  and cue index rows because the current projection tables are not SQLite-FK
  constrained
- `readyz` uses the same DB concurrency and timeout guard as normal DB-backed
  requests, but returns only coarse readiness state
- service shutdown drains through Axum graceful shutdown on Ctrl-C and Unix
  SIGTERM
- `maintenance` runs SQLite optimize and WAL checkpointing, with optional vacuum
  and explicit orphan repair
- `maintenance` also owns explicit audit retention, either by Unix millisecond
  cutoff, newest-row count, or both
- `backup` uses SQLite's online backup API and refuses to overwrite an existing
  backup path
- `restore` replaces the active database only behind an explicit confirmation
  flag and re-runs schema/health checks after restore
- `rebuild-projection` clears only derived projection tables and ledger
  projection markers, then replays the append-only ledger behind an explicit
  confirmation flag
- service audit events are persisted in SQLite so backup/restore includes the
  operational trail for the memory database

## Commands

```bash
cargo run -p beater-memory -- init
cargo run -p beater-memory -- remember --tenant local --project demo --kind gotcha \
  "Checkout fails when DATABASE_URL is missing. Fix by setting DATABASE_URL."
cargo run -p beater-memory -- query --tenant local --project demo \
  "How do I fix checkout database failures?"
cargo run -p beater-memory -- health --json
cargo run -p beater-memory -- maintenance
cargo run -p beater-memory -- maintenance --repair-orphans
cargo run -p beater-memory -- maintenance --retain-audit-events 10000
cargo run -p beater-memory -- rebuild-projection --yes-clear-projections
cargo run -p beater-memory -- backup --path ./backups/memory.db
BEATER_MEMORY_TOKEN=dev-secret cargo run -p beater-memory -- serve
curl http://127.0.0.1:8765/readyz
```

Useful service reads:

```bash
curl -H "Authorization: Bearer $BEATER_MEMORY_TOKEN" http://127.0.0.1:8765/v1/metrics
curl -H "Authorization: Bearer $BEATER_MEMORY_TOKEN" 'http://127.0.0.1:8765/v1/audit?limit=50'
```

Import sibling repo data:

```bash
cargo run -p beater-memory -- import-beater-js \
  --journal ../beater.js/examples/hello/.beater/journal.db \
  --tenant local --project hello

cargo run -p beater-memory -- import-jsonl \
  --path ./spans.jsonl --tenant local --project observed-agent
```

## Quality Gates

Every feature slice should pass:

```bash
cargo fmt --all --check
cargo test
cargo clippy --workspace --all-targets -- -D warnings
```

Before publishing a slice, inspect the diff, commit only intended files, open a
PR, and merge only after local checks and GitHub CI are clean.
