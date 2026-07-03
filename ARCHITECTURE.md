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
   - Validation: every append validates required identifiers, non-empty memory
     text, positive sequence numbers, and non-negative event timestamps before
     the row can enter the ledger.
   - Hot path: no model call is required to record an event.

2. **Distiller**
   - Trait: `Distiller`
   - Current implementations: `HeuristicDistiller` and provider-backed
     `ProviderDistiller<P>`. CLI and server entrypoints select either the
     heuristic distiller or a command-backed provider adapter with
     `--distiller provider-command --distiller-command <path>`.
   - Output schema: `DistilledMemory { ADD | UPDATE | INVALIDATE | NOOP }`;
     provider JSON uses the enum's snake_case wire values (`"add"`,
     `"update"`, `"invalidate"`, `"noop"`). Provider output is parsed through a
     strict JSON batch schema; malformed JSON or invalid memories get bounded
     repair attempts before rejection.
   - Projection validates required text, cited-span provenance, scoped target
     IDs, and invalidation/noop shape before touching graph tables. Targetless
     invalidations with no resolvable neighbor are normalized to adds instead
     of silently invalidating nothing.
   - Provider-backed distillation runs before projection write transactions.
     Rejected provider output leaves the ledger event pending and reports
     provider, repair, rejection, token, and elapsed counters.
   - Late-arrival replay is enabled only for distillers that opt into
     deterministic replay. Provider-backed distillers skip late replay unless
     they can safely replay a durable accepted batch for the event.
   - Projection rebuild is allowed for replay-safe distillers and for
     provider-command projections with complete durable distillation batches for
     the selected replay fingerprint. Command-provider rebuild checks batch
     coverage before clearing derived projection tables, then replays stored
     batches without provider calls. Replayable provider batches require
     explicit `target_node_id` values for `update` and `invalidate`, preventing
     rebuild from re-resolving targetless revisions against a different neighbor
     set. The current fingerprint covers the command path string, JSON args, and
     effective repair budget; provider binary contents, environment variables,
     cwd behavior, timeout, and model settings must be represented in args if
     they should affect replay identity.

3. **Typed Graph Projection**
   - Nodes: `Episode`, `Fact`, `EntityCue`, `Tag`, `Procedure`, `State`,
     `Gotcha`, `AntiMemory`, `Topic`
   - Edges: `mentions`, `derived_from`, `observed_in`, `supersedes`,
     `contradicts`, plus causal/procedural edge kinds reserved in the model.
   - Contradicted memory is invalidated with `valid_to_unix_ms`; it is not
     deleted.

4. **Read Path**
   - Validate query scope, question text, token budget, and enabled memory modes
     before retrieval begins.
   - HTTP query requests are normalized into the same `MemoryQuery` type and
     validated before entering the DB-backed retrieval worker.
   - Deterministic router: query-shape rules choose an effective subset of the
     caller-allowed memory modes (`semantic`, `episodic`, `procedural`,
     `gotcha`, `state`) before store reads when the query uses the default
     all-mode set. Explicit non-default `modes` are treated as the caller's
     exact route for compatibility. Ambiguous markerless queries and
     empty-evidence narrowed routes fall back to the allowed modes; multi-intent
     queries keep a conservative multi-mode route. When active reconstruction
     widens the read beyond the initial route, the answer records the
     reconstruction modes separately. `EntityCue` nodes remain graph support
     rather than answer evidence.
   - Tier 0: routed lexical cue seeding through `cue_index`
   - Tier 1: LLM-free graph activation using personalized PageRank-style
     propagation, ACT-R-like base-level activation, edge weights, and freshness
   - Reads are bitemporal: `as_of_unix_ms` is the observed validity boundary,
     and `known_at_unix_ms` is the ledger transaction-time boundary based on
     event ingestion. Future-known facts and invalidations are excluded until
     their source events are known.
   - Tier 2: optional active reconstruction that escalates forced queries and
     hard auto queries into bounded graph expansion through an
     `ActiveReconstructor` policy. The default policy is deterministic and
     provider-neutral; future model-backed providers must return validated
     accept/prune/stop decisions under the same step and token budgets.
   - Return type: `MemoryAnswer`, not raw chunks.

5. **Service API**
   - CLI command: `beater-memory serve`
   - Public endpoints: `GET /livez`, `GET /readyz`
   - Authenticated endpoints: `/v1/health`, `/v1/stats`, `/v1/remember`,
     `/v1/manage`, `/v1/project`, `/v1/query`, `/v1/maintenance`, `/v1/metrics`,
     `/v1/metrics/prometheus`, `/v1/audit`
   - Auth: bearer token from `--bearer-token` or `BEATER_MEMORY_TOKEN` by
     default; configured tokens are trimmed, blank tokens are rejected, and
     unauthenticated serving requires explicit `--allow-no-auth`
   - Limits: max request body bytes, max projection batch size, max query token
     budget, audit page size, max concurrent blocking SQLite tasks, and DB task
     timeout are configurable at startup. Public request limits must be
     positive; the rate limiter, DB concurrency limiter, and DB timeout keep
     their documented zero-value controls.
     Provider-command distillation has its own per-call timeout, and server
     startup rejects provider timeouts greater than or equal to the DB task
     timeout so a single provider call cannot race the HTTP timeout boundary.
   - Controls: a fixed-window per-actor limiter protects `/v1/*`; in-memory
     JSON and Prometheus service metrics expose request totals, DB saturation,
     DB timeouts, distillation provider counters, and query counts/latency/token
     totals by retrieval tier; durable audit rows record successes, failures,
     denied auth, throttled attempts, and projection summaries.
   - Economics telemetry: projection reports include source-token estimates,
     projected memory-token estimates, and distillation provider counters;
     store stats expose total/active stored-memory tokens and active node counts
     by kind.
   - Correlation: every HTTP response carries `x-request-id`; valid incoming
     request IDs are echoed and audit details include the same ID.
   - Writes: direct `remember` calls can carry an idempotency key so client
     retries map to the same ledger event. The compatibility default still
     manages projections immediately, but CLI `--no-project` and HTTP
     `"project": false` keep the operation append-only until an explicit
     `manage`/`project` call.
   - Shutdown: `serve` uses Axum graceful shutdown on Ctrl-C and SIGTERM on
     Unix; `serve_with_shutdown` exposes the same server loop with an injected
     shutdown future for embedding and tests.

6. **Evaluation Harness**
   - CLI command: `beater-memory eval --suite <path>`
   - Suites are deterministic JSON fixtures with LongMemEval-V2-style ability
     labels, ledger observations, query settings, explicit content expectation
     checks over answers, evidence, stale assumptions, and contradictions, plus
     an optional retrieval-tier gate. Cases are isolated into per-case project
     scopes by default; suites that model shared-haystack benchmarks opt in
     with `shared_haystack: true`.
   - The runner builds an isolated in-memory store, projects the suite, queries
     every case, and reports accuracy by ability, context-saturation shortfall
     when a full-context baseline score is supplied, write tokens per stored
     memory, projected/source token ratio, latency by tier, and tokens placed
     into answer context as selected evidence. It intentionally avoids F1/BLEU
     as a correctness signal.

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
- each ledger event's prepared distilled memories are applied inside
  `BEGIN IMMEDIATE ... COMMIT`
- `beater.js` and canonical JSONL imports append inside one immediate
  transaction, so malformed rows abort the batch with row context instead of
  leaving a partial import behind
- projection rechecks `projected_at_unix_ms IS NULL` inside the transaction so
  concurrent workers cannot double-count a stale pending row; provider-backed
  distillation happens before that transaction and is discarded if the row is no
  longer pending
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
- `x-request-id` is generated or echoed for every HTTP response and included in
  durable audit detail for authenticated, denied, failed, and throttled calls
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
- projection batch sizes and optional rebuild event caps must be positive so a
  rebuild cannot clear projections and replay zero events
- store APIs check SQLite `LIMIT` parameters before binding them so oversized
  embedded limits cannot wrap into unbounded reads or retention operations
- service audit events are persisted in SQLite so backup/restore includes the
  operational trail for the memory database
- audit records and audit-retention options are validated before insert or
  pruning, including non-negative retention cutoffs and checked SQLite
  retention limits

## Commands

```bash
cargo run -p beater-memory -- init
cargo run -p beater-memory -- remember --tenant local --project demo --kind gotcha \
  "Checkout fails when DATABASE_URL is missing. Fix by setting DATABASE_URL."
cargo run -p beater-memory -- remember --no-project --tenant local --project demo --kind fact \
  "Append this observation without managing projections yet."
cargo run -p beater-memory -- manage --limit 100
cargo run -p beater-memory -- query --tenant local --project demo \
  "How do I fix checkout database failures?"
cargo run -p beater-memory -- query --tenant local --project demo \
  --as-of-unix-ms 1782864000000 --known-at-unix-ms 1782867600000 \
  "How did checkout look at that time?"
cargo run -p beater-memory -- query --tenant local --project demo \
  --reconstruction-mode auto \
  "why did checkout database failures recover?"
cargo run -p beater-memory -- health --json
cargo run -p beater-memory -- maintenance
cargo run -p beater-memory -- maintenance --repair-orphans
cargo run -p beater-memory -- maintenance --retain-audit-events 10000
cargo run -p beater-memory -- rebuild-projection --yes-clear-projections
cargo run -p beater-memory -- backup --path ./backups/memory.db
cargo run -p beater-memory -- eval --suite ./memory-eval.json
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
