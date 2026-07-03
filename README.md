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
- provider-safe distillation boundary with strict JSON schema parsing,
  validation, bounded repair attempts, rejection metrics, and no provider calls
  inside projection write transactions
- opt-in command-backed provider distillation for CLI and HTTP manage paths,
  with a per-call timeout and bounded repair attempts
- provider distillation can emit typed relation edges to scoped neighbor
  memories; projected edges carry source-event provenance so bitemporal
  `known_at_unix_ms` reads traverse only relationships known at that time
- opt-in command-backed active reconstruction for CLI and HTTP query paths,
  with strict decision parsing, candidate validation, and provider metrics
- memory economics telemetry for projection source/stored-token estimates,
  active stored-token totals, active node counts by kind, and query tier
  latency/token counters
- distiller output validation before graph projection writes
- typed nodes and edges for facts, episodes, procedures, state, gotchas, and
  anti-memory
- deterministic typed-substore read routing with conservative fallback and
  response/audit diagnostics
- lexical cue seeding plus graph activation ranking
- `beater.js` journal import from `.beater/journal.db`
- canonical span JSONL import aligned with `beater-agents` span kinds
- atomic imports that roll back malformed batches and report the bad source row
- CLI commands for `init`, `remember`, `manage`, `project`, `query`, and import
  flows
- authenticated HTTP API for service deployments
- optional idempotency keys for retry-safe direct `remember` writes
- production operations for schema/integrity health checks and SQLite
  maintenance, backup, and restore
- graph projection integrity checks and orphan repair for edges, citations, and
  cue index entries
- bitemporal query windows: `as_of_unix_ms` filters observed validity, while
  `known_at_unix_ms` filters what the ledger had ingested by that transaction
  time
- optional Tier 2 active reconstruction for hard or forced read-time graph
  exploration, with bounded steps, token budget, and a diagnostic report
- ledger validation that rejects malformed events before they enter the
  append-only log
- query validation that rejects malformed scopes, empty questions, and unusable
  token budgets before retrieval
- HTTP query requests are validated through the same `MemoryQuery` contract used
  by the engine before any retrieval task runs
- projection limit validation that rejects zero-sized project and rebuild
  batches before mutating derived graph state
- checked SQLite limit binding so oversized embedded read or retention limits
  cannot become unbounded queries
- audit record validation and non-negative audit-retention cutoffs before
  writing or pruning the durable operational trail
- guarded projection rebuild from the append-only ledger
- explicit audit retention pruning by age or newest-row count
- database identity checks that reject unrelated SQLite files instead of
  silently migrating them
- service metrics, persisted audit events, fixed-window request limiting, and
  bounded DB work concurrency
- `x-request-id` propagation for HTTP responses and durable audit rows
- deterministic evaluation harness for LongMemEval-style ability cases,
  expectation-based judging, context-saturation gap placeholders, and
  read/write economics reports; cases are isolated by default unless the suite
  opts into a shared haystack, and eval runs can opt into the same provider
  distillation and active reconstruction adapters as CLI/HTTP runtime paths
- answer-shaped `MemoryAnswer` with citations, stale assumptions,
  contradictions, suggested follow-up queries, and token estimates

## Quick Start

```bash
cargo run -p beater-memory -- init

cargo run -p beater-memory -- remember \
  --idempotency-key demo-checkout-db-gotcha \
  --tenant local --project demo --kind gotcha \
  "Checkout fails when DATABASE_URL is missing. Fix by setting DATABASE_URL."

cargo run -p beater-memory -- remember --no-project \
  --tenant local --project demo --kind fact \
  "Append this observation without managing projections yet."

cargo run -p beater-memory -- manage --limit 100

cargo run -p beater-memory -- query \
  --tenant local --project demo \
  "How do I fix checkout database failures?"

cargo run -p beater-memory -- query \
  --tenant local --project demo \
  --as-of-unix-ms 1782864000000 \
  --known-at-unix-ms 1782867600000 \
  "How did checkout look at that time?"

cargo run -p beater-memory -- query \
  --tenant local --project demo \
  --reconstruction-mode auto \
  "why did checkout database failures recover?"

cargo run -p beater-memory -- \
  --reconstructor provider-command \
  --reconstructor-command ./reconstruct-provider \
  query --tenant local --project demo \
  --reconstruction-mode force \
  "why did checkout database failures recover?"

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

Import commands project by default. Pass `--no-project` to append imported
ledger events only, then run `manage` when you want to build projections.

Run a deterministic memory evaluation suite:

```bash
cargo run -p beater-memory -- eval --suite ./memory-eval.json
```

Provider-backed evals use the same global adapter flags as manage/query:

```bash
cargo run -p beater-memory -- \
  --distiller provider-command \
  --distiller-command ./distill-provider \
  --reconstructor provider-command \
  --reconstructor-command ./reconstruct-provider \
  eval --suite ./memory-eval.json --reconstruction-mode force
```

`eval` emits machine-readable JSON on stdout by default. Passing suites keep
stderr empty; failing suites still print the JSON report to stdout and then emit
a human error on stderr with a nonzero exit code.

The default DB path is `.beater-memory/memory.db`; override with `--db`.

For retryable direct writes, pass an idempotency key. Reusing the same key in
the same tenant, project, environment, and memory kind maps the write to the same
ledger event; repeated submissions return `ingested: false` over HTTP instead of
appending duplicate ledger work.
CLI `remember` and HTTP `/v1/remember` keep their compatibility default of
managing projections immediately; pass `--no-project` on the CLI or
`"project": false` over HTTP for a write-only append, then run `manage` or
`POST /v1/manage` later.

### Provider-backed distillation

Projection commands use the deterministic heuristic distiller by default. To
run a local provider adapter at manage time, pass global distiller flags before
the subcommand:

```bash
cargo run -p beater-memory -- \
  --distiller provider-command \
  --distiller-command ./distill-provider \
  --distiller-timeout-ms 25000 \
  remember --no-project --tenant local --project demo --kind fact \
  "Append this first, then manage it with the provider."

cargo run -p beater-memory -- \
  --distiller provider-command \
  --distiller-command ./distill-provider \
  --distiller-arg --model \
  --distiller-arg gpt-4.1-mini \
  manage --limit 100
```

The same flags work for `serve`:

```bash
BEATER_MEMORY_TOKEN=dev-secret cargo run -p beater-memory -- \
  --distiller provider-command \
  --distiller-command ./distill-provider \
  serve --bind 127.0.0.1:8765
```

`--distiller-arg` can pass provider flags that start with `-`; repeating the
flag passes multiple command arguments. The provider command timeout defaults to
25s and must be lower than the server DB task timeout when used with `serve`.
Increase `--db-task-timeout-ms` if a provider, repair budget, or manage batch
needs more wall-clock time.

The command receives one JSON request on stdin with `request: "distill"` or
`request: "repair"`, the ledger `event`, `neighbors`, and repair fields for
malformed output. It must write provider JSON on stdout:

```json
{
  "memories": [
    {
      "op": "add",
      "node_kind": "fact",
      "text": "Distilled memory text",
      "target_node_id": null,
      "relation_edges": [
        {"kind": "fixes", "target_node_id": "existing-scoped-neighbor-id"}
      ],
      "cited_spans": [
        {"tenant_id": "local", "project_id": "demo", "trace_id": "trace", "span_id": "span", "seq": 1}
      ]
    }
  ]
}
```

Provider projection reports, HTTP metrics, Prometheus metrics, and audit detail
include provider calls, provider errors, schema errors, repairs, rejections,
durable replay batches, token estimates, and elapsed milliseconds. Accepted
command-provider batches are stored with the projected ledger event, so
`rebuild-projection` can replay those batches without calling the provider when
the same command replay fingerprint is selected. Rebuild fails before clearing
projections if any already projected event is missing a matching durable batch.
Replay-safe provider batches may use `add` and `noop` freely; `update` and
`invalidate` must include an explicit `target_node_id` after validation so
rebuild does not re-resolve targets against a different neighbor set.
`relation_edges` may point from the emitted memory to an existing scoped
neighbor by `target_node_id`. Providers may emit domain relation kinds
`caused_by`, `fixes`, `before`, `after`, `part_of`, `blocks`, and `enables`;
revision and projection-scaffolding kinds such as `supersedes`, `contradicts`,
`mentions`, `derived_from`, and `observed_in` are reserved for the engine.
Projected relation edges record the source ledger event, allowing
`known_at_unix_ms` reads and active reconstruction to use only relationships
that were known by the query's transaction-time boundary.
The current command fingerprint includes the command path string, JSON command
arguments, and effective repair budget; it does not include provider file
contents, environment variables, cwd-dependent behavior, timeout, or model
settings unless those settings are encoded in the command arguments.

### Provider-backed active reconstruction

Queries use the deterministic active reconstructor by default. To run a local
provider adapter during forced or auto Tier 2 reads, pass global reconstructor
flags before `query` or `serve`:

```bash
cargo run -p beater-memory -- \
  --reconstructor provider-command \
  --reconstructor-command ./reconstruct-provider \
  --reconstructor-timeout-ms 25000 \
  query --tenant local --project demo \
  --reconstruction-mode force \
  "incident alpha"

BEATER_MEMORY_TOKEN=dev-secret cargo run -p beater-memory -- \
  --reconstructor provider-command \
  --reconstructor-command ./reconstruct-provider \
  --reconstructor-timeout-ms 5000 \
  serve --bind 127.0.0.1:8765
```

`--reconstructor-arg` repeats like `--distiller-arg` and can pass values that
start with `-`. The command timeout defaults to 25s and must be lower than the
server DB task timeout when used with `serve`. CLI query and server startup
reject missing, non-file, or non-executable command paths. HTTP query requests
using provider reconstruction are rejected when
`--reconstructor-timeout-ms * max_reconstruction_steps` is greater than or equal
to `--db-task-timeout-ms`, so long-running providers should either use a lower
per-call timeout, fewer reconstruction steps, or a larger DB task timeout.

The command receives one `ReconstructionStep` JSON document on stdin with
`question`, `step_index`, `expanded_node_id`, `remaining_tokens`, and
`candidates`. It must write one decision JSON document on stdout:

```json
{"decision":"accept","node_id":"node-id-from-candidates"}
```

Valid decisions are `accept`, `prune`, and `stop`. `accept` and `prune` must
name a candidate node ID from the same step; unknown fields, unknown decisions,
and non-candidate node IDs are treated as schema failures. Provider transport or
schema failures stop active reconstruction for that query and are reported in
the `ReconstructionReport`; they do not fail the whole query. Query responses,
HTTP metrics, Prometheus metrics, and audit detail include reconstruction
provider calls, provider errors, schema errors, input/output token estimates,
and elapsed milliseconds.

### Evaluation suites

Eval fixtures use `contract_version: 1`; omitted versions default to v1 for
older local fixtures, while unsupported future versions are rejected. A fixture
can also include `source` metadata with `name`, `uri`, and `revision`; CLI
reports echo that metadata plus the suite file path.

By default each case runs in an isolated per-case project. Set
`"shared_haystack": true` only for chronological shared-haystack benchmarks:
the runner ingests, projects, and queries each case in order, so later cases can
see earlier observations but earlier cases cannot see future observations.

Report `score` is `effective_expectation_pass_rate`: content expectations are
matched against the answer, evidence, stale assumptions, and contradictions,
then hard gates such as `expected_tier` force the case score to `0.0` when they
fail. `content_score` preserves the raw content-only match rate for debugging.
Top-level `passed`/`failed` count full case pass/fail, `checks_*` count content
expectations, and `context_saturation_gap` is the clamped shortfall versus the
average full-context baseline over cases that supply one. Each case report
includes effective scope, modes, routing/reconstruction telemetry, expectation
match rows, compact answer/evidence excerpts, and write/read economics.

## Operations

Manage/projection is atomic per ledger event. The engine uses an immediate
SQLite transaction, rechecks that the event is still pending inside the
transaction, then commits the memory nodes, edges, cue index, citations, and
projected marker together. New databases are stamped with the Beater Memory SQLite
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
and ledger events remain intact. Schema upgrades that introduce source-event
edge provenance reset legacy source-less projection rows when needed so the next
manage/rebuild pass regenerates edges with bitemporal provenance. Audit
retention is explicit: maintenance can
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
  -d '{"question":"checkout database failure","scope":{"tenant_id":"local","project_id":"demo","environment_id":null,"as_of_unix_ms":null,"known_at_unix_ms":null},"reconstruction_mode":"auto","max_reconstruction_steps":4,"max_reconstruction_tokens":2000}' \
  http://127.0.0.1:8765/v1/query
```

## Crate API

Key public API exports include:

- `MemoryEngine`
- `ProjectReport` and `ProjectionRebuildReport`
- `SqliteMemoryStore`
- `MemoryServerConfig`, `try_memory_router`, `memory_router`, `serve`, and
  `serve_with_shutdown`
- `StoreHealth`, `StoreStats`, `MaintenanceOptions`, `MaintenanceReport`,
  `GraphIntegrityReport`, `GraphRepairReport`, `AuditPruneReport`,
  `ProjectionResetReport`, `BackupReport`, `RestoreReport`, `AuditRecord`, and
  `AuditEvent`
- `LiveResponse`, `ReadyResponse`, `ServiceMetricsSnapshot`, and
  `QueryTierMetrics`
- `LedgerEvent`
- `Distiller`, `HeuristicDistiller`, `ProviderDistiller`,
  `DistillationProvider`, `DistillationPrompt`, `DistillationRepairPrompt`,
  `DistillationReplayKey`, `DistillOutcome`, and `DistillMetrics`
- `ActiveReconstructor`, `DeterministicReconstructor`,
  `ProviderReconstructor`, `CommandReconstructionProvider`,
  `CommandReconstructionProviderConfig`, `ReconstructionProvider`,
  `ReconstructorConfig`, `RuntimeReconstructor`,
  `ReconstructionCandidate`, `ReconstructionDecision`,
  `ReconstructionDecisionOutcome`, `ReconstructionMetrics`, and
  `ReconstructionStep`
- `MemoryQuery` and `MemoryAnswer`
- `MemoryTier`, `MemoryMode`, `MemoryNodeKind`, `MemoryEdgeKind`,
  `BeliefRevisionOp`, `DistilledEdge`, `RoutingReason`, `RoutingReport`,
  `ReconstructionMode`, `ReconstructionOptions`, `ReconstructionReason`, and
  `ReconstructionReport`
- `EVAL_CONTRACT_VERSION`, `EvalSuite`, `EvalSuiteSource`, `EvalCase`,
  `EvalEvent`, `EvalOptions`, `EvalRuntimeOptions`, `EvalReport`, `EvalReportSource`,
  `EvalExpectationReport`, `EvalAbility`, `EvalScoreKind`,
  `EvalAbilitySummary`, `EvalTierSummary`, `run_eval_suite`, and
  `run_eval_suite_with_source`, and `run_eval_suite_with_source_and_runtime`
- import helpers for `beater.js` journals and canonical JSONL
- evidence token budgeting helpers

Run checks:

```bash
cargo fmt --all --check
cargo test
scripts/e2e-memory.sh
cargo clippy --workspace --all-targets -- -D warnings
```

## Design Constraints

- Memory is `write / manage / read`, not just `retrieve`.
- The write hot path should stay append-only and robust.
- `project` remains a compatibility alias for manage-time projection; new code
  should prefer `manage` when it is intentionally distilling pending ledger
  observations.
- Provider-backed distillation must happen outside projection write
  transactions, use constrained schemas with snake_case JSON values such as
  `"add"` and `"invalidate"`, repair or reject malformed output, and emit
  provider/repair/rejection counters before writing projections. Provider
  relation edges must target scoped neighbor memories and are stored with source
  event provenance for bitemporal graph traversal.
- Retrieval returns an answer-shaped evidence bundle, not raw chunks.
- Token cost, read latency, write amplification, and context pollution are
  first-class metrics.
- Evaluation reports use versioned deterministic content expectation checks or
  future calibrated judges; they do not use F1/BLEU as the correctness signal.
  Retrieval-tier expectations are hard gates that zero the effective case score
  when they fail, context-saturation gap is the clamped full-context shortfall,
  and selected evidence tokens are reported as the answer-context token load.
- Contradictions are graph edges and stale assumptions, not silently overwritten
  summaries.

## Feature Workflow

For each coherent feature slice: implement, self-review the diff, run focused
tests plus the workspace checks, commit only intended files, open a PR, and
merge only after local checks and GitHub CI are clean.
