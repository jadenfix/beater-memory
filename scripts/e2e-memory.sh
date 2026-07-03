#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

cargo build -p beater-memory

TMP_DIR="$(mktemp -d)"
SERVER_PID=""

stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}

cleanup() {
  stop_server
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

BIN="$ROOT/target/debug/beater-memory"
DB="$TMP_DIR/memory.db"
PROVIDER_DB="$TMP_DIR/provider-memory.db"
PROVIDER_HTTP_DB="$TMP_DIR/provider-http-memory.db"
TOKEN="e2e-secret"
PORT=""
BASE_URL=""

json_assert() {
  local path="$1"
  local code="$2"
  python3 - "$path" "$code" <<'PY'
import json
import sys

path, code = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as handle:
    data = json.load(handle)
expr = code.strip()
if not expr.startswith("assert "):
    raise SystemExit("json_assert snippets must start with 'assert '")
expr = expr[len("assert "):].strip()
if not eval(expr, {"__builtins__": {}}, {"data": data, "any": any, "all": all, "len": len}):
    raise SystemExit(f"json assertion failed: {expr}")
PY
}

allocate_port() {
  python3 - <<'PY'
import socket

sock = socket.socket()
sock.bind(("127.0.0.1", 0))
print(sock.getsockname()[1])
sock.close()
PY
}

start_server() {
  local server_db="${1:-$DB}"
  shift || true
  local attempt
  for attempt in $(seq 1 10); do
    PORT="$(allocate_port)"
    BASE_URL="http://127.0.0.1:$PORT"
    : > "$TMP_DIR/server.log"
    local -a command=("$BIN" "--db" "$server_db")
    if (($# > 0)); then
      command+=("$@")
    fi
    command+=("serve" "--bind" "127.0.0.1:$PORT")
    BEATER_MEMORY_TOKEN="$TOKEN" "${command[@]}" \
      > "$TMP_DIR/server.log" 2>&1 &
    SERVER_PID="$!"

    for _ in $(seq 1 80); do
      if curl -fsS "$BASE_URL/readyz" > "$TMP_DIR/readyz.json" 2>/dev/null; then
        return 0
      fi
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        break
      fi
      sleep 0.25
    done

    if kill -0 "$SERVER_PID" 2>/dev/null; then
      cat "$TMP_DIR/server.log" >&2
      return 1
    fi
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    if grep -qiE "address already in use|os error 48|os error 98" "$TMP_DIR/server.log"; then
      continue
    fi
    cat "$TMP_DIR/server.log" >&2
    return 1
  done
  echo "server failed to bind after port retries" >&2
  cat "$TMP_DIR/server.log" >&2
  return 1
}

api_get() {
  curl -fsS -H "Authorization: Bearer $TOKEN" "$BASE_URL$1"
}

api_post() {
  local path="$1"
  local body="$2"
  curl -fsS \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "$body" \
    "$BASE_URL$path"
}

PROVIDER="$TMP_DIR/provider-distiller.py"
cat > "$PROVIDER" <<'PY'
#!/usr/bin/env python3
import json
import sys

request = json.load(sys.stdin)
event = request["event"]
span = {
    "tenant_id": event["tenant_id"],
    "project_id": event["project_id"],
    "trace_id": event["trace_id"],
    "span_id": event["span_id"],
    "seq": event["seq"],
}
kind = event.get("name", "fact")
if kind not in {"episode", "fact", "procedure", "state", "gotcha", "anti_memory", "topic"}:
    kind = "fact"
text = "Provider distilled: " + event["text"]
print(json.dumps({
    "memories": [
        {
            "op": "add",
            "node_kind": kind,
            "text": text,
            "target_node_id": None,
            "cited_spans": [span],
        }
    ]
}))
PY
chmod +x "$PROVIDER"

"$BIN" --db "$DB" init > "$TMP_DIR/init.json"
json_assert "$TMP_DIR/init.json" 'assert data["ledger_events"] == 0'

"$BIN" --db "$DB" remember \
  --tenant local \
  --project demo \
  --kind fact \
  --no-project \
  "Write-only memory marker beta." \
  > "$TMP_DIR/remember-write-only.json"
json_assert "$TMP_DIR/remember-write-only.json" 'assert data["events_projected"] == 0 and data["memories_added"] == 0'
"$BIN" --db "$DB" stats > "$TMP_DIR/stats-write-only.json"
json_assert "$TMP_DIR/stats-write-only.json" 'assert data["ledger_events"] == 1 and data["pending_events"] == 1 and data["nodes"] == 0'
"$BIN" --db "$DB" manage --limit 1 > "$TMP_DIR/manage-write-only.json"
json_assert "$TMP_DIR/manage-write-only.json" 'assert data["events_projected"] == 1 and data["memories_added"] >= 1'
"$BIN" --db "$DB" stats > "$TMP_DIR/stats-managed.json"
json_assert "$TMP_DIR/stats-managed.json" 'assert data["pending_events"] == 0 and data["nodes"] > 0'
"$BIN" --db "$DB" --distiller provider-command stats > "$TMP_DIR/stats-lazy-distiller.json"
json_assert "$TMP_DIR/stats-lazy-distiller.json" 'assert data["pending_events"] == 0 and data["nodes"] > 0'

"$BIN" --db "$PROVIDER_DB" remember \
  --tenant local \
  --project provider \
  --kind fact \
  --no-project \
  "Provider command CLI marker gamma." \
  > "$TMP_DIR/provider-cli-remember.json"
json_assert "$TMP_DIR/provider-cli-remember.json" 'assert data["events_projected"] == 0 and data["distillation_provider_calls"] == 0'
"$BIN" --db "$PROVIDER_DB" \
  --distiller provider-command \
  --distiller-command "$PROVIDER" \
  --distiller-arg --ignored-provider-flag \
  manage --limit 10 \
  > "$TMP_DIR/provider-cli-manage.json"
json_assert "$TMP_DIR/provider-cli-manage.json" 'assert data["events_projected"] == 1 and data["memories_added"] == 1'
json_assert "$TMP_DIR/provider-cli-manage.json" 'assert data["distillation_provider_calls"] == 1 and data["distillation_provider_errors"] == 0 and data["distillation_rejections"] == 0'
"$BIN" --db "$PROVIDER_DB" query \
  --tenant local \
  --project provider \
  --json \
  "Provider command CLI marker" \
  > "$TMP_DIR/provider-cli-query.json"
json_assert "$TMP_DIR/provider-cli-query.json" 'assert data["evidence"] and any("Provider distilled:" in item["text"] for item in data["evidence"])'
set +e
"$BIN" --db "$PROVIDER_DB" \
  --distiller provider-command \
  --distiller-command "$PROVIDER" \
  rebuild-projection --yes-clear-projections \
  > "$TMP_DIR/provider-cli-rebuild.json" 2> "$TMP_DIR/provider-cli-rebuild.err"
provider_rebuild_status=$?
set -e
if [ "$provider_rebuild_status" -eq 0 ]; then
  echo "expected provider-backed rebuild to be rejected" >&2
  exit 1
fi
grep -q "replay-safe distiller" "$TMP_DIR/provider-cli-rebuild.err"

"$BIN" --db "$DB" remember \
  --tenant local \
  --project demo \
  --kind gotcha \
  --idempotency-key checkout-db \
  "Checkout fails when DATABASE_URL is missing. Fix by setting DATABASE_URL." \
  > "$TMP_DIR/remember-gotcha.json"
json_assert "$TMP_DIR/remember-gotcha.json" 'assert data["events_projected"] == 1 and data["memories_added"] >= 2'
json_assert "$TMP_DIR/remember-gotcha.json" 'assert data["distillation_outputs"] >= 2 and data["distillation_provider_calls"] == 0 and data["distillation_rejections"] == 0'
json_assert "$TMP_DIR/remember-gotcha.json" 'assert data["source_token_estimate"] > 0 and data["projected_memory_token_estimate"] > 0 and data["stored_memories_touched"] >= data["memories_added"]'

"$BIN" --db "$DB" remember \
  --tenant local \
  --project demo \
  --kind fact \
  "Use the old checkout token for deploys." \
  > "$TMP_DIR/remember-old-token.json"
json_assert "$TMP_DIR/remember-old-token.json" 'assert data["events_projected"] == 1'

"$BIN" --db "$DB" remember \
  --tenant local \
  --project demo \
  --kind fact \
  "Do not use the old checkout token; it is deprecated. Use the scoped deploy token instead of the old token." \
  > "$TMP_DIR/remember-new-token.json"
json_assert "$TMP_DIR/remember-new-token.json" 'assert data["events_projected"] == 1 and data["memories_invalidated"] >= 1'
json_assert "$TMP_DIR/remember-new-token.json" 'assert "distillation_schema_errors" in data and "distillation_elapsed_ms" in data'

"$BIN" --db "$DB" query \
  --tenant local \
  --project demo \
  --json \
  "old checkout token" \
  > "$TMP_DIR/query-token.json"
json_assert "$TMP_DIR/query-token.json" 'assert data["evidence"] and data["contradictions"] and data["stale_assumptions"]'

"$BIN" --db "$DB" remember \
  --tenant local \
  --project demo \
  --kind fact \
  "Incident alpha blocked deploys." \
  > "$TMP_DIR/remember-incident-alpha.json"
json_assert "$TMP_DIR/remember-incident-alpha.json" 'assert data["events_projected"] == 1'

"$BIN" --db "$DB" query \
  --tenant local \
  --project demo \
  --modes semantic,episodic \
  --max-tokens 10 \
  --reconstruction-mode force \
  --max-reconstruction-steps 2 \
  --max-reconstruction-tokens 200 \
  --json \
  "incident alpha" \
  > "$TMP_DIR/query-reconstruction-expands.json"
json_assert "$TMP_DIR/query-reconstruction-expands.json" 'assert data["tier_used"] == "active_reconstruction" and data["reconstruction"]["reason"] == "forced"'
json_assert "$TMP_DIR/query-reconstruction-expands.json" 'assert data["routing"]["routed_modes"] == ["semantic", "episodic"]'
json_assert "$TMP_DIR/query-reconstruction-expands.json" 'assert data["routing"]["reconstruction_modes"] == ["semantic", "episodic"]'
json_assert "$TMP_DIR/query-reconstruction-expands.json" 'assert data["reconstruction"]["accepted_node_ids"] and data["reconstruction"]["tokens_spent"] <= 200 and data["reconstruction"]["steps_used"] <= 2'

cat > "$TMP_DIR/spans.jsonl" <<'JSONL'
{"tenant_id":"local","project_id":"demo","trace_id":"trace-jsonl","span_id":"span-write","seq":1,"name":"procedure","status":"ok","attributes":{"beater.span.kind":"memory.write"},"start_time_unix_ms":1782864000000,"payload":{"memory":"Deploy procedure: run cargo test before merging memory changes."}}
JSONL

"$BIN" --db "$DB" import-jsonl \
  --path "$TMP_DIR/spans.jsonl" \
  > "$TMP_DIR/import-jsonl.json"
json_assert "$TMP_DIR/import-jsonl.json" 'assert data["import"]["rows_seen"] == 1 and data["import"]["events_inserted"] == 1 and data["project"]["events_projected"] == 1'

"$BIN" --db "$DB" query \
  --tenant local \
  --project demo \
  --modes procedural \
  --json \
  "deploy procedure cargo test" \
  > "$TMP_DIR/query-procedure.json"
json_assert "$TMP_DIR/query-procedure.json" 'assert data["evidence"] and any(item["kind"] == "procedure" for item in data["evidence"])'
json_assert "$TMP_DIR/query-procedure.json" 'assert data["routing"]["routed_modes"] == ["procedural"]'

cat > "$TMP_DIR/temporal-old-spans.jsonl" <<'JSONL'
{"tenant_id":"local","project_id":"demo","trace_id":"trace-temporal","span_id":"span-old","seq":1,"name":"fact","status":"ok","attributes":{"beater.span.kind":"memory.write"},"start_time_unix_ms":1000,"payload":{"memory":"Use the legacy API token for deploys."}}
JSONL

"$BIN" --db "$DB" import-jsonl \
  --path "$TMP_DIR/temporal-old-spans.jsonl" \
  > "$TMP_DIR/import-temporal-old-jsonl.json"
json_assert "$TMP_DIR/import-temporal-old-jsonl.json" 'assert data["import"]["rows_seen"] == 1 and data["import"]["events_inserted"] == 1 and data["project"]["events_projected"] == 1'
KNOWN_AFTER_TEMPORAL_OLD="$(python3 - <<'PY'
import time
print(int(time.time() * 1000))
PY
)"
sleep 0.02

cat > "$TMP_DIR/temporal-new-spans.jsonl" <<'JSONL'
{"tenant_id":"local","project_id":"demo","trace_id":"trace-temporal","span_id":"span-new","seq":2,"name":"fact","status":"ok","attributes":{"beater.span.kind":"memory.write"},"start_time_unix_ms":2000,"payload":{"memory":"Do not use the legacy API token; it is deprecated. Use the scoped API token instead."}}
JSONL

"$BIN" --db "$DB" import-jsonl \
  --path "$TMP_DIR/temporal-new-spans.jsonl" \
  > "$TMP_DIR/import-temporal-new-jsonl.json"
json_assert "$TMP_DIR/import-temporal-new-jsonl.json" 'assert data["import"]["rows_seen"] == 1 and data["import"]["events_inserted"] == 1 and data["project"]["events_projected"] == 1'

"$BIN" --db "$DB" query \
  --tenant local \
  --project demo \
  --modes semantic \
  --as-of-unix-ms 1500 \
  --reconstruction-mode force \
  --max-reconstruction-steps 2 \
  --max-reconstruction-tokens 500 \
  --json \
  "legacy API token" \
  > "$TMP_DIR/query-temporal-reconstruction-before.json"
json_assert "$TMP_DIR/query-temporal-reconstruction-before.json" 'assert data["tier_used"] == "active_reconstruction" and data["reconstruction"]["mode"] == "force" and data["reconstruction"]["reason"] == "forced"'
json_assert "$TMP_DIR/query-temporal-reconstruction-before.json" 'assert data["routing"]["reconstruction_modes"] == ["semantic"]'
json_assert "$TMP_DIR/query-temporal-reconstruction-before.json" 'assert data["evidence"] and any("Use the legacy API token" in item["text"] for item in data["evidence"])'
json_assert "$TMP_DIR/query-temporal-reconstruction-before.json" 'assert not any("scoped API token" in item["text"] for item in data["evidence"])'

"$BIN" --db "$DB" query \
  --tenant local \
  --project demo \
  --modes semantic \
  --as-of-unix-ms 1500 \
  --json \
  "legacy API token" \
  > "$TMP_DIR/query-temporal-before.json"
json_assert "$TMP_DIR/query-temporal-before.json" 'assert data["evidence"] and any("Use the legacy API token" in item["text"] for item in data["evidence"])'
json_assert "$TMP_DIR/query-temporal-before.json" 'assert not any("scoped API token" in item["text"] for item in data["evidence"]) and not data["contradictions"] and not data["stale_assumptions"]'

"$BIN" --db "$DB" query \
  --tenant local \
  --project demo \
  --modes semantic \
  --as-of-unix-ms 2500 \
  --json \
  "legacy API token" \
  > "$TMP_DIR/query-temporal-after.json"
json_assert "$TMP_DIR/query-temporal-after.json" 'assert data["evidence"] and any("scoped API token" in item["text"] for item in data["evidence"])'
json_assert "$TMP_DIR/query-temporal-after.json" 'assert not any("for deploys" in item["text"] for item in data["evidence"]) and data["contradictions"] and data["stale_assumptions"]'

"$BIN" --db "$DB" query \
  --tenant local \
  --project demo \
  --modes semantic \
  --as-of-unix-ms 2500 \
  --known-at-unix-ms 1500 \
  --json \
  "legacy API token" \
  > "$TMP_DIR/query-temporal-before-known.json"
json_assert "$TMP_DIR/query-temporal-before-known.json" 'assert not data["evidence"] and not data["contradictions"] and not data["stale_assumptions"]'

"$BIN" --db "$DB" query \
  --tenant local \
  --project demo \
  --modes semantic \
  --as-of-unix-ms 2500 \
  --known-at-unix-ms "$KNOWN_AFTER_TEMPORAL_OLD" \
  --json \
  "legacy API token" \
  > "$TMP_DIR/query-temporal-old-known-new-unknown.json"
json_assert "$TMP_DIR/query-temporal-old-known-new-unknown.json" 'assert data["evidence"] and any("Use the legacy API token" in item["text"] for item in data["evidence"])'
json_assert "$TMP_DIR/query-temporal-old-known-new-unknown.json" 'assert not any("scoped API token" in item["text"] for item in data["evidence"]) and not data["contradictions"] and not data["stale_assumptions"]'

"$BIN" --db "$DB" query \
  --tenant local \
  --project demo \
  --modes semantic \
  --as-of-unix-ms 2500 \
  --known-at-unix-ms 9999999999999 \
  --json \
  "legacy API token" \
  > "$TMP_DIR/query-temporal-after-known.json"
json_assert "$TMP_DIR/query-temporal-after-known.json" 'assert data["evidence"] and any("scoped API token" in item["text"] for item in data["evidence"])'
json_assert "$TMP_DIR/query-temporal-after-known.json" 'assert data["contradictions"] and data["stale_assumptions"]'

cat > "$TMP_DIR/eval-suite.json" <<'JSON'
{
  "name": "lme-v2-shaped-smoke",
  "tenant_id": "local",
  "project_id": "eval",
  "cases": [
    {
      "id": "static-state",
      "ability": "static_state_recall",
      "question": "what is the production API base URL?",
      "modes": ["state"],
      "baseline_full_context_score": 1.0,
      "events": [
        {
          "kind": "state",
          "text": "The production API base URL is https://api.example.test.",
          "observed_at_unix_ms": 1000
        }
      ],
      "expected_evidence_contains": ["https://api.example.test"],
      "expected_tier": "activation"
    },
    {
      "id": "late-known-hidden",
      "ability": "dynamic_state_tracking",
      "question": "what is the late known eval flag?",
      "modes": ["semantic"],
      "as_of_unix_ms": 1500,
      "known_at_unix_ms": 2000,
      "baseline_full_context_score": 1.0,
      "events": [
        {
          "kind": "fact",
          "text": "The late known eval flag is beta.",
          "observed_at_unix_ms": 1000,
          "ingested_at_unix_ms": 3000
        }
      ],
      "expected_answer_contains": ["No matching memory"],
      "expected_tier": "activation"
    },
    {
      "id": "workflow",
      "ability": "workflow_knowledge",
      "question": "what is the deploy workflow?",
      "modes": ["procedural"],
      "reconstruction": {"mode": "force", "max_steps": 2, "max_tokens": 500},
      "baseline_full_context_score": 1.0,
      "events": [
        {
          "kind": "procedure",
          "text": "Deploy workflow: run migrations, restart workers, then check health.",
          "observed_at_unix_ms": 2000
        }
      ],
      "expected_evidence_contains": ["restart workers"],
      "expected_tier": "active_reconstruction"
    },
    {
      "id": "gotcha",
      "ability": "environment_gotcha",
      "question": "why does checkout return 500?",
      "modes": ["gotcha"],
      "baseline_full_context_score": 1.0,
      "events": [
        {
          "kind": "gotcha",
          "text": "Checkout gotcha: missing DATABASE_URL returns HTTP 500.",
          "observed_at_unix_ms": 3000
        }
      ],
      "expected_evidence_contains": ["DATABASE_URL"],
      "expected_tier": "activation"
    },
    {
      "id": "dynamic-state",
      "ability": "dynamic_state_tracking",
      "question": "is the checkout feature flag enabled?",
      "modes": ["state"],
      "baseline_full_context_score": 1.0,
      "events": [
        {
          "kind": "state",
          "text": "Checkout feature flag is enabled now.",
          "observed_at_unix_ms": 4000
        }
      ],
      "expected_evidence_contains": ["enabled"],
      "expected_tier": "activation"
    },
    {
      "id": "premise",
      "ability": "premise_awareness",
      "question": "should I use the legacy API token?",
      "modes": ["semantic"],
      "baseline_full_context_score": 1.0,
      "events": [
        {
          "kind": "fact",
          "text": "Use the legacy API token.",
          "observed_at_unix_ms": 5000
        },
        {
          "kind": "fact",
          "text": "Do not use the legacy API token; it is deprecated. Use the scoped API token.",
          "observed_at_unix_ms": 6000
        }
      ],
      "expected_evidence_contains": ["scoped API token"],
      "expected_stale_contains": ["legacy API token"],
      "expected_contradiction_contains": ["scoped API token"]
    }
  ]
}
JSON
"$BIN" eval --suite "$TMP_DIR/eval-suite.json" > "$TMP_DIR/eval-report.json"
json_assert "$TMP_DIR/eval-report.json" 'assert data["suite"] == "lme-v2-shaped-smoke" and data["cases"] == 6 and data["passed"] == 6 and data["failed"] == 0'
json_assert "$TMP_DIR/eval-report.json" 'assert data["score"] == 1.0 and data["context_saturation_gap"] == 0.0'
json_assert "$TMP_DIR/eval-report.json" 'assert data["source_tokens_per_stored_memory"] > 0 and data["projected_tokens_per_stored_memory"] > 0 and data["tokens_into_context_total"] > 0'
json_assert "$TMP_DIR/eval-report.json" 'assert any(row["ability"] == "premise_awareness" and row["score"] == 1.0 for row in data["ability_scores"])'
json_assert "$TMP_DIR/eval-report.json" 'assert any(row["tier"] == "active_reconstruction" and row["requests"] >= 1 for row in data["tier_metrics"])'
"$BIN" eval --suite "$TMP_DIR/eval-suite.json" --max-reconstruction-steps 1 > "$TMP_DIR/eval-report-step-override.json"
json_assert "$TMP_DIR/eval-report-step-override.json" 'assert data["passed"] == 6 and any(case["id"] == "workflow" and case["tier_used"] == "active_reconstruction" for case in data["case_reports"])'

cat > "$TMP_DIR/eval-failing-suite.json" <<'JSON'
{
  "name": "failing-smoke",
  "cases": [
    {
      "id": "missing",
      "question": "what database is configured?",
      "events": [
        {"kind": "fact", "text": "Checkout uses DATABASE_URL."}
      ],
      "expected_evidence_contains": ["REDIS_URL"]
    }
  ]
}
JSON
set +e
"$BIN" eval --suite "$TMP_DIR/eval-failing-suite.json" > "$TMP_DIR/eval-failing-report.json" 2> "$TMP_DIR/eval-failing.err"
eval_status=$?
set -e
if [ "$eval_status" -eq 0 ]; then
  echo "expected failing eval suite to exit nonzero" >&2
  exit 1
fi
json_assert "$TMP_DIR/eval-failing-report.json" 'assert data["failed"] == 1 and data["case_reports"][0]["failure_reasons"]'

start_server
json_assert "$TMP_DIR/readyz.json" 'assert data["status"] == "ok" and data["database"] == "ok"'

api_post "/v1/remember" '{"tenant_id":"local","project_id":"demo","kind":"gotcha","idempotency_key":"http-checkout-db","text":"HTTP checkout fails unless DATABASE_URL is configured before migrations."}' \
  > "$TMP_DIR/http-remember.json"
json_assert "$TMP_DIR/http-remember.json" 'assert data["ingested"] is True and data["project"]["events_projected"] == 1'
json_assert "$TMP_DIR/http-remember.json" 'assert data["project"]["distillation_outputs"] >= 2 and data["project"]["distillation_provider_calls"] == 0'

api_post "/v1/remember" '{"tenant_id":"local","project_id":"demo","kind":"gotcha","idempotency_key":"http-checkout-db","text":"HTTP checkout fails unless DATABASE_URL is configured before migrations."}' \
  > "$TMP_DIR/http-remember-duplicate.json"
json_assert "$TMP_DIR/http-remember-duplicate.json" 'assert data["ingested"] is False'

api_post "/v1/query" '{"question":"checkout database migrations","scope":{"tenant_id":"local","project_id":"demo","environment_id":null,"as_of_unix_ms":null},"max_tokens":800,"require_fresh":false,"modes":["semantic","episodic","procedural","gotcha","state"]}' \
  > "$TMP_DIR/http-query.json"
json_assert "$TMP_DIR/http-query.json" 'assert data["evidence"] and "DATABASE_URL" in data["answer"]'
json_assert "$TMP_DIR/http-query.json" 'assert data["routing"]["routed_modes"] == ["semantic", "episodic", "procedural", "gotcha", "state"]'

api_post "/v1/remember" '{"tenant_id":"local","project_id":"demo","kind":"fact","text":"HTTP incident alpha blocked deploys."}' \
  > "$TMP_DIR/http-remember-incident-alpha.json"
json_assert "$TMP_DIR/http-remember-incident-alpha.json" 'assert data["ingested"] is True and data["project"]["events_projected"] == 1'

api_post "/v1/query" '{"question":"HTTP incident alpha","scope":{"tenant_id":"local","project_id":"demo","environment_id":null,"as_of_unix_ms":null},"modes":["semantic","episodic"],"max_tokens":10,"reconstruction_mode":"force","max_reconstruction_steps":2,"max_reconstruction_tokens":200}' \
  > "$TMP_DIR/http-query-reconstruction-expands.json"
json_assert "$TMP_DIR/http-query-reconstruction-expands.json" 'assert data["tier_used"] == "active_reconstruction" and data["reconstruction"]["reason"] == "forced"'
json_assert "$TMP_DIR/http-query-reconstruction-expands.json" 'assert data["routing"]["routed_modes"] == ["semantic", "episodic"]'
json_assert "$TMP_DIR/http-query-reconstruction-expands.json" 'assert data["routing"]["reconstruction_modes"] == ["semantic", "episodic"]'
json_assert "$TMP_DIR/http-query-reconstruction-expands.json" 'assert data["reconstruction"]["accepted_node_ids"] and data["reconstruction"]["tokens_spent"] <= 200 and data["reconstruction"]["steps_used"] <= 2'

api_post "/v1/query" '{"question":"legacy API token","scope":{"tenant_id":"local","project_id":"demo","environment_id":null,"as_of_unix_ms":1500},"modes":["semantic"]}' \
  > "$TMP_DIR/http-query-temporal-before.json"
json_assert "$TMP_DIR/http-query-temporal-before.json" 'assert data["evidence"] and any("Use the legacy API token" in item["text"] for item in data["evidence"])'
json_assert "$TMP_DIR/http-query-temporal-before.json" 'assert not any("scoped API token" in item["text"] for item in data["evidence"]) and not data["contradictions"] and not data["stale_assumptions"]'

api_post "/v1/query" '{"question":"legacy API token","scope":{"tenant_id":"local","project_id":"demo","environment_id":null,"as_of_unix_ms":1500},"modes":["semantic"],"reconstruction_mode":"force","max_reconstruction_steps":2,"max_reconstruction_tokens":500}' \
  > "$TMP_DIR/http-query-temporal-reconstruction-before.json"
json_assert "$TMP_DIR/http-query-temporal-reconstruction-before.json" 'assert data["tier_used"] == "active_reconstruction" and data["reconstruction"]["mode"] == "force" and data["reconstruction"]["reason"] == "forced"'
json_assert "$TMP_DIR/http-query-temporal-reconstruction-before.json" 'assert data["routing"]["reconstruction_modes"] == ["semantic"]'
json_assert "$TMP_DIR/http-query-temporal-reconstruction-before.json" 'assert data["evidence"] and any("Use the legacy API token" in item["text"] for item in data["evidence"])'
json_assert "$TMP_DIR/http-query-temporal-reconstruction-before.json" 'assert not any("scoped API token" in item["text"] for item in data["evidence"])'

api_post "/v1/query" '{"question":"legacy API token","scope":{"tenant_id":"local","project_id":"demo","environment_id":null,"as_of_unix_ms":2500},"modes":["semantic"]}' \
  > "$TMP_DIR/http-query-temporal-after.json"
json_assert "$TMP_DIR/http-query-temporal-after.json" 'assert data["evidence"] and any("scoped API token" in item["text"] for item in data["evidence"])'
json_assert "$TMP_DIR/http-query-temporal-after.json" 'assert not any("for deploys" in item["text"] for item in data["evidence"]) and data["contradictions"] and data["stale_assumptions"]'

maintenance_status="$(
  curl -sS \
    -o "$TMP_DIR/http-maintenance-invalid.json" \
    -w "%{http_code}" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"prune_audit_before_unix_ms":-1,"retain_latest_audit_events":0}' \
    "$BASE_URL/v1/maintenance"
)"
if [[ "$maintenance_status" != "400" ]]; then
  cat "$TMP_DIR/http-maintenance-invalid.json" >&2
  exit 1
fi
json_assert "$TMP_DIR/http-maintenance-invalid.json" 'assert data["error"]["code"] == "bad_request"'

api_get "/v1/audit?limit=50" > "$TMP_DIR/http-audit.json"
json_assert "$TMP_DIR/http-audit.json" 'assert any(event["action"] == "maintenance" and event["outcome"] == "failure" and event["status_code"] == 400 for event in data["events"])'

api_get "/v1/stats" > "$TMP_DIR/http-stats.json"
json_assert "$TMP_DIR/http-stats.json" 'assert data["ledger_events"] >= 5 and data["nodes"] > 0 and data["audit_events"] > 0'
json_assert "$TMP_DIR/http-stats.json" 'assert data["total_node_tokens"] > 0 and data["active_node_tokens"] > 0 and data["active_fact_nodes"] >= 1'

api_get "/v1/metrics/prometheus" > "$TMP_DIR/prometheus.txt"
grep -q "beater_memory_http_requests_total" "$TMP_DIR/prometheus.txt"
python3 - "$TMP_DIR/prometheus.txt" <<'PY'
import re
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    text = handle.read()

def value(metric: str, labels: str) -> float:
    match = re.search(rf'^{re.escape(metric)}\{{{re.escape(labels)}\}} ([0-9.]+)$', text, re.M)
    if not match:
        raise SystemExit(f"missing {metric}{{{labels}}}")
    return float(match.group(1))

def plain_value(metric: str) -> float:
    match = re.search(rf'^{re.escape(metric)} ([0-9.]+)$', text, re.M)
    if not match:
        raise SystemExit(f"missing {metric}")
    return float(match.group(1))

assert value("beater_memory_query_tier_requests_total", 'tier="activation"') > 0
assert value("beater_memory_query_tier_requests_total", 'tier="active_reconstruction"') > 0
assert value("beater_memory_query_tier_tokens_total", 'tier="activation",token_kind="answer"') > 0
assert plain_value("beater_memory_distillation_provider_calls_total") == 0
PY

stop_server
"$BIN" --db "$PROVIDER_HTTP_DB" init > "$TMP_DIR/provider-http-init.json"
start_server "$PROVIDER_HTTP_DB" --distiller provider-command --distiller-command "$PROVIDER"
json_assert "$TMP_DIR/readyz.json" 'assert data["status"] == "ok" and data["database"] == "ok"'

api_post "/v1/remember" '{"tenant_id":"local","project_id":"provider-http","kind":"fact","text":"Provider command HTTP marker delta.","project":false}' \
  > "$TMP_DIR/provider-http-remember.json"
json_assert "$TMP_DIR/provider-http-remember.json" 'assert data["ingested"] is True and data["project"] is None'

api_post "/v1/manage" '{"limit":10}' > "$TMP_DIR/provider-http-manage.json"
json_assert "$TMP_DIR/provider-http-manage.json" 'assert data["events_projected"] == 1 and data["memories_added"] == 1'
json_assert "$TMP_DIR/provider-http-manage.json" 'assert data["distillation_provider_calls"] == 1 and data["distillation_provider_errors"] == 0 and data["distillation_rejections"] == 0'

api_post "/v1/query" '{"question":"Provider command HTTP marker","scope":{"tenant_id":"local","project_id":"provider-http","environment_id":null,"as_of_unix_ms":null},"modes":["semantic"]}' \
  > "$TMP_DIR/provider-http-query.json"
json_assert "$TMP_DIR/provider-http-query.json" 'assert data["evidence"] and any("Provider distilled:" in item["text"] for item in data["evidence"])'

api_get "/v1/metrics" > "$TMP_DIR/provider-http-metrics.json"
json_assert "$TMP_DIR/provider-http-metrics.json" 'assert data["manage_requests"] == 1 and data["distillation_provider_calls"] == 1 and data["distillation_rejections"] == 0'

api_get "/v1/audit?limit=20" > "$TMP_DIR/provider-http-audit.json"
json_assert "$TMP_DIR/provider-http-audit.json" 'assert any(event["action"] == "manage" and event["outcome"] == "success" and event["detail"]["project"]["distillation_provider_calls"] == 1 for event in data["events"])'

api_get "/v1/metrics/prometheus" > "$TMP_DIR/provider-http-prometheus.txt"
python3 - "$TMP_DIR/provider-http-prometheus.txt" <<'PY'
import re
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    text = handle.read()

match = re.search(r'^beater_memory_distillation_provider_calls_total ([0-9.]+)$', text, re.M)
if not match:
    raise SystemExit("missing provider calls metric")
assert float(match.group(1)) == 1
PY

echo "beater-memory e2e passed"
