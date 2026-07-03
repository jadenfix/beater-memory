#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

cargo build -p beater-memory

TMP_DIR="$(mktemp -d)"
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

BIN="$ROOT/target/debug/beater-memory"
DB="$TMP_DIR/memory.db"
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
  local attempt
  for attempt in $(seq 1 10); do
    PORT="$(allocate_port)"
    BASE_URL="http://127.0.0.1:$PORT"
    : > "$TMP_DIR/server.log"
    BEATER_MEMORY_TOKEN="$TOKEN" "$BIN" --db "$DB" serve --bind "127.0.0.1:$PORT" \
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

"$BIN" --db "$DB" init > "$TMP_DIR/init.json"
json_assert "$TMP_DIR/init.json" 'assert data["ledger_events"] == 0'

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

cat > "$TMP_DIR/temporal-spans.jsonl" <<'JSONL'
{"tenant_id":"local","project_id":"demo","trace_id":"trace-temporal","span_id":"span-old","seq":1,"name":"fact","status":"ok","attributes":{"beater.span.kind":"memory.write"},"start_time_unix_ms":1000,"payload":{"memory":"Use the legacy API token for deploys."}}
{"tenant_id":"local","project_id":"demo","trace_id":"trace-temporal","span_id":"span-new","seq":2,"name":"fact","status":"ok","attributes":{"beater.span.kind":"memory.write"},"start_time_unix_ms":2000,"payload":{"memory":"Do not use the legacy API token; it is deprecated. Use the scoped API token instead."}}
JSONL

"$BIN" --db "$DB" import-jsonl \
  --path "$TMP_DIR/temporal-spans.jsonl" \
  > "$TMP_DIR/import-temporal-jsonl.json"
json_assert "$TMP_DIR/import-temporal-jsonl.json" 'assert data["import"]["rows_seen"] == 2 and data["import"]["events_inserted"] == 2 and data["project"]["events_projected"] == 2'

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

assert value("beater_memory_query_tier_requests_total", 'tier="activation"') > 0
assert value("beater_memory_query_tier_requests_total", 'tier="active_reconstruction"') > 0
assert value("beater_memory_query_tier_tokens_total", 'tier="activation",token_kind="answer"') > 0
PY

echo "beater-memory e2e passed"
