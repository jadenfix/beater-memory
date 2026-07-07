#!/usr/bin/env python3
"""Check remi's public HTTP manifest against the Rust server and CLI."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "public-http-contract.json"
SERVER = ROOT / "crates" / "beater-memory" / "src" / "server.rs"
CLI = ROOT / "crates" / "beater-memory" / "src" / "bin" / "beater-memory.rs"
README = ROOT / "README.md"

ROUTE_RE = re.compile(r'\.route\(\s*"(?P<path>[^"]+)",\s*(?P<method>get|post)\(')
STRUCT_RE = re.compile(r"\b(?:pub\s+)?struct\s+(?P<name>[A-Za-z0-9_]+)\b")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def route_set(source: str) -> set[tuple[str, str]]:
    return {
        (match.group("method").upper(), match.group("path"))
        for match in ROUTE_RE.finditer(source)
    }


def declared_structs(source: str) -> set[str]:
    return {match.group("name") for match in STRUCT_RE.finditer(source)}


def main() -> int:
    manifest = load_json(MANIFEST)
    server = SERVER.read_text(encoding="utf-8")
    cli = CLI.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    violations: list[str] = []

    manifest_routes = {
        (endpoint["method"], endpoint["path"]) for endpoint in manifest["endpoints"]
    }
    source_routes = route_set(server)
    if manifest_routes != source_routes:
        missing = sorted(source_routes - manifest_routes)
        extra = sorted(manifest_routes - source_routes)
        if missing:
            violations.append(f"manifest missing source route(s): {missing}")
        if extra:
            violations.append(f"manifest has route(s) not in source: {extra}")

    operations = [endpoint["operation"] for endpoint in manifest["endpoints"]]
    if len(operations) != len(set(operations)):
        violations.append("endpoint operation names must be unique")
    for operation in operations:
        if not re.fullmatch(r"[a-z][a-zA-Z0-9]*", operation):
            violations.append(f"operation {operation!r} is not lower-camel")

    for endpoint in manifest["endpoints"]:
        path = endpoint["path"]
        auth = endpoint["auth"]
        if path.startswith("/v1/") and auth != "bearer":
            violations.append(f"{path}: /v1 route must use bearer auth")
        if not path.startswith("/v1/") and auth != "public":
            violations.append(f"{path}: non-/v1 probe must be public")

    structs = declared_structs(server)
    external_schemas = {
        "MemoryAnswer",
        "MaintenanceReport",
        "ProjectReport",
        "PrometheusText",
        "StoreHealth",
        "StoreStats",
    }
    for endpoint in manifest["endpoints"]:
        for key in ("request_schema", "query_schema", "response_schema"):
            schema = endpoint.get(key)
            if not schema or schema in external_schemas:
                continue
            if schema not in structs:
                violations.append(f"{endpoint['operation']}: {key} {schema} not found")

    required_server_snippets = [
        'const REQUEST_ID_HEADER: &str = "x-request-id";',
        "header::AUTHORIZATION",
        'value.strip_prefix("Bearer ")',
        "header::WWW_AUTHENTICATE",
        "header::RETRY_AFTER",
        'code: "unauthorized"',
        'code: "rate_limited"',
        'code: "service_busy"',
        'code: "service_timeout"',
        'code: "bad_request"',
        'code: "internal_error"',
    ]
    for snippet in required_server_snippets:
        if snippet not in server:
            violations.append(f"server missing public contract snippet: {snippet}")

    required_cli_snippets = [
        "token: Option<String>",
        "token_file: Option<PathBuf>",
        'default_value = "REMI_TOKEN"',
        "read_to_string",
        "allow_no_auth",
        "refusing to start without auth",
    ]
    for snippet in required_cli_snippets:
        if snippet not in cli:
            violations.append(f"CLI missing auth/config snippet: {snippet}")

    if "docs/public-http-contract.json" not in readme:
        violations.append("README must link docs/public-http-contract.json")

    auth = manifest["auth"]
    if auth.get("scheme") != "bearer" or auth.get("format") != "Bearer <token>":
        violations.append("manifest auth must use Bearer token shape")
    if auth.get("env") != "REMI_TOKEN":
        violations.append("manifest auth env must be REMI_TOKEN")
    expected_flags = ["--token", "--token-file", "--token-env", "--allow-no-auth"]
    if auth.get("cli_flags") != expected_flags:
        violations.append(f"manifest auth cli_flags must be {expected_flags}")
    expected_precedence = ["flag", "token_file", "env"]
    if manifest["client_shape"].get("auth_precedence") != expected_precedence:
        violations.append(
            f"manifest client auth_precedence must be {expected_precedence}"
        )
    if manifest["client_shape"].get("token_field") != "token":
        violations.append("manifest client token field must be token")
    if manifest["sdk"].get("status") != "not_published":
        violations.append("manifest should not claim a published SDK yet")
    if manifest["mcp"].get("status") != "not_exposed":
        violations.append("manifest should not claim an exposed MCP surface yet")

    if violations:
        print(f"{len(violations)} public HTTP contract violation(s):")
        for violation in violations:
            print(f"  - {violation}")
        return 1

    print(
        "public HTTP contract check passed: "
        f"{len(manifest_routes)} route(s), auth={auth['scheme']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
