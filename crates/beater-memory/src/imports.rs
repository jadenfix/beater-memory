use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use chrono::DateTime;
use rusqlite::{Connection, OpenFlags};

use crate::{
    error::{MemoryError, MemoryResult},
    model::MemoryNodeKind,
    store::{LedgerEvent, SqliteMemoryStore},
    text::{concise, json_text, now_unix_ms, stable_id},
};

/// Importer for the SQLite journal created by `beater.js`.
#[derive(Clone, Debug)]
pub struct BeaterJsJournal {
    path: PathBuf,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BeaterJsImportReport {
    pub rows_seen: usize,
    pub rows_skipped: usize,
    pub events_inserted: usize,
    pub events_duplicate: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CanonicalJsonlImportReport {
    pub rows_seen: usize,
    pub events_inserted: usize,
    pub events_duplicate: usize,
}

impl BeaterJsJournal {
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn import_into(
        &self,
        store: &SqliteMemoryStore,
        tenant_id: &str,
        project_id: &str,
        environment_id: Option<&str>,
    ) -> MemoryResult<BeaterJsImportReport> {
        let conn = Connection::open_with_flags(&self.path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        let mut stmt = conn.prepare(
            "
            SELECT r.id, r.agent, r.status, r.input, r.created_at,
                   s.seq, s.kind, s.status, s.request, s.result, s.tool_name,
                   s.tool_use_id, s.attempt, s.started_at, s.finished_at
            FROM runs r
            JOIN steps s ON s.run_id = r.id
            ORDER BY r.created_at, s.seq
            ",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(BeaterJsStep {
                run_id: row.get(0)?,
                agent: row.get(1)?,
                run_status: row.get(2)?,
                run_input: row.get(3)?,
                run_created_at: row.get(4)?,
                seq: row.get(5)?,
                kind: row.get(6)?,
                status: row.get(7)?,
                request_json: row.get(8)?,
                result_json: row.get(9)?,
                tool_name: row.get(10)?,
                tool_use_id: row.get(11)?,
                attempt: row.get(12)?,
                started_at: row.get(13)?,
                finished_at: row.get(14)?,
            })
        })?;

        store.with_immediate_transaction(|store| {
            let mut report = BeaterJsImportReport::default();
            for row in rows {
                let step = row?;
                report.rows_seen += 1;
                if should_skip_beater_js_step(&step) {
                    report.rows_skipped += 1;
                    continue;
                }
                let event = step.into_event(tenant_id, project_id, environment_id)?;
                event.validate().map_err(|err| {
                    MemoryError::invalid(format!(
                        "invalid beater.js journal row run_id={} seq={}: {err}",
                        event.trace_id, event.seq
                    ))
                })?;
                if store.append_event(&event)? {
                    report.events_inserted += 1;
                } else {
                    report.events_duplicate += 1;
                }
            }
            Ok(report)
        })
    }
}

pub fn import_canonical_jsonl(
    path: impl AsRef<Path>,
    store: &SqliteMemoryStore,
    tenant_id: Option<&str>,
    project_id: Option<&str>,
    environment_id: Option<&str>,
) -> MemoryResult<CanonicalJsonlImportReport> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    store.with_immediate_transaction(|store| {
        let mut report = CanonicalJsonlImportReport::default();
        for (index, line) in reader.lines().enumerate() {
            let line_number = index + 1;
            let line = line.map_err(|err| {
                MemoryError::invalid(format!("failed reading JSONL line {line_number}: {err}"))
            })?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            report.rows_seen += 1;
            let value: serde_json::Value = serde_json::from_str(trimmed).map_err(|err| {
                MemoryError::invalid(format!("invalid canonical JSONL line {line_number}: {err}"))
            })?;
            let event = canonical_json_event(&value, tenant_id, project_id, environment_id);
            event.validate().map_err(|err| {
                MemoryError::invalid(format!("invalid canonical JSONL line {line_number}: {err}"))
            })?;
            if store.append_event(&event)? {
                report.events_inserted += 1;
            } else {
                report.events_duplicate += 1;
            }
        }
        Ok(report)
    })
}

#[derive(Debug)]
struct BeaterJsStep {
    run_id: String,
    agent: String,
    run_status: String,
    run_input: String,
    run_created_at: i64,
    seq: u64,
    kind: String,
    status: String,
    request_json: String,
    result_json: Option<String>,
    tool_name: Option<String>,
    tool_use_id: Option<String>,
    attempt: i64,
    started_at: i64,
    finished_at: Option<i64>,
}

impl BeaterJsStep {
    fn into_event(
        self,
        tenant_id: &str,
        project_id: &str,
        environment_id: Option<&str>,
    ) -> MemoryResult<LedgerEvent> {
        let request: serde_json::Value = serde_json::from_str(&self.request_json)
            .map_err(|err| self.invalid_json_error("request", err))?;
        let result: serde_json::Value = match self.result_json.as_deref() {
            Some(result_json) => serde_json::from_str(result_json)
                .map_err(|err| self.invalid_json_error("result", err))?,
            None => serde_json::Value::Null,
        };
        let span_kind = beater_js_span_kind(&self.kind, self.tool_name.as_deref());
        let name = beater_js_event_name(
            span_kind,
            self.tool_name.as_deref(),
            &request,
            &result,
            &self.agent,
            &self.kind,
        );
        let text = beater_js_event_text(span_kind, &self.run_input, &request, &result);
        Ok(LedgerEvent {
            id: None,
            source: "beater.js.journal".to_string(),
            tenant_id: tenant_id.to_string(),
            project_id: project_id.to_string(),
            environment_id: environment_id.map(str::to_string),
            trace_id: self.run_id,
            span_id: self
                .tool_use_id
                .clone()
                .unwrap_or_else(|| format!("step-{}", self.seq)),
            seq: self.seq,
            span_kind: span_kind.to_string(),
            name,
            status: self.status,
            text,
            payload: serde_json::json!({
                "agent": self.agent,
                "run_status": self.run_status,
                "run_created_at": self.run_created_at,
                "request": request,
                "result": result,
                "tool_name": self.tool_name,
                "tool_use_id": self.tool_use_id,
                "attempt": self.attempt,
                "finished_at": self.finished_at,
            }),
            observed_at_unix_ms: self.started_at.saturating_mul(1_000),
            ingested_at_unix_ms: now_unix_ms(),
            projected_at_unix_ms: None,
        })
    }

    fn invalid_json_error(&self, field: &str, err: serde_json::Error) -> MemoryError {
        MemoryError::invalid(format!(
            "invalid beater.js journal {field} JSON for run_id={} seq={}: {err}",
            self.run_id, self.seq
        ))
    }
}

fn beater_js_event_name(
    span_kind: &str,
    tool_name: Option<&str>,
    request: &serde_json::Value,
    result: &serde_json::Value,
    agent: &str,
    kind: &str,
) -> String {
    if span_kind == "memory.write" {
        return memory_kind_from_value(request)
            .or_else(|| memory_kind_from_value(result))
            .unwrap_or(MemoryNodeKind::Fact)
            .as_str()
            .to_string();
    }
    if span_kind == "memory.read" {
        return MemoryNodeKind::Episode.as_str().to_string();
    }
    tool_name
        .map(str::to_string)
        .unwrap_or_else(|| format!("{agent}:{kind}"))
}

fn beater_js_event_text(
    span_kind: &str,
    run_input: &str,
    request: &serde_json::Value,
    result: &serde_json::Value,
) -> String {
    if span_kind == "memory.write"
        && let Some(memory) =
            memory_write_text_from_value(request).or_else(|| memory_write_text_from_value(result))
    {
        return concise(memory, 2_400);
    }
    if span_kind == "memory.read"
        && let Some(read_text) = memory_read_text(request, result)
    {
        return concise(&read_text, 2_400);
    }
    concise(
        &format!(
            "run input: {} request: {} result: {}",
            run_input,
            json_text(request),
            json_text(result)
        ),
        2_400,
    )
}

fn is_terminal_beater_js_status(status: &str) -> bool {
    matches!(
        status.trim().to_ascii_lowercase().as_str(),
        "completed"
            | "complete"
            | "succeeded"
            | "success"
            | "failed"
            | "error"
            | "errored"
            | "cancelled"
            | "canceled"
    )
}

fn is_success_beater_js_status(status: &str) -> bool {
    matches!(
        status.trim().to_ascii_lowercase().as_str(),
        "completed" | "complete" | "succeeded" | "success"
    )
}

fn beater_js_span_kind<'a>(kind: &'a str, tool_name: Option<&str>) -> &'a str {
    match kind {
        "llm_call" => "llm.call",
        "tool_call" => match tool_name {
            Some("memory.read" | "memory_read") => "memory.read",
            Some("memory.write" | "memory_write") => "memory.write",
            _ => "tool.call",
        },
        other => other,
    }
}

fn should_skip_beater_js_step(step: &BeaterJsStep) -> bool {
    if !is_terminal_beater_js_status(&step.status) {
        return true;
    }
    beater_js_span_kind(&step.kind, step.tool_name.as_deref()) == "memory.write"
        && !is_success_beater_js_status(&step.status)
}

fn first_string_field<'a>(value: &'a serde_json::Value, fields: &[&str]) -> Option<&'a str> {
    fields.iter().find_map(|field| {
        value
            .get(*field)
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
    })
}

fn string_at_path<'a>(value: &'a serde_json::Value, path: &[&str]) -> Option<&'a str> {
    let mut cursor = value;
    for key in path {
        cursor = cursor.get(*key)?;
    }
    cursor
        .as_str()
        .map(str::trim)
        .filter(|text| !text.is_empty())
}

fn parse_memory_kind(value: Option<&str>) -> Option<MemoryNodeKind> {
    let value = value?.trim().to_ascii_lowercase();
    match value.as_str() {
        "semantic" => Some(MemoryNodeKind::Fact),
        "episodic" => Some(MemoryNodeKind::Episode),
        "runbook" | "workflow" => Some(MemoryNodeKind::Procedure),
        "failure" => Some(MemoryNodeKind::Gotcha),
        kind => kind.parse::<MemoryNodeKind>().ok(),
    }
}

fn memory_kind_from_value(value: &serde_json::Value) -> Option<MemoryNodeKind> {
    parse_memory_kind(string_at_path(value, &["attributes", "beater.memory.kind"]))
        .or_else(|| parse_memory_kind(string_at_path(value, &["attributes", "memory.kind"])))
        .or_else(|| parse_memory_kind(string_at_path(value, &["input", "kind"])))
        .or_else(|| parse_memory_kind(string_at_path(value, &["input", "memory_kind"])))
        .or_else(|| parse_memory_kind(string_at_path(value, &["payload", "kind"])))
        .or_else(|| parse_memory_kind(string_at_path(value, &["payload", "memory_kind"])))
        .or_else(|| parse_memory_kind(string_at_path(value, &["output", "kind"])))
        .or_else(|| parse_memory_kind(string_at_path(value, &["output", "memory_kind"])))
        .or_else(|| parse_memory_kind(string_at_path(value, &["memory_kind"])))
        .or_else(|| parse_memory_kind(string_at_path(value, &["node_kind"])))
        .or_else(|| parse_memory_kind(string_at_path(value, &["name"])))
        .or_else(|| parse_memory_kind(string_at_path(value, &["kind"])))
}

fn memory_write_text_from_value(value: &serde_json::Value) -> Option<&str> {
    first_string_field(value, &["memory", "text", "content", "value"])
        .or_else(|| first_string_field(&value["input"], &["memory", "text", "content", "value"]))
        .or_else(|| {
            value["input"]
                .as_str()
                .map(str::trim)
                .filter(|text| !text.is_empty())
        })
        .or_else(|| first_string_field(&value["payload"], &["memory", "text", "content", "value"]))
        .or_else(|| first_string_field(&value["output"], &["memory", "text", "content", "value"]))
        .or_else(|| {
            value["output"]
                .as_str()
                .map(str::trim)
                .filter(|text| !text.is_empty())
        })
}

fn memory_read_text(request: &serde_json::Value, result: &serde_json::Value) -> Option<String> {
    let question = string_value(request)
        .or_else(|| first_string_field(request, &["question", "query", "input", "text"]))
        .or_else(|| first_string_field(&request["input"], &["question", "query", "text"]));
    let answer = string_value(result)
        .or_else(|| first_string_field(result, &["answer", "summary", "output", "text"]))
        .or_else(|| first_string_field(&result["output"], &["answer", "summary", "text"]));
    match (question, answer) {
        (Some(question), Some(answer)) => {
            Some(format!("memory read question: {question} answer: {answer}"))
        }
        (Some(question), None) => Some(format!("memory read question: {question}")),
        (None, Some(answer)) => Some(format!("memory read answer: {answer}")),
        (None, None) => None,
    }
}

fn string_value(value: &serde_json::Value) -> Option<&str> {
    value
        .as_str()
        .map(str::trim)
        .filter(|text| !text.is_empty())
}

fn canonical_json_event(
    value: &serde_json::Value,
    tenant_id: Option<&str>,
    project_id: Option<&str>,
    environment_id: Option<&str>,
) -> LedgerEvent {
    let tenant = tenant_id
        .map(str::to_string)
        .or_else(|| value["tenant_id"].as_str().map(str::to_string))
        .unwrap_or_else(|| "local".to_string());
    let project = project_id
        .map(str::to_string)
        .or_else(|| value["project_id"].as_str().map(str::to_string))
        .unwrap_or_else(|| "default".to_string());
    let env = environment_id
        .map(str::to_string)
        .or_else(|| value["environment_id"].as_str().map(str::to_string));
    let trace_id = value["trace_id"]
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| stable_id("trace", &[&json_text(value)]));
    let span_id = value["span_id"]
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| stable_id("span", &[&trace_id, &json_text(value)]));
    let seq = value["seq"].as_u64().unwrap_or(1);
    let span_kind = canonical_span_kind(value);
    let name = canonical_event_name(value, span_kind);
    let text = canonical_event_text(value, span_kind);
    let now = now_unix_ms();
    LedgerEvent {
        id: None,
        source: "beater-agents.canonical-jsonl".to_string(),
        tenant_id: tenant,
        project_id: project,
        environment_id: env,
        trace_id,
        span_id,
        seq,
        span_kind: span_kind.to_string(),
        name,
        status: value["status"].as_str().unwrap_or("ok").to_string(),
        text,
        payload: value.clone(),
        observed_at_unix_ms: canonical_observed_at_unix_ms(value, now),
        ingested_at_unix_ms: now,
        projected_at_unix_ms: None,
    }
}

fn canonical_span_kind(value: &serde_json::Value) -> &str {
    value["span_kind"]
        .as_str()
        .or_else(|| value["attributes"]["openinference.span.kind"].as_str())
        .or_else(|| value["attributes"]["beater.span.kind"].as_str())
        .or_else(|| value["kind"].as_str())
        .unwrap_or("agent.step")
}

fn canonical_event_name(value: &serde_json::Value, span_kind: &str) -> String {
    if span_kind == "memory.write" {
        return memory_kind_from_value(value)
            .unwrap_or(MemoryNodeKind::Fact)
            .as_str()
            .to_string();
    }
    if span_kind == "memory.read" {
        return MemoryNodeKind::Episode.as_str().to_string();
    }
    value["name"].as_str().unwrap_or(span_kind).to_string()
}

fn canonical_event_text(value: &serde_json::Value, span_kind: &str) -> String {
    if span_kind == "memory.write"
        && let Some(memory) = memory_write_text_from_value(value)
    {
        return concise(memory, 2_400);
    }
    if span_kind == "memory.read"
        && let Some(read_text) = memory_read_text(&value["input"], &value["output"])
    {
        return concise(&read_text, 2_400);
    }
    concise(&json_text(value), 2_400)
}

fn canonical_observed_at_unix_ms(value: &serde_json::Value, now: i64) -> i64 {
    value["start_time_unix_ms"]
        .as_i64()
        .or_else(|| value["start_time_ms"].as_i64())
        .or_else(|| value["timestamp_unix_ms"].as_i64())
        .or_else(|| timestamp_value_to_unix_ms(&value["start_time"]))
        .or_else(|| timestamp_value_to_unix_ms(&value["startTime"]))
        .or_else(|| timestamp_value_to_unix_ms(&value["timestamp"]))
        .unwrap_or(now)
}

fn timestamp_value_to_unix_ms(value: &serde_json::Value) -> Option<i64> {
    if let Some(value) = value.as_i64() {
        return Some(unix_number_to_ms(value as f64));
    }
    if let Some(value) = value.as_f64() {
        if !value.is_finite() || value < 0.0 {
            return None;
        }
        return Some(unix_number_to_ms(value));
    }
    value
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .and_then(parse_timestamp_string_to_unix_ms)
}

fn parse_timestamp_string_to_unix_ms(value: &str) -> Option<i64> {
    DateTime::parse_from_rfc3339(value)
        .map(|time| time.timestamp_millis())
        .ok()
        .or_else(|| {
            value.parse::<f64>().ok().and_then(|number| {
                if number.is_finite() && number >= 0.0 {
                    Some(unix_number_to_ms(number))
                } else {
                    None
                }
            })
        })
}

fn unix_number_to_ms(value: f64) -> i64 {
    if value >= 1_000_000_000_000.0 {
        value.round() as i64
    } else if value >= 1_000_000_000.0 {
        (value * 1_000.0).round() as i64
    } else {
        value.round() as i64
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, time::SystemTime};

    use rusqlite::params;

    use super::*;

    #[test]
    fn imports_beater_js_journal_steps() -> MemoryResult<()> {
        let dir = std::env::temp_dir().join(format!(
            "beater-memory-test-{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::create_dir_all(&dir)?;
        let journal_path = dir.join("journal.db");
        let conn = Connection::open(&journal_path)?;
        conn.execute_batch(
            "
            CREATE TABLE runs(
               id TEXT PRIMARY KEY, agent TEXT NOT NULL, status TEXT NOT NULL,
               input TEXT NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL);
            CREATE TABLE steps(
               run_id TEXT NOT NULL, seq INTEGER NOT NULL,
               kind TEXT NOT NULL, status TEXT NOT NULL,
               request TEXT NOT NULL, result TEXT,
               tool_name TEXT, tool_use_id TEXT,
               attempt INTEGER NOT NULL DEFAULT 1,
               started_at INTEGER NOT NULL, finished_at INTEGER,
               PRIMARY KEY(run_id, seq));
            ",
        )?;
        conn.execute(
            "INSERT INTO runs VALUES (?1, ?2, ?3, ?4, ?5, ?5)",
            params!["run-1", "support", "completed", "remember checkout", 1_i64],
        )?;
        conn.execute(
            "INSERT INTO steps VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 1, ?9, ?10)",
            params![
                "run-1",
                1_i64,
                "tool_call",
                "completed",
                serde_json::json!({"memory": "Checkout token zeta requires DATABASE_URL"})
                    .to_string(),
                serde_json::json!({"ok": true}).to_string(),
                "memory.write",
                "tool-use-1",
                2_i64,
                3_i64,
            ],
        )?;
        conn.execute(
            "INSERT INTO steps VALUES (?1, ?2, ?3, ?4, ?5, NULL, ?6, ?7, 1, ?8, NULL)",
            params![
                "run-1",
                2_i64,
                "tool_call",
                "started",
                serde_json::json!({"memory": "Started rows must not import"}).to_string(),
                "memory.write",
                "tool-use-started",
                4_i64,
            ],
        )?;
        conn.execute(
            "INSERT INTO steps VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 1, ?9, ?10)",
            params![
                "run-1",
                3_i64,
                "tool_call",
                "failed",
                serde_json::json!({"memory": "Failed writes must not become facts"}).to_string(),
                serde_json::json!({"error": "write failed"}).to_string(),
                "memory.write",
                "tool-use-failed",
                5_i64,
                6_i64,
            ],
        )?;
        drop(conn);

        let store = SqliteMemoryStore::in_memory()?;
        let report =
            BeaterJsJournal::new(&journal_path).import_into(&store, "tenant", "project", None)?;

        assert_eq!(report.rows_seen, 3);
        assert_eq!(report.rows_skipped, 2);
        assert_eq!(report.events_inserted, 1);
        let events = store.pending_events(10)?;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].span_kind, "memory.write");
        assert_eq!(events[0].name, "fact");
        assert_eq!(events[0].text, "Checkout token zeta requires DATABASE_URL");
        fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    fn beater_js_import_is_atomic_on_invalid_row() -> MemoryResult<()> {
        let dir = std::env::temp_dir().join(format!(
            "beater-memory-test-invalid-{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::create_dir_all(&dir)?;
        let journal_path = dir.join("journal.db");
        let conn = Connection::open(&journal_path)?;
        conn.execute_batch(
            "
            CREATE TABLE runs(
               id TEXT PRIMARY KEY, agent TEXT NOT NULL, status TEXT NOT NULL,
               input TEXT NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL);
            CREATE TABLE steps(
               run_id TEXT NOT NULL, seq INTEGER NOT NULL,
               kind TEXT NOT NULL, status TEXT NOT NULL,
               request TEXT NOT NULL, result TEXT,
               tool_name TEXT, tool_use_id TEXT,
               attempt INTEGER NOT NULL DEFAULT 1,
               started_at INTEGER NOT NULL, finished_at INTEGER,
               PRIMARY KEY(run_id, seq));
            ",
        )?;
        conn.execute(
            "INSERT INTO runs VALUES (?1, ?2, ?3, ?4, ?5, ?5)",
            params!["run-1", "support", "completed", "remember checkout", 1_i64],
        )?;
        conn.execute(
            "INSERT INTO steps VALUES (?1, ?2, ?3, ?4, ?5, ?6, NULL, NULL, 1, ?7, ?8)",
            params![
                "run-1",
                1_i64,
                "llm_call",
                "completed",
                serde_json::json!({"messages": [{"role": "user", "content": "checkout fails"}]})
                    .to_string(),
                serde_json::json!({"content": [{"type": "text", "text": "Set DATABASE_URL"}]})
                    .to_string(),
                2_i64,
                3_i64,
            ],
        )?;
        conn.execute(
            "INSERT INTO steps VALUES (?1, ?2, ?3, ?4, ?5, NULL, NULL, NULL, 1, ?6, NULL)",
            params!["run-1", 2_i64, "llm_call", "failed", "{not-json", 4_i64,],
        )?;
        drop(conn);

        let store = SqliteMemoryStore::in_memory()?;
        let err = BeaterJsJournal::new(&journal_path)
            .import_into(&store, "tenant", "project", None)
            .unwrap_err();

        assert!(err.to_string().contains("run_id=run-1 seq=2"));
        assert_eq!(store.stats()?.ledger_events, 0);
        fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    fn canonical_jsonl_uses_beater_span_kind_attrs() -> MemoryResult<()> {
        let dir = std::env::temp_dir().join(format!(
            "beater-memory-jsonl-test-{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::create_dir_all(&dir)?;
        let path = dir.join("spans.jsonl");
        let write = serde_json::json!({
            "tenant_id": "tenant",
            "project_id": "project",
            "trace_id": "trace",
            "span_id": "span-write",
            "seq": 7,
            "attributes": {
                "beater.span.kind": "memory.write",
                "beater.memory.kind": "Workflow"
            },
            "name": "memory.write",
            "start_time": "2026-07-01T00:00:00Z",
            "input": {
                "memory": "Checkout uses DATABASE_URL",
                "kind": "procedure"
            }
        });
        let read = serde_json::json!({
            "tenant_id": "tenant",
            "project_id": "project",
            "trace_id": "trace",
            "span_id": "span-read",
            "seq": 8,
            "span_kind": "memory.read",
            "name": "memory.read",
            "start_time": "2026-07-01T00:00:01Z",
            "input": "checkout env",
            "output": "Checkout uses DATABASE_URL"
        });
        fs::write(&path, format!("{write}\n{read}\n"))?;
        let store = SqliteMemoryStore::in_memory()?;
        let report = import_canonical_jsonl(&path, &store, None, None, None)?;

        assert_eq!(report.rows_seen, 2);
        assert_eq!(report.events_inserted, 2);
        let events = store.pending_events(10)?;
        let write_event = events
            .iter()
            .find(|event| event.span_id == "span-write")
            .expect("write event");
        assert_eq!(write_event.span_kind, "memory.write");
        assert_eq!(write_event.name, "procedure");
        assert_eq!(write_event.text, "Checkout uses DATABASE_URL");
        assert_eq!(write_event.observed_at_unix_ms, 1_782_864_000_000);
        let read_event = events
            .iter()
            .find(|event| event.span_id == "span-read")
            .expect("read event");
        assert_eq!(read_event.span_kind, "memory.read");
        assert_eq!(read_event.name, "episode");
        assert!(
            read_event
                .text
                .contains("memory read question: checkout env")
        );
        fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    fn canonical_jsonl_import_is_atomic_on_invalid_line() -> MemoryResult<()> {
        let dir = std::env::temp_dir().join(format!(
            "beater-memory-jsonl-invalid-test-{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::create_dir_all(&dir)?;
        let path = dir.join("spans.jsonl");
        let valid = serde_json::json!({
            "tenant_id": "tenant",
            "project_id": "project",
            "trace_id": "trace",
            "span_id": "span",
            "seq": 1,
            "attributes": {"beater.span.kind": "memory.write"},
            "name": "fact",
            "output": "Checkout uses DATABASE_URL"
        });
        fs::write(&path, format!("{valid}\n{{not-json\n"))?;
        let store = SqliteMemoryStore::in_memory()?;

        let err = import_canonical_jsonl(&path, &store, None, None, None).unwrap_err();

        assert!(err.to_string().contains("line 2"));
        assert_eq!(store.stats()?.ledger_events, 0);
        fs::remove_dir_all(dir)?;
        Ok(())
    }
}
