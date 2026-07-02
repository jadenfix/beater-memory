use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use rusqlite::{Connection, OpenFlags};

use crate::{
    error::{MemoryError, MemoryResult},
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
        let span_kind = match self.kind.as_str() {
            "llm_call" => "llm.call",
            "tool_call" => match self.tool_name.as_deref() {
                Some("memory.read" | "memory_read") => "memory.read",
                Some("memory.write" | "memory_write") => "memory.write",
                _ => "tool.call",
            },
            other => other,
        };
        let name = self
            .tool_name
            .clone()
            .unwrap_or_else(|| format!("{}:{}", self.agent, self.kind));
        let mut text = format!(
            "run input: {} request: {} result: {}",
            self.run_input,
            json_text(&request),
            json_text(&result)
        );
        text = concise(&text, 2_400);
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
    let span_kind = value["kind"]
        .as_str()
        .or_else(|| value["span_kind"].as_str())
        .or_else(|| value["attributes"]["openinference.span.kind"].as_str())
        .or_else(|| value["attributes"]["beater.span.kind"].as_str())
        .unwrap_or("agent.step");
    let name = value["name"].as_str().unwrap_or(span_kind);
    let text = concise(&json_text(value), 2_400);
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
        name: name.to_string(),
        status: value["status"].as_str().unwrap_or("ok").to_string(),
        text,
        payload: value.clone(),
        observed_at_unix_ms: value["start_time_unix_ms"].as_i64().unwrap_or(now),
        ingested_at_unix_ms: now,
        projected_at_unix_ms: None,
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
        drop(conn);

        let store = SqliteMemoryStore::in_memory()?;
        let report =
            BeaterJsJournal::new(&journal_path).import_into(&store, "tenant", "project", None)?;

        assert_eq!(report.rows_seen, 1);
        assert_eq!(report.events_inserted, 1);
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
        fs::write(
            &path,
            serde_json::json!({
                "tenant_id": "tenant",
                "project_id": "project",
                "trace_id": "trace",
                "span_id": "span",
                "seq": 7,
                "attributes": {"beater.span.kind": "memory.write"},
                "name": "fact",
                "output": "Checkout uses DATABASE_URL"
            })
            .to_string(),
        )?;
        let store = SqliteMemoryStore::in_memory()?;
        let report = import_canonical_jsonl(&path, &store, None, None, None)?;

        assert_eq!(report.events_inserted, 1);
        let events = store.pending_events(10)?;
        assert_eq!(events[0].span_kind, "memory.write");
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
