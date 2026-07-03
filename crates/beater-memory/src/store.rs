use std::{collections::BTreeSet, path::Path, str::FromStr, time::Duration};

use rusqlite::types::Value;
use rusqlite::{Connection, MAIN_DB, OpenFlags, OptionalExtension, params, params_from_iter};
use serde::{Deserialize, Serialize};

use crate::{
    error::{MemoryError, MemoryResult},
    model::{CitedSpan, MemoryEdgeKind, MemoryNodeKind, estimate_tokens},
    text::{canonical_key, now_unix_ms, stable_id, terms},
};

const SCHEMA_VERSION: u32 = 3;
const SQLITE_APPLICATION_ID: i64 = 0x424D_454D;

/// An append-only observation imported from a journal, span store, or direct write.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LedgerEvent {
    pub id: Option<i64>,
    pub source: String,
    pub tenant_id: String,
    pub project_id: String,
    pub environment_id: Option<String>,
    pub trace_id: String,
    pub span_id: String,
    pub seq: u64,
    pub span_kind: String,
    pub name: String,
    pub status: String,
    pub text: String,
    pub payload: serde_json::Value,
    pub observed_at_unix_ms: i64,
    pub ingested_at_unix_ms: i64,
    pub projected_at_unix_ms: Option<i64>,
}

impl LedgerEvent {
    #[must_use]
    pub fn direct_memory_write(
        tenant_id: impl Into<String>,
        project_id: impl Into<String>,
        kind: MemoryNodeKind,
        text: impl Into<String>,
    ) -> Self {
        let now = now_unix_ms();
        let text = text.into();
        let trace_id = stable_id("direct_trace", &[&text, &now.to_string()]);
        Self {
            id: None,
            source: "beater-memory.direct".to_string(),
            tenant_id: tenant_id.into(),
            project_id: project_id.into(),
            environment_id: None,
            trace_id: trace_id.clone(),
            span_id: stable_id("direct_span", &[&trace_id]),
            seq: 1,
            span_kind: "memory.write".to_string(),
            name: kind.as_str().to_string(),
            status: "ok".to_string(),
            text,
            payload: serde_json::json!({ "kind": kind.as_str() }),
            observed_at_unix_ms: now,
            ingested_at_unix_ms: now,
            projected_at_unix_ms: None,
        }
    }

    #[must_use]
    pub fn direct_memory_write_with_idempotency_key(
        tenant_id: impl Into<String>,
        project_id: impl Into<String>,
        kind: MemoryNodeKind,
        text: impl Into<String>,
        idempotency_key: &str,
    ) -> Self {
        Self::direct_memory_write(tenant_id, project_id, kind, text)
            .with_idempotency_key(idempotency_key)
    }

    #[must_use]
    pub fn with_idempotency_key(mut self, idempotency_key: &str) -> Self {
        self.apply_idempotency_key(idempotency_key);
        self
    }

    pub fn apply_idempotency_key(&mut self, idempotency_key: &str) {
        let idempotency_key = idempotency_key.trim();
        let environment_id = self.environment_id.as_deref().unwrap_or("");
        self.trace_id = stable_id(
            "direct_trace_idempotent",
            &[
                &self.tenant_id,
                &self.project_id,
                environment_id,
                &self.name,
                idempotency_key,
            ],
        );
        self.span_id = stable_id("direct_span", &[&self.trace_id]);
        if let Some(payload) = self.payload.as_object_mut() {
            payload.insert(
                "idempotency_key_hash".to_string(),
                serde_json::json!(stable_id("idempotency_key", &[idempotency_key])),
            );
        }
    }

    pub fn validate(&self) -> MemoryResult<()> {
        validate_required_identifier("source", &self.source)?;
        validate_required_identifier("tenant_id", &self.tenant_id)?;
        validate_required_identifier("project_id", &self.project_id)?;
        if let Some(environment_id) = self.environment_id.as_deref() {
            validate_required_identifier("environment_id", environment_id)?;
        }
        validate_required_identifier("trace_id", &self.trace_id)?;
        validate_required_identifier("span_id", &self.span_id)?;
        validate_required_identifier("span_kind", &self.span_kind)?;
        validate_required_identifier("name", &self.name)?;
        validate_required_identifier("status", &self.status)?;
        validate_required_text("text", &self.text)?;
        if self.seq == 0 {
            return Err(MemoryError::invalid("seq must be greater than 0"));
        }
        if self.observed_at_unix_ms < 0 {
            return Err(MemoryError::invalid(
                "observed_at_unix_ms must be non-negative",
            ));
        }
        if self.ingested_at_unix_ms < 0 {
            return Err(MemoryError::invalid(
                "ingested_at_unix_ms must be non-negative",
            ));
        }
        if self
            .projected_at_unix_ms
            .is_some_and(|projected_at| projected_at < 0)
        {
            return Err(MemoryError::invalid(
                "projected_at_unix_ms must be non-negative",
            ));
        }
        Ok(())
    }

    #[must_use]
    pub fn cited_span(&self) -> CitedSpan {
        CitedSpan {
            tenant_id: self.tenant_id.clone(),
            project_id: self.project_id.clone(),
            trace_id: self.trace_id.clone(),
            span_id: self.span_id.clone(),
            seq: self.seq,
        }
    }
}

/// A projected memory node.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryNode {
    pub id: String,
    pub tenant_id: String,
    pub project_id: String,
    pub environment_id: Option<String>,
    pub kind: MemoryNodeKind,
    pub text: String,
    pub canonical_key: String,
    pub created_at_unix_ms: i64,
    pub updated_at_unix_ms: i64,
    pub valid_from_unix_ms: i64,
    pub valid_to_unix_ms: Option<i64>,
    pub valid_to_event_id: Option<i64>,
    pub confidence: f32,
    pub token_estimate: u32,
    pub observation_count: u32,
}

impl MemoryNode {
    #[must_use]
    pub fn is_active_at(&self, as_of_unix_ms: Option<i64>) -> bool {
        let as_of = as_of_unix_ms.unwrap_or_else(now_unix_ms);
        self.valid_from_unix_ms <= as_of
            && self
                .valid_to_unix_ms
                .map(|valid_to| valid_to > as_of)
                .unwrap_or(true)
    }
}

/// A typed relationship between projected nodes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryEdge {
    pub id: i64,
    pub tenant_id: String,
    pub project_id: String,
    pub environment_id: Option<String>,
    pub from_node_id: String,
    pub to_node_id: String,
    pub kind: MemoryEdgeKind,
    pub weight: f32,
    pub created_at_unix_ms: i64,
}

#[derive(Clone, Copy)]
pub(crate) struct StoreScope<'a> {
    pub tenant_id: &'a str,
    pub project_id: &'a str,
    pub environment_id: Option<&'a str>,
}

impl<'a> StoreScope<'a> {
    pub fn new(tenant_id: &'a str, project_id: &'a str, environment_id: Option<&'a str>) -> Self {
        Self {
            tenant_id,
            project_id,
            environment_id,
        }
    }
}

/// Operational counts for the memory database.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct StoreStats {
    pub ledger_events: i64,
    pub pending_events: i64,
    pub nodes: i64,
    pub active_nodes: i64,
    #[serde(default)]
    pub total_node_tokens: i64,
    #[serde(default)]
    pub active_node_tokens: i64,
    #[serde(default)]
    pub active_episode_nodes: i64,
    #[serde(default)]
    pub active_fact_nodes: i64,
    #[serde(default)]
    pub active_entity_cue_nodes: i64,
    #[serde(default)]
    pub active_tag_nodes: i64,
    #[serde(default)]
    pub active_procedure_nodes: i64,
    #[serde(default)]
    pub active_state_nodes: i64,
    #[serde(default)]
    pub active_gotcha_nodes: i64,
    #[serde(default)]
    pub active_anti_memory_nodes: i64,
    #[serde(default)]
    pub active_topic_nodes: i64,
    pub edges: i64,
    pub audit_events: i64,
}

/// Health snapshot suitable for CLI output and service health endpoints.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreHealth {
    pub application_id: i64,
    pub expected_application_id: i64,
    pub schema_version: u32,
    pub expected_schema_version: u32,
    pub integrity_ok: bool,
    pub integrity_messages: Vec<String>,
    pub foreign_key_violations: i64,
    pub graph_integrity_ok: bool,
    pub graph_integrity: GraphIntegrityReport,
    pub stats: StoreStats,
}

/// Projection graph integrity counts not covered by SQLite foreign keys.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphIntegrityReport {
    pub orphan_edges_from: i64,
    pub orphan_edges_to: i64,
    pub orphan_node_spans: i64,
    pub orphan_cue_index_entries: i64,
}

impl GraphIntegrityReport {
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.orphan_edges_from == 0
            && self.orphan_edges_to == 0
            && self.orphan_node_spans == 0
            && self.orphan_cue_index_entries == 0
    }
}

/// Rows removed by graph integrity repair.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphRepairReport {
    pub memory_edges_removed: i64,
    pub node_spans_removed: i64,
    pub cue_index_entries_removed: i64,
}

/// Rows removed by audit retention maintenance.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditPruneReport {
    pub audit_events_removed: i64,
}

/// Options for a local maintenance pass.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaintenanceOptions {
    pub vacuum: bool,
    pub repair_orphans: bool,
    pub prune_audit_before_unix_ms: Option<i64>,
    pub retain_latest_audit_events: Option<usize>,
}

impl MaintenanceOptions {
    pub fn validate(&self) -> MemoryResult<()> {
        if self
            .prune_audit_before_unix_ms
            .is_some_and(|cutoff| cutoff < 0)
        {
            return Err(MemoryError::invalid(
                "prune_audit_before_unix_ms must be non-negative",
            ));
        }
        if let Some(retain_latest) = self.retain_latest_audit_events {
            sqlite_limit("retain_latest audit limit", retain_latest)?;
        }
        Ok(())
    }
}

/// Result of a local maintenance pass.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaintenanceReport {
    pub optimized: bool,
    pub wal_checkpoint_busy: i64,
    pub wal_checkpoint_log_frames: i64,
    pub wal_checkpoint_checkpointed_frames: i64,
    pub vacuumed: bool,
    pub repaired_orphans: bool,
    pub pruned_audit_events: bool,
    pub graph_integrity_before: GraphIntegrityReport,
    pub graph_integrity_after: GraphIntegrityReport,
    pub graph_repair: GraphRepairReport,
    pub audit_prune: AuditPruneReport,
}

/// Result of writing a SQLite backup file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackupReport {
    pub path: String,
    pub bytes: u64,
    pub schema_version: u32,
    pub integrity_ok: bool,
}

/// Result of restoring the active database from a SQLite backup file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RestoreReport {
    pub path: String,
    pub schema_version: u32,
    pub integrity_ok: bool,
    pub stats: StoreStats,
}

/// Rows removed or reset when rebuilding derived projections from the ledger.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectionResetReport {
    pub ledger_events_reset: i64,
    pub nodes_removed: i64,
    pub edges_removed: i64,
    pub node_spans_removed: i64,
    pub cue_index_entries_removed: i64,
}

/// A durable service audit event.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AuditEvent {
    pub id: i64,
    pub occurred_at_unix_ms: i64,
    pub actor: String,
    pub action: String,
    pub outcome: String,
    pub route: Option<String>,
    pub status_code: Option<u16>,
    pub detail: serde_json::Value,
}

/// Input for writing a durable service audit event.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AuditRecord {
    pub actor: String,
    pub action: String,
    pub outcome: String,
    pub route: Option<String>,
    pub status_code: Option<u16>,
    pub detail: serde_json::Value,
}

impl AuditRecord {
    pub fn validate(&self) -> MemoryResult<()> {
        validate_required_identifier("audit actor", &self.actor)?;
        validate_required_identifier("audit action", &self.action)?;
        validate_required_identifier("audit outcome", &self.outcome)?;
        if let Some(route) = self.route.as_deref() {
            validate_required_identifier("audit route", route)?;
        }
        if self
            .status_code
            .is_some_and(|status_code| !(100..=599).contains(&status_code))
        {
            return Err(MemoryError::invalid(
                "audit status_code must be a valid HTTP status code",
            ));
        }
        Ok(())
    }
}

/// SQLite-backed local memory store.
pub struct SqliteMemoryStore {
    conn: Connection,
}

impl SqliteMemoryStore {
    pub fn open(path: impl AsRef<Path>) -> MemoryResult<Self> {
        if let Some(parent) = path.as_ref().parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        configure_connection(&conn)?;
        let store = Self { conn };
        store.migrate()?;
        configure_persistent_database(&store.conn)?;
        Ok(store)
    }

    pub fn in_memory() -> MemoryResult<Self> {
        let store = Self {
            conn: Connection::open_in_memory()?,
        };
        configure_connection(&store.conn)?;
        store.migrate()?;
        configure_persistent_database(&store.conn)?;
        Ok(store)
    }

    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    fn migrate(&self) -> MemoryResult<()> {
        self.validate_database_identity()?;
        let user_version: u32 = self
            .conn
            .query_row("PRAGMA user_version", [], |row| row.get(0))?;
        if user_version > SCHEMA_VERSION {
            return Err(MemoryError::invalid(format!(
                "database schema version {user_version} is newer than supported version {SCHEMA_VERSION}"
            )));
        }
        let needs_v3_upgrade = user_version < 3;
        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS ledger_events(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                environment_id TEXT,
                trace_id TEXT NOT NULL,
                span_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                span_kind TEXT NOT NULL,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                text TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                observed_at_unix_ms INTEGER NOT NULL,
                ingested_at_unix_ms INTEGER NOT NULL,
                projected_at_unix_ms INTEGER,
                UNIQUE(tenant_id, project_id, trace_id, span_id, seq)
            );

            CREATE INDEX IF NOT EXISTS idx_ledger_pending
                ON ledger_events(projected_at_unix_ms, ingested_at_unix_ms);
            CREATE INDEX IF NOT EXISTS idx_ledger_scope
                ON ledger_events(tenant_id, project_id, environment_id, observed_at_unix_ms);

            CREATE TABLE IF NOT EXISTS memory_nodes(
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                environment_id TEXT,
                kind TEXT NOT NULL,
                text TEXT NOT NULL,
                canonical_key TEXT NOT NULL,
                created_at_unix_ms INTEGER NOT NULL,
                updated_at_unix_ms INTEGER NOT NULL,
                valid_from_unix_ms INTEGER NOT NULL,
                valid_to_unix_ms INTEGER,
                valid_to_event_id INTEGER,
                confidence REAL NOT NULL,
                token_estimate INTEGER NOT NULL,
                observation_count INTEGER NOT NULL,
                UNIQUE(tenant_id, project_id, environment_id, kind, canonical_key)
            );

            CREATE INDEX IF NOT EXISTS idx_nodes_scope
                ON memory_nodes(tenant_id, project_id, environment_id, kind, valid_to_unix_ms);
            CREATE INDEX IF NOT EXISTS idx_nodes_updated
                ON memory_nodes(tenant_id, project_id, updated_at_unix_ms);

            CREATE TABLE IF NOT EXISTS memory_edges(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                environment_id TEXT,
                from_node_id TEXT NOT NULL,
                to_node_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                weight REAL NOT NULL,
                created_at_unix_ms INTEGER NOT NULL,
                UNIQUE(from_node_id, to_node_id, kind)
            );

            CREATE INDEX IF NOT EXISTS idx_edges_from ON memory_edges(from_node_id);
            CREATE INDEX IF NOT EXISTS idx_edges_to ON memory_edges(to_node_id);
            CREATE INDEX IF NOT EXISTS idx_edges_scope
                ON memory_edges(tenant_id, project_id, environment_id, kind);

            CREATE TABLE IF NOT EXISTS node_spans(
                node_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                span_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                PRIMARY KEY(node_id, tenant_id, project_id, trace_id, span_id, seq)
            );

            CREATE TABLE IF NOT EXISTS cue_index(
                term TEXT NOT NULL,
                node_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                environment_id TEXT,
                weight REAL NOT NULL,
                PRIMARY KEY(term, node_id)
            );

            CREATE INDEX IF NOT EXISTS idx_cue_scope
                ON cue_index(tenant_id, project_id, environment_id, term);

            CREATE TABLE IF NOT EXISTS audit_events(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_at_unix_ms INTEGER NOT NULL,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                outcome TEXT NOT NULL,
                route TEXT,
                status_code INTEGER,
                detail_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_audit_events_time
                ON audit_events(occurred_at_unix_ms, id);
            CREATE INDEX IF NOT EXISTS idx_audit_events_action
                ON audit_events(action, outcome, occurred_at_unix_ms);

            ",
        )?;
        if !self.column_exists("memory_nodes", "valid_to_event_id")? {
            self.conn.execute(
                "ALTER TABLE memory_nodes ADD COLUMN valid_to_event_id INTEGER",
                [],
            )?;
        }
        self.backfill_valid_to_event_ids()?;
        if needs_v3_upgrade && self.projection_requires_v3_rebuild()? {
            let _ = self.reset_projection_in_transaction()?;
        }
        self.conn
            .pragma_update(None, "user_version", SCHEMA_VERSION)?;
        Ok(())
    }

    fn column_exists(&self, table: &str, column: &str) -> MemoryResult<bool> {
        let mut stmt = self.conn.prepare(&format!("PRAGMA table_info({table})"))?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
        for row in rows {
            if row? == column {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn backfill_valid_to_event_ids(&self) -> MemoryResult<()> {
        self.conn.execute(
            "
            UPDATE memory_nodes AS n
            SET valid_to_event_id = COALESCE(
                (
                    SELECT MIN(close_event.id)
                    FROM memory_edges edge
                    JOIN node_spans from_span
                      ON from_span.node_id = edge.from_node_id
                    JOIN ledger_events close_event
                      ON close_event.tenant_id = from_span.tenant_id
                     AND close_event.project_id = from_span.project_id
                     AND close_event.trace_id = from_span.trace_id
                     AND close_event.span_id = from_span.span_id
                     AND close_event.seq = from_span.seq
                    WHERE edge.to_node_id = n.id
                      AND edge.kind IN ('contradicts', 'supersedes')
                      AND close_event.observed_at_unix_ms = n.valid_to_unix_ms
                ),
                (
                    SELECT MIN(successor_event.id)
                    FROM memory_nodes successor
                    JOIN node_spans successor_span
                      ON successor_span.node_id = successor.id
                    JOIN ledger_events successor_event
                      ON successor_event.tenant_id = successor_span.tenant_id
                     AND successor_event.project_id = successor_span.project_id
                     AND successor_event.trace_id = successor_span.trace_id
                     AND successor_event.span_id = successor_span.span_id
                     AND successor_event.seq = successor_span.seq
                    WHERE successor.id != n.id
                      AND successor.tenant_id = n.tenant_id
                      AND successor.project_id = n.project_id
                      AND COALESCE(successor.environment_id, '') = COALESCE(n.environment_id, '')
                      AND successor.kind = n.kind
                      AND successor.valid_from_unix_ms = n.valid_to_unix_ms
                      AND (
                          CASE
                              WHEN instr(successor.canonical_key, '|rev:') > 0
                              THEN substr(successor.canonical_key, 1, instr(successor.canonical_key, '|rev:') - 1)
                              ELSE successor.canonical_key
                          END
                      ) = (
                          CASE
                              WHEN instr(n.canonical_key, '|rev:') > 0
                              THEN substr(n.canonical_key, 1, instr(n.canonical_key, '|rev:') - 1)
                              ELSE n.canonical_key
                          END
                      )
                      AND successor_event.observed_at_unix_ms = n.valid_to_unix_ms
                )
            )
            WHERE n.valid_to_unix_ms IS NOT NULL
              AND n.valid_to_event_id IS NULL
            ",
            [],
        )?;
        Ok(())
    }

    fn projection_requires_v3_rebuild(&self) -> MemoryResult<bool> {
        let needs_rebuild: i64 = self.conn.query_row(
            "
            SELECT EXISTS(
                SELECT 1
                FROM memory_nodes n
                WHERE n.valid_to_unix_ms IS NOT NULL
                  AND (
                      n.valid_to_event_id IS NULL
                      OR EXISTS (
                          SELECT 1
                          FROM memory_edges edge
                          JOIN node_spans from_span
                            ON from_span.node_id = edge.from_node_id
                          JOIN ledger_events close_event
                            ON close_event.tenant_id = from_span.tenant_id
                           AND close_event.project_id = from_span.project_id
                           AND close_event.trace_id = from_span.trace_id
                           AND close_event.span_id = from_span.span_id
                           AND close_event.seq = from_span.seq
                          WHERE edge.to_node_id = n.id
                            AND edge.kind IN ('contradicts', 'supersedes')
                            AND close_event.observed_at_unix_ms = n.valid_to_unix_ms
                          GROUP BY edge.from_node_id
                          HAVING COUNT(DISTINCT close_event.id) > 1
                      )
                  )
            )
            ",
            [],
            |row| row.get(0),
        )?;
        Ok(needs_rebuild != 0)
    }

    fn validate_database_identity(&self) -> MemoryResult<()> {
        validate_database_identity(&self.conn, true)
    }

    pub(crate) fn with_immediate_transaction<T>(
        &self,
        f: impl FnOnce(&Self) -> MemoryResult<T>,
    ) -> MemoryResult<T> {
        self.conn.execute_batch("BEGIN IMMEDIATE")?;
        match f(self) {
            Ok(value) => match self.conn.execute_batch("COMMIT") {
                Ok(()) => Ok(value),
                Err(err) => {
                    let _ = self.conn.execute_batch("ROLLBACK");
                    Err(err.into())
                }
            },
            Err(err) => {
                let _ = self.conn.execute_batch("ROLLBACK");
                Err(err)
            }
        }
    }

    pub fn append_event(&self, event: &LedgerEvent) -> MemoryResult<bool> {
        event.validate()?;
        let inserted = self.conn.execute(
            "
            INSERT OR IGNORE INTO ledger_events(
                source, tenant_id, project_id, environment_id, trace_id, span_id, seq,
                span_kind, name, status, text, payload_json, observed_at_unix_ms,
                ingested_at_unix_ms, projected_at_unix_ms
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)
            ",
            params![
                event.source,
                event.tenant_id,
                event.project_id,
                event.environment_id,
                event.trace_id,
                event.span_id,
                event.seq,
                event.span_kind,
                event.name,
                event.status,
                event.text,
                serde_json::to_string(&event.payload)?,
                event.observed_at_unix_ms,
                event.ingested_at_unix_ms,
                event.projected_at_unix_ms,
            ],
        )?;
        Ok(inserted == 1)
    }

    pub fn pending_events(&self, limit: usize) -> MemoryResult<Vec<LedgerEvent>> {
        let limit = sqlite_limit("pending_events limit", limit)?;
        let mut stmt = self.conn.prepare(
            "
            SELECT id, source, tenant_id, project_id, environment_id, trace_id, span_id, seq,
                   span_kind, name, status, text, payload_json, observed_at_unix_ms,
                   ingested_at_unix_ms, projected_at_unix_ms
            FROM ledger_events
            WHERE projected_at_unix_ms IS NULL
            ORDER BY observed_at_unix_ms, id
            LIMIT ?1
            ",
        )?;
        let rows = stmt.query_map(params![limit], read_ledger_event)?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub(crate) fn projected_events_after(
        &self,
        scope: StoreScope<'_>,
        after_event: &LedgerEvent,
        before_unix_ms: Option<i64>,
        before_event_id: Option<i64>,
    ) -> MemoryResult<Vec<LedgerEvent>> {
        let mut stmt = self.conn.prepare(
            "
            SELECT id, source, tenant_id, project_id, environment_id, trace_id, span_id, seq,
                   span_kind, name, status, text, payload_json, observed_at_unix_ms,
                   ingested_at_unix_ms, projected_at_unix_ms
            FROM ledger_events
            WHERE tenant_id = ?1
              AND project_id = ?2
              AND COALESCE(environment_id, '') = COALESCE(?3, '')
              AND projected_at_unix_ms IS NOT NULL
              AND (
                    observed_at_unix_ms > ?4
                    OR (
                        ?5 IS NOT NULL
                        AND observed_at_unix_ms = ?4
                        AND id > ?5
                    )
                  )
              AND (
                    ?6 IS NULL
                    OR observed_at_unix_ms < ?6
                    OR (
                        ?7 IS NOT NULL
                        AND observed_at_unix_ms = ?6
                        AND id < ?7
                    )
                  )
            ORDER BY observed_at_unix_ms, id
            ",
        )?;
        let rows = stmt.query_map(
            params![
                scope.tenant_id,
                scope.project_id,
                scope.environment_id,
                after_event.observed_at_unix_ms,
                after_event.id,
                before_unix_ms,
                before_event_id,
            ],
            read_ledger_event,
        )?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn mark_projected(&self, event_id: i64, projected_at_unix_ms: i64) -> MemoryResult<()> {
        self.conn.execute(
            "UPDATE ledger_events SET projected_at_unix_ms = ?2 WHERE id = ?1",
            params![event_id, projected_at_unix_ms],
        )?;
        Ok(())
    }

    pub(crate) fn event_is_pending(&self, event_id: i64) -> MemoryResult<bool> {
        let pending = self
            .conn
            .query_row(
                "SELECT projected_at_unix_ms IS NULL FROM ledger_events WHERE id = ?1",
                params![event_id],
                |row| row.get::<_, bool>(0),
            )
            .optional()?
            .unwrap_or(false);
        Ok(pending)
    }

    pub fn active_neighbors(
        &self,
        tenant_id: &str,
        project_id: &str,
        environment_id: Option<&str>,
        text: &str,
        limit: usize,
        as_of_unix_ms: i64,
    ) -> MemoryResult<Vec<MemoryNode>> {
        let query_terms = terms(text);
        if query_terms.is_empty() {
            return Ok(Vec::new());
        }
        let mut scored = Vec::new();
        let sql_limit = sqlite_limit("active_neighbors limit", limit)?;
        let mut stmt = self.conn.prepare(
            "
            SELECT DISTINCT n.id, n.tenant_id, n.project_id, n.environment_id, n.kind,
                   n.text, n.canonical_key, n.created_at_unix_ms, n.updated_at_unix_ms,
                   n.valid_from_unix_ms, n.valid_to_unix_ms, n.valid_to_event_id, n.confidence,
                   n.token_estimate, n.observation_count
            FROM cue_index c
            JOIN memory_nodes n ON n.id = c.node_id
            WHERE c.tenant_id = ?1
              AND c.project_id = ?2
              AND COALESCE(c.environment_id, '') = COALESCE(?3, '')
              AND c.term = ?4
              AND n.valid_from_unix_ms <= ?5
              AND (n.valid_to_unix_ms IS NULL OR n.valid_to_unix_ms > ?5)
            ORDER BY c.weight DESC,
                     n.valid_from_unix_ms DESC,
                     COALESCE((
                         SELECT e.id
                         FROM node_spans s
                         JOIN ledger_events e
                           ON e.tenant_id = s.tenant_id
                          AND e.project_id = s.project_id
                          AND e.trace_id = s.trace_id
                          AND e.span_id = s.span_id
                          AND e.seq = s.seq
                         WHERE s.node_id = n.id
                         ORDER BY e.observed_at_unix_ms, e.id
                         LIMIT 1
                     ), 0) DESC,
                     n.id
            LIMIT ?6
            ",
        )?;
        for term in query_terms {
            let rows = stmt.query_map(
                params![
                    tenant_id,
                    project_id,
                    environment_id,
                    term,
                    as_of_unix_ms,
                    sql_limit
                ],
                read_node,
            )?;
            for row in rows {
                scored.push(row?);
            }
        }
        let mut seen = BTreeSet::new();
        scored.retain(|node| seen.insert(node.id.clone()));
        scored.truncate(limit);
        Ok(scored)
    }

    pub(crate) fn projection_neighbors_for_event(
        &self,
        event: &LedgerEvent,
        text: &str,
        limit: usize,
    ) -> MemoryResult<Vec<MemoryNode>> {
        let query_terms = terms(text);
        if query_terms.is_empty() {
            return Ok(Vec::new());
        }
        let mut scored = Vec::new();
        let sql_limit = sqlite_limit("projection_neighbors_for_event limit", limit)?;
        let mut stmt = self.conn.prepare(
            "
            SELECT DISTINCT n.id, n.tenant_id, n.project_id, n.environment_id, n.kind,
                   n.text, n.canonical_key, n.created_at_unix_ms, n.updated_at_unix_ms,
                   n.valid_from_unix_ms, n.valid_to_unix_ms, n.valid_to_event_id, n.confidence,
                   n.token_estimate, n.observation_count
            FROM cue_index c
            JOIN memory_nodes n ON n.id = c.node_id
            WHERE c.tenant_id = ?1
              AND c.project_id = ?2
              AND COALESCE(c.environment_id, '') = COALESCE(?3, '')
              AND c.term = ?4
              AND n.valid_from_unix_ms <= ?5
	              AND (
	                  n.valid_to_unix_ms IS NULL
	                  OR n.valid_to_unix_ms > ?5
	                  OR (
	                      n.valid_to_unix_ms = ?5
	                      AND ?6 IS NOT NULL
	                      AND n.valid_to_event_id >= ?6
	                  )
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM ledger_events first_event
                  WHERE first_event.id = (
                      SELECT e.id
                      FROM node_spans s
                      JOIN ledger_events e
                        ON e.tenant_id = s.tenant_id
                       AND e.project_id = s.project_id
                       AND e.trace_id = s.trace_id
                       AND e.span_id = s.span_id
                       AND e.seq = s.seq
                      WHERE s.node_id = n.id
                      ORDER BY e.observed_at_unix_ms, e.id
                      LIMIT 1
                  )
                    AND first_event.tenant_id = ?1
                    AND first_event.project_id = ?2
                    AND COALESCE(first_event.environment_id, '') = COALESCE(?3, '')
                    AND first_event.observed_at_unix_ms = ?5
                    AND ?6 IS NOT NULL
                    AND first_event.id >= ?6
              )
            ORDER BY c.weight DESC,
                     n.valid_from_unix_ms DESC,
                     COALESCE((
                         SELECT e.id
                         FROM node_spans s
                         JOIN ledger_events e
                           ON e.tenant_id = s.tenant_id
                          AND e.project_id = s.project_id
                          AND e.trace_id = s.trace_id
                          AND e.span_id = s.span_id
                          AND e.seq = s.seq
                         WHERE s.node_id = n.id
                         ORDER BY e.observed_at_unix_ms, e.id
                         LIMIT 1
                     ), 0) DESC,
                     n.id
            LIMIT ?7
            ",
        )?;
        for term in query_terms {
            let rows = stmt.query_map(
                params![
                    event.tenant_id,
                    event.project_id,
                    event.environment_id,
                    term,
                    event.observed_at_unix_ms,
                    event.id,
                    sql_limit
                ],
                read_node,
            )?;
            for row in rows {
                scored.push(row?);
            }
        }
        let mut seen = BTreeSet::new();
        scored.retain(|node| seen.insert(node.id.clone()));
        scored.truncate(limit);
        Ok(scored)
    }

    pub(crate) fn upsert_node(
        &self,
        scope: StoreScope<'_>,
        kind: MemoryNodeKind,
        text: &str,
        valid_from_unix_ms: i64,
        cited_spans: &[CitedSpan],
    ) -> MemoryResult<(MemoryNode, bool)> {
        let base_key = if kind == MemoryNodeKind::Episode {
            cited_spans
                .first()
                .map(|span| format!("episode:{}:{}:{}", span.trace_id, span.span_id, span.seq))
                .unwrap_or_else(|| canonical_key(kind.as_str(), text))
        } else {
            canonical_key(kind.as_str(), text)
        };
        let now = now_unix_ms();
        let token_estimate = estimate_tokens(text);
        let family = self.node_family_by_scope_key(
            scope.tenant_id,
            scope.project_id,
            scope.environment_id,
            kind,
            &base_key,
        )?;
        let projection_event_id = self.event_id_for_spans(cited_spans)?;
        let existing = family
            .iter()
            .find(|node| node.valid_from_unix_ms == valid_from_unix_ms)
            .cloned();
        match existing {
            Some(mut node) => {
                let first_event_id = self.first_event_id_for_node(&node.id)?;
                if projection_event_id
                    .zip(first_event_id)
                    .is_some_and(|(current, first)| current > first)
                {
                    self.attach_spans(&node.id, cited_spans)?;
                    return Ok((node, false));
                }
                let merged_text = merge_node_text(&node.text, text);
                let token_estimate = estimate_tokens(&merged_text);
                self.conn.execute(
                    "
                    UPDATE memory_nodes
                    SET text = ?2,
                        updated_at_unix_ms = ?3,
                        confidence = MIN(confidence + 0.05, 1.0),
                        token_estimate = ?4,
                        observation_count = observation_count + 1
                    WHERE id = ?1
                    ",
                    params![node.id, merged_text, now, token_estimate],
                )?;
                node.text = merged_text;
                node.updated_at_unix_ms = now;
                node.confidence = (node.confidence + 0.05).min(1.0);
                node.token_estimate = token_estimate;
                node.observation_count += 1;
                self.attach_spans(&node.id, cited_spans)?;
                self.reindex_node(&node)?;
                Ok((node, false))
            }
            None => {
                let future_successor = family
                    .iter()
                    .filter(|node| node.valid_from_unix_ms > valid_from_unix_ms)
                    .min_by_key(|node| node.valid_from_unix_ms);
                let valid_to_unix_ms = future_successor.map(|node| node.valid_from_unix_ms);
                let valid_to_event_id = if let Some(node) = future_successor {
                    self.first_event_id_for_node(&node.id)?
                } else {
                    None
                };
                self.close_previous_family_versions(
                    scope,
                    kind,
                    &base_key,
                    valid_from_unix_ms,
                    projection_event_id,
                )?;
                let key = if kind == MemoryNodeKind::Episode {
                    base_key
                } else {
                    revision_key(&base_key, valid_from_unix_ms)
                };
                let id = stable_id(
                    "node",
                    &[
                        scope.tenant_id,
                        scope.project_id,
                        scope.environment_id.unwrap_or(""),
                        kind.as_str(),
                        &key,
                    ],
                );
                self.conn.execute(
                    "
                    INSERT INTO memory_nodes(
                        id, tenant_id, project_id, environment_id, kind, text, canonical_key,
                        created_at_unix_ms, updated_at_unix_ms, valid_from_unix_ms,
                        valid_to_unix_ms, valid_to_event_id, confidence, token_estimate,
                        observation_count
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?8, ?9, ?10, ?11, ?12, ?13, 1)
                    ",
                    params![
                        id,
                        scope.tenant_id,
                        scope.project_id,
                        scope.environment_id,
                        kind.as_str(),
                        text,
                        key,
                        now,
                        valid_from_unix_ms,
                        valid_to_unix_ms,
                        valid_to_event_id,
                        0.65_f32,
                        token_estimate,
                    ],
                )?;
                let node = MemoryNode {
                    id,
                    tenant_id: scope.tenant_id.to_string(),
                    project_id: scope.project_id.to_string(),
                    environment_id: scope.environment_id.map(str::to_string),
                    kind,
                    text: text.to_string(),
                    canonical_key: key,
                    created_at_unix_ms: now,
                    updated_at_unix_ms: now,
                    valid_from_unix_ms,
                    valid_to_unix_ms,
                    valid_to_event_id,
                    confidence: 0.65,
                    token_estimate,
                    observation_count: 1,
                };
                self.attach_spans(&node.id, cited_spans)?;
                self.reindex_node(&node)?;
                Ok((node, true))
            }
        }
    }

    pub fn node_by_id(&self, id: &str) -> MemoryResult<Option<MemoryNode>> {
        self.conn
            .query_row(
                "
                SELECT id, tenant_id, project_id, environment_id, kind, text, canonical_key,
                       created_at_unix_ms, updated_at_unix_ms, valid_from_unix_ms,
                       valid_to_unix_ms, valid_to_event_id, confidence, token_estimate,
                       observation_count
                FROM memory_nodes
                WHERE id = ?1
                ",
                params![id],
                read_node,
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn node_by_scope_key(
        &self,
        tenant_id: &str,
        project_id: &str,
        environment_id: Option<&str>,
        kind: MemoryNodeKind,
        canonical_key: &str,
    ) -> MemoryResult<Option<MemoryNode>> {
        self.conn
            .query_row(
                "
                SELECT id, tenant_id, project_id, environment_id, kind, text, canonical_key,
                       created_at_unix_ms, updated_at_unix_ms, valid_from_unix_ms,
                       valid_to_unix_ms, valid_to_event_id, confidence, token_estimate,
                       observation_count
                FROM memory_nodes
                WHERE tenant_id = ?1
                  AND project_id = ?2
                  AND COALESCE(environment_id, '') = COALESCE(?3, '')
                  AND kind = ?4
                  AND canonical_key = ?5
                ",
                params![
                    tenant_id,
                    project_id,
                    environment_id,
                    kind.as_str(),
                    canonical_key
                ],
                read_node,
            )
            .optional()
            .map_err(Into::into)
    }

    pub(crate) fn node_version_by_text(
        &self,
        scope: StoreScope<'_>,
        kind: MemoryNodeKind,
        text: &str,
        valid_from_unix_ms: i64,
    ) -> MemoryResult<Option<MemoryNode>> {
        let base_key = canonical_key(kind.as_str(), text);
        Ok(self
            .node_family_by_scope_key(
                scope.tenant_id,
                scope.project_id,
                scope.environment_id,
                kind,
                &base_key,
            )?
            .into_iter()
            .find(|node| node.valid_from_unix_ms == valid_from_unix_ms))
    }

    fn node_family_by_scope_key(
        &self,
        tenant_id: &str,
        project_id: &str,
        environment_id: Option<&str>,
        kind: MemoryNodeKind,
        canonical_key: &str,
    ) -> MemoryResult<Vec<MemoryNode>> {
        let revision_glob = format!("{canonical_key}|rev:*");
        let mut stmt = self.conn.prepare(
            "
            SELECT id, tenant_id, project_id, environment_id, kind, text, canonical_key,
                   created_at_unix_ms, updated_at_unix_ms, valid_from_unix_ms,
                   valid_to_unix_ms, valid_to_event_id, confidence, token_estimate,
                   observation_count
            FROM memory_nodes
            WHERE tenant_id = ?1
              AND project_id = ?2
              AND COALESCE(environment_id, '') = COALESCE(?3, '')
              AND kind = ?4
              AND (canonical_key = ?5 OR canonical_key GLOB ?6)
            ORDER BY valid_from_unix_ms DESC, updated_at_unix_ms DESC
            ",
        )?;
        let rows = stmt.query_map(
            params![
                tenant_id,
                project_id,
                environment_id,
                kind.as_str(),
                canonical_key,
                revision_glob
            ],
            read_node,
        )?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    fn close_previous_family_versions(
        &self,
        scope: StoreScope<'_>,
        kind: MemoryNodeKind,
        canonical_key: &str,
        valid_from_unix_ms: i64,
        valid_to_event_id: Option<i64>,
    ) -> MemoryResult<()> {
        let revision_glob = format!("{canonical_key}|rev:*");
        self.conn.execute(
            "
            UPDATE memory_nodes
            SET valid_to_unix_ms = ?7,
                valid_to_event_id = ?8
            WHERE tenant_id = ?1
              AND project_id = ?2
              AND COALESCE(environment_id, '') = COALESCE(?3, '')
              AND kind = ?4
              AND (canonical_key = ?5 OR canonical_key GLOB ?6)
              AND valid_from_unix_ms < ?7
              AND (valid_to_unix_ms IS NULL OR valid_to_unix_ms > ?7)
            ",
            params![
                scope.tenant_id,
                scope.project_id,
                scope.environment_id,
                kind.as_str(),
                canonical_key,
                revision_glob,
                valid_from_unix_ms,
                valid_to_event_id,
            ],
        )?;
        Ok(())
    }

    pub fn invalidate_node(
        &self,
        node_id: &str,
        valid_to_unix_ms: i64,
        valid_to_event_id: Option<i64>,
    ) -> MemoryResult<bool> {
        let changed = self.conn.execute(
            "
            UPDATE memory_nodes
            SET valid_to_unix_ms = ?2,
                valid_to_event_id = ?3
            WHERE id = ?1
              AND valid_from_unix_ms <= ?2
              AND (
                    valid_to_unix_ms IS NULL
                    OR valid_to_unix_ms > ?2
                    OR (
                        valid_to_unix_ms = ?2
                        AND ?3 IS NOT NULL
                        AND (valid_to_event_id IS NULL OR valid_to_event_id > ?3)
                    )
                  )
            ",
            params![node_id, valid_to_unix_ms, valid_to_event_id],
        )?;
        Ok(changed > 0)
    }

    pub(crate) fn insert_edge(
        &self,
        scope: StoreScope<'_>,
        from_node_id: &str,
        to_node_id: &str,
        kind: MemoryEdgeKind,
        weight: f32,
        created_at_unix_ms: i64,
    ) -> MemoryResult<bool> {
        if from_node_id == to_node_id {
            return Ok(false);
        }
        let inserted = self.conn.execute(
            "
            INSERT OR IGNORE INTO memory_edges(
                tenant_id, project_id, environment_id, from_node_id, to_node_id,
                kind, weight, created_at_unix_ms
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            ",
            params![
                scope.tenant_id,
                scope.project_id,
                scope.environment_id,
                from_node_id,
                to_node_id,
                kind.as_str(),
                weight,
                created_at_unix_ms,
            ],
        )?;
        Ok(inserted == 1)
    }

    pub fn seed_nodes(
        &self,
        tenant_id: &str,
        project_id: &str,
        environment_id: Option<&str>,
        query_terms: &[String],
        limit: usize,
    ) -> MemoryResult<Vec<MemoryNode>> {
        self.seed_nodes_observed_by(
            tenant_id,
            project_id,
            environment_id,
            query_terms,
            limit,
            None,
        )
    }

    pub fn seed_nodes_observed_by(
        &self,
        tenant_id: &str,
        project_id: &str,
        environment_id: Option<&str>,
        query_terms: &[String],
        limit: usize,
        as_of_unix_ms: Option<i64>,
    ) -> MemoryResult<Vec<MemoryNode>> {
        self.seed_nodes_observed_by_kinds(
            StoreScope::new(tenant_id, project_id, environment_id),
            query_terms,
            limit,
            as_of_unix_ms,
            &[],
        )
    }

    pub(crate) fn seed_nodes_observed_by_kinds(
        &self,
        scope: StoreScope<'_>,
        query_terms: &[String],
        limit: usize,
        as_of_unix_ms: Option<i64>,
        kinds: &[MemoryNodeKind],
    ) -> MemoryResult<Vec<MemoryNode>> {
        let sql_limit = sqlite_limit("seed_nodes limit", limit)?;
        if query_terms.is_empty() {
            return self.recent_nodes_observed_by_kinds(
                scope.tenant_id,
                scope.project_id,
                scope.environment_id,
                limit,
                as_of_unix_ms,
                kinds,
            );
        }
        let mut nodes = Vec::new();
        let kind_filter = kind_filter_clause("n", 7, kinds.len());
        let sql = format!(
            "
            SELECT DISTINCT n.id, n.tenant_id, n.project_id, n.environment_id, n.kind,
                   n.text, n.canonical_key, n.created_at_unix_ms, n.updated_at_unix_ms,
                   n.valid_from_unix_ms, n.valid_to_unix_ms, n.valid_to_event_id, n.confidence,
                   n.token_estimate, n.observation_count
            FROM cue_index c
            JOIN memory_nodes n ON n.id = c.node_id
            WHERE c.tenant_id = ?1
              AND c.project_id = ?2
              AND COALESCE(c.environment_id, '') = COALESCE(?3, '')
              AND c.term = ?4
              AND (?6 IS NULL OR n.valid_from_unix_ms <= ?6)
              {kind_filter}
            ORDER BY c.weight DESC,
                     n.valid_from_unix_ms DESC,
                     COALESCE((
                         SELECT e.id
                         FROM node_spans s
                         JOIN ledger_events e
                           ON e.tenant_id = s.tenant_id
                          AND e.project_id = s.project_id
                          AND e.trace_id = s.trace_id
                          AND e.span_id = s.span_id
                          AND e.seq = s.seq
                         WHERE s.node_id = n.id
                         ORDER BY e.observed_at_unix_ms, e.id
                         LIMIT 1
                     ), 0) DESC,
                     n.id
            LIMIT ?5
            "
        );
        let mut stmt = self.conn.prepare(&sql)?;
        for term in query_terms {
            let values = query_values(
                scope.tenant_id,
                scope.project_id,
                scope.environment_id,
                Some(term.as_str()),
                sql_limit,
                as_of_unix_ms,
                kinds,
            );
            let rows = stmt.query_map(params_from_iter(values.iter()), read_node)?;
            for row in rows {
                nodes.push(row?);
            }
        }
        let mut seen = BTreeSet::new();
        nodes.retain(|node| seen.insert(node.id.clone()));
        nodes.truncate(limit);
        Ok(nodes)
    }

    pub fn recent_nodes(
        &self,
        tenant_id: &str,
        project_id: &str,
        environment_id: Option<&str>,
        limit: usize,
    ) -> MemoryResult<Vec<MemoryNode>> {
        self.recent_nodes_observed_by(tenant_id, project_id, environment_id, limit, None)
    }

    pub fn recent_nodes_observed_by(
        &self,
        tenant_id: &str,
        project_id: &str,
        environment_id: Option<&str>,
        limit: usize,
        as_of_unix_ms: Option<i64>,
    ) -> MemoryResult<Vec<MemoryNode>> {
        self.recent_nodes_observed_by_kinds(
            tenant_id,
            project_id,
            environment_id,
            limit,
            as_of_unix_ms,
            &[],
        )
    }

    pub fn recent_nodes_observed_by_kinds(
        &self,
        tenant_id: &str,
        project_id: &str,
        environment_id: Option<&str>,
        limit: usize,
        as_of_unix_ms: Option<i64>,
        kinds: &[MemoryNodeKind],
    ) -> MemoryResult<Vec<MemoryNode>> {
        let limit = sqlite_limit("recent_nodes limit", limit)?;
        let kind_filter = kind_filter_clause("memory_nodes", 6, kinds.len());
        let sql = format!(
            "
            SELECT id, tenant_id, project_id, environment_id, kind, text, canonical_key,
                   created_at_unix_ms, updated_at_unix_ms, valid_from_unix_ms,
                   valid_to_unix_ms, valid_to_event_id, confidence, token_estimate,
                   observation_count
            FROM memory_nodes
            WHERE tenant_id = ?1
              AND project_id = ?2
              AND COALESCE(environment_id, '') = COALESCE(?3, '')
              AND (?5 IS NULL OR valid_from_unix_ms <= ?5)
              {kind_filter}
            ORDER BY valid_from_unix_ms DESC,
                     COALESCE((
                         SELECT e.id
                         FROM node_spans s
                         JOIN ledger_events e
                           ON e.tenant_id = s.tenant_id
                          AND e.project_id = s.project_id
                          AND e.trace_id = s.trace_id
                          AND e.span_id = s.span_id
                          AND e.seq = s.seq
                         WHERE s.node_id = memory_nodes.id
                         ORDER BY e.observed_at_unix_ms, e.id
                         LIMIT 1
                     ), 0) DESC,
                     id
            LIMIT ?4
            "
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let values = query_values(
            tenant_id,
            project_id,
            environment_id,
            None,
            limit,
            as_of_unix_ms,
            kinds,
        );
        let rows = stmt.query_map(params_from_iter(values.iter()), read_node)?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn all_nodes(
        &self,
        tenant_id: &str,
        project_id: &str,
        environment_id: Option<&str>,
    ) -> MemoryResult<Vec<MemoryNode>> {
        self.recent_nodes(tenant_id, project_id, environment_id, 10_000)
    }

    pub fn all_nodes_observed_by(
        &self,
        tenant_id: &str,
        project_id: &str,
        environment_id: Option<&str>,
        as_of_unix_ms: i64,
    ) -> MemoryResult<Vec<MemoryNode>> {
        self.recent_nodes_observed_by(
            tenant_id,
            project_id,
            environment_id,
            10_000,
            Some(as_of_unix_ms),
        )
    }

    pub fn all_nodes_observed_by_kinds(
        &self,
        tenant_id: &str,
        project_id: &str,
        environment_id: Option<&str>,
        as_of_unix_ms: i64,
        kinds: &[MemoryNodeKind],
    ) -> MemoryResult<Vec<MemoryNode>> {
        self.recent_nodes_observed_by_kinds(
            tenant_id,
            project_id,
            environment_id,
            10_000,
            Some(as_of_unix_ms),
            kinds,
        )
    }

    pub fn edges_for_scope(
        &self,
        tenant_id: &str,
        project_id: &str,
        environment_id: Option<&str>,
    ) -> MemoryResult<Vec<MemoryEdge>> {
        let mut stmt = self.conn.prepare(
            "
            SELECT id, tenant_id, project_id, environment_id, from_node_id, to_node_id,
                   kind, weight, created_at_unix_ms
            FROM memory_edges
            WHERE tenant_id = ?1
              AND project_id = ?2
              AND COALESCE(environment_id, '') = COALESCE(?3, '')
            ",
        )?;
        let rows = stmt.query_map(params![tenant_id, project_id, environment_id], read_edge)?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn cited_spans_for_node(&self, node_id: &str) -> MemoryResult<Vec<CitedSpan>> {
        self.cited_spans_for_node_as_of(node_id, None)
    }

    pub fn cited_spans_for_node_as_of(
        &self,
        node_id: &str,
        as_of_unix_ms: Option<i64>,
    ) -> MemoryResult<Vec<CitedSpan>> {
        let mut stmt = self.conn.prepare(
            "
            SELECT s.tenant_id, s.project_id, s.trace_id, s.span_id, s.seq
            FROM node_spans s
            LEFT JOIN ledger_events e
              ON e.tenant_id = s.tenant_id
             AND e.project_id = s.project_id
             AND e.trace_id = s.trace_id
             AND e.span_id = s.span_id
             AND e.seq = s.seq
            WHERE s.node_id = ?1
              AND (?2 IS NULL OR e.observed_at_unix_ms IS NULL OR e.observed_at_unix_ms <= ?2)
            ORDER BY s.seq
            ",
        )?;
        let rows = stmt.query_map(params![node_id, as_of_unix_ms], |row| {
            Ok(CitedSpan {
                tenant_id: row.get(0)?,
                project_id: row.get(1)?,
                trace_id: row.get(2)?,
                span_id: row.get(3)?,
                seq: row.get(4)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn stats(&self) -> MemoryResult<StoreStats> {
        let ledger_events: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM ledger_events", [], |row| row.get(0))?;
        let pending_events: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM ledger_events WHERE projected_at_unix_ms IS NULL",
            [],
            |row| row.get(0),
        )?;
        let nodes: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM memory_nodes", [], |row| row.get(0))?;
        let now = now_unix_ms();
        let active_nodes: i64 = self.conn.query_row(
            "
            SELECT COUNT(*)
            FROM memory_nodes
            WHERE valid_from_unix_ms <= ?1
              AND (valid_to_unix_ms IS NULL OR valid_to_unix_ms > ?1)
            ",
            params![now],
            |row| row.get(0),
        )?;
        let total_node_tokens: i64 = self.conn.query_row(
            "SELECT COALESCE(SUM(token_estimate), 0) FROM memory_nodes",
            [],
            |row| row.get(0),
        )?;
        let active_node_tokens: i64 = self.conn.query_row(
            "
            SELECT COALESCE(SUM(token_estimate), 0)
            FROM memory_nodes
            WHERE valid_from_unix_ms <= ?1
              AND (valid_to_unix_ms IS NULL OR valid_to_unix_ms > ?1)
            ",
            params![now],
            |row| row.get(0),
        )?;
        let active_episode_nodes = self.active_node_count_by_kind(MemoryNodeKind::Episode, now)?;
        let active_fact_nodes = self.active_node_count_by_kind(MemoryNodeKind::Fact, now)?;
        let active_entity_cue_nodes =
            self.active_node_count_by_kind(MemoryNodeKind::EntityCue, now)?;
        let active_tag_nodes = self.active_node_count_by_kind(MemoryNodeKind::Tag, now)?;
        let active_procedure_nodes =
            self.active_node_count_by_kind(MemoryNodeKind::Procedure, now)?;
        let active_state_nodes = self.active_node_count_by_kind(MemoryNodeKind::State, now)?;
        let active_gotcha_nodes = self.active_node_count_by_kind(MemoryNodeKind::Gotcha, now)?;
        let active_anti_memory_nodes =
            self.active_node_count_by_kind(MemoryNodeKind::AntiMemory, now)?;
        let active_topic_nodes = self.active_node_count_by_kind(MemoryNodeKind::Topic, now)?;
        let edges: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM memory_edges", [], |row| row.get(0))?;
        let audit_events: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM audit_events", [], |row| row.get(0))?;
        Ok(StoreStats {
            ledger_events,
            pending_events,
            nodes,
            active_nodes,
            total_node_tokens,
            active_node_tokens,
            active_episode_nodes,
            active_fact_nodes,
            active_entity_cue_nodes,
            active_tag_nodes,
            active_procedure_nodes,
            active_state_nodes,
            active_gotcha_nodes,
            active_anti_memory_nodes,
            active_topic_nodes,
            edges,
            audit_events,
        })
    }

    fn active_node_count_by_kind(&self, kind: MemoryNodeKind, now: i64) -> MemoryResult<i64> {
        self.conn
            .query_row(
                "
                SELECT COUNT(*)
                FROM memory_nodes
                WHERE valid_from_unix_ms <= ?2
                  AND (valid_to_unix_ms IS NULL OR valid_to_unix_ms > ?2)
                  AND kind = ?1
                ",
                params![kind.as_str(), now],
                |row| row.get(0),
            )
            .map_err(Into::into)
    }

    pub fn append_audit(&self, record: &AuditRecord) -> MemoryResult<i64> {
        record.validate()?;
        self.conn.execute(
            "
            INSERT INTO audit_events(
                occurred_at_unix_ms, actor, action, outcome, route, status_code, detail_json
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            ",
            params![
                now_unix_ms(),
                record.actor,
                record.action,
                record.outcome,
                record.route,
                record.status_code,
                serde_json::to_string(&record.detail)?,
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn recent_audit_events(&self, limit: usize) -> MemoryResult<Vec<AuditEvent>> {
        let limit = sqlite_limit("recent_audit_events limit", limit)?;
        let mut stmt = self.conn.prepare(
            "
            SELECT id, occurred_at_unix_ms, actor, action, outcome, route, status_code, detail_json
            FROM audit_events
            ORDER BY occurred_at_unix_ms DESC, id DESC
            LIMIT ?1
            ",
        )?;
        let rows = stmt.query_map(params![limit], read_audit_event)?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn schema_version(&self) -> MemoryResult<u32> {
        self.conn
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .map_err(Into::into)
    }

    pub fn application_id(&self) -> MemoryResult<i64> {
        self.conn
            .query_row("PRAGMA application_id", [], |row| row.get(0))
            .map_err(Into::into)
    }

    pub fn health(&self) -> MemoryResult<StoreHealth> {
        let schema_version = self.schema_version()?;
        let application_id = self.application_id()?;
        let graph_integrity = self.graph_integrity()?;
        let mut stmt = self.conn.prepare("PRAGMA integrity_check")?;
        let integrity_messages = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let integrity_ok =
            integrity_messages.len() == 1 && integrity_messages.first().is_some_and(|m| m == "ok");
        let mut fk_stmt = self.conn.prepare("PRAGMA foreign_key_check")?;
        let foreign_key_violations = fk_stmt.query_map([], |_| Ok(()))?.count() as i64;
        Ok(StoreHealth {
            schema_version,
            expected_schema_version: SCHEMA_VERSION,
            integrity_ok,
            integrity_messages,
            foreign_key_violations,
            graph_integrity_ok: graph_integrity.is_clean(),
            graph_integrity,
            stats: self.stats()?,
            application_id,
            expected_application_id: SQLITE_APPLICATION_ID,
        })
    }

    pub fn maintenance(&self, vacuum: bool) -> MemoryResult<MaintenanceReport> {
        self.maintenance_with_options(MaintenanceOptions {
            vacuum,
            ..MaintenanceOptions::default()
        })
    }

    pub fn maintenance_with_options(
        &self,
        options: MaintenanceOptions,
    ) -> MemoryResult<MaintenanceReport> {
        options.validate()?;
        self.conn.execute_batch("PRAGMA optimize")?;
        let graph_integrity_before = self.graph_integrity()?;
        let graph_repair = if options.repair_orphans {
            self.with_immediate_transaction(|store| store.repair_graph_orphans())?
        } else {
            GraphRepairReport::default()
        };
        let pruned_audit_events = options.prune_audit_before_unix_ms.is_some()
            || options.retain_latest_audit_events.is_some();
        let audit_prune = if pruned_audit_events {
            self.with_immediate_transaction(|store| {
                store.prune_audit_events_in_transaction(
                    options.prune_audit_before_unix_ms,
                    options.retain_latest_audit_events,
                )
            })?
        } else {
            AuditPruneReport::default()
        };
        let graph_integrity_after = self.graph_integrity()?;
        let (busy, log_frames, checkpointed_frames): (i64, i64, i64) =
            self.conn
                .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                })?;
        if options.vacuum {
            self.conn.execute_batch("VACUUM")?;
        }
        Ok(MaintenanceReport {
            optimized: true,
            wal_checkpoint_busy: busy,
            wal_checkpoint_log_frames: log_frames,
            wal_checkpoint_checkpointed_frames: checkpointed_frames,
            vacuumed: options.vacuum,
            repaired_orphans: options.repair_orphans,
            pruned_audit_events,
            graph_integrity_before,
            graph_integrity_after,
            graph_repair,
            audit_prune,
        })
    }

    pub fn graph_integrity(&self) -> MemoryResult<GraphIntegrityReport> {
        let orphan_edges_from: i64 = self.conn.query_row(
            "
            SELECT COUNT(*)
            FROM memory_edges e
            LEFT JOIN memory_nodes n ON n.id = e.from_node_id
            WHERE n.id IS NULL
            ",
            [],
            |row| row.get(0),
        )?;
        let orphan_edges_to: i64 = self.conn.query_row(
            "
            SELECT COUNT(*)
            FROM memory_edges e
            LEFT JOIN memory_nodes n ON n.id = e.to_node_id
            WHERE n.id IS NULL
            ",
            [],
            |row| row.get(0),
        )?;
        let orphan_node_spans: i64 = self.conn.query_row(
            "
            SELECT COUNT(*)
            FROM node_spans s
            LEFT JOIN memory_nodes n ON n.id = s.node_id
            WHERE n.id IS NULL
            ",
            [],
            |row| row.get(0),
        )?;
        let orphan_cue_index_entries: i64 = self.conn.query_row(
            "
            SELECT COUNT(*)
            FROM cue_index c
            LEFT JOIN memory_nodes n ON n.id = c.node_id
            WHERE n.id IS NULL
            ",
            [],
            |row| row.get(0),
        )?;
        Ok(GraphIntegrityReport {
            orphan_edges_from,
            orphan_edges_to,
            orphan_node_spans,
            orphan_cue_index_entries,
        })
    }

    pub fn reset_projection(&self) -> MemoryResult<ProjectionResetReport> {
        self.with_immediate_transaction(|store| store.reset_projection_in_transaction())
    }

    fn reset_projection_in_transaction(&self) -> MemoryResult<ProjectionResetReport> {
        let edges_removed = self.conn.execute("DELETE FROM memory_edges", [])? as i64;
        let node_spans_removed = self.conn.execute("DELETE FROM node_spans", [])? as i64;
        let cue_index_entries_removed = self.conn.execute("DELETE FROM cue_index", [])? as i64;
        let nodes_removed = self.conn.execute("DELETE FROM memory_nodes", [])? as i64;
        let ledger_events_reset = self.conn.execute(
            "UPDATE ledger_events SET projected_at_unix_ms = NULL WHERE projected_at_unix_ms IS NOT NULL",
            [],
        )? as i64;
        Ok(ProjectionResetReport {
            ledger_events_reset,
            nodes_removed,
            edges_removed,
            node_spans_removed,
            cue_index_entries_removed,
        })
    }

    fn repair_graph_orphans(&self) -> MemoryResult<GraphRepairReport> {
        let memory_edges_removed = self.conn.execute(
            "
            DELETE FROM memory_edges
            WHERE from_node_id NOT IN (SELECT id FROM memory_nodes)
               OR to_node_id NOT IN (SELECT id FROM memory_nodes)
            ",
            [],
        )? as i64;
        let node_spans_removed = self.conn.execute(
            "
            DELETE FROM node_spans
            WHERE node_id NOT IN (SELECT id FROM memory_nodes)
            ",
            [],
        )? as i64;
        let cue_index_entries_removed = self.conn.execute(
            "
            DELETE FROM cue_index
            WHERE node_id NOT IN (SELECT id FROM memory_nodes)
            ",
            [],
        )? as i64;
        Ok(GraphRepairReport {
            memory_edges_removed,
            node_spans_removed,
            cue_index_entries_removed,
        })
    }

    pub fn prune_audit_events(
        &self,
        before_unix_ms: Option<i64>,
        retain_latest: Option<usize>,
    ) -> MemoryResult<AuditPruneReport> {
        let options = MaintenanceOptions {
            prune_audit_before_unix_ms: before_unix_ms,
            retain_latest_audit_events: retain_latest,
            ..MaintenanceOptions::default()
        };
        options.validate()?;
        if options.prune_audit_before_unix_ms.is_none()
            && options.retain_latest_audit_events.is_none()
        {
            return Ok(AuditPruneReport::default());
        }
        self.with_immediate_transaction(|store| {
            store.prune_audit_events_in_transaction(
                options.prune_audit_before_unix_ms,
                options.retain_latest_audit_events,
            )
        })
    }

    fn prune_audit_events_in_transaction(
        &self,
        before_unix_ms: Option<i64>,
        retain_latest: Option<usize>,
    ) -> MemoryResult<AuditPruneReport> {
        let mut audit_events_removed = 0_i64;
        if let Some(cutoff) = before_unix_ms {
            audit_events_removed += self.conn.execute(
                "DELETE FROM audit_events WHERE occurred_at_unix_ms < ?1",
                params![cutoff],
            )? as i64;
        }
        if let Some(limit) = retain_latest {
            let limit = sqlite_limit("retain_latest audit limit", limit)?;
            audit_events_removed += self.conn.execute(
                "
                DELETE FROM audit_events
                WHERE id NOT IN (
                    SELECT id
                    FROM audit_events
                    ORDER BY occurred_at_unix_ms DESC, id DESC
                    LIMIT ?1
                )
                ",
                params![limit],
            )? as i64;
        }
        Ok(AuditPruneReport {
            audit_events_removed,
        })
    }

    pub fn backup_to(&self, path: impl AsRef<Path>) -> MemoryResult<BackupReport> {
        let path = path.as_ref();
        if path.exists() {
            return Err(MemoryError::invalid(format!(
                "backup path already exists: {}",
                path.display()
            )));
        }
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        self.conn.backup(MAIN_DB, path, None)?;
        let health = self.health()?;
        Ok(BackupReport {
            path: path.display().to_string(),
            bytes: std::fs::metadata(path)?.len(),
            schema_version: health.schema_version,
            integrity_ok: health.integrity_ok,
        })
    }

    pub fn restore_from(&mut self, path: impl AsRef<Path>) -> MemoryResult<RestoreReport> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(MemoryError::invalid(format!(
                "restore path is not a file: {}",
                path.display()
            )));
        }
        validate_restore_source(path)?;
        self.conn.restore(
            MAIN_DB,
            path,
            Option::<fn(rusqlite::backup::Progress)>::None,
        )?;
        self.migrate()?;
        let health = self.health()?;
        Ok(RestoreReport {
            path: path.display().to_string(),
            schema_version: health.schema_version,
            integrity_ok: health.integrity_ok,
            stats: health.stats,
        })
    }

    fn attach_spans(&self, node_id: &str, cited_spans: &[CitedSpan]) -> MemoryResult<()> {
        for span in cited_spans {
            self.conn.execute(
                "
                INSERT OR IGNORE INTO node_spans(node_id, tenant_id, project_id, trace_id, span_id, seq)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                ",
                params![
                    node_id,
                    span.tenant_id,
                    span.project_id,
                    span.trace_id,
                    span.span_id,
                    span.seq,
                ],
            )?;
        }
        Ok(())
    }

    fn event_id_for_spans(&self, cited_spans: &[CitedSpan]) -> MemoryResult<Option<i64>> {
        for span in cited_spans {
            if let Some(event_id) = self.event_id_for_span(span)? {
                return Ok(Some(event_id));
            }
        }
        Ok(None)
    }

    fn first_event_id_for_node(&self, node_id: &str) -> MemoryResult<Option<i64>> {
        self.conn
            .query_row(
                "
                SELECT e.id
                FROM node_spans s
                JOIN ledger_events e
                  ON e.tenant_id = s.tenant_id
                 AND e.project_id = s.project_id
                 AND e.trace_id = s.trace_id
                 AND e.span_id = s.span_id
                 AND e.seq = s.seq
                WHERE s.node_id = ?1
                ORDER BY e.observed_at_unix_ms, e.id
                LIMIT 1
                ",
                params![node_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(Into::into)
    }

    fn event_id_for_span(&self, span: &CitedSpan) -> MemoryResult<Option<i64>> {
        self.conn
            .query_row(
                "
                SELECT id
                FROM ledger_events
                WHERE tenant_id = ?1
                  AND project_id = ?2
                  AND trace_id = ?3
                  AND span_id = ?4
                  AND seq = ?5
                ",
                params![
                    span.tenant_id,
                    span.project_id,
                    span.trace_id,
                    span.span_id,
                    span.seq
                ],
                |row| row.get(0),
            )
            .optional()
            .map_err(Into::into)
    }

    fn reindex_node(&self, node: &MemoryNode) -> MemoryResult<()> {
        self.conn
            .execute("DELETE FROM cue_index WHERE node_id = ?1", params![node.id])?;
        let mut weight = 1.0_f32;
        for term in terms(&node.text).into_iter().take(64) {
            self.conn.execute(
                "
                INSERT OR REPLACE INTO cue_index(term, node_id, tenant_id, project_id, environment_id, weight)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                ",
                params![
                    term,
                    node.id,
                    node.tenant_id,
                    node.project_id,
                    node.environment_id,
                    weight,
                ],
            )?;
            weight = (weight * 0.97).max(0.2);
        }
        Ok(())
    }
}

fn merge_node_text(existing: &str, incoming: &str) -> String {
    let incoming = incoming.trim();
    if incoming.is_empty() || existing.contains(incoming) {
        existing.to_string()
    } else if existing.trim().is_empty() {
        incoming.to_string()
    } else {
        format!("{} {}", existing.trim(), incoming)
    }
}

fn revision_key(canonical_key: &str, valid_from_unix_ms: i64) -> String {
    format!("{canonical_key}|rev:{valid_from_unix_ms}")
}

fn validate_required_identifier(field: &str, value: &str) -> MemoryResult<()> {
    if value.trim().is_empty() {
        return Err(MemoryError::invalid(format!("{field} must not be empty")));
    }
    if value.trim() != value {
        return Err(MemoryError::invalid(format!(
            "{field} must not have leading or trailing whitespace"
        )));
    }
    Ok(())
}

fn validate_required_text(field: &str, value: &str) -> MemoryResult<()> {
    if value.trim().is_empty() {
        Err(MemoryError::invalid(format!("{field} must not be empty")))
    } else {
        Ok(())
    }
}

fn sqlite_limit(field: &str, limit: usize) -> MemoryResult<i64> {
    i64::try_from(limit)
        .map_err(|_| MemoryError::invalid(format!("{field} exceeds SQLite LIMIT range")))
}

fn kind_filter_clause(alias: &str, first_index: usize, count: usize) -> String {
    if count == 0 {
        return String::new();
    }
    let placeholders = (first_index..first_index + count)
        .map(|index| format!("?{index}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("AND {alias}.kind IN ({placeholders})")
}

fn query_values(
    tenant_id: &str,
    project_id: &str,
    environment_id: Option<&str>,
    term: Option<&str>,
    limit: i64,
    as_of_unix_ms: Option<i64>,
    kinds: &[MemoryNodeKind],
) -> Vec<Value> {
    let mut values = vec![
        Value::Text(tenant_id.to_string()),
        Value::Text(project_id.to_string()),
        optional_text(environment_id),
    ];
    if let Some(term) = term {
        values.push(Value::Text(term.to_string()));
    }
    values.push(Value::Integer(limit));
    values.push(as_of_unix_ms.map(Value::Integer).unwrap_or(Value::Null));
    values.extend(
        kinds
            .iter()
            .map(|kind| Value::Text(kind.as_str().to_string())),
    );
    values
}

fn optional_text(value: Option<&str>) -> Value {
    value
        .map(|value| Value::Text(value.to_string()))
        .unwrap_or(Value::Null)
}

fn configure_connection(conn: &Connection) -> MemoryResult<()> {
    conn.busy_timeout(Duration::from_secs(5))?;
    conn.execute_batch(
        "
        PRAGMA foreign_keys = ON;
        ",
    )?;
    Ok(())
}

fn configure_persistent_database(conn: &Connection) -> MemoryResult<()> {
    conn.execute_batch(
        "
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        ",
    )?;
    Ok(())
}

fn validate_restore_source(path: &Path) -> MemoryResult<()> {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    validate_database_identity(&conn, false)?;
    let user_version: u32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if user_version > SCHEMA_VERSION {
        return Err(MemoryError::invalid(format!(
            "restore source schema version {user_version} is newer than supported version {SCHEMA_VERSION}"
        )));
    }
    Ok(())
}

fn validate_database_identity(conn: &Connection, initialize_empty: bool) -> MemoryResult<()> {
    let application_id: i64 = conn.query_row("PRAGMA application_id", [], |row| row.get(0))?;
    if application_id == SQLITE_APPLICATION_ID {
        return Ok(());
    }
    if application_id != 0 {
        return Err(MemoryError::invalid(format!(
            "database application_id {application_id} is not beater-memory application_id {SQLITE_APPLICATION_ID}"
        )));
    }
    let user_version: u32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if initialize_empty && user_version == 0 && !database_has_user_objects(conn)? {
        set_application_id(conn)?;
        return Ok(());
    }
    Err(MemoryError::invalid(
        "refusing to initialize a SQLite database that was not created by beater-memory",
    ))
}

fn database_has_user_objects(conn: &Connection) -> MemoryResult<bool> {
    let count: i64 = conn.query_row(
        "
        SELECT COUNT(*)
        FROM sqlite_master
        WHERE name NOT LIKE 'sqlite_%'
          AND type IN ('table', 'index', 'trigger', 'view')
        ",
        [],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

fn set_application_id(conn: &Connection) -> MemoryResult<()> {
    conn.execute_batch(&format!("PRAGMA application_id = {SQLITE_APPLICATION_ID};"))?;
    Ok(())
}

fn read_ledger_event(row: &rusqlite::Row<'_>) -> rusqlite::Result<LedgerEvent> {
    let payload_json: String = row.get(12)?;
    let payload = serde_json::from_str(&payload_json).map_err(|err| {
        rusqlite::Error::FromSqlConversionFailure(12, rusqlite::types::Type::Text, Box::new(err))
    })?;
    Ok(LedgerEvent {
        id: row.get(0)?,
        source: row.get(1)?,
        tenant_id: row.get(2)?,
        project_id: row.get(3)?,
        environment_id: row.get(4)?,
        trace_id: row.get(5)?,
        span_id: row.get(6)?,
        seq: row.get(7)?,
        span_kind: row.get(8)?,
        name: row.get(9)?,
        status: row.get(10)?,
        text: row.get(11)?,
        payload,
        observed_at_unix_ms: row.get(13)?,
        ingested_at_unix_ms: row.get(14)?,
        projected_at_unix_ms: row.get(15)?,
    })
}

fn read_node(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryNode> {
    let kind: String = row.get(4)?;
    let kind = MemoryNodeKind::from_str(&kind).map_err(|err| {
        rusqlite::Error::FromSqlConversionFailure(
            4,
            rusqlite::types::Type::Text,
            Box::new(MemoryError::invalid(err)),
        )
    })?;
    Ok(MemoryNode {
        id: row.get(0)?,
        tenant_id: row.get(1)?,
        project_id: row.get(2)?,
        environment_id: row.get(3)?,
        kind,
        text: row.get(5)?,
        canonical_key: row.get(6)?,
        created_at_unix_ms: row.get(7)?,
        updated_at_unix_ms: row.get(8)?,
        valid_from_unix_ms: row.get(9)?,
        valid_to_unix_ms: row.get(10)?,
        valid_to_event_id: row.get(11)?,
        confidence: row.get(12)?,
        token_estimate: row.get(13)?,
        observation_count: row.get(14)?,
    })
}

fn read_edge(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryEdge> {
    let kind: String = row.get(6)?;
    let kind = MemoryEdgeKind::from_str(&kind).map_err(|err| {
        rusqlite::Error::FromSqlConversionFailure(
            6,
            rusqlite::types::Type::Text,
            Box::new(MemoryError::invalid(err)),
        )
    })?;
    Ok(MemoryEdge {
        id: row.get(0)?,
        tenant_id: row.get(1)?,
        project_id: row.get(2)?,
        environment_id: row.get(3)?,
        from_node_id: row.get(4)?,
        to_node_id: row.get(5)?,
        kind,
        weight: row.get(7)?,
        created_at_unix_ms: row.get(8)?,
    })
}

fn read_audit_event(row: &rusqlite::Row<'_>) -> rusqlite::Result<AuditEvent> {
    let detail_json: String = row.get(7)?;
    let detail = serde_json::from_str(&detail_json).map_err(|err| {
        rusqlite::Error::FromSqlConversionFailure(7, rusqlite::types::Type::Text, Box::new(err))
    })?;
    Ok(AuditEvent {
        id: row.get(0)?,
        occurred_at_unix_ms: row.get(1)?,
        actor: row.get(2)?,
        action: row.get(3)?,
        outcome: row.get(4)?,
        route: row.get(5)?,
        status_code: row.get(6)?,
        detail,
    })
}

#[cfg(test)]
mod tests {
    use std::time::SystemTime;

    use super::*;

    #[test]
    fn store_is_idempotent_for_same_ledger_event() -> MemoryResult<()> {
        let store = SqliteMemoryStore::in_memory()?;
        let event = LedgerEvent::direct_memory_write(
            "tenant",
            "project",
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL.",
        );

        assert!(store.append_event(&event)?);
        assert!(!store.append_event(&event)?);
        assert_eq!(store.stats()?.ledger_events, 1);
        Ok(())
    }

    #[test]
    fn append_event_rejects_malformed_ledger_events() -> MemoryResult<()> {
        let store = SqliteMemoryStore::in_memory()?;

        assert_invalid_event(
            &store,
            |event| event.tenant_id = " ".to_string(),
            "tenant_id",
        )?;
        assert_invalid_event(
            &store,
            |event| event.trace_id = " trace".to_string(),
            "trace_id",
        )?;
        assert_invalid_event(
            &store,
            |event| event.environment_id = Some(String::new()),
            "environment_id",
        )?;
        assert_invalid_event(&store, |event| event.text = "\t".to_string(), "text")?;
        assert_invalid_event(&store, |event| event.seq = 0, "seq")?;
        assert_invalid_event(
            &store,
            |event| event.observed_at_unix_ms = -1,
            "observed_at_unix_ms",
        )?;
        assert_invalid_event(
            &store,
            |event| event.ingested_at_unix_ms = -1,
            "ingested_at_unix_ms",
        )?;
        assert_invalid_event(
            &store,
            |event| event.projected_at_unix_ms = Some(-1),
            "projected_at_unix_ms",
        )?;

        assert_eq!(store.stats()?.ledger_events, 0);
        Ok(())
    }

    #[test]
    fn store_rejects_limits_outside_sqlite_range() -> MemoryResult<()> {
        let Some(limit) = oversized_sqlite_limit() else {
            return Ok(());
        };
        let store = SqliteMemoryStore::in_memory()?;

        let err = store.pending_events(limit).unwrap_err();
        assert!(err.to_string().contains("pending_events limit"));

        let err = store
            .seed_nodes(
                "tenant",
                "project",
                None,
                &[String::from("checkout")],
                limit,
            )
            .unwrap_err();
        assert!(err.to_string().contains("seed_nodes limit"));

        let err = store
            .recent_nodes("tenant", "project", None, limit)
            .unwrap_err();
        assert!(err.to_string().contains("recent_nodes limit"));

        let err = store.recent_audit_events(limit).unwrap_err();
        assert!(err.to_string().contains("recent_audit_events limit"));

        let err = store.prune_audit_events(None, Some(limit)).unwrap_err();
        assert!(err.to_string().contains("retain_latest audit limit"));
        Ok(())
    }

    #[test]
    fn direct_memory_idempotency_key_stabilizes_ledger_key() -> MemoryResult<()> {
        let store = SqliteMemoryStore::in_memory()?;
        let mut first = LedgerEvent::direct_memory_write(
            "tenant",
            "project",
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL.",
        );
        first.environment_id = Some("prod".to_string());
        first.apply_idempotency_key("retry-key");
        let mut second = LedgerEvent::direct_memory_write(
            "tenant",
            "project",
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL.",
        );
        second.environment_id = Some("prod".to_string());
        second.apply_idempotency_key("retry-key");

        assert_eq!(first.trace_id, second.trace_id);
        assert_eq!(first.span_id, second.span_id);
        assert_eq!(
            first.payload["idempotency_key_hash"],
            second.payload["idempotency_key_hash"]
        );
        assert!(store.append_event(&first)?);
        assert!(!store.append_event(&second)?);
        assert_eq!(store.stats()?.ledger_events, 1);
        Ok(())
    }

    #[test]
    fn nodes_are_upserted_by_canonical_key() -> MemoryResult<()> {
        let store = SqliteMemoryStore::in_memory()?;
        let span = CitedSpan {
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            trace_id: "trace".to_string(),
            span_id: "span".to_string(),
            seq: 1,
        };

        let (_, created) = store.upsert_node(
            StoreScope::new("tenant", "project", None),
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL.",
            1,
            std::slice::from_ref(&span),
        )?;
        let (node, second_created) = store.upsert_node(
            StoreScope::new("tenant", "project", None),
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL.",
            1,
            &[span],
        )?;

        assert!(created);
        assert!(!second_created);
        assert_eq!(node.observation_count, 2);
        assert_eq!(store.cited_spans_for_node(&node.id)?.len(), 1);
        Ok(())
    }

    #[test]
    fn stats_report_token_totals_and_active_kind_counts() -> MemoryResult<()> {
        let store = SqliteMemoryStore::in_memory()?;
        let span = CitedSpan {
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            trace_id: "trace".to_string(),
            span_id: "span".to_string(),
            seq: 1,
        };
        let (fact, _) = store.upsert_node(
            StoreScope::new("tenant", "project", None),
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL.",
            1,
            std::slice::from_ref(&span),
        )?;
        let (procedure, _) = store.upsert_node(
            StoreScope::new("tenant", "project", None),
            MemoryNodeKind::Procedure,
            "Run migrations before restarting checkout workers.",
            2,
            &[span],
        )?;
        let (future, _) = store.upsert_node(
            StoreScope::new("tenant", "project", None),
            MemoryNodeKind::Fact,
            "Future checkout token is active later.",
            now_unix_ms() + 1_000_000,
            &[],
        )?;
        assert!(store.invalidate_node(&fact.id, 3, Some(30))?);
        assert!(store.invalidate_node(&procedure.id, now_unix_ms() + 1_000_000, Some(31))?);

        let stats = store.stats()?;

        assert_eq!(stats.nodes, 3);
        assert_eq!(stats.active_nodes, 1);
        assert_eq!(stats.active_fact_nodes, 0);
        assert_eq!(stats.active_procedure_nodes, 1);
        assert_eq!(
            stats.active_node_tokens,
            i64::from(procedure.token_estimate)
        );
        assert_eq!(
            stats.total_node_tokens,
            i64::from(fact.token_estimate)
                + i64::from(procedure.token_estimate)
                + i64::from(future.token_estimate)
        );
        Ok(())
    }

    #[test]
    fn store_stats_deserializes_without_memory_economics_fields() {
        let stats: StoreStats = serde_json::from_value(serde_json::json!({
            "ledger_events": 1,
            "pending_events": 0,
            "nodes": 2,
            "active_nodes": 2,
            "edges": 1,
            "audit_events": 0
        }))
        .unwrap_or_else(|err| panic!("{err}"));

        assert_eq!(stats.total_node_tokens, 0);
        assert_eq!(stats.active_node_tokens, 0);
        assert_eq!(stats.active_fact_nodes, 0);
        assert_eq!(stats.active_procedure_nodes, 0);
    }

    #[test]
    fn restating_invalidated_memory_creates_new_temporal_version() -> MemoryResult<()> {
        let store = SqliteMemoryStore::in_memory()?;
        let span = CitedSpan {
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            trace_id: "trace".to_string(),
            span_id: "span".to_string(),
            seq: 1,
        };
        let (old, created) = store.upsert_node(
            StoreScope::new("tenant", "project", None),
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL.",
            1_000,
            std::slice::from_ref(&span),
        )?;
        assert!(created);
        assert!(store.invalidate_node(&old.id, 2_000, None)?);

        let (new, second_created) = store.upsert_node(
            StoreScope::new("tenant", "project", None),
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL.",
            3_000,
            &[span],
        )?;

        assert!(second_created);
        assert_ne!(old.id, new.id);
        assert_eq!(
            store.node_by_id(&old.id)?.unwrap().valid_to_unix_ms,
            Some(2_000)
        );
        assert_eq!(new.valid_from_unix_ms, 3_000);
        assert_eq!(new.valid_to_unix_ms, None);
        assert_eq!(store.stats()?.nodes, 2);
        Ok(())
    }

    #[test]
    fn equal_time_invalidation_can_replace_later_close_event() -> MemoryResult<()> {
        let store = SqliteMemoryStore::in_memory()?;
        let span = CitedSpan {
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            trace_id: "trace".to_string(),
            span_id: "span".to_string(),
            seq: 1,
        };
        let (node, _) = store.upsert_node(
            StoreScope::new("tenant", "project", None),
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL.",
            1_000,
            &[span],
        )?;

        assert!(store.invalidate_node(&node.id, 3_000, Some(30))?);
        assert!(store.invalidate_node(&node.id, 3_000, Some(20))?);
        assert!(!store.invalidate_node(&node.id, 3_000, Some(40))?);
        let node = store.node_by_id(&node.id)?.unwrap();
        assert_eq!(node.valid_to_unix_ms, Some(3_000));
        assert_eq!(node.valid_to_event_id, Some(20));
        Ok(())
    }

    #[test]
    fn observed_by_queries_filter_future_rows_before_limits() -> MemoryResult<()> {
        let store = SqliteMemoryStore::in_memory()?;
        let old_span = CitedSpan {
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            trace_id: "old".to_string(),
            span_id: "span".to_string(),
            seq: 1,
        };
        let future_span = CitedSpan {
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            trace_id: "future".to_string(),
            span_id: "span".to_string(),
            seq: 1,
        };
        let (old, _) = store.upsert_node(
            StoreScope::new("tenant", "project", None),
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL.",
            1_000,
            &[old_span],
        )?;
        store.upsert_node(
            StoreScope::new("tenant", "project", None),
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL future detail.",
            3_000,
            &[future_span],
        )?;

        let recent = store.recent_nodes_observed_by("tenant", "project", None, 1, Some(1_500))?;
        let seeded = store.seed_nodes_observed_by(
            "tenant",
            "project",
            None,
            &[String::from("checkout")],
            1,
            Some(1_500),
        )?;

        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].id, old.id);
        assert_eq!(seeded.len(), 1);
        assert_eq!(seeded[0].id, old.id);
        Ok(())
    }

    #[test]
    fn observed_by_kind_filter_applies_before_limits() -> MemoryResult<()> {
        let store = SqliteMemoryStore::in_memory()?;
        for index in 0..3 {
            let span = CitedSpan {
                tenant_id: "tenant".to_string(),
                project_id: "project".to_string(),
                trace_id: format!("fact-{index}"),
                span_id: "span".to_string(),
                seq: 1,
            };
            store.upsert_node(
                StoreScope::new("tenant", "project", None),
                MemoryNodeKind::Fact,
                &format!("Deploy noisy fact {index}."),
                2_000 + i64::from(index),
                &[span],
            )?;
        }
        let procedure_span = CitedSpan {
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            trace_id: "procedure".to_string(),
            span_id: "span".to_string(),
            seq: 1,
        };
        let (procedure, _) = store.upsert_node(
            StoreScope::new("tenant", "project", None),
            MemoryNodeKind::Procedure,
            "Deploy procedure: run migrations.",
            1_000,
            &[procedure_span],
        )?;

        let unfiltered = store.recent_nodes_observed_by("tenant", "project", None, 1, None)?;
        let filtered = store.recent_nodes_observed_by_kinds(
            "tenant",
            "project",
            None,
            1,
            None,
            &[MemoryNodeKind::Procedure],
        )?;

        assert_eq!(unfiltered.len(), 1);
        assert_eq!(unfiltered[0].kind, MemoryNodeKind::Fact);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, procedure.id);
        Ok(())
    }

    #[test]
    fn seed_kind_filter_applies_before_limits() -> MemoryResult<()> {
        let store = SqliteMemoryStore::in_memory()?;
        for index in 0..3 {
            let span = CitedSpan {
                tenant_id: "tenant".to_string(),
                project_id: "project".to_string(),
                trace_id: format!("fact-seed-{index}"),
                span_id: "span".to_string(),
                seq: 1,
            };
            store.upsert_node(
                StoreScope::new("tenant", "project", None),
                MemoryNodeKind::Fact,
                &format!("Deploy seed noisy fact {index}."),
                2_000 + i64::from(index),
                &[span],
            )?;
        }
        let procedure_span = CitedSpan {
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            trace_id: "procedure-seed".to_string(),
            span_id: "span".to_string(),
            seq: 1,
        };
        let (procedure, _) = store.upsert_node(
            StoreScope::new("tenant", "project", None),
            MemoryNodeKind::Procedure,
            "Deploy seed procedure: run migrations.",
            1_000,
            &[procedure_span],
        )?;

        let filtered = store.seed_nodes_observed_by_kinds(
            StoreScope::new("tenant", "project", None),
            &[String::from("deploy")],
            1,
            None,
            &[MemoryNodeKind::Procedure],
        )?;

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, procedure.id);
        Ok(())
    }

    #[test]
    fn projection_neighbors_respect_same_timestamp_ledger_order() -> MemoryResult<()> {
        let store = SqliteMemoryStore::in_memory()?;
        for label in ["earlier", "current", "later"] {
            let mut event = LedgerEvent::direct_memory_write(
                "tenant",
                "project",
                MemoryNodeKind::Fact,
                format!("{label} token alpha"),
            );
            event.trace_id = label.to_string();
            event.span_id = "span".to_string();
            event.observed_at_unix_ms = 3_000;
            event.ingested_at_unix_ms = 3_000;
            store.append_event(&event)?;
        }
        let events = store.pending_events(10)?;
        for event in &events {
            store.upsert_node(
                StoreScope::new("tenant", "project", None),
                MemoryNodeKind::Fact,
                &event.text,
                event.observed_at_unix_ms,
                &[event.cited_span()],
            )?;
        }
        let later = events
            .iter()
            .find(|event| event.trace_id == "later")
            .unwrap();
        store.upsert_node(
            StoreScope::new("tenant", "project", None),
            MemoryNodeKind::Fact,
            "earlier token alpha",
            3_000,
            &[later.cited_span()],
        )?;
        let earlier_node = store
            .all_nodes("tenant", "project", None)?
            .into_iter()
            .find(|node| node.text == "earlier token alpha")
            .unwrap();
        assert_eq!(earlier_node.observation_count, 1);
        let current = events
            .iter()
            .find(|event| event.trace_id == "current")
            .unwrap();
        let neighbors = store.projection_neighbors_for_event(current, "token alpha", 10)?;
        let texts: Vec<_> = neighbors.iter().map(|node| node.text.as_str()).collect();

        assert!(texts.contains(&"earlier token alpha"));
        assert!(!texts.contains(&"current token alpha"));
        assert!(!texts.contains(&"later token alpha"));
        Ok(())
    }

    #[test]
    fn projection_neighbors_respect_same_timestamp_closure_order() -> MemoryResult<()> {
        let store = SqliteMemoryStore::in_memory()?;
        let mut origin = LedgerEvent::direct_memory_write(
            "tenant",
            "project",
            MemoryNodeKind::Fact,
            "origin boundary token alpha",
        );
        origin.trace_id = "origin".to_string();
        origin.span_id = "span".to_string();
        origin.observed_at_unix_ms = 1_000;
        origin.ingested_at_unix_ms = 1_000;
        store.append_event(&origin)?;
        for label in ["earlier", "current", "later"] {
            let mut event = LedgerEvent::direct_memory_write(
                "tenant",
                "project",
                MemoryNodeKind::Fact,
                format!("{label} closer"),
            );
            event.trace_id = label.to_string();
            event.span_id = "span".to_string();
            event.observed_at_unix_ms = 3_000;
            event.ingested_at_unix_ms = 3_000;
            store.append_event(&event)?;
        }
        let events = store.pending_events(10)?;
        let event_for = |trace_id: &str| {
            events
                .iter()
                .find(|event| event.trace_id == trace_id)
                .unwrap()
        };
        let origin = event_for("origin");
        for label in ["earlier", "current", "later"] {
            let target_text = format!("{label} boundary token alpha");
            let closer = event_for(label);
            let (target, _) = store.upsert_node(
                StoreScope::new("tenant", "project", None),
                MemoryNodeKind::Fact,
                &target_text,
                1_000,
                &[origin.cited_span()],
            )?;
            let (closer_node, _) = store.upsert_node(
                StoreScope::new("tenant", "project", None),
                MemoryNodeKind::Fact,
                &closer.text,
                3_000,
                &[closer.cited_span()],
            )?;
            store.invalidate_node(&target.id, 3_000, closer.id)?;
            store.insert_edge(
                StoreScope::new("tenant", "project", None),
                &closer_node.id,
                &target.id,
                MemoryEdgeKind::Contradicts,
                0.8,
                3_000,
            )?;
        }

        let current = event_for("current");
        let neighbors =
            store.projection_neighbors_for_event(current, "boundary token alpha", 10)?;
        let texts: Vec<_> = neighbors.iter().map(|node| node.text.as_str()).collect();

        assert!(!texts.contains(&"earlier boundary token alpha"));
        assert!(texts.contains(&"current boundary token alpha"));
        assert!(texts.contains(&"later boundary token alpha"));
        Ok(())
    }

    #[test]
    fn late_older_version_closes_at_next_projected_version() -> MemoryResult<()> {
        let store = SqliteMemoryStore::in_memory()?;
        let future_span = CitedSpan {
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            trace_id: "future".to_string(),
            span_id: "span".to_string(),
            seq: 1,
        };
        let older_span = CitedSpan {
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            trace_id: "older".to_string(),
            span_id: "span".to_string(),
            seq: 1,
        };
        let (future, _) = store.upsert_node(
            StoreScope::new("tenant", "project", None),
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL.",
            3_000,
            &[future_span],
        )?;
        let (older, created) = store.upsert_node(
            StoreScope::new("tenant", "project", None),
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL.",
            1_000,
            &[older_span],
        )?;

        assert!(created);
        assert_ne!(future.id, older.id);
        assert_eq!(older.valid_to_unix_ms, Some(3_000));
        assert!(older.is_active_at(Some(1_500)));
        assert!(!older.is_active_at(Some(3_500)));
        assert!(future.is_active_at(Some(3_500)));
        Ok(())
    }

    #[test]
    fn health_reports_schema_and_integrity() -> MemoryResult<()> {
        let store = SqliteMemoryStore::in_memory()?;
        let health = store.health()?;

        assert_eq!(health.application_id, SQLITE_APPLICATION_ID);
        assert_eq!(health.expected_application_id, SQLITE_APPLICATION_ID);
        assert_eq!(health.schema_version, SCHEMA_VERSION);
        assert_eq!(health.expected_schema_version, SCHEMA_VERSION);
        assert!(health.integrity_ok);
        assert_eq!(health.foreign_key_violations, 0);
        assert!(health.graph_integrity_ok);
        assert!(health.graph_integrity.is_clean());
        Ok(())
    }

    #[test]
    fn new_database_is_stamped_with_beater_memory_identity() -> MemoryResult<()> {
        let path = temp_path("identity.db");
        let store = SqliteMemoryStore::open(&path)?;

        assert_eq!(store.application_id()?, SQLITE_APPLICATION_ID);
        assert_eq!(store.schema_version()?, SCHEMA_VERSION);

        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn version_one_database_migrates_after_identity_check() -> MemoryResult<()> {
        let path = temp_path("v1.db");
        {
            let conn = Connection::open(&path)?;
            set_application_id(&conn)?;
            conn.execute_batch(
                "
                CREATE TABLE ledger_events(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    environment_id TEXT,
                    trace_id TEXT NOT NULL,
                    span_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    span_kind TEXT NOT NULL,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    text TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    observed_at_unix_ms INTEGER NOT NULL,
                    ingested_at_unix_ms INTEGER NOT NULL,
                    projected_at_unix_ms INTEGER,
                    UNIQUE(tenant_id, project_id, trace_id, span_id, seq)
                );
                INSERT INTO ledger_events(
                    source, tenant_id, project_id, trace_id, span_id, seq, span_kind,
                    name, status, text, payload_json, observed_at_unix_ms, ingested_at_unix_ms
                ) VALUES (
                    'test', 'tenant', 'project', 'trace', 'span', 1, 'memory.write',
                    'fact', 'ok', 'Existing v1 data survives migration.', '{}', 1, 1
                );
                PRAGMA user_version = 1;
                ",
            )?;
        }

        let store = SqliteMemoryStore::open(&path)?;

        assert_eq!(store.application_id()?, SQLITE_APPLICATION_ID);
        assert_eq!(store.schema_version()?, SCHEMA_VERSION);
        assert_eq!(store.stats()?.ledger_events, 1);
        assert_eq!(store.stats()?.audit_events, 0);

        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn foreign_sqlite_database_is_rejected_without_restamping() -> MemoryResult<()> {
        let path = temp_path("foreign.db");
        {
            let conn = Connection::open(&path)?;
            conn.execute_batch(
                "
                CREATE TABLE foreign_data(id INTEGER PRIMARY KEY, value TEXT NOT NULL);
                INSERT INTO foreign_data(value) VALUES ('do not touch');
                ",
            )?;
        }

        let err = match SqliteMemoryStore::open(&path) {
            Ok(_) => panic!("foreign database should not open as beater-memory"),
            Err(err) => err,
        };
        let conn = Connection::open(&path)?;
        let application_id: i64 = conn.query_row("PRAGMA application_id", [], |row| row.get(0))?;
        let value: String =
            conn.query_row("SELECT value FROM foreign_data WHERE id = 1", [], |row| {
                row.get(0)
            })?;

        assert!(err.to_string().contains("refusing to initialize"));
        assert_eq!(application_id, 0);
        assert_eq!(value, "do not touch");

        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn maintenance_reports_checkpoint_state() -> MemoryResult<()> {
        let store = SqliteMemoryStore::in_memory()?;
        let report = store.maintenance(false)?;

        assert!(report.optimized);
        assert!(!report.vacuumed);
        assert!(!report.repaired_orphans);
        assert!(report.graph_integrity_before.is_clean());
        assert!(report.graph_integrity_after.is_clean());
        assert!(report.wal_checkpoint_busy >= 0);
        Ok(())
    }

    #[test]
    fn graph_integrity_reports_and_repairs_orphan_projection_rows() -> MemoryResult<()> {
        let store = SqliteMemoryStore::in_memory()?;
        insert_orphan_projection_rows(&store)?;

        let health = store.health()?;
        assert!(!health.graph_integrity_ok);
        assert_eq!(health.graph_integrity.orphan_edges_from, 2);
        assert_eq!(health.graph_integrity.orphan_edges_to, 2);
        assert_eq!(health.graph_integrity.orphan_node_spans, 1);
        assert_eq!(health.graph_integrity.orphan_cue_index_entries, 1);

        let dry_run = store.maintenance_with_options(MaintenanceOptions {
            vacuum: false,
            repair_orphans: false,
            ..MaintenanceOptions::default()
        })?;
        assert!(!dry_run.repaired_orphans);
        assert!(!dry_run.graph_integrity_after.is_clean());

        let repaired = store.maintenance_with_options(MaintenanceOptions {
            vacuum: false,
            repair_orphans: true,
            ..MaintenanceOptions::default()
        })?;

        assert!(repaired.repaired_orphans);
        assert!(!repaired.graph_integrity_before.is_clean());
        assert!(repaired.graph_integrity_after.is_clean());
        assert_eq!(repaired.graph_repair.memory_edges_removed, 2);
        assert_eq!(repaired.graph_repair.node_spans_removed, 1);
        assert_eq!(repaired.graph_repair.cue_index_entries_removed, 1);
        assert!(store.health()?.graph_integrity_ok);
        Ok(())
    }

    #[test]
    fn backup_and_restore_round_trip_memory_data() -> MemoryResult<()> {
        let store = SqliteMemoryStore::in_memory()?;
        let event = LedgerEvent::direct_memory_write(
            "tenant",
            "project",
            MemoryNodeKind::Fact,
            "The support API health route is /api/health.",
        );
        store.append_event(&event)?;
        let span = event.cited_span();
        store.upsert_node(
            StoreScope::new("tenant", "project", None),
            MemoryNodeKind::Fact,
            "The support API health route is /api/health.",
            event.observed_at_unix_ms,
            &[span],
        )?;

        let backup_path = temp_path("backup.db");
        let backup = store.backup_to(&backup_path)?;
        assert!(backup.bytes > 0);
        assert!(backup.integrity_ok);

        let mut restored = SqliteMemoryStore::in_memory()?;
        let report = restored.restore_from(&backup_path)?;

        assert!(report.integrity_ok);
        assert_eq!(report.stats.ledger_events, 1);
        assert_eq!(report.stats.nodes, 1);
        let _ = std::fs::remove_file(backup_path);
        Ok(())
    }

    #[test]
    fn restore_rejects_foreign_sqlite_before_replacing_active_data() -> MemoryResult<()> {
        let mut store = SqliteMemoryStore::in_memory()?;
        let event = LedgerEvent::direct_memory_write(
            "tenant",
            "project",
            MemoryNodeKind::Fact,
            "This active memory must survive failed restore.",
        );
        store.append_event(&event)?;
        let foreign_path = temp_path("foreign-restore.db");
        {
            let conn = Connection::open(&foreign_path)?;
            conn.execute_batch(
                "
                CREATE TABLE foreign_data(id INTEGER PRIMARY KEY, value TEXT NOT NULL);
                INSERT INTO foreign_data(value) VALUES ('not a memory backup');
                ",
            )?;
        }

        let err = match store.restore_from(&foreign_path) {
            Ok(_) => panic!("foreign database should not restore as beater-memory"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("refusing to initialize"));
        assert_eq!(store.stats()?.ledger_events, 1);

        let _ = std::fs::remove_file(foreign_path);
        Ok(())
    }

    #[test]
    fn audit_events_are_durable_and_recent_first() -> MemoryResult<()> {
        let store = SqliteMemoryStore::in_memory()?;
        store.append_audit(&AuditRecord {
            actor: "bearer:test".to_string(),
            action: "remember".to_string(),
            outcome: "success".to_string(),
            route: Some("/v1/remember".to_string()),
            status_code: Some(200),
            detail: serde_json::json!({"projected": true}),
        })?;
        store.append_audit(&AuditRecord {
            actor: "bearer:test".to_string(),
            action: "query".to_string(),
            outcome: "failure".to_string(),
            route: Some("/v1/query".to_string()),
            status_code: Some(400),
            detail: serde_json::json!({"code": "bad_request"}),
        })?;

        let events = store.recent_audit_events(10)?;

        assert_eq!(store.stats()?.audit_events, 2);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].action, "query");
        assert_eq!(events[0].detail["code"], "bad_request");
        Ok(())
    }

    #[test]
    fn append_audit_rejects_malformed_records() -> MemoryResult<()> {
        let store = SqliteMemoryStore::in_memory()?;

        assert_invalid_audit(
            &store,
            |record| record.actor = " bearer:test".to_string(),
            "audit actor",
        )?;
        assert_invalid_audit(
            &store,
            |record| record.action = String::new(),
            "audit action",
        )?;
        assert_invalid_audit(
            &store,
            |record| record.outcome = "success ".to_string(),
            "audit outcome",
        )?;
        assert_invalid_audit(
            &store,
            |record| record.route = Some(" ".to_string()),
            "audit route",
        )?;
        assert_invalid_audit(
            &store,
            |record| record.status_code = Some(99),
            "status_code",
        )?;

        assert_eq!(store.stats()?.audit_events, 0);
        Ok(())
    }

    #[test]
    fn audit_pruning_removes_old_rows_and_retains_newest() -> MemoryResult<()> {
        let store = SqliteMemoryStore::in_memory()?;
        insert_audit_event(&store, 1_000, "oldest")?;
        insert_audit_event(&store, 2_000, "old")?;
        insert_audit_event(&store, 3_000, "new")?;
        insert_audit_event(&store, 4_000, "newest")?;

        let cutoff_report = store.prune_audit_events(Some(2_500), None)?;

        assert_eq!(cutoff_report.audit_events_removed, 2);
        assert_eq!(store.stats()?.audit_events, 2);
        let events = store.recent_audit_events(10)?;
        assert_eq!(events[0].action, "newest");
        assert_eq!(events[1].action, "new");

        let retain_report = store.prune_audit_events(None, Some(1))?;

        assert_eq!(retain_report.audit_events_removed, 1);
        assert_eq!(store.stats()?.audit_events, 1);
        let events = store.recent_audit_events(10)?;
        assert_eq!(events[0].action, "newest");

        let clear_report = store.prune_audit_events(None, Some(0))?;

        assert_eq!(clear_report.audit_events_removed, 1);
        assert_eq!(store.stats()?.audit_events, 0);
        Ok(())
    }

    #[test]
    fn audit_pruning_rejects_invalid_retention_before_deleting_rows() -> MemoryResult<()> {
        let store = SqliteMemoryStore::in_memory()?;
        insert_audit_event(&store, 1_000, "oldest")?;
        insert_audit_event(&store, 2_000, "newest")?;

        let err = store.prune_audit_events(Some(-1), Some(0)).unwrap_err();

        assert!(err.to_string().contains("prune_audit_before_unix_ms"));
        assert_eq!(store.stats()?.audit_events, 2);

        let err = store
            .maintenance_with_options(MaintenanceOptions {
                prune_audit_before_unix_ms: Some(-1),
                retain_latest_audit_events: Some(0),
                ..MaintenanceOptions::default()
            })
            .unwrap_err();

        assert!(err.to_string().contains("prune_audit_before_unix_ms"));
        assert_eq!(store.stats()?.audit_events, 2);
        Ok(())
    }

    #[test]
    fn maintenance_reports_audit_pruning() -> MemoryResult<()> {
        let store = SqliteMemoryStore::in_memory()?;
        insert_audit_event(&store, 1_000, "oldest")?;
        insert_audit_event(&store, 2_000, "old")?;
        insert_audit_event(&store, 3_000, "new")?;
        insert_audit_event(&store, 4_000, "newest")?;

        let report = store.maintenance_with_options(MaintenanceOptions {
            vacuum: false,
            repair_orphans: false,
            prune_audit_before_unix_ms: Some(2_500),
            retain_latest_audit_events: Some(1),
        })?;

        assert!(report.pruned_audit_events);
        assert_eq!(report.audit_prune.audit_events_removed, 3);
        assert_eq!(store.stats()?.audit_events, 1);
        let events = store.recent_audit_events(10)?;
        assert_eq!(events[0].action, "newest");
        Ok(())
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "beater-memory-{}-{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
            name
        ))
    }

    fn assert_invalid_event(
        store: &SqliteMemoryStore,
        edit: impl FnOnce(&mut LedgerEvent),
        expected_message: &str,
    ) -> MemoryResult<()> {
        let mut event = LedgerEvent::direct_memory_write(
            "tenant",
            "project",
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL.",
        );
        edit(&mut event);
        let err = store.append_event(&event).unwrap_err();
        assert!(
            err.to_string().contains(expected_message),
            "expected error containing {expected_message:?}, got {err}"
        );
        Ok(())
    }

    fn assert_invalid_audit(
        store: &SqliteMemoryStore,
        edit: impl FnOnce(&mut AuditRecord),
        expected_message: &str,
    ) -> MemoryResult<()> {
        let mut record = AuditRecord {
            actor: "bearer:test".to_string(),
            action: "remember".to_string(),
            outcome: "success".to_string(),
            route: Some("/v1/remember".to_string()),
            status_code: Some(200),
            detail: serde_json::json!({}),
        };
        edit(&mut record);

        let err = store.append_audit(&record).unwrap_err();

        assert!(
            err.to_string().contains(expected_message),
            "expected error containing {expected_message:?}, got {err}"
        );
        Ok(())
    }

    fn oversized_sqlite_limit() -> Option<usize> {
        usize::try_from(i64::MAX)
            .ok()
            .and_then(|max| max.checked_add(1))
    }

    fn insert_audit_event(
        store: &SqliteMemoryStore,
        occurred_at_unix_ms: i64,
        action: &str,
    ) -> MemoryResult<i64> {
        store.connection().execute(
            "
            INSERT INTO audit_events(
                occurred_at_unix_ms, actor, action, outcome, route, status_code, detail_json
            ) VALUES (?1, 'test-actor', ?2, 'success', '/test', 200, '{}')
            ",
            params![occurred_at_unix_ms, action],
        )?;
        Ok(store.connection().last_insert_rowid())
    }

    fn insert_orphan_projection_rows(store: &SqliteMemoryStore) -> MemoryResult<()> {
        store.connection().execute_batch(
            "
            INSERT INTO memory_edges(
                tenant_id, project_id, from_node_id, to_node_id, kind, weight, created_at_unix_ms
            ) VALUES
                ('tenant', 'project', 'missing-from', 'missing-to', 'mentions', 1.0, 1),
                ('tenant', 'project', 'missing-from-2', 'missing-to-2', 'derived_from', 1.0, 1);
            INSERT INTO node_spans(node_id, tenant_id, project_id, trace_id, span_id, seq)
            VALUES ('missing-node', 'tenant', 'project', 'trace', 'span', 1);
            INSERT INTO cue_index(term, node_id, tenant_id, project_id, weight)
            VALUES ('missing', 'missing-node', 'tenant', 'project', 1.0);
            ",
        )?;
        Ok(())
    }
}
