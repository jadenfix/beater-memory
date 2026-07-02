use std::{path::Path, str::FromStr, time::Duration};

use rusqlite::{Connection, MAIN_DB, OpenFlags, OptionalExtension, params};
use serde::{Deserialize, Serialize};

use crate::{
    error::{MemoryError, MemoryResult},
    model::{CitedSpan, MemoryEdgeKind, MemoryNodeKind, estimate_tokens},
    text::{canonical_key, now_unix_ms, stable_id, terms},
};

const SCHEMA_VERSION: u32 = 2;
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
pub struct StoreStats {
    pub ledger_events: i64,
    pub pending_events: i64,
    pub nodes: i64,
    pub active_nodes: i64,
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

            PRAGMA user_version = 2;
            ",
        )?;
        Ok(())
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
        let rows = stmt.query_map(params![limit as i64], read_ledger_event)?;
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
    ) -> MemoryResult<Vec<MemoryNode>> {
        let query_terms = terms(text);
        if query_terms.is_empty() {
            return Ok(Vec::new());
        }
        let mut scored = Vec::new();
        for term in query_terms {
            for node in self.seed_nodes(tenant_id, project_id, environment_id, &[term], limit)? {
                if node.valid_to_unix_ms.is_none() {
                    scored.push(node);
                }
            }
        }
        scored.sort_by(|left, right| right.updated_at_unix_ms.cmp(&left.updated_at_unix_ms));
        scored.dedup_by(|left, right| left.id == right.id);
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
        let key = if kind == MemoryNodeKind::Episode {
            cited_spans
                .first()
                .map(|span| format!("episode:{}:{}:{}", span.trace_id, span.span_id, span.seq))
                .unwrap_or_else(|| canonical_key(kind.as_str(), text))
        } else {
            canonical_key(kind.as_str(), text)
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
        let now = now_unix_ms();
        let token_estimate = estimate_tokens(text);
        let existing = self.node_by_scope_key(
            scope.tenant_id,
            scope.project_id,
            scope.environment_id,
            kind,
            &key,
        )?;
        match existing {
            Some(mut node) => {
                let merged_text = merge_node_text(&node.text, text);
                let token_estimate = estimate_tokens(&merged_text);
                self.conn.execute(
                    "
                    UPDATE memory_nodes
                    SET text = ?2,
                        updated_at_unix_ms = ?3,
                        valid_to_unix_ms = NULL,
                        confidence = MIN(confidence + 0.05, 1.0),
                        token_estimate = ?4,
                        observation_count = observation_count + 1
                    WHERE id = ?1
                    ",
                    params![node.id, merged_text, now, token_estimate],
                )?;
                node.text = merged_text;
                node.updated_at_unix_ms = now;
                node.valid_to_unix_ms = None;
                node.confidence = (node.confidence + 0.05).min(1.0);
                node.token_estimate = token_estimate;
                node.observation_count += 1;
                self.attach_spans(&node.id, cited_spans)?;
                self.reindex_node(&node)?;
                Ok((node, false))
            }
            None => {
                self.conn.execute(
                    "
                    INSERT INTO memory_nodes(
                        id, tenant_id, project_id, environment_id, kind, text, canonical_key,
                        created_at_unix_ms, updated_at_unix_ms, valid_from_unix_ms,
                        valid_to_unix_ms, confidence, token_estimate, observation_count
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?8, ?9, NULL, ?10, ?11, 1)
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
                    valid_to_unix_ms: None,
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
                       valid_to_unix_ms, confidence, token_estimate, observation_count
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
                       valid_to_unix_ms, confidence, token_estimate, observation_count
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

    pub fn invalidate_node(&self, node_id: &str, valid_to_unix_ms: i64) -> MemoryResult<bool> {
        let changed = self.conn.execute(
            "
            UPDATE memory_nodes
            SET valid_to_unix_ms = ?2, updated_at_unix_ms = ?2
            WHERE id = ?1 AND valid_to_unix_ms IS NULL
            ",
            params![node_id, valid_to_unix_ms],
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
                now_unix_ms(),
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
        if query_terms.is_empty() {
            return self.recent_nodes(tenant_id, project_id, environment_id, limit);
        }
        let mut nodes = Vec::new();
        let mut stmt = self.conn.prepare(
            "
            SELECT DISTINCT n.id, n.tenant_id, n.project_id, n.environment_id, n.kind,
                   n.text, n.canonical_key, n.created_at_unix_ms, n.updated_at_unix_ms,
                   n.valid_from_unix_ms, n.valid_to_unix_ms, n.confidence,
                   n.token_estimate, n.observation_count
            FROM cue_index c
            JOIN memory_nodes n ON n.id = c.node_id
            WHERE c.tenant_id = ?1
              AND c.project_id = ?2
              AND COALESCE(c.environment_id, '') = COALESCE(?3, '')
              AND c.term = ?4
            ORDER BY c.weight DESC, n.updated_at_unix_ms DESC
            LIMIT ?5
            ",
        )?;
        for term in query_terms {
            let rows = stmt.query_map(
                params![tenant_id, project_id, environment_id, term, limit as i64],
                read_node,
            )?;
            for row in rows {
                nodes.push(row?);
            }
        }
        nodes.sort_by(|left, right| {
            right
                .updated_at_unix_ms
                .cmp(&left.updated_at_unix_ms)
                .then_with(|| left.id.cmp(&right.id))
        });
        nodes.dedup_by(|left, right| left.id == right.id);
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
        let mut stmt = self.conn.prepare(
            "
            SELECT id, tenant_id, project_id, environment_id, kind, text, canonical_key,
                   created_at_unix_ms, updated_at_unix_ms, valid_from_unix_ms,
                   valid_to_unix_ms, confidence, token_estimate, observation_count
            FROM memory_nodes
            WHERE tenant_id = ?1
              AND project_id = ?2
              AND COALESCE(environment_id, '') = COALESCE(?3, '')
            ORDER BY updated_at_unix_ms DESC
            LIMIT ?4
            ",
        )?;
        let rows = stmt.query_map(
            params![tenant_id, project_id, environment_id, limit as i64],
            read_node,
        )?;
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
        let mut stmt = self.conn.prepare(
            "
            SELECT tenant_id, project_id, trace_id, span_id, seq
            FROM node_spans
            WHERE node_id = ?1
            ORDER BY seq
            ",
        )?;
        let rows = stmt.query_map(params![node_id], |row| {
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
        let active_nodes: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM memory_nodes WHERE valid_to_unix_ms IS NULL",
            [],
            |row| row.get(0),
        )?;
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
            edges,
            audit_events,
        })
    }

    pub fn append_audit(&self, record: &AuditRecord) -> MemoryResult<i64> {
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
        let mut stmt = self.conn.prepare(
            "
            SELECT id, occurred_at_unix_ms, actor, action, outcome, route, status_code, detail_json
            FROM audit_events
            ORDER BY occurred_at_unix_ms DESC, id DESC
            LIMIT ?1
            ",
        )?;
        let rows = stmt.query_map(params![limit as i64], read_audit_event)?;
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
        if before_unix_ms.is_none() && retain_latest.is_none() {
            return Ok(AuditPruneReport::default());
        }
        self.with_immediate_transaction(|store| {
            store.prune_audit_events_in_transaction(before_unix_ms, retain_latest)
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
                params![limit as i64],
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
        confidence: row.get(11)?,
        token_estimate: row.get(12)?,
        observation_count: row.get(13)?,
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
            2,
            &[span],
        )?;

        assert!(created);
        assert!(!second_created);
        assert_eq!(node.observation_count, 2);
        assert_eq!(store.cited_spans_for_node(&node.id)?.len(), 1);
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
