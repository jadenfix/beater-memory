use std::{path::Path, str::FromStr};

use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};

use crate::{
    error::{MemoryError, MemoryResult},
    model::{CitedSpan, MemoryEdgeKind, MemoryNodeKind, estimate_tokens},
    text::{canonical_key, now_unix_ms, stable_id, terms},
};

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
        let store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    pub fn in_memory() -> MemoryResult<Self> {
        let store = Self {
            conn: Connection::open_in_memory()?,
        };
        store.migrate()?;
        Ok(store)
    }

    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    fn migrate(&self) -> MemoryResult<()> {
        self.conn.execute_batch(
            "
            PRAGMA foreign_keys = ON;
            PRAGMA journal_mode = WAL;

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
            ",
        )?;
        Ok(())
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

    pub fn stats(&self) -> MemoryResult<serde_json::Value> {
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
        Ok(serde_json::json!({
            "ledger_events": ledger_events,
            "pending_events": pending_events,
            "nodes": nodes,
            "active_nodes": active_nodes,
            "edges": edges,
        }))
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

#[cfg(test)]
mod tests {
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
        assert_eq!(
            store.stats()?["ledger_events"],
            serde_json::Value::Number(1.into())
        );
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
}
