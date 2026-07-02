use crate::{
    distill::{Distiller, HeuristicDistiller},
    error::{MemoryError, MemoryResult},
    graph::answer_query,
    model::{
        ActivationWeights, BeliefRevisionOp, DistilledMemory, MemoryAnswer, MemoryEdgeKind,
        MemoryNodeKind, MemoryQuery,
    },
    store::{LedgerEvent, MemoryNode, ProjectionResetReport, SqliteMemoryStore, StoreScope},
    text::{now_unix_ms, overlap_score, top_terms},
};
use serde::{Deserialize, Serialize};

/// Result of one projection pass.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectReport {
    pub events_seen: usize,
    pub events_projected: usize,
    pub events_skipped: usize,
    pub memories_added: usize,
    pub memories_updated: usize,
    pub memories_invalidated: usize,
    pub memories_nooped: usize,
    pub edges_added: usize,
}

/// Result of resetting derived projections and replaying the ledger.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectionRebuildReport {
    pub reset: ProjectionResetReport,
    pub project: ProjectReport,
    pub batch_size: usize,
    pub max_events: Option<usize>,
    pub completed: bool,
}

/// Memory engine facade: ledger import, projection, and answer-shaped reads.
pub struct MemoryEngine<D = HeuristicDistiller> {
    store: SqliteMemoryStore,
    distiller: D,
    activation_weights: ActivationWeights,
}

impl MemoryEngine<HeuristicDistiller> {
    pub fn open(path: impl AsRef<std::path::Path>) -> MemoryResult<Self> {
        Ok(Self::new(
            SqliteMemoryStore::open(path)?,
            HeuristicDistiller::default(),
        ))
    }

    pub fn in_memory() -> MemoryResult<Self> {
        Ok(Self::new(
            SqliteMemoryStore::in_memory()?,
            HeuristicDistiller::default(),
        ))
    }
}

impl<D: Distiller> MemoryEngine<D> {
    #[must_use]
    pub fn new(store: SqliteMemoryStore, distiller: D) -> Self {
        Self {
            store,
            distiller,
            activation_weights: ActivationWeights::default(),
        }
    }

    #[must_use]
    pub fn with_activation_weights(mut self, activation_weights: ActivationWeights) -> Self {
        self.activation_weights = activation_weights;
        self
    }

    #[must_use]
    pub fn store(&self) -> &SqliteMemoryStore {
        &self.store
    }

    pub fn ingest_event(&self, event: &LedgerEvent) -> MemoryResult<bool> {
        self.store.append_event(event)
    }

    pub fn project_pending(&self, limit: usize) -> MemoryResult<ProjectReport> {
        if limit == 0 {
            return Err(MemoryError::invalid("project limit must be greater than 0"));
        }
        let events = self.store.pending_events(limit)?;
        let mut report = ProjectReport::default();
        for event in events {
            let event_report = self.store.with_immediate_transaction(|store| {
                let mut event_report = ProjectReport {
                    events_seen: 1,
                    ..ProjectReport::default()
                };
                let Some(event_id) = event.id else {
                    event_report.events_skipped = 1;
                    return Ok(event_report);
                };
                if !store.event_is_pending(event_id)? {
                    event_report.events_skipped = 1;
                    return Ok(event_report);
                }

                let neighbors = store.active_neighbors(
                    &event.tenant_id,
                    &event.project_id,
                    event.environment_id.as_deref(),
                    &event.text,
                    24,
                )?;
                let memories = self.distiller.distill(&event, &neighbors);
                let mut projected_nodes = Vec::new();
                for memory in memories {
                    match self.apply_distilled(&event, memory, &neighbors)? {
                        ApplyOutcome::Added(node) => {
                            event_report.memories_added += 1;
                            projected_nodes.push(node);
                        }
                        ApplyOutcome::Updated(node) => {
                            event_report.memories_updated += 1;
                            projected_nodes.push(node);
                        }
                        ApplyOutcome::Invalidated { replacement } => {
                            event_report.memories_invalidated += 1;
                            if let Some(node) = replacement {
                                projected_nodes.push(node);
                            }
                        }
                        ApplyOutcome::Noop => event_report.memories_nooped += 1,
                    }
                }
                event_report.edges_added += self.link_projected_nodes(&event, &projected_nodes)?;
                store.mark_projected(event_id, now_unix_ms())?;
                event_report.events_projected = 1;
                Ok(event_report)
            })?;
            report.absorb(event_report);
        }
        Ok(report)
    }

    pub fn rebuild_projection(
        &self,
        batch_size: usize,
        max_events: Option<usize>,
    ) -> MemoryResult<ProjectionRebuildReport> {
        if batch_size == 0 {
            return Err(MemoryError::invalid("batch_size must be greater than 0"));
        }
        if max_events.is_some_and(|max_events| max_events == 0) {
            return Err(MemoryError::invalid("max_events must be greater than 0"));
        }
        let reset = self.store.reset_projection()?;
        let mut project = ProjectReport::default();
        let mut remaining = max_events.unwrap_or(usize::MAX);
        let completed;
        loop {
            if remaining == 0 {
                completed = self.store.stats()?.pending_events == 0;
                break;
            }
            let limit = batch_size.min(remaining);
            let batch = self.project_pending(limit)?;
            let projected = batch.events_projected;
            project.absorb(batch);
            if projected == 0 {
                completed = true;
                break;
            }
            remaining = remaining.saturating_sub(projected);
        }
        Ok(ProjectionRebuildReport {
            reset,
            project,
            batch_size,
            max_events,
            completed,
        })
    }

    pub fn query(&self, query: &MemoryQuery) -> MemoryResult<MemoryAnswer> {
        query.validate()?;
        answer_query(&self.store, query, self.activation_weights)
    }

    fn apply_distilled(
        &self,
        event: &LedgerEvent,
        memory: DistilledMemory,
        neighbors: &[MemoryNode],
    ) -> MemoryResult<ApplyOutcome> {
        match memory.op {
            BeliefRevisionOp::Noop => Ok(ApplyOutcome::Noop),
            BeliefRevisionOp::Add | BeliefRevisionOp::Update => {
                let (node, created) = self.store.upsert_node(
                    StoreScope::new(
                        &event.tenant_id,
                        &event.project_id,
                        event.environment_id.as_deref(),
                    ),
                    memory.node_kind,
                    &memory.text,
                    event.observed_at_unix_ms,
                    &memory.cited_spans,
                )?;
                self.ensure_entity_cues(event, &node)?;
                if created {
                    Ok(ApplyOutcome::Added(node))
                } else {
                    Ok(ApplyOutcome::Updated(node))
                }
            }
            BeliefRevisionOp::Invalidate => {
                let replacement = if !memory.text.trim().is_empty() {
                    let (node, _created) = self.store.upsert_node(
                        StoreScope::new(
                            &event.tenant_id,
                            &event.project_id,
                            event.environment_id.as_deref(),
                        ),
                        memory.node_kind,
                        &memory.text,
                        event.observed_at_unix_ms,
                        &memory.cited_spans,
                    )?;
                    self.ensure_entity_cues(event, &node)?;
                    Some(node)
                } else {
                    None
                };
                let targets = self.invalidation_targets(&memory, neighbors, replacement.as_ref());
                for target in targets {
                    if self
                        .store
                        .invalidate_node(&target.id, event.observed_at_unix_ms)?
                        && let Some(newer) = replacement.as_ref()
                    {
                        self.store.insert_edge(
                            StoreScope::new(
                                &event.tenant_id,
                                &event.project_id,
                                event.environment_id.as_deref(),
                            ),
                            &newer.id,
                            &target.id,
                            MemoryEdgeKind::Supersedes,
                            1.0,
                        )?;
                        self.store.insert_edge(
                            StoreScope::new(
                                &event.tenant_id,
                                &event.project_id,
                                event.environment_id.as_deref(),
                            ),
                            &newer.id,
                            &target.id,
                            MemoryEdgeKind::Contradicts,
                            0.8,
                        )?;
                    }
                }
                Ok(ApplyOutcome::Invalidated { replacement })
            }
        }
    }

    fn invalidation_targets(
        &self,
        memory: &DistilledMemory,
        neighbors: &[MemoryNode],
        replacement: Option<&MemoryNode>,
    ) -> Vec<MemoryNode> {
        if let Some(target_id) = memory.target_node_id.as_deref() {
            return neighbors
                .iter()
                .filter(|node| node.id == target_id)
                .cloned()
                .collect();
        }
        neighbors
            .iter()
            .filter(|node| replacement.map(|newer| newer.id.as_str()) != Some(node.id.as_str()))
            .filter(|node| overlap_score(&memory.text, &node.text) >= 0.12)
            .take(3)
            .cloned()
            .collect()
    }

    fn link_projected_nodes(
        &self,
        event: &LedgerEvent,
        projected_nodes: &[MemoryNode],
    ) -> MemoryResult<usize> {
        let mut edges = 0;
        for pair in projected_nodes.windows(2) {
            let from = &pair[0];
            let to = &pair[1];
            if self.store.insert_edge(
                StoreScope::new(
                    &event.tenant_id,
                    &event.project_id,
                    event.environment_id.as_deref(),
                ),
                &to.id,
                &from.id,
                MemoryEdgeKind::DerivedFrom,
                0.8,
            )? {
                edges += 1;
            }
            if from.kind == MemoryNodeKind::Episode
                && self.store.insert_edge(
                    StoreScope::new(
                        &event.tenant_id,
                        &event.project_id,
                        event.environment_id.as_deref(),
                    ),
                    &to.id,
                    &from.id,
                    MemoryEdgeKind::ObservedIn,
                    0.9,
                )?
            {
                edges += 1;
            }
        }
        Ok(edges)
    }

    fn ensure_entity_cues(&self, event: &LedgerEvent, node: &MemoryNode) -> MemoryResult<()> {
        let cited_span = event.cited_span();
        for cue in top_terms(&node.text, 8) {
            let (cue_node, _) = self.store.upsert_node(
                StoreScope::new(
                    &event.tenant_id,
                    &event.project_id,
                    event.environment_id.as_deref(),
                ),
                MemoryNodeKind::EntityCue,
                &cue,
                event.observed_at_unix_ms,
                std::slice::from_ref(&cited_span),
            )?;
            self.store.insert_edge(
                StoreScope::new(
                    &event.tenant_id,
                    &event.project_id,
                    event.environment_id.as_deref(),
                ),
                &node.id,
                &cue_node.id,
                MemoryEdgeKind::Mentions,
                0.55,
            )?;
        }
        Ok(())
    }
}

impl ProjectReport {
    fn absorb(&mut self, other: Self) {
        self.events_seen += other.events_seen;
        self.events_projected += other.events_projected;
        self.events_skipped += other.events_skipped;
        self.memories_added += other.memories_added;
        self.memories_updated += other.memories_updated;
        self.memories_invalidated += other.memories_invalidated;
        self.memories_nooped += other.memories_nooped;
        self.edges_added += other.edges_added;
    }
}

enum ApplyOutcome {
    Added(MemoryNode),
    Updated(MemoryNode),
    Invalidated { replacement: Option<MemoryNode> },
    Noop,
}

#[cfg(test)]
mod tests {
    use crate::model::{MemoryScope, MemoryTier};

    use super::*;

    #[test]
    fn engine_projects_and_answers_e2e() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        let event = LedgerEvent::direct_memory_write(
            "tenant",
            "project",
            MemoryNodeKind::Gotcha,
            "Checkout route fails with 500 when DATABASE_URL is missing. Fix by setting DATABASE_URL.",
        );
        engine.ingest_event(&event)?;

        let report = engine.project_pending(100)?;
        assert_eq!(report.events_seen, 1);
        assert_eq!(report.events_projected, 1);
        assert_eq!(report.events_skipped, 0);
        assert!(report.memories_added >= 2);

        let answer = engine.query(&MemoryQuery::new(
            "How do we fix checkout database failures?",
            MemoryScope::new("tenant", "project"),
        ))?;

        assert_eq!(answer.tier_used, MemoryTier::Activation);
        assert!(answer.answer.contains("DATABASE_URL"));
        assert!(!answer.evidence.is_empty());
        assert!(!answer.cited_spans.is_empty());
        Ok(())
    }

    #[test]
    fn query_rejects_invalid_request_before_retrieval() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        let query = MemoryQuery::new("what changed?", MemoryScope::new("tenant", "project"))
            .with_max_tokens(0);

        let err = engine.query(&query).unwrap_err();

        assert!(err.to_string().contains("max_tokens"));
        Ok(())
    }

    #[test]
    fn projection_is_idempotent_after_event_is_marked() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        engine.ingest_event(&LedgerEvent::direct_memory_write(
            "tenant",
            "project",
            MemoryNodeKind::Fact,
            "The API health route is /api/health.",
        ))?;

        let first = engine.project_pending(100)?;
        let second = engine.project_pending(100)?;

        assert_eq!(first.events_projected, 1);
        assert_eq!(second.events_seen, 0);
        assert_eq!(engine.store().stats()?.pending_events, 0);
        Ok(())
    }

    #[test]
    fn project_pending_rejects_zero_limit() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;

        let err = engine.project_pending(0).unwrap_err();

        assert!(err.to_string().contains("project limit"));
        Ok(())
    }

    #[test]
    fn rebuild_projection_replays_ledger_and_preserves_audit() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        engine.store().append_audit(&crate::AuditRecord {
            actor: "test".to_string(),
            action: "setup".to_string(),
            outcome: "success".to_string(),
            route: None,
            status_code: None,
            detail: serde_json::json!({}),
        })?;
        engine.ingest_event(&LedgerEvent::direct_memory_write(
            "tenant",
            "project",
            MemoryNodeKind::Fact,
            "The support API health route is /api/health.",
        ))?;
        engine.ingest_event(&LedgerEvent::direct_memory_write(
            "tenant",
            "project",
            MemoryNodeKind::Gotcha,
            "Checkout requires DATABASE_URL before running migrations.",
        ))?;
        engine.project_pending(100)?;
        let before = engine.store().stats()?;
        assert_eq!(before.ledger_events, 2);
        assert_eq!(before.pending_events, 0);
        assert!(before.nodes > 0);

        let report = engine.rebuild_projection(1, None)?;

        let after = engine.store().stats()?;
        assert_eq!(report.reset.ledger_events_reset, 2);
        assert!(report.reset.nodes_removed > 0);
        assert_eq!(report.project.events_projected, 2);
        assert!(report.completed);
        assert_eq!(after.ledger_events, 2);
        assert_eq!(after.audit_events, 1);
        assert_eq!(after.pending_events, 0);
        assert!(after.nodes > 0);
        let answer = engine.query(&MemoryQuery::new(
            "checkout migrations database",
            MemoryScope::new("tenant", "project"),
        ))?;
        assert!(answer.answer.contains("DATABASE_URL"));
        Ok(())
    }

    #[test]
    fn rebuild_projection_rejects_zero_max_events_before_reset() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        engine.ingest_event(&LedgerEvent::direct_memory_write(
            "tenant",
            "project",
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL.",
        ))?;
        engine.project_pending(100)?;
        let before = engine.store().stats()?;

        let err = engine.rebuild_projection(10, Some(0)).unwrap_err();
        let after = engine.store().stats()?;

        assert!(err.to_string().contains("max_events"));
        assert_eq!(after.pending_events, before.pending_events);
        assert_eq!(after.nodes, before.nodes);
        assert_eq!(after.edges, before.edges);
        Ok(())
    }

    #[test]
    fn rebuild_projection_can_stop_after_max_events() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        engine.ingest_event(&LedgerEvent::direct_memory_write(
            "tenant",
            "project",
            MemoryNodeKind::Fact,
            "First deploy step sets DATABASE_URL.",
        ))?;
        engine.ingest_event(&LedgerEvent::direct_memory_write(
            "tenant",
            "project",
            MemoryNodeKind::Fact,
            "Second deploy step runs migrations.",
        ))?;
        engine.project_pending(100)?;

        let report = engine.rebuild_projection(10, Some(1))?;

        assert_eq!(report.project.events_projected, 1);
        assert!(!report.completed);
        assert_eq!(engine.store().stats()?.pending_events, 1);
        Ok(())
    }

    #[test]
    fn invalidation_surfaces_stale_assumptions() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        engine.ingest_event(&LedgerEvent::direct_memory_write(
            "tenant",
            "project",
            MemoryNodeKind::Fact,
            "Use the old checkout token for deploys.",
        ))?;
        engine.project_pending(100)?;
        engine.ingest_event(&LedgerEvent::direct_memory_write(
            "tenant",
            "project",
            MemoryNodeKind::Fact,
            "Do not use the old checkout token; it is deprecated. Use the scoped deploy token.",
        ))?;
        engine.project_pending(100)?;

        let answer = engine.query(&MemoryQuery::new(
            "old checkout token",
            MemoryScope::new("tenant", "project"),
        ))?;

        assert!(!answer.contradictions.is_empty());
        assert!(!answer.stale_assumptions.is_empty());
        Ok(())
    }
}
