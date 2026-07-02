use crate::{
    distill::{Distiller, HeuristicDistiller},
    error::MemoryResult,
    graph::answer_query,
    model::{
        ActivationWeights, BeliefRevisionOp, DistilledMemory, MemoryAnswer, MemoryEdgeKind,
        MemoryNodeKind, MemoryQuery,
    },
    store::{LedgerEvent, MemoryNode, SqliteMemoryStore, StoreScope},
    text::{now_unix_ms, overlap_score, top_terms},
};

/// Result of one projection pass.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProjectReport {
    pub events_seen: usize,
    pub memories_added: usize,
    pub memories_updated: usize,
    pub memories_invalidated: usize,
    pub memories_nooped: usize,
    pub edges_added: usize,
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
        let events = self.store.pending_events(limit)?;
        let mut report = ProjectReport {
            events_seen: events.len(),
            ..ProjectReport::default()
        };
        for event in events {
            let neighbors = self.store.active_neighbors(
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
                        report.memories_added += 1;
                        projected_nodes.push(node);
                    }
                    ApplyOutcome::Updated(node) => {
                        report.memories_updated += 1;
                        projected_nodes.push(node);
                    }
                    ApplyOutcome::Invalidated { replacement } => {
                        report.memories_invalidated += 1;
                        if let Some(node) = replacement {
                            projected_nodes.push(node);
                        }
                    }
                    ApplyOutcome::Noop => report.memories_nooped += 1,
                }
            }
            report.edges_added += self.link_projected_nodes(&event, &projected_nodes)?;
            if let Some(id) = event.id {
                self.store.mark_projected(id, now_unix_ms())?;
            }
        }
        Ok(report)
    }

    pub fn query(&self, query: &MemoryQuery) -> MemoryResult<MemoryAnswer> {
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
