use crate::{
    distill::{Distiller, HeuristicDistiller},
    error::{MemoryError, MemoryResult},
    graph::answer_query,
    model::{
        ActivationWeights, BeliefRevisionOp, DistilledMemory, MemoryAnswer, MemoryEdgeKind,
        MemoryNodeKind, MemoryQuery,
    },
    reconstruct::{ActiveReconstructor, DeterministicReconstructor},
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
pub struct MemoryEngine<D = HeuristicDistiller, R = DeterministicReconstructor> {
    store: SqliteMemoryStore,
    distiller: D,
    reconstructor: R,
    activation_weights: ActivationWeights,
}

impl MemoryEngine<HeuristicDistiller, DeterministicReconstructor> {
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
        Self::new_with_reconstructor(store, distiller, DeterministicReconstructor)
    }
}

impl<D: Distiller, R: ActiveReconstructor> MemoryEngine<D, R> {
    #[must_use]
    pub fn new_with_reconstructor(
        store: SqliteMemoryStore,
        distiller: D,
        reconstructor: R,
    ) -> Self {
        Self {
            store,
            distiller,
            reconstructor,
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
                    event.observed_at_unix_ms,
                )?;
                let memories = self.distiller.distill(&event, &neighbors);
                let mut projected_nodes = Vec::new();
                for memory in memories {
                    memory.validate().map_err(|err| {
                        MemoryError::invalid(format!(
                            "invalid distilled memory for event trace_id={} span_id={} seq={}: {err}",
                            event.trace_id, event.span_id, event.seq
                        ))
                    })?;
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
        answer_query(
            &self.store,
            query,
            self.activation_weights,
            &self.reconstructor,
        )
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
                self.apply_projected_invalidations_for_late_node(event, &node)?;
                if created {
                    Ok(ApplyOutcome::Added(node))
                } else {
                    Ok(ApplyOutcome::Updated(node))
                }
            }
            BeliefRevisionOp::Invalidate => {
                let replacement = self.replacement_for_invalidation(event, &memory)?;
                let targets = self.invalidation_targets(&memory, neighbors, replacement.as_ref());
                for target in targets {
                    if self.store.invalidate_node(
                        &target.id,
                        event.observed_at_unix_ms,
                        event.id,
                    )? && let Some(newer) = replacement.as_ref()
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
                            event.observed_at_unix_ms,
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
                            event.observed_at_unix_ms,
                        )?;
                    }
                }
                if let Some(node) = replacement.as_ref() {
                    self.apply_projected_invalidations_for_late_node(event, node)?;
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
            .filter(|node| node.kind == memory.node_kind)
            .filter(|node| replacement.map(|newer| newer.id.as_str()) != Some(node.id.as_str()))
            .filter(|node| overlap_score(&memory.text, &node.text) >= 0.12)
            .take(3)
            .cloned()
            .collect()
    }

    fn replacement_for_invalidation(
        &self,
        event: &LedgerEvent,
        memory: &DistilledMemory,
    ) -> MemoryResult<Option<MemoryNode>> {
        if memory.text.trim().is_empty() {
            return Ok(None);
        }
        let scope = StoreScope::new(
            &event.tenant_id,
            &event.project_id,
            event.environment_id.as_deref(),
        );
        let (node, created) = if let Some(node) = self.store.node_version_by_text(
            scope,
            memory.node_kind,
            &memory.text,
            event.observed_at_unix_ms,
        )? {
            (node, false)
        } else {
            self.store.upsert_node(
                scope,
                memory.node_kind,
                &memory.text,
                event.observed_at_unix_ms,
                &memory.cited_spans,
            )?
        };
        if created {
            self.ensure_entity_cues(event, &node)?;
        }
        Ok(Some(node))
    }

    fn apply_projected_invalidations_for_late_node(
        &self,
        event: &LedgerEvent,
        node: &MemoryNode,
    ) -> MemoryResult<()> {
        if matches!(
            node.kind,
            MemoryNodeKind::Episode | MemoryNodeKind::EntityCue
        ) {
            return Ok(());
        }
        let projected_events = self.store.projected_events_after(
            StoreScope::new(
                &event.tenant_id,
                &event.project_id,
                event.environment_id.as_deref(),
            ),
            event,
            node.valid_to_unix_ms,
            node.valid_to_event_id,
        )?;
        for projected_event in projected_events {
            let neighbors = self.store.projection_neighbors_for_event(
                &projected_event,
                &projected_event.text,
                24,
            )?;
            if !neighbors.iter().any(|candidate| candidate.id == node.id) {
                continue;
            }

            let memories = self.distiller.distill(&projected_event, &neighbors);
            for memory in memories {
                memory.validate().map_err(|err| {
                    MemoryError::invalid(format!(
                        "invalid late distilled memory for event trace_id={} span_id={} seq={}: {err}",
                        projected_event.trace_id, projected_event.span_id, projected_event.seq
                    ))
                })?;
                if memory.op != BeliefRevisionOp::Invalidate {
                    continue;
                }
                let targets = self.invalidation_targets(&memory, &neighbors, None);
                if !targets.iter().any(|target| target.id == node.id) {
                    continue;
                }
                if self.apply_late_invalidation(&projected_event, memory, node)? {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    fn apply_late_invalidation(
        &self,
        event: &LedgerEvent,
        memory: DistilledMemory,
        target: &MemoryNode,
    ) -> MemoryResult<bool> {
        let replacement = self.replacement_for_invalidation(event, &memory)?;
        let invalidated =
            self.store
                .invalidate_node(&target.id, event.observed_at_unix_ms, event.id)?;
        if invalidated && let Some(newer) = replacement.as_ref() {
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
                event.observed_at_unix_ms,
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
                event.observed_at_unix_ms,
            )?;
        }
        if let Some(node) = replacement.as_ref() {
            self.apply_projected_invalidations_for_late_node(event, node)?;
        }
        Ok(invalidated)
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
                event.observed_at_unix_ms,
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
                    event.observed_at_unix_ms,
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
                event.observed_at_unix_ms,
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
    use crate::{
        model::{
            MemoryMode, MemoryScope, MemoryTier, ReconstructionMode, ReconstructionOptions,
            ReconstructionReason, RoutingReason,
        },
        reconstruct::{ReconstructionDecision, ReconstructionStep},
    };

    use super::*;

    #[derive(Clone)]
    struct BadReconstructor;

    impl ActiveReconstructor for BadReconstructor {
        fn decide(&self, _step: &ReconstructionStep) -> ReconstructionDecision {
            ReconstructionDecision::Accept {
                node_id: "missing-node".to_string(),
            }
        }
    }

    #[derive(Clone)]
    struct BadPruneReconstructor;

    impl ActiveReconstructor for BadPruneReconstructor {
        fn decide(&self, _step: &ReconstructionStep) -> ReconstructionDecision {
            ReconstructionDecision::Prune {
                node_id: "missing-node".to_string(),
            }
        }
    }

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
    fn default_query_routes_to_procedural_substore() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        for index in 0..20 {
            engine.ingest_event(&event_at(
                MemoryNodeKind::Fact,
                &format!("Deploy workflow noisy fact {index}."),
                1_000 + index,
            ))?;
        }
        engine.ingest_event(&event_at(
            MemoryNodeKind::Procedure,
            "Deploy workflow steps: run migrations then restart workers.",
            2_000,
        ))?;
        engine.project_pending(100)?;

        let answer = engine.query(&MemoryQuery::new(
            "deploy workflow steps",
            MemoryScope::new("tenant", "project"),
        ))?;

        let routing = answer.routing.as_ref().expect("routing report");
        assert_eq!(routing.reason, RoutingReason::ProceduralIntent);
        assert_eq!(routing.routed_modes, vec![MemoryMode::Procedural]);
        assert!(
            answer
                .evidence
                .iter()
                .any(|item| item.kind == MemoryNodeKind::Procedure)
        );
        assert!(
            !answer
                .evidence
                .iter()
                .any(|item| item.kind == MemoryNodeKind::Fact)
        );
        Ok(())
    }

    #[test]
    fn explicit_query_modes_constrain_routing() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Deploy workflow fact uses the release checklist.",
            1_000,
        ))?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Procedure,
            "Deploy workflow steps: run migrations then restart workers.",
            2_000,
        ))?;
        engine.project_pending(100)?;

        let answer = engine.query(
            &MemoryQuery::new(
                "deploy workflow steps",
                MemoryScope::new("tenant", "project"),
            )
            .with_modes(vec![MemoryMode::Semantic]),
        )?;

        let routing = answer.routing.as_ref().expect("routing report");
        assert_eq!(routing.routed_modes, vec![MemoryMode::Semantic]);
        assert!(
            answer
                .evidence
                .iter()
                .any(|item| item.kind == MemoryNodeKind::Fact)
        );
        assert!(
            !answer
                .evidence
                .iter()
                .any(|item| item.kind == MemoryNodeKind::Procedure)
        );
        Ok(())
    }

    #[test]
    fn explicit_all_query_modes_are_not_narrowed() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Deploy workflow fact uses the release checklist.",
            1_000,
        ))?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Procedure,
            "Deploy workflow steps: run migrations then restart workers.",
            2_000,
        ))?;
        engine.project_pending(100)?;

        let modes = MemoryNodeKind::default_modes();
        let answer = engine.query(
            &MemoryQuery::new(
                "deploy workflow steps",
                MemoryScope::new("tenant", "project"),
            )
            .with_modes(modes.clone()),
        )?;

        let routing = answer.routing.as_ref().expect("routing report");
        assert_eq!(routing.routed_modes, modes);
        assert_eq!(routing.reason, RoutingReason::AmbiguousFallback);
        assert!(
            answer
                .evidence
                .iter()
                .any(|item| item.kind == MemoryNodeKind::Fact)
        );
        assert!(
            answer
                .evidence
                .iter()
                .any(|item| item.kind == MemoryNodeKind::Procedure)
        );
        Ok(())
    }

    #[test]
    fn forced_reconstruction_can_expand_allowed_modes_outside_route() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Procedure,
            "Deploy workflow steps: restart checkout workers.",
            1_000,
        ))?;
        engine.project_pending(100)?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Checkout workers require credential beta.",
            1_001,
        ))?;
        engine.project_pending(100)?;

        let nodes = engine.store().all_nodes("tenant", "project", None)?;
        let procedure = nodes
            .iter()
            .find(|node| node.kind == MemoryNodeKind::Procedure)
            .expect("procedure exists");
        let fact = nodes
            .iter()
            .find(|node| {
                node.kind == MemoryNodeKind::Fact
                    && node.text == "Checkout workers require credential beta."
            })
            .expect("fact exists");
        engine.store().insert_edge(
            StoreScope::new("tenant", "project", None),
            &procedure.id,
            &fact.id,
            MemoryEdgeKind::Fixes,
            1.0,
            1_001,
        )?;

        let answer = engine.query(
            &MemoryQuery::new(
                "deploy workflow steps",
                MemoryScope::new("tenant", "project"),
            )
            .with_reconstruction(ReconstructionOptions::force()),
        )?;

        let routing = answer.routing.as_ref().expect("routing report");
        let reconstruction = answer
            .reconstruction
            .as_ref()
            .expect("reconstruction report");
        assert_eq!(routing.routed_modes, vec![MemoryMode::Procedural]);
        assert_eq!(
            routing.reconstruction_modes,
            Some(MemoryNodeKind::default_modes())
        );
        assert!(reconstruction.accepted_node_ids.contains(&fact.id));
        assert!(answer.evidence.iter().any(|item| item.node_id == fact.id));
        Ok(())
    }

    #[test]
    fn forced_reconstruction_expands_linked_evidence() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?.with_activation_weights(ActivationWeights {
            ppr: 0.0,
            base_level: 0.0,
            edge_type: 0.0,
            freshness: 0.0,
        });
        let (_source, target) = linked_reconstruction_fixture(&engine, 1_000, 1_001, 1_001)?;

        let answer = engine.query(
            &MemoryQuery::new("incident alpha", MemoryScope::new("tenant", "project"))
                .with_modes(vec![MemoryMode::Semantic])
                .with_reconstruction(ReconstructionOptions::force()),
        )?;

        let reconstruction = answer
            .reconstruction
            .as_ref()
            .expect("forced query should run reconstruction");
        assert_eq!(answer.tier_used, MemoryTier::ActiveReconstruction);
        assert_eq!(reconstruction.reason, ReconstructionReason::Forced);
        assert!(reconstruction.accepted_node_ids.contains(&target.id));
        assert!(answer.evidence.iter().any(|item| item.node_id == target.id));
        Ok(())
    }

    #[test]
    fn auto_reconstruction_escalates_compositional_queries() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        linked_reconstruction_fixture(&engine, 1_000, 1_001, 1_001)?;

        let answer = engine.query(
            &MemoryQuery::new(
                "why was incident alpha fixed",
                MemoryScope::new("tenant", "project"),
            )
            .with_modes(vec![MemoryMode::Semantic])
            .with_reconstruction(ReconstructionOptions {
                mode: ReconstructionMode::Auto,
                ..ReconstructionOptions::default()
            }),
        )?;

        assert_eq!(answer.tier_used, MemoryTier::ActiveReconstruction);
        assert_eq!(
            answer.reconstruction.as_ref().map(|report| report.reason),
            Some(ReconstructionReason::CompositionalQuery)
        );
        Ok(())
    }

    #[test]
    fn reconstruction_respects_as_of_visibility() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        let (_source, future_target) = linked_reconstruction_fixture(&engine, 1_000, 3_000, 3_000)?;

        let answer = engine.query(
            &MemoryQuery::new(
                "incident alpha",
                MemoryScope::new("tenant", "project").as_of_unix_ms(1_500),
            )
            .with_modes(vec![MemoryMode::Semantic])
            .with_reconstruction(ReconstructionOptions::force()),
        )?;

        assert_eq!(answer.tier_used, MemoryTier::ActiveReconstruction);
        assert!(
            !answer
                .evidence
                .iter()
                .any(|item| item.node_id == future_target.id)
        );
        assert!(
            !answer
                .reconstruction
                .as_ref()
                .expect("forced query should run reconstruction")
                .accepted_node_ids
                .contains(&future_target.id)
        );
        Ok(())
    }

    #[test]
    fn invalid_reconstructor_decisions_are_ignored() -> MemoryResult<()> {
        let engine = MemoryEngine::new_with_reconstructor(
            SqliteMemoryStore::in_memory()?,
            HeuristicDistiller::default(),
            BadReconstructor,
        )
        .with_activation_weights(ActivationWeights {
            ppr: 0.0,
            base_level: 0.0,
            edge_type: 0.0,
            freshness: 0.0,
        });
        let (_source, target) = linked_reconstruction_fixture(&engine, 1_000, 1_001, 1_001)?;

        let answer = engine.query(
            &MemoryQuery::new("incident alpha", MemoryScope::new("tenant", "project"))
                .with_modes(vec![MemoryMode::Semantic])
                .with_reconstruction(ReconstructionOptions::force()),
        )?;

        let reconstruction = answer
            .reconstruction
            .as_ref()
            .expect("forced query should run reconstruction");
        assert_eq!(answer.tier_used, MemoryTier::ActiveReconstruction);
        assert!(!answer.evidence.iter().any(|item| item.node_id == target.id));
        assert!(!reconstruction.accepted_node_ids.contains(&target.id));
        assert!(
            !reconstruction
                .pruned_node_ids
                .iter()
                .any(|node_id| node_id == "missing-node")
        );
        Ok(())
    }

    #[test]
    fn invalid_prune_decisions_are_ignored() -> MemoryResult<()> {
        let engine = MemoryEngine::new_with_reconstructor(
            SqliteMemoryStore::in_memory()?,
            HeuristicDistiller::default(),
            BadPruneReconstructor,
        )
        .with_activation_weights(ActivationWeights {
            ppr: 0.0,
            base_level: 0.0,
            edge_type: 0.0,
            freshness: 0.0,
        });
        linked_reconstruction_fixture(&engine, 1_000, 1_001, 1_001)?;

        let answer = engine.query(
            &MemoryQuery::new("incident alpha", MemoryScope::new("tenant", "project"))
                .with_modes(vec![MemoryMode::Semantic])
                .with_reconstruction(ReconstructionOptions::force()),
        )?;

        let reconstruction = answer
            .reconstruction
            .as_ref()
            .expect("forced query should run reconstruction");
        assert!(
            !reconstruction
                .pruned_node_ids
                .iter()
                .any(|node_id| node_id == "missing-node")
        );
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
    fn project_pending_rejects_invalid_distilled_memory_and_rolls_back() -> MemoryResult<()> {
        #[derive(Clone)]
        struct InvalidDistiller;

        impl Distiller for InvalidDistiller {
            fn distill(
                &self,
                event: &LedgerEvent,
                _neighbors: &[MemoryNode],
            ) -> Vec<DistilledMemory> {
                vec![
                    DistilledMemory::add(
                        MemoryNodeKind::Fact,
                        "Checkout uses DATABASE_URL.",
                        event.cited_span(),
                    ),
                    DistilledMemory::add(MemoryNodeKind::Fact, " ", event.cited_span()),
                ]
            }
        }

        let engine = MemoryEngine::new(SqliteMemoryStore::in_memory()?, InvalidDistiller);
        engine.ingest_event(&LedgerEvent::direct_memory_write(
            "tenant",
            "project",
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL.",
        ))?;

        let err = engine.project_pending(100).unwrap_err();
        let stats = engine.store().stats()?;

        assert!(err.to_string().contains("invalid distilled memory"));
        assert_eq!(stats.pending_events, 1);
        assert_eq!(stats.nodes, 0);
        assert_eq!(stats.edges, 0);
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

    #[test]
    fn as_of_query_uses_fact_validity_windows() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Use the legacy checkout token for deploys.",
            1_000,
        ))?;
        engine.project_pending(100)?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Do not use the legacy checkout token; it is deprecated. Use the scoped deploy token.",
            2_000,
        ))?;
        engine.project_pending(100)?;

        let before = engine.query(
            &MemoryQuery::new(
                "legacy checkout token",
                MemoryScope::new("tenant", "project").as_of_unix_ms(1_500),
            )
            .with_modes(vec![MemoryMode::Semantic]),
        )?;

        assert!(before.evidence.iter().any(|item| {
            item.text
                .contains("Use the legacy checkout token for deploys.")
        }));
        assert!(
            !before
                .evidence
                .iter()
                .any(|item| item.text.contains("scoped deploy token"))
        );
        assert!(before.contradictions.is_empty());
        assert!(before.stale_assumptions.is_empty());

        let after = engine.query(
            &MemoryQuery::new(
                "legacy checkout token",
                MemoryScope::new("tenant", "project").as_of_unix_ms(2_500),
            )
            .with_modes(vec![MemoryMode::Semantic]),
        )?;

        assert!(
            after
                .evidence
                .iter()
                .any(|item| item.text.contains("scoped deploy token"))
        );
        assert!(!after.evidence.iter().any(|item| {
            item.text
                .contains("Use the legacy checkout token for deploys.")
        }));
        assert!(!after.contradictions.is_empty());
        assert!(!after.stale_assumptions.is_empty());
        Ok(())
    }

    #[test]
    fn as_of_query_does_not_resurrect_restatement_before_its_valid_time() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Use the legacy checkout token for deploys.",
            1_000,
        ))?;
        engine.project_pending(100)?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Do not use the legacy checkout token; it is deprecated. Use the scoped deploy token.",
            2_000,
        ))?;
        engine.project_pending(100)?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Use the legacy checkout token for deploys.",
            3_000,
        ))?;
        engine.project_pending(100)?;

        let during_invalidation = engine.query(
            &MemoryQuery::new(
                "legacy checkout token",
                MemoryScope::new("tenant", "project").as_of_unix_ms(2_500),
            )
            .with_modes(vec![MemoryMode::Semantic]),
        )?;

        assert!(
            during_invalidation
                .evidence
                .iter()
                .any(|item| item.text.contains("scoped deploy token"))
        );
        assert!(!during_invalidation.evidence.iter().any(|item| {
            item.text
                .contains("Use the legacy checkout token for deploys.")
        }));
        assert!(!during_invalidation.stale_assumptions.is_empty());
        Ok(())
    }

    #[test]
    fn late_older_invalidation_does_not_invalidate_future_fact() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Use prod deploy key alpha.",
            2_000,
        ))?;
        engine.project_pending(100)?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Do not use prod deploy key alpha; it is deprecated.",
            1_000,
        ))?;
        engine.project_pending(100)?;

        let future_fact = engine
            .store()
            .all_nodes("tenant", "project", None)?
            .into_iter()
            .find(|node| node.text == "Use prod deploy key alpha.")
            .expect("future fact should exist");

        assert_eq!(future_fact.valid_from_unix_ms, 2_000);
        assert_eq!(future_fact.valid_to_unix_ms, None);
        Ok(())
    }

    #[test]
    fn invalidation_target_lookup_filters_future_matches_before_limit() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Use crowd token alpha.",
            1_000,
        ))?;
        engine.project_pending(100)?;
        for index in 0..40 {
            engine.ingest_event(&event_at(
                MemoryNodeKind::Fact,
                &format!("Use crowd token alpha future-{index}."),
                3_000 + index,
            ))?;
        }
        engine.project_pending(100)?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Do not use crowd token alpha; it is deprecated. Use scoped crowd token alpha.",
            2_000,
        ))?;
        engine.project_pending(100)?;

        let answer = engine.query(
            &MemoryQuery::new(
                "crowd token alpha",
                MemoryScope::new("tenant", "project").as_of_unix_ms(2_500),
            )
            .with_modes(vec![MemoryMode::Semantic]),
        )?;

        assert!(
            answer
                .evidence
                .iter()
                .any(|item| item.text.contains("scoped crowd token alpha"))
        );
        assert!(
            !answer
                .evidence
                .iter()
                .any(|item| item.text == "Use crowd token alpha.")
        );
        assert!(!answer.contradictions.is_empty());
        assert!(!answer.stale_assumptions.is_empty());
        Ok(())
    }

    #[test]
    fn target_only_invalidation_surfaces_stale_assumption() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Legacy target-only token remains active.",
            1_000,
        ))?;
        engine.project_pending(100)?;
        let target = engine
            .store()
            .all_nodes("tenant", "project", None)?
            .into_iter()
            .find(|node| {
                node.kind == MemoryNodeKind::Fact
                    && node.text == "Legacy target-only token remains active."
            })
            .expect("fact should exist");
        assert!(engine.store().invalidate_node(&target.id, 2_000, None)?);

        let answer = engine.query(
            &MemoryQuery::new(
                "legacy target-only token",
                MemoryScope::new("tenant", "project").as_of_unix_ms(2_500),
            )
            .with_modes(vec![MemoryMode::Semantic]),
        )?;

        assert!(answer.evidence.is_empty());
        assert_eq!(answer.stale_assumptions.len(), 1);
        assert_eq!(answer.stale_assumptions[0].node_id, target.id);
        Ok(())
    }

    #[test]
    fn as_of_query_filters_future_citations_on_active_memory() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL.",
            1_000,
        ))?;
        engine.project_pending(100)?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL.",
            3_000,
        ))?;
        engine.project_pending(100)?;

        let answer = engine.query(
            &MemoryQuery::new(
                "checkout database_url",
                MemoryScope::new("tenant", "project").as_of_unix_ms(1_500),
            )
            .with_modes(vec![MemoryMode::Semantic]),
        )?;

        assert!(!answer.evidence.is_empty());
        assert!(
            answer
                .cited_spans
                .iter()
                .all(|span| span.trace_id == "trace-1000")
        );
        assert!(
            !answer
                .cited_spans
                .iter()
                .any(|span| span.trace_id == "trace-3000")
        );
        Ok(())
    }

    #[test]
    fn as_of_query_score_is_stable_after_future_restatement() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL.",
            1_000,
        ))?;
        engine.project_pending(100)?;
        let before = engine.query(
            &MemoryQuery::new(
                "checkout database_url",
                MemoryScope::new("tenant", "project").as_of_unix_ms(1_500),
            )
            .with_modes(vec![MemoryMode::Semantic]),
        )?;
        let before_score = before.evidence[0].score;

        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL.",
            3_000,
        ))?;
        engine.project_pending(100)?;
        let after = engine.query(
            &MemoryQuery::new(
                "checkout database_url",
                MemoryScope::new("tenant", "project").as_of_unix_ms(1_500),
            )
            .with_modes(vec![MemoryMode::Semantic]),
        )?;

        assert_eq!(after.evidence[0].node_id, before.evidence[0].node_id);
        assert!((after.evidence[0].score - before_score).abs() < f32::EPSILON);
        Ok(())
    }

    #[test]
    fn as_of_query_score_is_stable_before_future_invalidation() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL.",
            1_000,
        ))?;
        engine.project_pending(100)?;
        let before = engine.query(
            &MemoryQuery::new(
                "checkout database_url",
                MemoryScope::new("tenant", "project").as_of_unix_ms(1_500),
            )
            .with_modes(vec![MemoryMode::Semantic]),
        )?;
        let before_score = before.evidence[0].score;

        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Do not use Checkout uses DATABASE_URL; it is deprecated.",
            900_000_000_000,
        ))?;
        engine.project_pending(100)?;
        let after = engine.query(
            &MemoryQuery::new(
                "checkout database_url",
                MemoryScope::new("tenant", "project").as_of_unix_ms(1_500),
            )
            .with_modes(vec![MemoryMode::Semantic]),
        )?;

        assert_eq!(after.evidence[0].node_id, before.evidence[0].node_id);
        assert!((after.evidence[0].score - before_score).abs() < f32::EPSILON);
        assert!(after.contradictions.is_empty());
        assert!(after.stale_assumptions.is_empty());
        Ok(())
    }

    #[test]
    fn current_query_hides_future_invalidation_edges() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL.",
            1_000,
        ))?;
        engine.project_pending(100)?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Do not use Checkout uses DATABASE_URL; it is deprecated.",
            4_000_000_000_000,
        ))?;
        engine.project_pending(100)?;

        let answer = engine.query(
            &MemoryQuery::new(
                "checkout database_url",
                MemoryScope::new("tenant", "project"),
            )
            .with_modes(vec![MemoryMode::Semantic]),
        )?;

        assert!(
            answer
                .evidence
                .iter()
                .any(|item| item.text == "Checkout uses DATABASE_URL.")
        );
        assert!(answer.contradictions.is_empty());
        assert!(answer.stale_assumptions.is_empty());
        Ok(())
    }

    #[test]
    fn current_query_filters_future_citations() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL.",
            1_000,
        ))?;
        engine.project_pending(100)?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL.",
            4_000_000_000_000,
        ))?;
        engine.project_pending(100)?;

        let answer = engine.query(
            &MemoryQuery::new(
                "checkout database_url",
                MemoryScope::new("tenant", "project"),
            )
            .with_modes(vec![MemoryMode::Semantic]),
        )?;

        assert!(!answer.cited_spans.is_empty());
        assert!(
            answer
                .cited_spans
                .iter()
                .all(|span| span.trace_id == "trace-1000")
        );
        assert!(
            !answer
                .cited_spans
                .iter()
                .any(|span| span.trace_id == "trace-4000000000000")
        );
        Ok(())
    }

    #[test]
    fn same_family_rollover_does_not_surface_false_stale_assumption() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL.",
            3_000,
        ))?;
        engine.project_pending(100)?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL.",
            1_000,
        ))?;
        engine.project_pending(100)?;

        let answer = engine.query(
            &MemoryQuery::new(
                "checkout database_url",
                MemoryScope::new("tenant", "project").as_of_unix_ms(3_500),
            )
            .with_modes(vec![MemoryMode::Semantic]),
        )?;

        assert!(!answer.evidence.is_empty());
        assert!(answer.stale_assumptions.is_empty());
        assert!(answer.contradictions.is_empty());
        Ok(())
    }

    #[test]
    fn late_arrival_versions_converge_after_rebuild() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL.",
            3_000,
        ))?;
        engine.project_pending(100)?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL.",
            1_000,
        ))?;
        engine.project_pending(100)?;

        let query = MemoryQuery::new(
            "checkout database_url",
            MemoryScope::new("tenant", "project").as_of_unix_ms(3_500),
        )
        .with_modes(vec![MemoryMode::Semantic]);
        let incremental = engine.query(&query)?;
        engine.rebuild_projection(100, None)?;
        let rebuilt = engine.query(&query)?;

        assert_eq!(evidence_texts(&incremental), evidence_texts(&rebuilt));
        assert_evidence_scores_close(&incremental, &rebuilt);
        assert_eq!(incremental.stale_assumptions, rebuilt.stale_assumptions);
        assert_eq!(incremental.contradictions, rebuilt.contradictions);
        Ok(())
    }

    #[test]
    fn late_arrival_invalidation_target_converges_after_rebuild() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Use crowd token alpha.",
            3_000,
        ))?;
        engine.project_pending(100)?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Do not use crowd token alpha; it is deprecated. Use scoped crowd token alpha.",
            2_000,
        ))?;
        engine.project_pending(100)?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            "Use crowd token alpha.",
            1_000,
        ))?;
        engine.project_pending(100)?;

        let query = MemoryQuery::new(
            "crowd token alpha",
            MemoryScope::new("tenant", "project").as_of_unix_ms(2_500),
        )
        .with_modes(vec![MemoryMode::Semantic]);
        let incremental = engine.query(&query)?;
        engine.rebuild_projection(100, None)?;
        let rebuilt = engine.query(&query)?;

        assert_eq!(evidence_texts(&incremental), evidence_texts(&rebuilt));
        assert_evidence_scores_close(&incremental, &rebuilt);
        assert_eq!(incremental.stale_assumptions, rebuilt.stale_assumptions);
        assert_eq!(incremental.contradictions, rebuilt.contradictions);
        assert!(!rebuilt.stale_assumptions.is_empty());
        assert!(!rebuilt.contradictions.is_empty());
        Ok(())
    }

    #[test]
    fn late_invalidation_replays_replacement_against_later_invalidations() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        let old = "Use crowd token alpha.";
        let scoped =
            "Do not use crowd token alpha; it is deprecated. Use scoped crowd token alpha.";
        let rotated =
            "Do not use scoped crowd token alpha; it is deprecated. Use rotated crowd token alpha.";
        engine.ingest_event(&event_at(MemoryNodeKind::Fact, old, 1_000))?;
        engine.project_pending(100)?;
        engine.ingest_event(&event_at(MemoryNodeKind::Fact, rotated, 3_000))?;
        engine.project_pending(100)?;
        engine.ingest_event(&event_at(MemoryNodeKind::Fact, scoped, 2_000))?;
        engine.project_pending(100)?;

        let before_later_invalidation = engine.query(
            &MemoryQuery::new(
                "scoped crowd token alpha",
                MemoryScope::new("tenant", "project").as_of_unix_ms(2_500),
            )
            .with_modes(vec![MemoryMode::Semantic]),
        )?;
        assert!(
            before_later_invalidation
                .evidence
                .iter()
                .any(|item| item.text.contains(scoped))
        );

        let after_later_invalidation_query = MemoryQuery::new(
            "scoped crowd token alpha",
            MemoryScope::new("tenant", "project").as_of_unix_ms(3_500),
        )
        .with_modes(vec![MemoryMode::Semantic]);
        let incremental = engine.query(&after_later_invalidation_query)?;
        assert!(
            incremental
                .evidence
                .iter()
                .any(|item| item.text.contains(rotated))
        );
        assert!(
            !incremental
                .evidence
                .iter()
                .any(|item| item.text.contains(scoped))
        );

        let nodes = engine.store().all_nodes("tenant", "project", None)?;
        let scoped_node = nodes.iter().find(|node| node.text == scoped).unwrap();
        let rotated_node = nodes.iter().find(|node| node.text == rotated).unwrap();
        assert_eq!(scoped_node.valid_to_unix_ms, Some(3_000));
        assert!(rotated_node.is_active_at(Some(3_500)));
        let edges = engine.store().edges_for_scope("tenant", "project", None)?;
        assert!(edges.iter().any(|edge| {
            edge.from_node_id == rotated_node.id
                && edge.to_node_id == scoped_node.id
                && edge.kind == MemoryEdgeKind::Contradicts
                && edge.created_at_unix_ms == 3_000
        }));
        assert!(edges.iter().any(|edge| {
            edge.from_node_id == rotated_node.id
                && edge.to_node_id == scoped_node.id
                && edge.kind == MemoryEdgeKind::Supersedes
                && edge.created_at_unix_ms == 3_000
        }));

        engine.rebuild_projection(100, None)?;
        let rebuilt = engine.query(&after_later_invalidation_query)?;
        assert_eq!(evidence_texts(&incremental), evidence_texts(&rebuilt));
        assert_evidence_scores_close(&incremental, &rebuilt);
        Ok(())
    }

    #[test]
    fn late_replay_uses_full_pre_event_neighbors_for_target_choice() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        let better_target = "Use premium beta token for deploys.";
        let weak_late_node = "Use beta token.";
        let future_invalidation = "Do not use premium beta token for deploys; it is deprecated. Use gamma token for deploys.";
        engine.ingest_event(&event_at(MemoryNodeKind::Fact, better_target, 1_000))?;
        engine.project_pending(100)?;
        engine.ingest_event(&event_at(MemoryNodeKind::Fact, future_invalidation, 3_000))?;
        engine.project_pending(100)?;
        engine.ingest_event(&event_at(MemoryNodeKind::Fact, weak_late_node, 2_000))?;
        engine.project_pending(100)?;

        let incremental_nodes = engine.store().all_nodes("tenant", "project", None)?;
        assert_eq!(
            node_valid_to(&incremental_nodes, better_target),
            Some(Some(3_000))
        );
        assert_eq!(
            node_valid_to(&incremental_nodes, weak_late_node),
            Some(None)
        );

        engine.rebuild_projection(100, None)?;
        let rebuilt_nodes = engine.store().all_nodes("tenant", "project", None)?;
        assert_eq!(node_valid_to(&rebuilt_nodes, weak_late_node), Some(None));
        Ok(())
    }

    fn linked_reconstruction_fixture<D, R>(
        engine: &MemoryEngine<D, R>,
        source_observed_at_unix_ms: i64,
        target_observed_at_unix_ms: i64,
        edge_observed_at_unix_ms: i64,
    ) -> MemoryResult<(MemoryNode, MemoryNode)>
    where
        D: Distiller,
        R: ActiveReconstructor,
    {
        let source_text = "Incident alpha blocked deploys.";
        let target_text = "Credential beta rotation restored deploys.";
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            source_text,
            source_observed_at_unix_ms,
        ))?;
        engine.project_pending(100)?;
        engine.ingest_event(&event_at(
            MemoryNodeKind::Fact,
            target_text,
            target_observed_at_unix_ms,
        ))?;
        engine.project_pending(100)?;

        let nodes = engine.store().all_nodes("tenant", "project", None)?;
        let source = nodes
            .iter()
            .find(|node| node.text == source_text && node.kind == MemoryNodeKind::Fact)
            .cloned()
            .expect("source memory should exist");
        let target = nodes
            .iter()
            .find(|node| node.text == target_text && node.kind == MemoryNodeKind::Fact)
            .cloned()
            .expect("target memory should exist");
        engine.store().insert_edge(
            StoreScope::new("tenant", "project", None),
            &source.id,
            &target.id,
            MemoryEdgeKind::Fixes,
            1.0,
            edge_observed_at_unix_ms,
        )?;
        Ok((source, target))
    }

    fn event_at(kind: MemoryNodeKind, text: &str, observed_at_unix_ms: i64) -> LedgerEvent {
        let mut event = LedgerEvent::direct_memory_write("tenant", "project", kind, text);
        event.trace_id = format!("trace-{observed_at_unix_ms}");
        event.span_id = format!("span-{observed_at_unix_ms}");
        event.observed_at_unix_ms = observed_at_unix_ms;
        event.ingested_at_unix_ms = observed_at_unix_ms;
        event
    }

    fn evidence_texts(answer: &MemoryAnswer) -> Vec<String> {
        answer
            .evidence
            .iter()
            .map(|item| item.text.clone())
            .collect()
    }

    fn evidence_scores(answer: &MemoryAnswer) -> Vec<u32> {
        answer
            .evidence
            .iter()
            .map(|item| (item.score * 1_000_000.0).round() as u32)
            .collect()
    }

    fn assert_evidence_scores_close(left: &MemoryAnswer, right: &MemoryAnswer) {
        let left = evidence_scores(left);
        let right = evidence_scores(right);
        assert_eq!(left.len(), right.len());
        for (left, right) in left.iter().zip(right.iter()) {
            assert!(
                left.abs_diff(*right) <= 1,
                "evidence score mismatch: {left} vs {right}"
            );
        }
    }

    fn node_valid_to(nodes: &[MemoryNode], text: &str) -> Option<Option<i64>> {
        nodes
            .iter()
            .find(|node| node.text == text)
            .map(|node| node.valid_to_unix_ms)
    }
}
