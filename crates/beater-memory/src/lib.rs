//! Agent-first memory primitives for Beater.
//!
//! This crate starts with the stable API surface for a memory engine that is a
//! projection over ledgered agent traces. Storage, Tantivy cue seeding, and graph
//! projection can grow behind these types without changing the caller contract.

/// Which memory substores a query is allowed to consult.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryMode {
    Semantic,
    Episodic,
    Procedural,
    Gotcha,
    State,
}

/// The highest retrieval tier used to produce an answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryTier {
    /// Lexical/entity cue seeding only.
    CueSeed,
    /// LLM-free graph activation, e.g. PPR blended with recency/frequency decay.
    Activation,
    /// Budgeted LLM-guided path exploration for hard compositional queries.
    ActiveReconstruction,
}

/// Typed memory node families in the projection graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryNodeKind {
    Episode,
    Fact,
    EntityCue,
    Tag,
    Procedure,
    State,
    Gotcha,
    AntiMemory,
    Topic,
}

/// Typed graph edges used by the memory projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryEdgeKind {
    Mentions,
    CausedBy,
    Fixes,
    Contradicts,
    Supersedes,
    Before,
    After,
    PartOf,
    DerivedFrom,
    Blocks,
    Enables,
    ObservedIn,
}

/// Belief revision operation emitted by the offline distiller.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BeliefRevisionOp {
    Add,
    Update,
    Invalidate,
    Noop,
}

/// Tenant/project/time scope for a memory query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryScope {
    pub tenant_id: String,
    pub project_id: String,
    pub environment_id: Option<String>,
    pub as_of_unix_ms: Option<i64>,
}

impl MemoryScope {
    #[must_use]
    pub fn new(tenant_id: impl Into<String>, project_id: impl Into<String>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            project_id: project_id.into(),
            environment_id: None,
            as_of_unix_ms: None,
        }
    }

    #[must_use]
    pub fn with_environment(mut self, environment_id: impl Into<String>) -> Self {
        self.environment_id = Some(environment_id.into());
        self
    }

    #[must_use]
    pub fn as_of_unix_ms(mut self, as_of_unix_ms: i64) -> Self {
        self.as_of_unix_ms = Some(as_of_unix_ms);
        self
    }
}

/// Answer-shaped memory request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryQuery {
    pub question: String,
    pub scope: MemoryScope,
    pub max_tokens: u32,
    pub require_fresh: bool,
    pub modes: Vec<MemoryMode>,
}

impl MemoryQuery {
    #[must_use]
    pub fn new(question: impl Into<String>, scope: MemoryScope) -> Self {
        Self {
            question: question.into(),
            scope,
            max_tokens: 1_200,
            require_fresh: false,
            modes: vec![
                MemoryMode::Semantic,
                MemoryMode::Episodic,
                MemoryMode::Procedural,
                MemoryMode::Gotcha,
                MemoryMode::State,
            ],
        }
    }

    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    #[must_use]
    pub fn requiring_fresh(mut self) -> Self {
        self.require_fresh = true;
        self
    }

    #[must_use]
    pub fn with_modes(mut self, modes: Vec<MemoryMode>) -> Self {
        self.modes = modes;
        self
    }
}

/// Span provenance for a returned memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CitedSpan {
    pub tenant_id: String,
    pub project_id: String,
    pub trace_id: String,
    pub span_id: String,
    pub seq: u64,
}

/// One compact evidence item that can be placed into model context.
#[derive(Clone, Debug, PartialEq)]
pub struct Evidence {
    pub node_id: String,
    pub kind: MemoryNodeKind,
    pub text: String,
    pub score: f32,
    pub token_estimate: u32,
    pub cited_spans: Vec<CitedSpan>,
}

impl Evidence {
    #[must_use]
    pub fn new(
        node_id: impl Into<String>,
        kind: MemoryNodeKind,
        text: impl Into<String>,
        score: f32,
    ) -> Self {
        let text = text.into();
        let token_estimate = estimate_tokens(&text);
        Self {
            node_id: node_id.into(),
            kind,
            text,
            score,
            token_estimate,
            cited_spans: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_cited_span(mut self, span: CitedSpan) -> Self {
        self.cited_spans.push(span);
        self
    }
}

/// A contradiction surfaced instead of silently collapsed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Contradiction {
    pub older_node_id: String,
    pub newer_node_id: String,
    pub summary: String,
}

/// A premise that may be stale under the query's requested time/freshness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StaleAssumption {
    pub node_id: String,
    pub summary: String,
    pub invalidated_at_unix_ms: Option<i64>,
}

/// Answer returned by the memory engine.
#[derive(Clone, Debug, PartialEq)]
pub struct MemoryAnswer {
    pub answer: String,
    pub evidence: Vec<Evidence>,
    pub cited_spans: Vec<CitedSpan>,
    pub contradictions: Vec<Contradiction>,
    pub stale_assumptions: Vec<StaleAssumption>,
    pub suggested_next_queries: Vec<String>,
    pub token_estimate: u32,
    pub tier_used: MemoryTier,
}

impl MemoryAnswer {
    #[must_use]
    pub fn empty(tier_used: MemoryTier) -> Self {
        Self {
            answer: String::new(),
            evidence: Vec::new(),
            cited_spans: Vec::new(),
            contradictions: Vec::new(),
            stale_assumptions: Vec::new(),
            suggested_next_queries: Vec::new(),
            token_estimate: 0,
            tier_used,
        }
    }
}

/// Output of an offline distillation pass before it is applied to projections.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DistilledMemory {
    pub op: BeliefRevisionOp,
    pub node_kind: MemoryNodeKind,
    pub text: String,
    pub target_node_id: Option<String>,
    pub cited_spans: Vec<CitedSpan>,
}

/// Blend factors for LLM-free activation ranking.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActivationWeights {
    pub ppr: f32,
    pub base_level: f32,
    pub edge_type: f32,
    pub freshness: f32,
}

impl Default for ActivationWeights {
    fn default() -> Self {
        Self {
            ppr: 0.45,
            base_level: 0.30,
            edge_type: 0.15,
            freshness: 0.10,
        }
    }
}

/// Conservative token estimate for budgeting evidence before it reaches a model.
#[must_use]
pub fn estimate_tokens(text: &str) -> u32 {
    let non_ws = text.chars().filter(|ch| !ch.is_whitespace()).count() as u32;
    non_ws.div_ceil(4).max(u32::from(!text.is_empty()))
}

/// Weighted activation score. Inputs are expected to already be normalized.
#[must_use]
pub fn blend_activation(
    ppr: f32,
    base_level: f32,
    edge_type: f32,
    freshness: f32,
    weights: ActivationWeights,
) -> f32 {
    let score = ppr * weights.ppr
        + base_level * weights.base_level
        + edge_type * weights.edge_type
        + freshness * weights.freshness;
    score.clamp(0.0, 1.0)
}

/// Select highest-scoring evidence that fits within a token budget.
#[must_use]
pub fn budget_evidence(mut evidence: Vec<Evidence>, max_tokens: u32) -> Vec<Evidence> {
    evidence.sort_by(|left, right| right.score.total_cmp(&left.score));

    let mut used = 0;
    let mut selected = Vec::new();
    for item in evidence {
        if item.token_estimate > max_tokens {
            continue;
        }
        if used + item.token_estimate > max_tokens {
            continue;
        }
        used += item.token_estimate;
        selected.push(item);
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_defaults_to_all_modes_and_a_budget() {
        let query = MemoryQuery::new("what changed?", MemoryScope::new("tenant", "project"));

        assert_eq!(query.max_tokens, 1_200);
        assert!(query.modes.contains(&MemoryMode::Semantic));
        assert!(query.modes.contains(&MemoryMode::Gotcha));
        assert!(!query.require_fresh);
    }

    #[test]
    fn token_estimate_is_nonzero_for_nonempty_text() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
    }

    #[test]
    fn activation_score_is_bounded() {
        let score = blend_activation(2.0, 2.0, 2.0, 2.0, ActivationWeights::default());

        assert_eq!(score, 1.0);
    }

    #[test]
    fn evidence_budget_prefers_high_score_items_that_fit() {
        let selected = budget_evidence(
            vec![
                Evidence::new("low", MemoryNodeKind::Fact, "tiny", 0.1),
                Evidence::new("big", MemoryNodeKind::Procedure, "x".repeat(80), 1.0),
                Evidence::new("high", MemoryNodeKind::Gotcha, "small", 0.9),
            ],
            3,
        );

        let ids: Vec<&str> = selected.iter().map(|item| item.node_id.as_str()).collect();
        assert_eq!(ids, vec!["high", "low"]);
    }
}
