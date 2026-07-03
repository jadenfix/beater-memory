use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};

use crate::error::{MemoryError, MemoryResult};

/// Which memory substores a query is allowed to consult.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryMode {
    Semantic,
    Episodic,
    Procedural,
    Gotcha,
    State,
}

impl MemoryMode {
    #[must_use]
    pub fn accepts(self, kind: MemoryNodeKind) -> bool {
        match self {
            Self::Semantic => matches!(
                kind,
                MemoryNodeKind::Fact
                    | MemoryNodeKind::EntityCue
                    | MemoryNodeKind::Tag
                    | MemoryNodeKind::Topic
            ),
            Self::Episodic => kind == MemoryNodeKind::Episode,
            Self::Procedural => kind == MemoryNodeKind::Procedure,
            Self::Gotcha => matches!(kind, MemoryNodeKind::Gotcha | MemoryNodeKind::AntiMemory),
            Self::State => kind == MemoryNodeKind::State,
        }
    }
}

/// The highest retrieval tier used to produce an answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryTier {
    /// Lexical/entity cue seeding only.
    CueSeed,
    /// LLM-free graph activation, e.g. PPR blended with recency/frequency decay.
    Activation,
    /// Budgeted LLM-guided path exploration for hard compositional queries.
    ActiveReconstruction,
}

/// Read-time active reconstruction policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReconstructionMode {
    #[default]
    Off,
    Auto,
    Force,
}

impl ReconstructionMode {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Auto => "auto",
            Self::Force => "force",
        }
    }
}

/// Why a query did or did not escalate into active reconstruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReconstructionReason {
    Forced,
    EmptyEvidence,
    LowConfidence,
    AmbiguousEvidence,
    CompositionalQuery,
}

/// Why the read router selected a query's effective memory substores.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingReason {
    SemanticIntent,
    EpisodicIntent,
    ProceduralIntent,
    GotchaIntent,
    StateIntent,
    AmbiguousFallback,
    EmptyRouteFallback,
}

/// Diagnostic report for deterministic typed-substore routing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoutingReport {
    pub allowed_modes: Vec<MemoryMode>,
    pub routed_modes: Vec<MemoryMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reconstruction_modes: Option<Vec<MemoryMode>>,
    pub reason: RoutingReason,
    pub confidence: f32,
}

/// Query-time bounds for active reconstruction.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconstructionOptions {
    #[serde(default)]
    pub mode: ReconstructionMode,
    #[serde(default = "default_reconstruction_max_steps")]
    pub max_steps: u8,
    #[serde(default = "default_reconstruction_max_tokens")]
    pub max_tokens: u32,
}

const fn default_reconstruction_max_steps() -> u8 {
    4
}

const fn default_reconstruction_max_tokens() -> u32 {
    2_000
}

impl Default for ReconstructionOptions {
    fn default() -> Self {
        Self {
            mode: ReconstructionMode::Off,
            max_steps: default_reconstruction_max_steps(),
            max_tokens: default_reconstruction_max_tokens(),
        }
    }
}

impl ReconstructionOptions {
    #[must_use]
    pub fn off() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn force() -> Self {
        Self {
            mode: ReconstructionMode::Force,
            ..Self::default()
        }
    }

    pub fn validate(&self) -> MemoryResult<()> {
        if self.max_steps == 0 {
            return Err(MemoryError::invalid(
                "max_reconstruction_steps must be greater than 0",
            ));
        }
        if self.max_tokens == 0 {
            return Err(MemoryError::invalid(
                "max_reconstruction_tokens must be greater than 0",
            ));
        }
        Ok(())
    }
}

/// Typed memory node families in the projection graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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

impl MemoryNodeKind {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Episode => "episode",
            Self::Fact => "fact",
            Self::EntityCue => "entity_cue",
            Self::Tag => "tag",
            Self::Procedure => "procedure",
            Self::State => "state",
            Self::Gotcha => "gotcha",
            Self::AntiMemory => "anti_memory",
            Self::Topic => "topic",
        }
    }

    #[must_use]
    pub fn default_modes() -> Vec<MemoryMode> {
        vec![
            MemoryMode::Semantic,
            MemoryMode::Episodic,
            MemoryMode::Procedural,
            MemoryMode::Gotcha,
            MemoryMode::State,
        ]
    }
}

impl fmt::Display for MemoryNodeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for MemoryNodeKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "episode" => Ok(Self::Episode),
            "fact" => Ok(Self::Fact),
            "entity_cue" | "entity" | "cue" => Ok(Self::EntityCue),
            "tag" => Ok(Self::Tag),
            "procedure" | "procedural" => Ok(Self::Procedure),
            "state" => Ok(Self::State),
            "gotcha" => Ok(Self::Gotcha),
            "anti_memory" | "antimemory" | "anti-memory" => Ok(Self::AntiMemory),
            "topic" => Ok(Self::Topic),
            other => Err(format!("unknown memory node kind {other:?}")),
        }
    }
}

/// Typed graph edges used by the memory projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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

impl MemoryEdgeKind {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mentions => "mentions",
            Self::CausedBy => "caused_by",
            Self::Fixes => "fixes",
            Self::Contradicts => "contradicts",
            Self::Supersedes => "supersedes",
            Self::Before => "before",
            Self::After => "after",
            Self::PartOf => "part_of",
            Self::DerivedFrom => "derived_from",
            Self::Blocks => "blocks",
            Self::Enables => "enables",
            Self::ObservedIn => "observed_in",
        }
    }
}

impl fmt::Display for MemoryEdgeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for MemoryEdgeKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "mentions" => Ok(Self::Mentions),
            "caused_by" => Ok(Self::CausedBy),
            "fixes" => Ok(Self::Fixes),
            "contradicts" => Ok(Self::Contradicts),
            "supersedes" => Ok(Self::Supersedes),
            "before" => Ok(Self::Before),
            "after" => Ok(Self::After),
            "part_of" => Ok(Self::PartOf),
            "derived_from" => Ok(Self::DerivedFrom),
            "blocks" => Ok(Self::Blocks),
            "enables" => Ok(Self::Enables),
            "observed_in" => Ok(Self::ObservedIn),
            other => Err(format!("unknown memory edge kind {other:?}")),
        }
    }
}

/// Belief revision operation emitted by the offline distiller.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BeliefRevisionOp {
    Add,
    Update,
    Invalidate,
    Noop,
}

impl BeliefRevisionOp {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Update => "update",
            Self::Invalidate => "invalidate",
            Self::Noop => "noop",
        }
    }
}

impl FromStr for BeliefRevisionOp {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "add" | "ADD" => Ok(Self::Add),
            "update" | "UPDATE" => Ok(Self::Update),
            "invalidate" | "INVALIDATE" | "delete" | "DELETE" => Ok(Self::Invalidate),
            "noop" | "NOOP" | "no_op" => Ok(Self::Noop),
            other => Err(format!("unknown belief revision op {other:?}")),
        }
    }
}

/// Tenant/project/time scope for a memory query.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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

    pub fn validate(&self) -> MemoryResult<()> {
        validate_required_identifier("tenant_id", &self.tenant_id)?;
        validate_required_identifier("project_id", &self.project_id)?;
        if let Some(environment_id) = self.environment_id.as_deref() {
            validate_required_identifier("environment_id", environment_id)?;
        }
        if self.as_of_unix_ms.is_some_and(|as_of| as_of < 0) {
            return Err(MemoryError::invalid("as_of_unix_ms must be non-negative"));
        }
        Ok(())
    }
}

/// Answer-shaped memory request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct MemoryQuery {
    pub question: String,
    pub scope: MemoryScope,
    pub max_tokens: u32,
    pub require_fresh: bool,
    pub modes: Vec<MemoryMode>,
    #[serde(skip_serializing)]
    pub modes_explicit: bool,
    #[serde(default, skip_serializing_if = "is_default_reconstruction_options")]
    pub reconstruction: ReconstructionOptions,
}

impl MemoryQuery {
    #[must_use]
    pub fn new(question: impl Into<String>, scope: MemoryScope) -> Self {
        Self {
            question: question.into(),
            scope,
            max_tokens: 1_200,
            require_fresh: false,
            modes: MemoryNodeKind::default_modes(),
            modes_explicit: false,
            reconstruction: ReconstructionOptions::default(),
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
        self.modes_explicit = true;
        self
    }

    #[must_use]
    pub fn with_reconstruction(mut self, reconstruction: ReconstructionOptions) -> Self {
        self.reconstruction = reconstruction;
        self
    }

    #[must_use]
    pub fn accepts_kind(&self, kind: MemoryNodeKind) -> bool {
        self.modes.iter().any(|mode| mode.accepts(kind))
    }

    pub fn validate(&self) -> MemoryResult<()> {
        self.scope.validate()?;
        validate_required_text("question", &self.question)?;
        if self.max_tokens == 0 {
            return Err(MemoryError::invalid("max_tokens must be greater than 0"));
        }
        if self.modes.is_empty() {
            return Err(MemoryError::invalid("modes must not be empty"));
        }
        self.reconstruction.validate()?;
        Ok(())
    }
}

impl<'de> Deserialize<'de> for MemoryQuery {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawMemoryQuery {
            question: String,
            scope: MemoryScope,
            max_tokens: u32,
            require_fresh: bool,
            #[serde(default)]
            modes: Option<Vec<MemoryMode>>,
            #[serde(default)]
            modes_explicit: bool,
            #[serde(default)]
            reconstruction: ReconstructionOptions,
        }

        let raw = RawMemoryQuery::deserialize(deserializer)?;
        let modes_explicit = raw.modes_explicit || raw.modes.is_some();
        Ok(Self {
            question: raw.question,
            scope: raw.scope,
            max_tokens: raw.max_tokens,
            require_fresh: raw.require_fresh,
            modes: raw.modes.unwrap_or_else(MemoryNodeKind::default_modes),
            modes_explicit,
            reconstruction: raw.reconstruction,
        })
    }
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

fn is_default_reconstruction_options(options: &ReconstructionOptions) -> bool {
    options == &ReconstructionOptions::default()
}

/// Span provenance for a returned memory.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CitedSpan {
    pub tenant_id: String,
    pub project_id: String,
    pub trace_id: String,
    pub span_id: String,
    pub seq: u64,
}

impl CitedSpan {
    pub fn validate(&self) -> MemoryResult<()> {
        validate_required_identifier("cited_span.tenant_id", &self.tenant_id)?;
        validate_required_identifier("cited_span.project_id", &self.project_id)?;
        validate_required_identifier("cited_span.trace_id", &self.trace_id)?;
        validate_required_identifier("cited_span.span_id", &self.span_id)?;
        if self.seq == 0 {
            return Err(MemoryError::invalid(
                "cited_span.seq must be greater than 0",
            ));
        }
        Ok(())
    }
}

/// One compact evidence item that can be placed into model context.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Contradiction {
    pub older_node_id: String,
    pub newer_node_id: String,
    pub summary: String,
}

/// A premise that may be stale under the query's requested time/freshness.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaleAssumption {
    pub node_id: String,
    pub summary: String,
    pub invalidated_at_unix_ms: Option<i64>,
}

/// Diagnostic report for a bounded active reconstruction pass.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconstructionReport {
    pub mode: ReconstructionMode,
    pub reason: ReconstructionReason,
    pub steps_used: u8,
    pub tokens_spent: u32,
    pub expanded_node_ids: Vec<String>,
    pub accepted_node_ids: Vec<String>,
    pub pruned_node_ids: Vec<String>,
}

/// Answer returned by the memory engine.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryAnswer {
    pub answer: String,
    pub evidence: Vec<Evidence>,
    pub cited_spans: Vec<CitedSpan>,
    pub contradictions: Vec<Contradiction>,
    pub stale_assumptions: Vec<StaleAssumption>,
    pub suggested_next_queries: Vec<String>,
    pub token_estimate: u32,
    pub tier_used: MemoryTier,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing: Option<RoutingReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reconstruction: Option<ReconstructionReport>,
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
            routing: None,
            reconstruction: None,
        }
    }
}

/// Output of an offline distillation pass before it is applied to projections.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistilledMemory {
    pub op: BeliefRevisionOp,
    pub node_kind: MemoryNodeKind,
    pub text: String,
    pub target_node_id: Option<String>,
    pub cited_spans: Vec<CitedSpan>,
}

impl DistilledMemory {
    #[must_use]
    pub fn add(node_kind: MemoryNodeKind, text: impl Into<String>, cited_span: CitedSpan) -> Self {
        Self {
            op: BeliefRevisionOp::Add,
            node_kind,
            text: text.into(),
            target_node_id: None,
            cited_spans: vec![cited_span],
        }
    }

    pub fn validate(&self) -> MemoryResult<()> {
        if self.cited_spans.is_empty() {
            return Err(MemoryError::invalid("cited_spans must not be empty"));
        }
        for span in &self.cited_spans {
            span.validate()?;
        }
        if let Some(target_node_id) = self.target_node_id.as_deref() {
            validate_required_identifier("target_node_id", target_node_id)?;
        }
        match self.op {
            BeliefRevisionOp::Add => {
                validate_required_text("text", &self.text)?;
            }
            BeliefRevisionOp::Update => {
                validate_required_text("text", &self.text)?;
            }
            BeliefRevisionOp::Invalidate => {
                let has_text = !self.text.trim().is_empty();
                let has_target = self.target_node_id.is_some();
                if !has_text && !has_target {
                    return Err(MemoryError::invalid(
                        "invalidate memory must include text or target_node_id",
                    ));
                }
            }
            BeliefRevisionOp::Noop => {
                if self.target_node_id.is_some() {
                    return Err(MemoryError::invalid(
                        "noop memory must not include target_node_id",
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Blend factors for LLM-free activation ranking.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
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
        assert!(!query.modes_explicit);
        assert_eq!(query.reconstruction, ReconstructionOptions::default());
        query.validate().unwrap_or_else(|err| panic!("{err}"));
    }

    #[test]
    fn query_deserialization_defaults_reconstruction_options() {
        let query: MemoryQuery = serde_json::from_value(serde_json::json!({
            "question": "what changed?",
            "scope": {
                "tenant_id": "tenant",
                "project_id": "project",
                "environment_id": null,
                "as_of_unix_ms": null
            },
            "max_tokens": 1200,
            "require_fresh": false,
            "modes": ["semantic"]
        }))
        .unwrap_or_else(|err| panic!("{err}"));

        assert_eq!(query.reconstruction, ReconstructionOptions::default());
        assert!(query.modes_explicit);

        let query: MemoryQuery = serde_json::from_value(serde_json::json!({
            "question": "what changed?",
            "scope": {
                "tenant_id": "tenant",
                "project_id": "project",
                "environment_id": null,
                "as_of_unix_ms": null
            },
            "max_tokens": 1200,
            "require_fresh": false,
            "modes": ["semantic"],
            "reconstruction": { "mode": "auto" }
        }))
        .unwrap_or_else(|err| panic!("{err}"));

        assert_eq!(query.reconstruction.mode, ReconstructionMode::Auto);
        assert_eq!(query.reconstruction.max_steps, 4);
        assert_eq!(query.reconstruction.max_tokens, 2_000);
        assert!(query.modes_explicit);
    }

    #[test]
    fn query_serialization_omits_default_reconstruction_options() {
        let query = MemoryQuery::new("what changed?", MemoryScope::new("tenant", "project"))
            .with_modes(vec![MemoryMode::Semantic]);

        let value = serde_json::to_value(query).unwrap_or_else(|err| panic!("{err}"));

        assert!(value.get("reconstruction").is_none());
        assert!(value.get("modes_explicit").is_none());
    }

    #[test]
    fn answer_serialization_omits_empty_reconstruction_report() {
        let answer = MemoryAnswer::empty(MemoryTier::Activation);

        let value = serde_json::to_value(answer).unwrap_or_else(|err| panic!("{err}"));

        assert!(value.get("reconstruction").is_none());
    }

    #[test]
    fn scope_validation_rejects_malformed_identifiers() {
        let err = MemoryScope::new(" tenant", "project")
            .validate()
            .unwrap_err();
        assert!(err.to_string().contains("tenant_id"));

        let err = MemoryScope::new("tenant", "").validate().unwrap_err();
        assert!(err.to_string().contains("project_id"));

        let err = MemoryScope::new("tenant", "project")
            .with_environment(" ")
            .validate()
            .unwrap_err();
        assert!(err.to_string().contains("environment_id"));

        let err = MemoryScope::new("tenant", "project")
            .as_of_unix_ms(-1)
            .validate()
            .unwrap_err();
        assert!(err.to_string().contains("as_of_unix_ms"));
    }

    #[test]
    fn query_validation_rejects_unusable_requests() {
        let scope = MemoryScope::new("tenant", "project");

        let err = MemoryQuery::new(" ", scope.clone()).validate().unwrap_err();
        assert!(err.to_string().contains("question"));

        let err = MemoryQuery::new("what changed?", scope.clone())
            .with_max_tokens(0)
            .validate()
            .unwrap_err();
        assert!(err.to_string().contains("max_tokens"));

        let err = MemoryQuery::new("what changed?", scope)
            .with_modes(Vec::new())
            .validate()
            .unwrap_err();
        assert!(err.to_string().contains("modes"));
    }

    #[test]
    fn cited_span_validation_rejects_malformed_provenance() {
        let mut span = valid_span();
        span.trace_id = " trace".to_string();
        let err = span.validate().unwrap_err();
        assert!(err.to_string().contains("cited_span.trace_id"));

        let mut span = valid_span();
        span.seq = 0;
        let err = span.validate().unwrap_err();
        assert!(err.to_string().contains("cited_span.seq"));
    }

    #[test]
    fn distilled_memory_validation_rejects_malformed_outputs() {
        DistilledMemory::add(
            MemoryNodeKind::Fact,
            "Checkout uses DATABASE_URL.",
            valid_span(),
        )
        .validate()
        .unwrap_or_else(|err| panic!("{err}"));

        let err = DistilledMemory::add(MemoryNodeKind::Fact, " ", valid_span())
            .validate()
            .unwrap_err();
        assert!(err.to_string().contains("text"));

        let err = DistilledMemory {
            op: BeliefRevisionOp::Noop,
            node_kind: MemoryNodeKind::Episode,
            text: String::new(),
            target_node_id: Some("node_1".to_string()),
            cited_spans: vec![valid_span()],
        }
        .validate()
        .unwrap_err();
        assert!(err.to_string().contains("target_node_id"));

        let err = DistilledMemory {
            op: BeliefRevisionOp::Invalidate,
            node_kind: MemoryNodeKind::Fact,
            text: String::new(),
            target_node_id: None,
            cited_spans: vec![valid_span()],
        }
        .validate()
        .unwrap_err();
        assert!(err.to_string().contains("invalidate memory"));

        let err = DistilledMemory {
            op: BeliefRevisionOp::Add,
            node_kind: MemoryNodeKind::Fact,
            text: "Checkout uses DATABASE_URL.".to_string(),
            target_node_id: None,
            cited_spans: Vec::new(),
        }
        .validate()
        .unwrap_err();
        assert!(err.to_string().contains("cited_spans"));
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

    fn valid_span() -> CitedSpan {
        CitedSpan {
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            trace_id: "trace".to_string(),
            span_id: "span".to_string(),
            seq: 1,
        }
    }
}
