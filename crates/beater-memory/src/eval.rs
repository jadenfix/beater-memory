use std::{
    collections::{BTreeMap, BTreeSet},
    time::Instant,
};

use serde::{Deserialize, Serialize};

use crate::{
    MemoryEngine, ProjectReport,
    error::{MemoryError, MemoryResult},
    model::{
        MemoryMode, MemoryNodeKind, MemoryQuery, MemoryScope, MemoryTier, ReconstructionMode,
        ReconstructionOptions, ReconstructionReason, RoutingReason, estimate_tokens,
    },
    store::{LedgerEvent, StoreStats},
    text::stable_id,
};

/// Current JSON fixture/report contract version for deterministic evals.
pub const EVAL_CONTRACT_VERSION: u32 = 1;

/// LongMemEval-style ability labels for deterministic memory suites.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum EvalAbility {
    StaticStateRecall,
    DynamicStateTracking,
    WorkflowKnowledge,
    EnvironmentGotcha,
    PremiseAwareness,
    CompositionalReasoning,
    #[default]
    Other,
}

/// Stable scoring semantics used by aggregate and per-case eval scores.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalScoreKind {
    /// Content expectation rate, with hard gate failures forcing the score to zero.
    #[default]
    EffectiveExpectationPassRate,
    /// Raw content expectation match rate, retained for older reports.
    ContentExpectationPassRate,
}

/// Optional provenance declared by an eval fixture.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalSuiteSource {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
}

/// Source metadata echoed in eval reports.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EvalReportSource {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suite_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suite: Option<EvalSuiteSource>,
}

/// A deterministic evaluation suite run against an isolated in-memory engine.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalSuite {
    #[serde(default = "default_eval_contract_version")]
    pub contract_version: u32,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<EvalSuiteSource>,
    #[serde(default = "default_eval_tenant")]
    pub tenant_id: String,
    #[serde(default = "default_eval_project")]
    pub project_id: String,
    #[serde(default)]
    pub environment_id: Option<String>,
    #[serde(default)]
    pub shared_haystack: bool,
    pub cases: Vec<EvalCase>,
}

/// One benchmark case with its own ledger observations and expectation checks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalCase {
    pub id: String,
    #[serde(default)]
    pub ability: EvalAbility,
    #[serde(default)]
    pub tenant_id: Option<String>,
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub environment_id: Option<String>,
    #[serde(default)]
    pub as_of_unix_ms: Option<i64>,
    #[serde(default)]
    pub known_at_unix_ms: Option<i64>,
    #[serde(default)]
    pub require_fresh: bool,
    pub question: String,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub modes: Option<Vec<MemoryMode>>,
    #[serde(default, deserialize_with = "deserialize_eval_reconstruction")]
    pub reconstruction: Option<ReconstructionOptions>,
    #[serde(default)]
    pub baseline_full_context_score: Option<f32>,
    pub events: Vec<EvalEvent>,
    #[serde(default)]
    pub expected_answer_contains: Vec<String>,
    #[serde(default)]
    pub expected_evidence_contains: Vec<String>,
    #[serde(default)]
    pub expected_stale_contains: Vec<String>,
    #[serde(default)]
    pub expected_contradiction_contains: Vec<String>,
    #[serde(default)]
    pub expected_tier: Option<MemoryTier>,
}

/// One observation to append before the suite is projected and queried.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalEvent {
    pub kind: MemoryNodeKind,
    pub text: String,
    #[serde(default)]
    pub observed_at_unix_ms: Option<i64>,
    #[serde(default)]
    pub ingested_at_unix_ms: Option<i64>,
}

/// Runtime overrides applied to every case in a suite.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EvalOptions {
    pub max_tokens: Option<u32>,
    pub reconstruction_mode: Option<ReconstructionMode>,
    pub max_reconstruction_steps: Option<u8>,
    pub max_reconstruction_tokens: Option<u32>,
}

/// Aggregate result of one deterministic suite run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EvalReport {
    #[serde(default = "default_eval_contract_version")]
    pub contract_version: u32,
    pub suite: String,
    #[serde(default)]
    pub source: EvalReportSource,
    pub cases: usize,
    pub passed: usize,
    pub failed: usize,
    #[serde(default = "default_legacy_score_kind")]
    pub score_kind: EvalScoreKind,
    pub score: f32,
    #[serde(default)]
    pub checks_total: usize,
    #[serde(default)]
    pub checks_matched: usize,
    #[serde(default)]
    pub checks_failed: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_full_context_score: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_saturation_gap: Option<f32>,
    pub source_tokens_per_stored_memory: f32,
    pub projected_tokens_per_stored_memory: f32,
    pub projected_to_source_token_ratio: f32,
    pub query_latency_ms_sum: u64,
    pub tokens_into_context_total: u32,
    pub answer_tokens_total: u32,
    pub evidence_tokens_total: u32,
    pub reconstruction_tokens_total: u32,
    pub project: crate::ProjectReport,
    pub stats: StoreStats,
    pub ability_scores: Vec<EvalAbilitySummary>,
    pub tier_metrics: Vec<EvalTierSummary>,
    pub case_reports: Vec<EvalCaseReport>,
}

/// Score summary for a LongMemEval-style ability bucket.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EvalAbilitySummary {
    pub ability: EvalAbility,
    pub cases: usize,
    pub passed: usize,
    pub score: f32,
}

/// Query economics summarized by retrieval tier.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EvalTierSummary {
    pub tier: MemoryTier,
    pub requests: usize,
    pub latency_ms_sum: u64,
    pub tokens_into_context: u32,
}

/// Per-case deterministic judge output and query telemetry.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[non_exhaustive]
pub struct EvalCaseReport {
    pub id: String,
    pub ability: EvalAbility,
    #[serde(default)]
    pub question: String,
    #[serde(default)]
    pub tenant_id: String,
    #[serde(default)]
    pub project_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub as_of_unix_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub known_at_unix_ms: Option<i64>,
    #[serde(default)]
    pub max_tokens: u32,
    #[serde(default)]
    pub modes: Vec<MemoryMode>,
    #[serde(default)]
    pub reconstruction_mode: ReconstructionMode,
    pub passed: bool,
    #[serde(default = "default_legacy_score_kind")]
    pub score_kind: EvalScoreKind,
    pub score: f32,
    #[serde(default)]
    pub content_score: f32,
    #[serde(default)]
    pub hard_gate_failed: bool,
    #[serde(default)]
    pub checks_total: usize,
    #[serde(default)]
    pub checks_matched: usize,
    #[serde(default)]
    pub checks_failed: usize,
    pub tier_used: MemoryTier,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing_reason: Option<RoutingReason>,
    #[serde(default)]
    pub routed_modes: Vec<MemoryMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing_confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reconstruction_reason: Option<ReconstructionReason>,
    #[serde(default)]
    pub reconstruction_steps: u8,
    #[serde(default)]
    pub reconstruction_provider_calls: usize,
    #[serde(default)]
    pub reconstruction_provider_errors: usize,
    #[serde(default)]
    pub reconstruction_provider_schema_errors: usize,
    pub latency_ms: u64,
    pub token_estimate: u32,
    pub answer_tokens: u32,
    pub evidence_tokens: u32,
    pub reconstruction_tokens: u32,
    pub evidence_count: usize,
    pub stale_assumption_count: usize,
    pub contradiction_count: usize,
    #[serde(default)]
    pub source_events: usize,
    #[serde(default)]
    pub events_projected: usize,
    #[serde(default)]
    pub events_skipped: usize,
    #[serde(default)]
    pub source_token_estimate: u32,
    #[serde(default)]
    pub projected_memory_token_estimate: u32,
    #[serde(default)]
    pub stored_memories_touched: usize,
    #[serde(default)]
    pub expectations: Vec<EvalExpectationReport>,
    #[serde(default)]
    pub evidence_node_ids: Vec<String>,
    #[serde(default)]
    pub evidence_texts: Vec<String>,
    #[serde(default)]
    pub answer_excerpt: String,
    pub failure_reasons: Vec<String>,
}

impl<'de> Deserialize<'de> for EvalCaseReport {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawEvalCaseReport {
            id: String,
            ability: EvalAbility,
            #[serde(default)]
            question: String,
            #[serde(default)]
            tenant_id: String,
            #[serde(default)]
            project_id: String,
            #[serde(default)]
            environment_id: Option<String>,
            #[serde(default)]
            as_of_unix_ms: Option<i64>,
            #[serde(default)]
            known_at_unix_ms: Option<i64>,
            #[serde(default)]
            max_tokens: u32,
            #[serde(default)]
            modes: Vec<MemoryMode>,
            #[serde(default)]
            reconstruction_mode: ReconstructionMode,
            passed: bool,
            #[serde(default = "default_legacy_score_kind")]
            score_kind: EvalScoreKind,
            score: f32,
            #[serde(default)]
            content_score: Option<f32>,
            #[serde(default)]
            hard_gate_failed: bool,
            #[serde(default)]
            checks_total: usize,
            #[serde(default)]
            checks_matched: usize,
            #[serde(default)]
            checks_failed: usize,
            tier_used: MemoryTier,
            #[serde(default)]
            routing_reason: Option<RoutingReason>,
            #[serde(default)]
            routed_modes: Vec<MemoryMode>,
            #[serde(default)]
            routing_confidence: Option<f32>,
            #[serde(default)]
            reconstruction_reason: Option<ReconstructionReason>,
            #[serde(default)]
            reconstruction_steps: u8,
            #[serde(default)]
            reconstruction_provider_calls: usize,
            #[serde(default)]
            reconstruction_provider_errors: usize,
            #[serde(default)]
            reconstruction_provider_schema_errors: usize,
            latency_ms: u64,
            token_estimate: u32,
            answer_tokens: u32,
            evidence_tokens: u32,
            reconstruction_tokens: u32,
            evidence_count: usize,
            stale_assumption_count: usize,
            contradiction_count: usize,
            #[serde(default)]
            source_events: usize,
            #[serde(default)]
            events_projected: usize,
            #[serde(default)]
            events_skipped: usize,
            #[serde(default)]
            source_token_estimate: u32,
            #[serde(default)]
            projected_memory_token_estimate: u32,
            #[serde(default)]
            stored_memories_touched: usize,
            #[serde(default)]
            expectations: Vec<EvalExpectationReport>,
            #[serde(default)]
            evidence_node_ids: Vec<String>,
            #[serde(default)]
            evidence_texts: Vec<String>,
            #[serde(default)]
            answer_excerpt: String,
            failure_reasons: Vec<String>,
        }

        let raw = RawEvalCaseReport::deserialize(deserializer)?;
        Ok(Self {
            id: raw.id,
            ability: raw.ability,
            question: raw.question,
            tenant_id: raw.tenant_id,
            project_id: raw.project_id,
            environment_id: raw.environment_id,
            as_of_unix_ms: raw.as_of_unix_ms,
            known_at_unix_ms: raw.known_at_unix_ms,
            max_tokens: raw.max_tokens,
            modes: raw.modes,
            reconstruction_mode: raw.reconstruction_mode,
            passed: raw.passed,
            score_kind: raw.score_kind,
            score: raw.score,
            content_score: raw.content_score.unwrap_or(raw.score),
            hard_gate_failed: raw.hard_gate_failed,
            checks_total: raw.checks_total,
            checks_matched: raw.checks_matched,
            checks_failed: raw.checks_failed,
            tier_used: raw.tier_used,
            routing_reason: raw.routing_reason,
            routed_modes: raw.routed_modes,
            routing_confidence: raw.routing_confidence,
            reconstruction_reason: raw.reconstruction_reason,
            reconstruction_steps: raw.reconstruction_steps,
            reconstruction_provider_calls: raw.reconstruction_provider_calls,
            reconstruction_provider_errors: raw.reconstruction_provider_errors,
            reconstruction_provider_schema_errors: raw.reconstruction_provider_schema_errors,
            latency_ms: raw.latency_ms,
            token_estimate: raw.token_estimate,
            answer_tokens: raw.answer_tokens,
            evidence_tokens: raw.evidence_tokens,
            reconstruction_tokens: raw.reconstruction_tokens,
            evidence_count: raw.evidence_count,
            stale_assumption_count: raw.stale_assumption_count,
            contradiction_count: raw.contradiction_count,
            source_events: raw.source_events,
            events_projected: raw.events_projected,
            events_skipped: raw.events_skipped,
            source_token_estimate: raw.source_token_estimate,
            projected_memory_token_estimate: raw.projected_memory_token_estimate,
            stored_memories_touched: raw.stored_memories_touched,
            expectations: raw.expectations,
            evidence_node_ids: raw.evidence_node_ids,
            evidence_texts: raw.evidence_texts,
            answer_excerpt: raw.answer_excerpt,
            failure_reasons: raw.failure_reasons,
        })
    }
}

/// Per-expectation match diagnostic for a case report.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EvalExpectationReport {
    pub field: String,
    pub expected: String,
    pub matched: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matched_text: Option<String>,
}

/// Run a deterministic memory evaluation suite in an isolated in-memory store.
pub fn run_eval_suite(suite: &EvalSuite, options: &EvalOptions) -> MemoryResult<EvalReport> {
    run_eval_suite_with_source(suite, options, None)
}

/// Run a deterministic suite while attaching external source metadata.
pub fn run_eval_suite_with_source(
    suite: &EvalSuite,
    options: &EvalOptions,
    suite_path: Option<String>,
) -> MemoryResult<EvalReport> {
    validate_options(options)?;
    validate_suite(suite)?;
    let engine = MemoryEngine::in_memory()?;
    let mut project = ProjectReport::default();
    let mut case_reports = Vec::with_capacity(suite.cases.len());
    for (case_index, case) in suite.cases.iter().enumerate() {
        let scope = case_scope(suite, case);
        let mut ingested_events = 0_usize;
        for (event_index, event) in case.events.iter().enumerate() {
            let mut ledger_event =
                LedgerEvent::direct_memory_write(&scope.0, &scope.1, event.kind, &event.text);
            ledger_event.environment_id = scope.2.clone();
            ledger_event.trace_id = stable_id(
                "eval_trace",
                &[&suite.name, &case.id, &event_index.to_string()],
            );
            ledger_event.span_id = stable_id("eval_span", &[&ledger_event.trace_id]);
            ledger_event.seq = u64::try_from(event_index + 1).unwrap_or(u64::MAX);
            ledger_event.observed_at_unix_ms = event
                .observed_at_unix_ms
                .unwrap_or_else(|| default_event_time(case_index, event_index));
            ledger_event.ingested_at_unix_ms = event
                .ingested_at_unix_ms
                .unwrap_or(ledger_event.observed_at_unix_ms);
            ledger_event.payload = serde_json::json!({
                "kind": event.kind.as_str(),
                "eval_suite": suite.name,
                "eval_contract_version": suite.contract_version,
                "eval_case_id": case.id,
            });
            if engine.ingest_event(&ledger_event)? {
                ingested_events += 1;
            }
        }
        let case_project = engine.project_pending(ingested_events.max(1))?;
        absorb_project_report(&mut project, &case_project);
        let query = build_query(suite, case, options);
        let started = Instant::now();
        let answer = engine.query(&query)?;
        let latency_ms = elapsed_ms(started);
        let judgement = judge_case(case, &answer);
        let evidence_tokens = answer
            .evidence
            .iter()
            .map(|item| item.token_estimate)
            .sum::<u32>();
        let answer_tokens = estimate_tokens(&answer.answer);
        let reconstruction_tokens = answer
            .reconstruction
            .as_ref()
            .map(|report| report.tokens_spent)
            .unwrap_or(0);
        let routing_reason = answer.routing.as_ref().map(|routing| routing.reason);
        let routed_modes = answer
            .routing
            .as_ref()
            .map(|routing| routing.routed_modes.clone())
            .unwrap_or_default();
        let routing_confidence = answer.routing.as_ref().map(|routing| routing.confidence);
        let reconstruction_reason = answer.reconstruction.as_ref().map(|report| report.reason);
        let reconstruction_steps = answer
            .reconstruction
            .as_ref()
            .map(|report| report.steps_used)
            .unwrap_or(0);
        let reconstruction_provider_calls = answer
            .reconstruction
            .as_ref()
            .map(|report| report.provider_calls)
            .unwrap_or(0);
        let reconstruction_provider_errors = answer
            .reconstruction
            .as_ref()
            .map(|report| report.provider_errors)
            .unwrap_or(0);
        let reconstruction_provider_schema_errors = answer
            .reconstruction
            .as_ref()
            .map(|report| report.provider_schema_errors)
            .unwrap_or(0);
        case_reports.push(EvalCaseReport {
            id: case.id.clone(),
            ability: case.ability,
            question: case.question.clone(),
            tenant_id: query.scope.tenant_id.clone(),
            project_id: query.scope.project_id.clone(),
            environment_id: query.scope.environment_id.clone(),
            as_of_unix_ms: query.scope.as_of_unix_ms,
            known_at_unix_ms: query.scope.known_at_unix_ms,
            max_tokens: query.max_tokens,
            modes: query.modes.clone(),
            reconstruction_mode: query.reconstruction.mode,
            passed: judgement.failure_reasons.is_empty(),
            score_kind: EvalScoreKind::EffectiveExpectationPassRate,
            score: judgement.score,
            content_score: judgement.content_score,
            hard_gate_failed: judgement.hard_gate_failed,
            checks_total: judgement.total_checks,
            checks_matched: judgement.matched_checks,
            checks_failed: judgement
                .total_checks
                .saturating_sub(judgement.matched_checks),
            tier_used: answer.tier_used,
            routing_reason,
            routed_modes,
            routing_confidence,
            reconstruction_reason,
            reconstruction_steps,
            reconstruction_provider_calls,
            reconstruction_provider_errors,
            reconstruction_provider_schema_errors,
            latency_ms,
            token_estimate: answer.token_estimate,
            answer_tokens,
            evidence_tokens,
            reconstruction_tokens,
            evidence_count: answer.evidence.len(),
            stale_assumption_count: answer.stale_assumptions.len(),
            contradiction_count: answer.contradictions.len(),
            source_events: case.events.len(),
            events_projected: case_project.events_projected,
            events_skipped: case_project.events_skipped,
            source_token_estimate: case_project.source_token_estimate,
            projected_memory_token_estimate: case_project.projected_memory_token_estimate,
            stored_memories_touched: case_project.stored_memories_touched,
            expectations: judgement.expectations,
            evidence_node_ids: answer
                .evidence
                .iter()
                .map(|item| item.node_id.clone())
                .collect(),
            evidence_texts: answer
                .evidence
                .iter()
                .map(|item| excerpt(&item.text))
                .collect(),
            answer_excerpt: excerpt(&answer.answer),
            failure_reasons: judgement.failure_reasons,
        });
    }
    let stats = engine.store().stats()?;

    let passed = case_reports.iter().filter(|case| case.passed).count();
    let cases = case_reports.len();
    let score = average(case_reports.iter().map(|case| case.score));
    let baseline_pairs = suite
        .cases
        .iter()
        .zip(case_reports.iter())
        .filter_map(|(case, report)| {
            case.baseline_full_context_score
                .map(|baseline| (report.score, baseline))
        })
        .collect::<Vec<_>>();
    let baseline_full_context_score =
        average_optional(baseline_pairs.iter().map(|(_, baseline)| *baseline));
    let context_saturation_gap = if baseline_pairs.is_empty() {
        None
    } else {
        let baseline_average = average(baseline_pairs.iter().map(|(_, baseline)| *baseline));
        let memory_average = average(baseline_pairs.iter().map(|(memory_score, _)| *memory_score));
        Some((baseline_average - memory_average).max(0.0))
    };
    let query_latency_ms_sum = case_reports.iter().map(|case| case.latency_ms).sum();
    let tokens_into_context_total = case_reports
        .iter()
        .map(|case| case.evidence_tokens)
        .sum::<u32>();
    let answer_tokens_total = case_reports
        .iter()
        .map(|case| case.answer_tokens)
        .sum::<u32>();
    let evidence_tokens_total = case_reports
        .iter()
        .map(|case| case.evidence_tokens)
        .sum::<u32>();
    let reconstruction_tokens_total = case_reports
        .iter()
        .map(|case| case.reconstruction_tokens)
        .sum::<u32>();
    let checks_total = case_reports
        .iter()
        .map(|case| case.checks_total)
        .sum::<usize>();
    let checks_matched = case_reports
        .iter()
        .map(|case| case.checks_matched)
        .sum::<usize>();
    let touched = project.stored_memories_touched.max(1) as f32;
    let source = project.source_token_estimate as f32;
    let projected = project.projected_memory_token_estimate as f32;

    Ok(EvalReport {
        contract_version: suite.contract_version,
        suite: suite.name.clone(),
        source: EvalReportSource {
            suite_path,
            suite: suite.source.clone(),
        },
        cases,
        passed,
        failed: cases.saturating_sub(passed),
        score_kind: EvalScoreKind::EffectiveExpectationPassRate,
        score,
        checks_total,
        checks_matched,
        checks_failed: checks_total.saturating_sub(checks_matched),
        baseline_full_context_score,
        context_saturation_gap,
        source_tokens_per_stored_memory: source / touched,
        projected_tokens_per_stored_memory: projected / touched,
        projected_to_source_token_ratio: if source > 0.0 {
            projected / source
        } else {
            0.0
        },
        query_latency_ms_sum,
        tokens_into_context_total,
        answer_tokens_total,
        evidence_tokens_total,
        reconstruction_tokens_total,
        project,
        stats,
        ability_scores: ability_summaries(&case_reports),
        tier_metrics: tier_summaries(&case_reports),
        case_reports,
    })
}

fn absorb_project_report(total: &mut ProjectReport, case: &ProjectReport) {
    total.events_seen += case.events_seen;
    total.events_projected += case.events_projected;
    total.events_skipped += case.events_skipped;
    total.source_token_estimate += case.source_token_estimate;
    total.projected_memory_token_estimate += case.projected_memory_token_estimate;
    total.stored_memories_touched += case.stored_memories_touched;
    total.distillation_outputs += case.distillation_outputs;
    total.distillation_provider_calls += case.distillation_provider_calls;
    total.distillation_provider_errors += case.distillation_provider_errors;
    total.distillation_schema_errors += case.distillation_schema_errors;
    total.distillation_repair_attempts += case.distillation_repair_attempts;
    total.distillation_repair_successes += case.distillation_repair_successes;
    total.distillation_rejections += case.distillation_rejections;
    total.distillation_replayed_batches += case.distillation_replayed_batches;
    total.distillation_input_tokens += case.distillation_input_tokens;
    total.distillation_output_tokens += case.distillation_output_tokens;
    total.distillation_elapsed_ms += case.distillation_elapsed_ms;
    total.memories_added += case.memories_added;
    total.memories_updated += case.memories_updated;
    total.memories_invalidated += case.memories_invalidated;
    total.memories_nooped += case.memories_nooped;
    total.edges_added += case.edges_added;
}

fn validate_options(options: &EvalOptions) -> MemoryResult<()> {
    if options.max_tokens == Some(0) {
        return Err(MemoryError::invalid(
            "eval max_tokens override must be greater than 0",
        ));
    }
    if options.max_reconstruction_steps == Some(0) {
        return Err(MemoryError::invalid(
            "eval max_reconstruction_steps override must be greater than 0",
        ));
    }
    if options.max_reconstruction_tokens == Some(0) {
        return Err(MemoryError::invalid(
            "eval max_reconstruction_tokens override must be greater than 0",
        ));
    }
    Ok(())
}

fn validate_suite(suite: &EvalSuite) -> MemoryResult<()> {
    if suite.contract_version != EVAL_CONTRACT_VERSION {
        return Err(MemoryError::invalid(format!(
            "unsupported eval contract_version {}; expected {}",
            suite.contract_version, EVAL_CONTRACT_VERSION
        )));
    }
    if suite.name.trim().is_empty() {
        return Err(MemoryError::invalid("eval suite name must not be empty"));
    }
    if let Some(source) = suite.source.as_ref() {
        validate_source(source)?;
    }
    if suite.cases.is_empty() {
        return Err(MemoryError::invalid(
            "eval suite must include at least one case",
        ));
    }
    let mut ids = BTreeSet::new();
    for case in &suite.cases {
        if case.id.trim().is_empty() {
            return Err(MemoryError::invalid("eval case id must not be empty"));
        }
        if !ids.insert(case.id.as_str()) {
            return Err(MemoryError::invalid(format!(
                "duplicate eval case id {:?}",
                case.id
            )));
        }
        if case.question.trim().is_empty() {
            return Err(MemoryError::invalid(format!(
                "eval case {:?} question must not be empty",
                case.id
            )));
        }
        validate_case_scope(suite, case)?;
        if case.events.is_empty() {
            return Err(MemoryError::invalid(format!(
                "eval case {:?} must include at least one event",
                case.id
            )));
        }
        for event in &case.events {
            if event.text.trim().is_empty() {
                return Err(MemoryError::invalid(format!(
                    "eval case {:?} event text must not be empty",
                    case.id
                )));
            }
        }
        validate_expectations(
            &case.id,
            "expected_answer_contains",
            &case.expected_answer_contains,
        )?;
        validate_expectations(
            &case.id,
            "expected_evidence_contains",
            &case.expected_evidence_contains,
        )?;
        validate_expectations(
            &case.id,
            "expected_stale_contains",
            &case.expected_stale_contains,
        )?;
        validate_expectations(
            &case.id,
            "expected_contradiction_contains",
            &case.expected_contradiction_contains,
        )?;
        let expectation_count = case.expected_answer_contains.len()
            + case.expected_evidence_contains.len()
            + case.expected_stale_contains.len()
            + case.expected_contradiction_contains.len();
        if expectation_count == 0 {
            return Err(MemoryError::invalid(format!(
                "eval case {:?} must include at least one content expectation",
                case.id
            )));
        }
        if let Some(max_tokens) = case.max_tokens
            && max_tokens == 0
        {
            return Err(MemoryError::invalid(format!(
                "eval case {:?} max_tokens must be greater than 0",
                case.id
            )));
        }
        if case.modes.as_ref().is_some_and(Vec::is_empty) {
            return Err(MemoryError::invalid(format!(
                "eval case {:?} modes must not be empty",
                case.id
            )));
        }
        if case.as_of_unix_ms.is_some_and(|as_of| as_of < 0) {
            return Err(MemoryError::invalid(format!(
                "eval case {:?} as_of_unix_ms must be non-negative",
                case.id
            )));
        }
        if case.known_at_unix_ms.is_some_and(|known_at| known_at < 0) {
            return Err(MemoryError::invalid(format!(
                "eval case {:?} known_at_unix_ms must be non-negative",
                case.id
            )));
        }
        for event in &case.events {
            if event
                .observed_at_unix_ms
                .is_some_and(|observed_at| observed_at < 0)
            {
                return Err(MemoryError::invalid(format!(
                    "eval case {:?} event observed_at_unix_ms must be non-negative",
                    case.id
                )));
            }
            if event
                .ingested_at_unix_ms
                .is_some_and(|ingested_at| ingested_at < 0)
            {
                return Err(MemoryError::invalid(format!(
                    "eval case {:?} event ingested_at_unix_ms must be non-negative",
                    case.id
                )));
            }
            if let (Some(observed_at), Some(ingested_at)) =
                (event.observed_at_unix_ms, event.ingested_at_unix_ms)
                && ingested_at < observed_at
            {
                return Err(MemoryError::invalid(format!(
                    "eval case {:?} event ingested_at_unix_ms must be greater than or equal to observed_at_unix_ms",
                    case.id
                )));
            }
        }
        if let Some(reconstruction) = case.reconstruction.as_ref() {
            reconstruction.validate()?;
        }
        if case
            .baseline_full_context_score
            .is_some_and(|score| !(0.0..=1.0).contains(&score))
        {
            return Err(MemoryError::invalid(format!(
                "eval case {:?} baseline_full_context_score must be between 0 and 1",
                case.id
            )));
        }
    }
    Ok(())
}

fn validate_case_scope(suite: &EvalSuite, case: &EvalCase) -> MemoryResult<()> {
    let (tenant_id, project_id, environment_id) = case_scope(suite, case);
    let mut scope = MemoryScope::new(tenant_id, project_id);
    if let Some(environment_id) = environment_id {
        scope = scope.with_environment(environment_id);
    }
    if let Some(as_of_unix_ms) = case.as_of_unix_ms {
        scope = scope.as_of_unix_ms(as_of_unix_ms);
    }
    if let Some(known_at_unix_ms) = case.known_at_unix_ms {
        scope = scope.known_at_unix_ms(known_at_unix_ms);
    }
    scope.validate().map_err(|err| {
        MemoryError::invalid(format!("eval case {:?} scope is invalid: {err}", case.id))
    })
}

fn validate_source(source: &EvalSuiteSource) -> MemoryResult<()> {
    for (field, value) in [
        ("source.name", source.name.as_ref()),
        ("source.uri", source.uri.as_ref()),
        ("source.revision", source.revision.as_ref()),
    ] {
        if value.is_some_and(|value| value.trim().is_empty()) {
            return Err(MemoryError::invalid(format!(
                "eval suite {field} must not be empty"
            )));
        }
    }
    Ok(())
}

fn validate_expectations(case_id: &str, field: &str, values: &[String]) -> MemoryResult<()> {
    if values.iter().any(|value| value.trim().is_empty()) {
        return Err(MemoryError::invalid(format!(
            "eval case {case_id:?} {field} values must not be empty"
        )));
    }
    Ok(())
}

fn build_query(suite: &EvalSuite, case: &EvalCase, options: &EvalOptions) -> MemoryQuery {
    let (tenant_id, project_id, environment_id) = case_scope(suite, case);
    let mut scope = MemoryScope::new(tenant_id, project_id);
    if let Some(environment_id) = environment_id {
        scope = scope.with_environment(environment_id);
    }
    if let Some(as_of_unix_ms) = case.as_of_unix_ms {
        scope = scope.as_of_unix_ms(as_of_unix_ms);
    }
    if let Some(known_at_unix_ms) = case.known_at_unix_ms {
        scope = scope.known_at_unix_ms(known_at_unix_ms);
    }
    let max_tokens = options.max_tokens.or(case.max_tokens).unwrap_or(1_200);
    let mut reconstruction = case.reconstruction.clone().unwrap_or_default();
    if let Some(mode) = options.reconstruction_mode {
        reconstruction.mode = mode;
    }
    if let Some(max_steps) = options.max_reconstruction_steps {
        reconstruction.max_steps = max_steps;
    }
    if let Some(max_tokens) = options.max_reconstruction_tokens {
        reconstruction.max_tokens = max_tokens;
    }
    let mut query = MemoryQuery::new(case.question.clone(), scope)
        .with_max_tokens(max_tokens)
        .with_reconstruction(reconstruction);
    if case.require_fresh {
        query = query.requiring_fresh();
    }
    if let Some(modes) = case.modes.clone() {
        query = query.with_modes(modes);
    }
    query
}

fn case_scope(suite: &EvalSuite, case: &EvalCase) -> (String, String, Option<String>) {
    let default_project_id = if suite.shared_haystack {
        suite.project_id.clone()
    } else {
        format!(
            "{}-{}",
            suite.project_id,
            stable_id("eval_case_project", &[&suite.name, &case.id])
        )
    };
    (
        case.tenant_id
            .clone()
            .unwrap_or_else(|| suite.tenant_id.clone()),
        case.project_id.clone().unwrap_or(default_project_id),
        case.environment_id
            .clone()
            .or_else(|| suite.environment_id.clone()),
    )
}

struct EvalCaseJudgement {
    score: f32,
    content_score: f32,
    hard_gate_failed: bool,
    matched_checks: usize,
    total_checks: usize,
    expectations: Vec<EvalExpectationReport>,
    failure_reasons: Vec<String>,
}

fn judge_case(case: &EvalCase, answer: &crate::MemoryAnswer) -> EvalCaseJudgement {
    let mut matched = 0_usize;
    let mut total = 0_usize;
    let mut failures = Vec::new();
    let mut expectations = Vec::new();

    check_expected(
        "answer",
        &case.expected_answer_contains,
        std::slice::from_ref(&answer.answer),
        &mut matched,
        &mut total,
        &mut expectations,
        &mut failures,
    );
    let evidence = answer
        .evidence
        .iter()
        .map(|item| item.text.as_str())
        .collect::<Vec<_>>();
    check_expected(
        "evidence",
        &case.expected_evidence_contains,
        &evidence,
        &mut matched,
        &mut total,
        &mut expectations,
        &mut failures,
    );
    let stale = answer
        .stale_assumptions
        .iter()
        .map(|item| item.summary.as_str())
        .collect::<Vec<_>>();
    check_expected(
        "stale assumption",
        &case.expected_stale_contains,
        &stale,
        &mut matched,
        &mut total,
        &mut expectations,
        &mut failures,
    );
    let contradictions = answer
        .contradictions
        .iter()
        .map(|item| item.summary.as_str())
        .collect::<Vec<_>>();
    check_expected(
        "contradiction",
        &case.expected_contradiction_contains,
        &contradictions,
        &mut matched,
        &mut total,
        &mut expectations,
        &mut failures,
    );
    let mut hard_gate_failed = false;
    if let Some(expected_tier) = case.expected_tier
        && answer.tier_used != expected_tier
    {
        hard_gate_failed = true;
        failures.push(format!(
            "expected tier {expected_tier:?}, got {:?}",
            answer.tier_used
        ));
    }

    if total == 0 {
        return EvalCaseJudgement {
            score: 0.0,
            content_score: 0.0,
            hard_gate_failed: false,
            matched_checks: 0,
            total_checks: 0,
            expectations,
            failure_reasons: vec!["case has no expectations".to_string()],
        };
    }
    let content_score = matched as f32 / total as f32;
    EvalCaseJudgement {
        score: if hard_gate_failed { 0.0 } else { content_score },
        content_score,
        hard_gate_failed,
        matched_checks: matched,
        total_checks: total,
        expectations,
        failure_reasons: failures,
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EvalReconstructionOptions {
    #[serde(default)]
    mode: Option<ReconstructionMode>,
    #[serde(default)]
    max_steps: Option<u8>,
    #[serde(default)]
    max_tokens: Option<u32>,
}

fn deserialize_eval_reconstruction<'de, D>(
    deserializer: D,
) -> Result<Option<ReconstructionOptions>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<EvalReconstructionOptions>::deserialize(deserializer).map(|options| {
        options.map(|options| {
            let defaults = ReconstructionOptions::default();
            ReconstructionOptions {
                mode: options.mode.unwrap_or(defaults.mode),
                max_steps: options.max_steps.unwrap_or(defaults.max_steps),
                max_tokens: options.max_tokens.unwrap_or(defaults.max_tokens),
            }
        })
    })
}

fn check_expected(
    label: &str,
    expected: &[String],
    haystacks: &[impl AsRef<str>],
    matched: &mut usize,
    total: &mut usize,
    reports: &mut Vec<EvalExpectationReport>,
    failures: &mut Vec<String>,
) {
    for expected in expected {
        *total += 1;
        let matched_text = haystacks
            .iter()
            .map(AsRef::as_ref)
            .find(|haystack| contains_case_insensitive(haystack, expected))
            .map(excerpt);
        if matched_text.is_some() {
            *matched += 1;
        } else {
            failures.push(format!("missing {label} substring {expected:?}"));
        }
        reports.push(EvalExpectationReport {
            field: label.to_string(),
            expected: expected.clone(),
            matched: matched_text.is_some(),
            matched_text,
        });
    }
}

fn contains_case_insensitive(haystack: &str, needle: &str) -> bool {
    haystack
        .to_ascii_lowercase()
        .contains(&needle.to_ascii_lowercase())
}

fn excerpt(value: &str) -> String {
    const MAX: usize = 240;
    if value.len() <= MAX {
        return value.to_string();
    }
    let mut end = MAX;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &value[..end])
}

fn ability_summaries(case_reports: &[EvalCaseReport]) -> Vec<EvalAbilitySummary> {
    let mut buckets: BTreeMap<EvalAbility, Vec<&EvalCaseReport>> = BTreeMap::new();
    for report in case_reports {
        buckets.entry(report.ability).or_default().push(report);
    }
    buckets
        .into_iter()
        .map(|(ability, reports)| {
            let cases = reports.len();
            let passed = reports.iter().filter(|case| case.passed).count();
            let score = average(reports.iter().map(|case| case.score));
            EvalAbilitySummary {
                ability,
                cases,
                passed,
                score,
            }
        })
        .collect()
}

fn tier_summaries(case_reports: &[EvalCaseReport]) -> Vec<EvalTierSummary> {
    [
        MemoryTier::CueSeed,
        MemoryTier::Activation,
        MemoryTier::ActiveReconstruction,
    ]
    .into_iter()
    .filter_map(|tier| {
        let reports = case_reports
            .iter()
            .filter(|report| report.tier_used == tier)
            .collect::<Vec<_>>();
        if reports.is_empty() {
            return None;
        }
        Some(EvalTierSummary {
            tier,
            requests: reports.len(),
            latency_ms_sum: reports.iter().map(|case| case.latency_ms).sum(),
            tokens_into_context: reports.iter().map(|case| case.evidence_tokens).sum(),
        })
    })
    .collect()
}

fn average(values: impl Iterator<Item = f32>) -> f32 {
    let mut count = 0_usize;
    let mut sum = 0.0_f32;
    for value in values {
        count += 1;
        sum += value;
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

fn average_optional(values: impl Iterator<Item = f32>) -> Option<f32> {
    let mut count = 0_usize;
    let mut sum = 0.0_f32;
    for value in values {
        count += 1;
        sum += value;
    }
    (count > 0).then_some(sum / count as f32)
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)
}

const fn default_eval_contract_version() -> u32 {
    EVAL_CONTRACT_VERSION
}

const fn default_legacy_score_kind() -> EvalScoreKind {
    EvalScoreKind::ContentExpectationPassRate
}

fn default_event_time(case_index: usize, event_index: usize) -> i64 {
    let case_index = i64::try_from(case_index).unwrap_or(i64::MAX / 10_000);
    let event_index = i64::try_from(event_index).unwrap_or(i64::MAX / 10_000);
    1_000 + (case_index * 10_000) + (event_index * 1_000)
}

fn default_eval_tenant() -> String {
    "local".to_string()
}

fn default_eval_project() -> String {
    "eval".to_string()
}

#[cfg(test)]
mod tests {
    use crate::model::{ReconstructionMode, ReconstructionOptions};

    use super::*;

    #[test]
    fn eval_suite_reports_ability_scores_and_economics() -> MemoryResult<()> {
        let suite = EvalSuite {
            contract_version: EVAL_CONTRACT_VERSION,
            name: "smoke".to_string(),
            source: None,
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            environment_id: None,
            shared_haystack: false,
            cases: vec![
                EvalCase {
                    id: "state".to_string(),
                    ability: EvalAbility::StaticStateRecall,
                    question: "what is the production api base url?".to_string(),
                    events: vec![EvalEvent {
                        kind: MemoryNodeKind::State,
                        text: "The production API base URL is https://api.example.test."
                            .to_string(),
                        observed_at_unix_ms: Some(1_000),
                        ingested_at_unix_ms: None,
                    }],
                    modes: Some(vec![MemoryMode::State]),
                    baseline_full_context_score: Some(1.0),
                    expected_evidence_contains: vec!["https://api.example.test".to_string()],
                    expected_tier: Some(MemoryTier::Activation),
                    ..case_defaults()
                },
                EvalCase {
                    id: "premise".to_string(),
                    ability: EvalAbility::PremiseAwareness,
                    question: "should I use the legacy api token?".to_string(),
                    events: vec![
                        EvalEvent {
                            kind: MemoryNodeKind::Fact,
                            text: "Use the legacy API token.".to_string(),
                            observed_at_unix_ms: Some(2_000),
                            ingested_at_unix_ms: None,
                        },
                        EvalEvent {
                            kind: MemoryNodeKind::Fact,
                            text: "Do not use the legacy API token; it is deprecated. Use the scoped API token.".to_string(),
                            observed_at_unix_ms: Some(3_000),
                            ingested_at_unix_ms: None,
                        },
                    ],
                    modes: Some(vec![MemoryMode::Semantic]),
                    reconstruction: Some(ReconstructionOptions {
                        mode: ReconstructionMode::Auto,
                        ..ReconstructionOptions::default()
                    }),
                    baseline_full_context_score: Some(1.0),
                    expected_evidence_contains: vec!["scoped API token".to_string()],
                    expected_stale_contains: vec!["legacy API token".to_string()],
                    expected_contradiction_contains: vec!["scoped API token".to_string()],
                    ..case_defaults()
                },
            ],
        };

        let report = run_eval_suite(&suite, &EvalOptions::default())?;

        assert_eq!(report.cases, 2);
        assert_eq!(report.contract_version, EVAL_CONTRACT_VERSION);
        assert_eq!(
            report.score_kind,
            EvalScoreKind::EffectiveExpectationPassRate
        );
        assert_eq!(report.passed, 2);
        assert_eq!(report.failed, 0);
        assert_eq!(report.score, 1.0);
        assert_eq!(report.checks_total, 4);
        assert_eq!(report.checks_matched, 4);
        assert_eq!(report.checks_failed, 0);
        assert_eq!(report.context_saturation_gap, Some(0.0));
        assert!(report.source_tokens_per_stored_memory > 0.0);
        assert!(report.projected_tokens_per_stored_memory > 0.0);
        assert_eq!(report.ability_scores.len(), 2);
        assert!(
            report
                .tier_metrics
                .iter()
                .any(|tier| tier.tier == MemoryTier::Activation && tier.requests >= 1)
        );
        let first_case = &report.case_reports[0];
        assert_eq!(
            first_case.score_kind,
            EvalScoreKind::EffectiveExpectationPassRate
        );
        assert_eq!(first_case.content_score, 1.0);
        assert!(!first_case.hard_gate_failed);
        assert_eq!(first_case.checks_total, 1);
        assert_eq!(first_case.checks_matched, 1);
        assert_eq!(first_case.expectations.len(), 1);
        assert!(first_case.expectations[0].matched);
        assert!(!first_case.answer_excerpt.is_empty());
        assert!(!first_case.evidence_node_ids.is_empty());
        assert!(!first_case.evidence_texts.is_empty());
        assert_eq!(first_case.tenant_id, "tenant");
        assert!(first_case.project_id.starts_with("project-"));
        assert_ne!(first_case.project_id, "project");
        assert_eq!(first_case.max_tokens, 1_200);
        assert_eq!(first_case.modes, vec![MemoryMode::State]);
        assert!(first_case.routing_reason.is_some());
        assert_eq!(first_case.source_events, 1);
        assert_eq!(first_case.events_projected, 1);
        Ok(())
    }

    #[test]
    fn eval_report_includes_source_metadata() -> MemoryResult<()> {
        let suite = EvalSuite {
            contract_version: EVAL_CONTRACT_VERSION,
            name: "source".to_string(),
            source: Some(EvalSuiteSource {
                name: Some("fixture-pack".to_string()),
                uri: Some("https://example.test/evals/source.json".to_string()),
                revision: Some("abc123".to_string()),
            }),
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            environment_id: None,
            shared_haystack: false,
            cases: vec![EvalCase {
                id: "case".to_string(),
                question: "what route is healthy?".to_string(),
                events: vec![EvalEvent {
                    kind: MemoryNodeKind::Fact,
                    text: "The healthy route is /readyz.".to_string(),
                    observed_at_unix_ms: Some(1_000),
                    ingested_at_unix_ms: None,
                }],
                expected_evidence_contains: vec!["/readyz".to_string()],
                ..case_defaults()
            }],
        };

        let report = run_eval_suite_with_source(
            &suite,
            &EvalOptions::default(),
            Some("/tmp/source.json".to_string()),
        )?;

        assert_eq!(
            report.source.suite_path.as_deref(),
            Some("/tmp/source.json")
        );
        assert_eq!(
            report
                .source
                .suite
                .as_ref()
                .and_then(|source| source.name.as_deref()),
            Some("fixture-pack")
        );
        Ok(())
    }

    #[test]
    fn eval_suite_rejects_unsupported_contract_version() {
        let suite = EvalSuite {
            contract_version: EVAL_CONTRACT_VERSION + 1,
            name: "future".to_string(),
            source: None,
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            environment_id: None,
            shared_haystack: false,
            cases: vec![EvalCase {
                id: "case".to_string(),
                question: "what changed?".to_string(),
                events: vec![EvalEvent {
                    kind: MemoryNodeKind::Fact,
                    text: "Checkout uses DATABASE_URL.".to_string(),
                    observed_at_unix_ms: Some(1_000),
                    ingested_at_unix_ms: None,
                }],
                expected_evidence_contains: vec!["DATABASE_URL".to_string()],
                ..case_defaults()
            }],
        };

        let err = run_eval_suite(&suite, &EvalOptions::default()).unwrap_err();

        assert!(err.to_string().contains("contract_version"));
    }

    #[test]
    fn eval_suite_applies_known_at_window() -> MemoryResult<()> {
        let suite = EvalSuite {
            contract_version: EVAL_CONTRACT_VERSION,
            name: "known-at".to_string(),
            source: None,
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            environment_id: None,
            shared_haystack: false,
            cases: vec![EvalCase {
                id: "late-known".to_string(),
                question: "what is the late known flag?".to_string(),
                as_of_unix_ms: Some(1_500),
                known_at_unix_ms: Some(2_000),
                events: vec![EvalEvent {
                    kind: MemoryNodeKind::Fact,
                    text: "The late known flag is beta.".to_string(),
                    observed_at_unix_ms: Some(1_000),
                    ingested_at_unix_ms: Some(3_000),
                }],
                expected_answer_contains: vec!["No matching memory".to_string()],
                expected_tier: Some(MemoryTier::Activation),
                ..case_defaults()
            }],
        };

        let report = run_eval_suite(&suite, &EvalOptions::default())?;

        assert_eq!(report.passed, 1);
        assert_eq!(report.failed, 0);
        Ok(())
    }

    #[test]
    fn eval_suite_rejects_cases_without_expectations() {
        let suite = EvalSuite {
            contract_version: EVAL_CONTRACT_VERSION,
            name: "invalid".to_string(),
            source: None,
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            environment_id: None,
            shared_haystack: false,
            cases: vec![EvalCase {
                id: "missing-expectations".to_string(),
                question: "what changed?".to_string(),
                events: vec![EvalEvent {
                    kind: MemoryNodeKind::Fact,
                    text: "Checkout uses DATABASE_URL.".to_string(),
                    observed_at_unix_ms: Some(1_000),
                    ingested_at_unix_ms: None,
                }],
                ..case_defaults()
            }],
        };

        let err = run_eval_suite(&suite, &EvalOptions::default()).unwrap_err();

        assert!(err.to_string().contains("expectation"));
    }

    #[test]
    fn eval_suite_rejects_tier_only_cases() {
        let suite = EvalSuite {
            contract_version: EVAL_CONTRACT_VERSION,
            name: "invalid".to_string(),
            source: None,
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            environment_id: None,
            shared_haystack: false,
            cases: vec![EvalCase {
                id: "tier-only".to_string(),
                question: "what tier did this use?".to_string(),
                events: vec![EvalEvent {
                    kind: MemoryNodeKind::Fact,
                    text: "Checkout uses DATABASE_URL.".to_string(),
                    observed_at_unix_ms: Some(1_000),
                    ingested_at_unix_ms: None,
                }],
                expected_tier: Some(MemoryTier::Activation),
                ..case_defaults()
            }],
        };

        let err = run_eval_suite(&suite, &EvalOptions::default()).unwrap_err();

        assert!(err.to_string().contains("content expectation"));
    }

    #[test]
    fn eval_suite_rejects_unknown_fixture_fields() {
        let err = serde_json::from_value::<EvalSuite>(serde_json::json!({
            "name": "typo",
            "unkown_field": true,
            "cases": []
        }))
        .unwrap_err();

        assert!(err.to_string().contains("unknown field"));

        let err = serde_json::from_value::<EvalSuite>(serde_json::json!({
            "name": "typo",
            "cases": [
                {
                    "id": "bad-reconstruction",
                    "question": "what changed?",
                    "events": [
                        {"kind": "fact", "text": "Checkout uses DATABASE_URL."}
                    ],
                    "reconstruction": {"mode": "force", "max_step": 1},
                    "expected_evidence_contains": ["DATABASE_URL"]
                }
            ]
        }))
        .unwrap_err();

        assert!(err.to_string().contains("unknown field"));
    }

    #[test]
    fn eval_suite_deserialization_defaults_contract_version() {
        let suite = serde_json::from_value::<EvalSuite>(serde_json::json!({
            "name": "old-fixture",
            "cases": [
                {
                    "id": "case",
                    "question": "what database is configured?",
                    "events": [
                        {"kind": "fact", "text": "Checkout uses DATABASE_URL."}
                    ],
                    "expected_evidence_contains": ["DATABASE_URL"]
                }
            ]
        }))
        .unwrap_or_else(|err| panic!("{err}"));

        assert_eq!(suite.contract_version, EVAL_CONTRACT_VERSION);
        assert!(suite.source.is_none());
    }

    #[test]
    fn eval_case_report_deserializes_without_new_telemetry() {
        let report = serde_json::from_value::<EvalCaseReport>(serde_json::json!({
            "id": "old",
            "ability": "other",
            "passed": false,
            "score": 1.0,
            "tier_used": "activation",
            "latency_ms": 3,
            "token_estimate": 10,
            "answer_tokens": 4,
            "evidence_tokens": 6,
            "reconstruction_tokens": 0,
            "evidence_count": 1,
            "stale_assumption_count": 0,
            "contradiction_count": 0,
            "failure_reasons": ["expected tier ActiveReconstruction, got Activation"]
        }))
        .unwrap_or_else(|err| panic!("{err}"));

        assert_eq!(report.score_kind, EvalScoreKind::ContentExpectationPassRate);
        assert_eq!(report.score, 1.0);
        assert_eq!(report.content_score, report.score);
        assert!(!report.hard_gate_failed);
        assert_eq!(report.checks_total, 0);
        assert!(report.expectations.is_empty());
        assert!(report.evidence_node_ids.is_empty());
    }

    #[test]
    fn eval_report_deserializes_legacy_score_kind() -> MemoryResult<()> {
        let suite = EvalSuite {
            contract_version: EVAL_CONTRACT_VERSION,
            name: "legacy-score-kind".to_string(),
            source: None,
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            environment_id: None,
            shared_haystack: false,
            cases: vec![EvalCase {
                id: "case".to_string(),
                question: "what database is configured?".to_string(),
                events: vec![EvalEvent {
                    kind: MemoryNodeKind::Fact,
                    text: "Checkout uses DATABASE_URL.".to_string(),
                    observed_at_unix_ms: Some(1_000),
                    ingested_at_unix_ms: None,
                }],
                expected_evidence_contains: vec!["DATABASE_URL".to_string()],
                expected_tier: Some(MemoryTier::ActiveReconstruction),
                ..case_defaults()
            }],
        };
        let report = run_eval_suite(&suite, &EvalOptions::default())?;
        let mut value = serde_json::to_value(report)?;
        value.as_object_mut().unwrap().remove("score_kind");
        let case = value["case_reports"][0].as_object_mut().unwrap();
        case.remove("score_kind");
        case.remove("content_score");
        case.insert("score".to_string(), serde_json::json!(1.0));
        value["score"] = serde_json::json!(1.0);

        let report = serde_json::from_value::<EvalReport>(value)?;

        assert_eq!(report.score_kind, EvalScoreKind::ContentExpectationPassRate);
        assert_eq!(report.score, 1.0);
        assert_eq!(
            report.case_reports[0].score_kind,
            EvalScoreKind::ContentExpectationPassRate
        );
        assert_eq!(report.case_reports[0].score, 1.0);
        assert_eq!(report.case_reports[0].content_score, 1.0);
        Ok(())
    }

    #[test]
    fn context_saturation_gap_uses_only_cases_with_baselines() -> MemoryResult<()> {
        let suite = EvalSuite {
            contract_version: EVAL_CONTRACT_VERSION,
            name: "baseline-subset".to_string(),
            source: None,
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            environment_id: None,
            shared_haystack: false,
            cases: vec![
                EvalCase {
                    id: "baseline".to_string(),
                    question: "what route is healthy?".to_string(),
                    events: vec![EvalEvent {
                        kind: MemoryNodeKind::Fact,
                        text: "The health route is /livez.".to_string(),
                        observed_at_unix_ms: Some(1_000),
                        ingested_at_unix_ms: None,
                    }],
                    baseline_full_context_score: Some(1.0),
                    expected_evidence_contains: vec!["/readyz".to_string()],
                    ..case_defaults()
                },
                EvalCase {
                    id: "no-baseline-pass".to_string(),
                    question: "what database is configured?".to_string(),
                    events: vec![EvalEvent {
                        kind: MemoryNodeKind::Fact,
                        text: "Checkout uses DATABASE_URL.".to_string(),
                        observed_at_unix_ms: Some(2_000),
                        ingested_at_unix_ms: None,
                    }],
                    expected_evidence_contains: vec!["DATABASE_URL".to_string()],
                    ..case_defaults()
                },
            ],
        };

        let report = run_eval_suite(&suite, &EvalOptions::default())?;

        assert_eq!(report.score, 0.5);
        assert_eq!(report.baseline_full_context_score, Some(1.0));
        assert_eq!(report.context_saturation_gap, Some(1.0));
        Ok(())
    }

    #[test]
    fn eval_score_tracks_partial_expectation_matches() -> MemoryResult<()> {
        let suite = EvalSuite {
            contract_version: EVAL_CONTRACT_VERSION,
            name: "partial".to_string(),
            source: None,
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            environment_id: None,
            shared_haystack: false,
            cases: vec![EvalCase {
                id: "partial".to_string(),
                question: "what database is configured?".to_string(),
                events: vec![EvalEvent {
                    kind: MemoryNodeKind::Fact,
                    text: "Checkout uses DATABASE_URL.".to_string(),
                    observed_at_unix_ms: Some(1_000),
                    ingested_at_unix_ms: None,
                }],
                expected_evidence_contains: vec![
                    "DATABASE_URL".to_string(),
                    "REDIS_URL".to_string(),
                ],
                ..case_defaults()
            }],
        };

        let report = run_eval_suite(&suite, &EvalOptions::default())?;
        let case = &report.case_reports[0];

        assert_eq!(report.score, 0.5);
        assert_eq!(report.passed, 0);
        assert_eq!(report.failed, 1);
        assert_eq!(case.content_score, 0.5);
        assert_eq!(case.score, 0.5);
        assert_eq!(case.checks_total, 2);
        assert_eq!(case.checks_matched, 1);
        assert_eq!(case.checks_failed, 1);
        assert_eq!(case.expectations.len(), 2);
        assert!(
            case.expectations.iter().any(|expectation| {
                expectation.expected == "DATABASE_URL" && expectation.matched
            })
        );
        assert!(
            case.expectations
                .iter()
                .any(|expectation| { expectation.expected == "REDIS_URL" && !expectation.matched })
        );
        Ok(())
    }

    #[test]
    fn eval_tier_gate_failure_zeroes_effective_score() -> MemoryResult<()> {
        let suite = EvalSuite {
            contract_version: EVAL_CONTRACT_VERSION,
            name: "tier-gate".to_string(),
            source: None,
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            environment_id: None,
            shared_haystack: false,
            cases: vec![EvalCase {
                id: "tier".to_string(),
                question: "what database is configured?".to_string(),
                events: vec![EvalEvent {
                    kind: MemoryNodeKind::Fact,
                    text: "Checkout uses DATABASE_URL.".to_string(),
                    observed_at_unix_ms: Some(1_000),
                    ingested_at_unix_ms: None,
                }],
                expected_evidence_contains: vec!["DATABASE_URL".to_string()],
                expected_tier: Some(MemoryTier::ActiveReconstruction),
                ..case_defaults()
            }],
        };

        let report = run_eval_suite(&suite, &EvalOptions::default())?;
        let case = &report.case_reports[0];

        assert_eq!(report.score, 0.0);
        assert_eq!(report.passed, 0);
        assert_eq!(report.failed, 1);
        assert_eq!(case.content_score, 1.0);
        assert_eq!(case.score, 0.0);
        assert!(case.hard_gate_failed);
        assert!(
            case.failure_reasons
                .iter()
                .any(|reason| reason.contains("expected tier"))
        );
        Ok(())
    }

    #[test]
    fn eval_cases_are_isolated_by_default() -> MemoryResult<()> {
        let mut suite = EvalSuite {
            contract_version: EVAL_CONTRACT_VERSION,
            name: "isolation".to_string(),
            source: None,
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            environment_id: None,
            shared_haystack: false,
            cases: vec![
                EvalCase {
                    id: "first".to_string(),
                    question: "where is the shared fixture token?".to_string(),
                    events: vec![EvalEvent {
                        kind: MemoryNodeKind::Fact,
                        text: "The first case only mentions alpha.".to_string(),
                        observed_at_unix_ms: Some(1_000),
                        ingested_at_unix_ms: None,
                    }],
                    expected_evidence_contains: vec!["shared fixture token is beta".to_string()],
                    ..case_defaults()
                },
                EvalCase {
                    id: "second".to_string(),
                    question: "where is the shared fixture token?".to_string(),
                    events: vec![EvalEvent {
                        kind: MemoryNodeKind::Fact,
                        text: "The shared fixture token is beta.".to_string(),
                        observed_at_unix_ms: Some(2_000),
                        ingested_at_unix_ms: None,
                    }],
                    expected_evidence_contains: vec!["shared fixture token is beta".to_string()],
                    ..case_defaults()
                },
                EvalCase {
                    id: "third".to_string(),
                    question: "what did the first case mention?".to_string(),
                    events: vec![EvalEvent {
                        kind: MemoryNodeKind::Fact,
                        text: "The third case has unrelated filler.".to_string(),
                        observed_at_unix_ms: Some(3_000),
                        ingested_at_unix_ms: None,
                    }],
                    expected_evidence_contains: vec!["first case only mentions alpha".to_string()],
                    ..case_defaults()
                },
            ],
        };

        let isolated = run_eval_suite(&suite, &EvalOptions::default())?;
        suite.shared_haystack = true;
        let shared = run_eval_suite(&suite, &EvalOptions::default())?;

        assert!(!isolated.case_reports[0].passed);
        assert!(isolated.case_reports[1].passed);
        assert!(!isolated.case_reports[2].passed);
        assert!(!shared.case_reports[0].passed);
        assert!(shared.case_reports[1].passed);
        assert!(shared.case_reports[2].passed);
        Ok(())
    }

    #[test]
    fn reconstruction_budget_override_preserves_case_mode() {
        let suite = EvalSuite {
            contract_version: EVAL_CONTRACT_VERSION,
            name: "override".to_string(),
            source: None,
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            environment_id: None,
            shared_haystack: false,
            cases: Vec::new(),
        };
        let case = EvalCase {
            id: "force".to_string(),
            question: "incident chain".to_string(),
            known_at_unix_ms: Some(1_500),
            events: vec![EvalEvent {
                kind: MemoryNodeKind::Fact,
                text: "Incident alpha blocked deploys.".to_string(),
                observed_at_unix_ms: Some(1_000),
                ingested_at_unix_ms: None,
            }],
            reconstruction: Some(ReconstructionOptions {
                mode: ReconstructionMode::Force,
                max_steps: 4,
                max_tokens: 2_000,
            }),
            expected_evidence_contains: vec!["Incident alpha".to_string()],
            ..case_defaults()
        };

        let query = build_query(
            &suite,
            &case,
            &EvalOptions {
                max_tokens: None,
                reconstruction_mode: None,
                max_reconstruction_steps: Some(2),
                max_reconstruction_tokens: Some(500),
            },
        );

        assert_eq!(query.reconstruction.mode, ReconstructionMode::Force);
        assert_eq!(query.reconstruction.max_steps, 2);
        assert_eq!(query.reconstruction.max_tokens, 500);
        assert_eq!(query.scope.known_at_unix_ms, Some(1_500));
    }

    fn case_defaults() -> EvalCase {
        EvalCase {
            id: String::new(),
            ability: EvalAbility::Other,
            tenant_id: None,
            project_id: None,
            environment_id: None,
            as_of_unix_ms: None,
            known_at_unix_ms: None,
            require_fresh: false,
            question: String::new(),
            max_tokens: None,
            modes: None,
            reconstruction: None,
            baseline_full_context_score: None,
            events: Vec::new(),
            expected_answer_contains: Vec::new(),
            expected_evidence_contains: Vec::new(),
            expected_stale_contains: Vec::new(),
            expected_contradiction_contains: Vec::new(),
            expected_tier: None,
        }
    }
}
