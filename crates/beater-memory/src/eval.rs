use std::{
    collections::{BTreeMap, BTreeSet},
    time::Instant,
};

use serde::{Deserialize, Serialize};

use crate::{
    MemoryEngine,
    error::{MemoryError, MemoryResult},
    model::{
        MemoryMode, MemoryNodeKind, MemoryQuery, MemoryScope, MemoryTier, ReconstructionMode,
        ReconstructionOptions, estimate_tokens,
    },
    store::{LedgerEvent, StoreStats},
    text::stable_id,
};

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

/// A deterministic evaluation suite run against an isolated in-memory engine.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalSuite {
    pub name: String,
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
    pub suite: String,
    pub cases: usize,
    pub passed: usize,
    pub failed: usize,
    pub score: f32,
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EvalCaseReport {
    pub id: String,
    pub ability: EvalAbility,
    pub passed: bool,
    pub score: f32,
    pub tier_used: MemoryTier,
    pub latency_ms: u64,
    pub token_estimate: u32,
    pub answer_tokens: u32,
    pub evidence_tokens: u32,
    pub reconstruction_tokens: u32,
    pub evidence_count: usize,
    pub stale_assumption_count: usize,
    pub contradiction_count: usize,
    pub failure_reasons: Vec<String>,
}

/// Run a deterministic memory evaluation suite in an isolated in-memory store.
pub fn run_eval_suite(suite: &EvalSuite, options: &EvalOptions) -> MemoryResult<EvalReport> {
    validate_suite(suite)?;
    validate_options(options)?;
    let engine = MemoryEngine::in_memory()?;
    let mut projected_events = 0_usize;
    for (case_index, case) in suite.cases.iter().enumerate() {
        let scope = case_scope(suite, case);
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
            ledger_event.ingested_at_unix_ms = ledger_event.observed_at_unix_ms;
            ledger_event.payload = serde_json::json!({
                "kind": event.kind.as_str(),
                "eval_suite": suite.name,
                "eval_case_id": case.id,
            });
            if engine.ingest_event(&ledger_event)? {
                projected_events += 1;
            }
        }
    }
    let project = engine.project_pending(projected_events.max(1))?;
    let stats = engine.store().stats()?;

    let mut case_reports = Vec::with_capacity(suite.cases.len());
    for case in &suite.cases {
        let query = build_query(suite, case, options);
        let started = Instant::now();
        let answer = engine.query(&query)?;
        let latency_ms = elapsed_ms(started);
        let (score, failure_reasons) = judge_case(case, &answer);
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
        case_reports.push(EvalCaseReport {
            id: case.id.clone(),
            ability: case.ability,
            passed: failure_reasons.is_empty(),
            score,
            tier_used: answer.tier_used,
            latency_ms,
            token_estimate: answer.token_estimate,
            answer_tokens,
            evidence_tokens,
            reconstruction_tokens,
            evidence_count: answer.evidence.len(),
            stale_assumption_count: answer.stale_assumptions.len(),
            contradiction_count: answer.contradictions.len(),
            failure_reasons,
        });
    }

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
    let touched = project.stored_memories_touched.max(1) as f32;
    let source = project.source_token_estimate as f32;
    let projected = project.projected_memory_token_estimate as f32;

    Ok(EvalReport {
        suite: suite.name.clone(),
        cases,
        passed,
        failed: cases.saturating_sub(passed),
        score,
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
    if suite.name.trim().is_empty() {
        return Err(MemoryError::invalid("eval suite name must not be empty"));
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

fn judge_case(case: &EvalCase, answer: &crate::MemoryAnswer) -> (f32, Vec<String>) {
    let mut matched = 0_usize;
    let mut total = 0_usize;
    let mut failures = Vec::new();

    check_expected(
        "answer",
        &case.expected_answer_contains,
        std::slice::from_ref(&answer.answer),
        &mut matched,
        &mut total,
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
        &mut failures,
    );
    if let Some(expected_tier) = case.expected_tier
        && answer.tier_used != expected_tier
    {
        failures.push(format!(
            "expected tier {expected_tier:?}, got {:?}",
            answer.tier_used
        ));
    }

    if total == 0 {
        return (0.0, vec!["case has no expectations".to_string()]);
    }
    (matched as f32 / total as f32, failures)
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
    failures: &mut Vec<String>,
) {
    for expected in expected {
        *total += 1;
        if haystacks
            .iter()
            .any(|haystack| contains_case_insensitive(haystack.as_ref(), expected))
        {
            *matched += 1;
        } else {
            failures.push(format!("missing {label} substring {expected:?}"));
        }
    }
}

fn contains_case_insensitive(haystack: &str, needle: &str) -> bool {
    haystack
        .to_ascii_lowercase()
        .contains(&needle.to_ascii_lowercase())
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
            name: "smoke".to_string(),
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
                        },
                        EvalEvent {
                            kind: MemoryNodeKind::Fact,
                            text: "Do not use the legacy API token; it is deprecated. Use the scoped API token.".to_string(),
                            observed_at_unix_ms: Some(3_000),
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
        assert_eq!(report.passed, 2);
        assert_eq!(report.failed, 0);
        assert_eq!(report.score, 1.0);
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
        Ok(())
    }

    #[test]
    fn eval_suite_rejects_cases_without_expectations() {
        let suite = EvalSuite {
            name: "invalid".to_string(),
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
            name: "invalid".to_string(),
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
    fn context_saturation_gap_uses_only_cases_with_baselines() -> MemoryResult<()> {
        let suite = EvalSuite {
            name: "baseline-subset".to_string(),
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
    fn eval_cases_are_isolated_by_default() -> MemoryResult<()> {
        let mut suite = EvalSuite {
            name: "isolation".to_string(),
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
                    }],
                    expected_evidence_contains: vec!["shared fixture token is beta".to_string()],
                    ..case_defaults()
                },
            ],
        };

        let isolated = run_eval_suite(&suite, &EvalOptions::default())?;
        suite.shared_haystack = true;
        let shared = run_eval_suite(&suite, &EvalOptions::default())?;

        assert!(!isolated.case_reports[0].passed);
        assert!(isolated.case_reports[1].passed);
        assert!(shared.case_reports[0].passed);
        assert!(shared.case_reports[1].passed);
        Ok(())
    }

    #[test]
    fn reconstruction_budget_override_preserves_case_mode() {
        let suite = EvalSuite {
            name: "override".to_string(),
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            environment_id: None,
            shared_haystack: false,
            cases: Vec::new(),
        };
        let case = EvalCase {
            id: "force".to_string(),
            question: "incident chain".to_string(),
            events: vec![EvalEvent {
                kind: MemoryNodeKind::Fact,
                text: "Incident alpha blocked deploys.".to_string(),
                observed_at_unix_ms: Some(1_000),
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
    }

    fn case_defaults() -> EvalCase {
        EvalCase {
            id: String::new(),
            ability: EvalAbility::Other,
            tenant_id: None,
            project_id: None,
            environment_id: None,
            as_of_unix_ms: None,
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
