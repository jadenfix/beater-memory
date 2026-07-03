use crate::{
    error::{MemoryError, MemoryResult},
    model::{BeliefRevisionOp, DistilledMemory, MemoryNodeKind, estimate_tokens},
    store::{LedgerEvent, MemoryNode},
    text::{concise, overlap_score},
};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Offline/sleep-time distillation boundary.
///
/// Provider-backed implementations should return only typed, validated output.
/// The engine revalidates this boundary before applying projection writes.
pub trait Distiller {
    fn distill(
        &self,
        event: &LedgerEvent,
        neighbors: &[MemoryNode],
    ) -> MemoryResult<DistillOutcome>;

    fn supports_late_replay(&self) -> bool {
        false
    }
}

/// Provider/economics counters from one distillation attempt.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistillMetrics {
    pub provider_calls: usize,
    pub provider_errors: usize,
    pub schema_errors: usize,
    pub repair_attempts: usize,
    pub repair_successes: usize,
    pub rejected_outputs: usize,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub elapsed_ms: u64,
}

/// Distilled memory bundle plus counters. Rejected outcomes are not projected.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistillOutcome {
    pub memories: Vec<DistilledMemory>,
    pub metrics: DistillMetrics,
    pub rejected: bool,
}

impl DistillOutcome {
    #[must_use]
    pub fn accepted(memories: Vec<DistilledMemory>) -> Self {
        Self {
            memories,
            metrics: DistillMetrics::default(),
            rejected: false,
        }
    }

    #[must_use]
    pub fn rejected(metrics: DistillMetrics) -> Self {
        Self {
            memories: Vec::new(),
            metrics,
            rejected: true,
        }
    }
}

/// Synchronous provider interface for constrained distillation JSON.
pub trait DistillationProvider {
    fn distill(&self, prompt: DistillationPrompt<'_>) -> MemoryResult<String>;

    fn repair(&self, _prompt: DistillationRepairPrompt<'_>) -> MemoryResult<String> {
        Err(MemoryError::invalid(
            "distillation provider does not support repair",
        ))
    }
}

/// Input passed to a provider-backed distiller.
#[derive(Clone, Copy, Debug)]
pub struct DistillationPrompt<'a> {
    pub event: &'a LedgerEvent,
    pub neighbors: &'a [MemoryNode],
}

/// Repair request for malformed provider output.
#[derive(Clone, Copy, Debug)]
pub struct DistillationRepairPrompt<'a> {
    pub event: &'a LedgerEvent,
    pub neighbors: &'a [MemoryNode],
    pub raw_output: &'a str,
    pub error: &'a str,
}

/// Provider-backed distiller with schema validation and bounded repair.
#[derive(Clone, Debug)]
pub struct ProviderDistiller<P> {
    provider: P,
    max_repairs: usize,
}

impl<P> ProviderDistiller<P> {
    #[must_use]
    pub fn new(provider: P) -> Self {
        Self {
            provider,
            max_repairs: 1,
        }
    }

    #[must_use]
    pub fn with_max_repairs(mut self, max_repairs: usize) -> Self {
        self.max_repairs = max_repairs;
        self
    }
}

/// Deterministic first-principles distiller used by the local MVP.
#[derive(Clone, Debug)]
pub struct HeuristicDistiller {
    max_memory_chars: usize,
}

impl Default for HeuristicDistiller {
    fn default() -> Self {
        Self {
            max_memory_chars: 900,
        }
    }
}

impl HeuristicDistiller {
    #[must_use]
    pub fn new(max_memory_chars: usize) -> Self {
        Self { max_memory_chars }
    }
}

impl Distiller for HeuristicDistiller {
    fn distill(
        &self,
        event: &LedgerEvent,
        neighbors: &[MemoryNode],
    ) -> MemoryResult<DistillOutcome> {
        let body = concise(&event.text, self.max_memory_chars);
        if body.trim().is_empty() {
            return Ok(DistillOutcome::accepted(vec![DistilledMemory {
                op: BeliefRevisionOp::Noop,
                node_kind: MemoryNodeKind::Episode,
                text: String::new(),
                target_node_id: None,
                cited_spans: vec![event.cited_span()],
            }]));
        }

        let cited_span = event.cited_span();
        let mut out = vec![DistilledMemory::add(
            MemoryNodeKind::Episode,
            format!("{} {}: {body}", event.span_kind, event.name),
            cited_span.clone(),
        )];

        let kind = classify_memory_kind(event, &body);
        let op = classify_op(&body);
        let target_node_id = if op == BeliefRevisionOp::Invalidate {
            best_target(&body, kind, neighbors)
        } else {
            None
        };

        out.push(DistilledMemory {
            op,
            node_kind: kind,
            text: body,
            target_node_id,
            cited_spans: vec![cited_span],
        });
        Ok(DistillOutcome::accepted(out))
    }

    fn supports_late_replay(&self) -> bool {
        true
    }
}

impl<P: DistillationProvider> Distiller for ProviderDistiller<P> {
    fn distill(
        &self,
        event: &LedgerEvent,
        neighbors: &[MemoryNode],
    ) -> MemoryResult<DistillOutcome> {
        let started = Instant::now();
        let mut metrics = DistillMetrics {
            input_tokens: estimate_distillation_input_tokens(event, neighbors),
            ..DistillMetrics::default()
        };
        metrics.provider_calls += 1;
        let mut raw = match self
            .provider
            .distill(DistillationPrompt { event, neighbors })
        {
            Ok(raw) => raw,
            Err(_) => {
                metrics.provider_errors += 1;
                metrics.rejected_outputs += 1;
                metrics.elapsed_ms = elapsed_ms(started);
                return Ok(DistillOutcome::rejected(metrics));
            }
        };
        metrics.output_tokens += estimate_tokens(&raw);

        let mut repairs_used = 0;
        loop {
            match parse_provider_output(&raw) {
                Ok(memories) => {
                    if repairs_used > 0 {
                        metrics.repair_successes += 1;
                    }
                    metrics.elapsed_ms = elapsed_ms(started);
                    return Ok(DistillOutcome {
                        memories,
                        metrics,
                        rejected: false,
                    });
                }
                Err(error) => {
                    metrics.schema_errors += 1;
                    if repairs_used >= self.max_repairs {
                        metrics.rejected_outputs += 1;
                        metrics.elapsed_ms = elapsed_ms(started);
                        return Ok(DistillOutcome::rejected(metrics));
                    }
                    metrics.repair_attempts += 1;
                    metrics.provider_calls += 1;
                    let repaired = match self.provider.repair(DistillationRepairPrompt {
                        event,
                        neighbors,
                        raw_output: &raw,
                        error: &error,
                    }) {
                        Ok(repaired) => repaired,
                        Err(_) => {
                            metrics.provider_errors += 1;
                            metrics.rejected_outputs += 1;
                            metrics.elapsed_ms = elapsed_ms(started);
                            return Ok(DistillOutcome::rejected(metrics));
                        }
                    };
                    metrics.output_tokens += estimate_tokens(&repaired);
                    raw = repaired;
                    repairs_used += 1;
                }
            }
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderOutput {
    memories: Vec<ProviderMemory>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderMemory {
    op: BeliefRevisionOp,
    node_kind: MemoryNodeKind,
    text: String,
    target_node_id: Option<String>,
    cited_spans: Vec<crate::model::CitedSpan>,
}

fn parse_provider_output(raw: &str) -> Result<Vec<DistilledMemory>, String> {
    let output: ProviderOutput =
        serde_json::from_str(raw).map_err(|err| format!("invalid provider JSON: {err}"))?;
    if output.memories.is_empty() {
        return Err("provider output must include at least one memory".to_string());
    }
    let memories = output
        .memories
        .into_iter()
        .map(|memory| DistilledMemory {
            op: memory.op,
            node_kind: memory.node_kind,
            text: memory.text,
            target_node_id: memory.target_node_id,
            cited_spans: memory.cited_spans,
        })
        .collect::<Vec<_>>();
    for (index, memory) in memories.iter().enumerate() {
        memory
            .validate()
            .map_err(|err| format!("invalid memory at index {index}: {err}"))?;
    }
    Ok(memories)
}

fn estimate_distillation_input_tokens(event: &LedgerEvent, neighbors: &[MemoryNode]) -> u32 {
    estimate_tokens(&event.text)
        + neighbors
            .iter()
            .map(|neighbor| estimate_tokens(&neighbor.text))
            .sum::<u32>()
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)
}

fn classify_memory_kind(event: &LedgerEvent, text: &str) -> MemoryNodeKind {
    let declared = event.name.to_ascii_lowercase();
    match declared.as_str() {
        "fact" | "semantic" => return MemoryNodeKind::Fact,
        "episode" | "episodic" => return MemoryNodeKind::Episode,
        "procedure" | "runbook" | "workflow" => return MemoryNodeKind::Procedure,
        "state" => return MemoryNodeKind::State,
        "gotcha" | "failure" => return MemoryNodeKind::Gotcha,
        "anti_memory" | "anti-memory" => return MemoryNodeKind::AntiMemory,
        _ => {}
    }

    let lower = text.to_ascii_lowercase();
    if lower.contains("do not use")
        || lower.contains("looked relevant")
        || lower.contains("misleading")
        || lower.contains("red herring")
    {
        MemoryNodeKind::AntiMemory
    } else if lower.contains("error")
        || lower.contains("failed")
        || lower.contains("failure")
        || lower.contains("panic")
        || lower.contains("regression")
        || lower.contains("gotcha")
        || lower.contains("blocked")
    {
        MemoryNodeKind::Gotcha
    } else if lower.contains("run ")
        || lower.contains("command")
        || lower.contains("step")
        || lower.contains("fix by")
        || lower.contains("workaround")
        || lower.contains("procedure")
    {
        MemoryNodeKind::Procedure
    } else if lower.contains("current ")
        || lower.contains("configured")
        || lower.contains("environment")
        || lower.contains("state")
        || lower.contains("uses ")
    {
        MemoryNodeKind::State
    } else {
        MemoryNodeKind::Fact
    }
}

fn classify_op(text: &str) -> BeliefRevisionOp {
    let lower = text.to_ascii_lowercase();
    if lower.contains("no longer")
        || lower.contains("deprecated")
        || lower.contains("invalidated")
        || lower.contains("stale")
        || lower.contains("not true")
        || lower.contains("replace ")
        || lower.contains("instead of")
        || lower.contains("do not use")
    {
        BeliefRevisionOp::Invalidate
    } else if lower.contains("update")
        || lower.contains("changed")
        || lower.contains("now ")
        || lower.contains("new ")
    {
        BeliefRevisionOp::Update
    } else {
        BeliefRevisionOp::Add
    }
}

fn best_target(text: &str, kind: MemoryNodeKind, neighbors: &[MemoryNode]) -> Option<String> {
    neighbors
        .iter()
        .filter(|node| node.kind != MemoryNodeKind::Episode)
        .filter(|node| node.kind != MemoryNodeKind::EntityCue)
        .map(|node| {
            let kind_bonus = if node.kind == kind { 0.08 } else { 0.0 };
            (overlap_score(text, &node.text) + kind_bonus, node)
        })
        .filter(|(score, _)| *score >= 0.12)
        .max_by(|left, right| left.0.total_cmp(&right.0))
        .map(|(_, node)| node.id.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(kind: MemoryNodeKind, text: &str) -> LedgerEvent {
        LedgerEvent::direct_memory_write("tenant", "project", kind, text)
    }

    #[test]
    fn emits_episode_plus_typed_memory() {
        let memories = HeuristicDistiller::default()
            .distill(
                &event(
                    MemoryNodeKind::Gotcha,
                    "Checkout failed with DATABASE_URL missing. Fix by setting it.",
                ),
                &[],
            )
            .unwrap_or_else(|err| panic!("{err}"))
            .memories;

        assert_eq!(memories.len(), 2);
        assert_eq!(memories[0].node_kind, MemoryNodeKind::Episode);
        assert_eq!(memories[1].node_kind, MemoryNodeKind::Gotcha);
    }

    #[test]
    fn invalidations_can_target_neighbors() {
        let neighbor = MemoryNode {
            id: "node_old".to_string(),
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            environment_id: None,
            kind: MemoryNodeKind::Fact,
            text: "Use the old checkout token.".to_string(),
            canonical_key: "fact:old checkout token".to_string(),
            created_at_unix_ms: 1,
            updated_at_unix_ms: 1,
            valid_from_unix_ms: 1,
            valid_to_unix_ms: None,
            valid_to_event_id: None,
            confidence: 0.7,
            token_estimate: 8,
            observation_count: 1,
        };
        let memories = HeuristicDistiller::default()
            .distill(
                &event(
                    MemoryNodeKind::Fact,
                    "Do not use the old checkout token; it is deprecated.",
                ),
                &[neighbor],
            )
            .unwrap_or_else(|err| panic!("{err}"))
            .memories;

        assert_eq!(memories[1].op, BeliefRevisionOp::Invalidate);
        assert_eq!(memories[1].target_node_id.as_deref(), Some("node_old"));
    }

    #[test]
    fn invalidations_prefer_typed_memory_over_episode_scaffolding() {
        let episode = MemoryNode {
            id: "node_episode".to_string(),
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            environment_id: None,
            kind: MemoryNodeKind::Episode,
            text: "memory.write fact: Use the old checkout token.".to_string(),
            canonical_key: "episode:trace:span:1".to_string(),
            created_at_unix_ms: 1,
            updated_at_unix_ms: 1,
            valid_from_unix_ms: 1,
            valid_to_unix_ms: None,
            valid_to_event_id: None,
            confidence: 0.7,
            token_estimate: 8,
            observation_count: 1,
        };
        let fact = MemoryNode {
            id: "node_fact".to_string(),
            tenant_id: "tenant".to_string(),
            project_id: "project".to_string(),
            environment_id: None,
            kind: MemoryNodeKind::Fact,
            text: "Use the old checkout token.".to_string(),
            canonical_key: "fact:old checkout token".to_string(),
            created_at_unix_ms: 1,
            updated_at_unix_ms: 1,
            valid_from_unix_ms: 1,
            valid_to_unix_ms: None,
            valid_to_event_id: None,
            confidence: 0.7,
            token_estimate: 8,
            observation_count: 1,
        };

        let memories = HeuristicDistiller::default()
            .distill(
                &event(
                    MemoryNodeKind::Fact,
                    "Do not use the old checkout token; it is deprecated.",
                ),
                &[episode, fact],
            )
            .unwrap_or_else(|err| panic!("{err}"))
            .memories;

        assert_eq!(memories[1].op, BeliefRevisionOp::Invalidate);
        assert_eq!(memories[1].target_node_id.as_deref(), Some("node_fact"));
    }

    #[derive(Clone)]
    struct FakeProvider {
        raw: Result<String, String>,
        repaired: Option<String>,
    }

    impl DistillationProvider for FakeProvider {
        fn distill(&self, _prompt: DistillationPrompt<'_>) -> MemoryResult<String> {
            self.raw
                .clone()
                .map_err(|err| MemoryError::invalid(format!("provider failed: {err}")))
        }

        fn repair(&self, prompt: DistillationRepairPrompt<'_>) -> MemoryResult<String> {
            assert!(!prompt.raw_output.is_empty());
            assert!(!prompt.error.is_empty());
            self.repaired
                .clone()
                .ok_or_else(|| MemoryError::invalid("repair unavailable"))
        }
    }

    fn provider_json(event: &LedgerEvent, text: &str) -> String {
        serde_json::json!({
            "memories": [{
                "op": "add",
                "node_kind": "fact",
                "text": text,
                "target_node_id": null,
                "cited_spans": [event.cited_span()]
            }]
        })
        .to_string()
    }

    #[test]
    fn provider_distiller_accepts_valid_schema() {
        let event = event(MemoryNodeKind::Fact, "Checkout uses DATABASE_URL.");
        let distiller = ProviderDistiller::new(FakeProvider {
            raw: Ok(provider_json(&event, "Checkout uses DATABASE_URL.")),
            repaired: None,
        });

        let outcome = distiller
            .distill(&event, &[])
            .unwrap_or_else(|err| panic!("{err}"));

        assert!(!outcome.rejected);
        assert_eq!(outcome.memories.len(), 1);
        assert_eq!(outcome.metrics.provider_calls, 1);
        assert_eq!(outcome.metrics.repair_attempts, 0);
        assert!(outcome.metrics.input_tokens > 0);
        assert!(outcome.metrics.output_tokens > 0);
    }

    #[test]
    fn provider_distiller_repairs_invalid_schema_once() {
        let event = event(MemoryNodeKind::Fact, "Checkout uses DATABASE_URL.");
        let distiller = ProviderDistiller::new(FakeProvider {
            raw: Ok("{\"memories\":".to_string()),
            repaired: Some(provider_json(&event, "Checkout uses DATABASE_URL.")),
        });

        let outcome = distiller
            .distill(&event, &[])
            .unwrap_or_else(|err| panic!("{err}"));

        assert!(!outcome.rejected);
        assert_eq!(outcome.metrics.provider_calls, 2);
        assert_eq!(outcome.metrics.schema_errors, 1);
        assert_eq!(outcome.metrics.repair_attempts, 1);
        assert_eq!(outcome.metrics.repair_successes, 1);
    }

    #[test]
    fn provider_distiller_rejects_after_failed_repair() {
        let event = event(MemoryNodeKind::Fact, "Checkout uses DATABASE_URL.");
        let distiller = ProviderDistiller::new(FakeProvider {
            raw: Ok("{\"memories\":".to_string()),
            repaired: Some("{\"memories\":".to_string()),
        });

        let outcome = distiller
            .distill(&event, &[])
            .unwrap_or_else(|err| panic!("{err}"));

        assert!(outcome.rejected);
        assert!(outcome.memories.is_empty());
        assert_eq!(outcome.metrics.schema_errors, 2);
        assert_eq!(outcome.metrics.rejected_outputs, 1);
    }

    #[test]
    fn provider_distiller_rejects_transport_failure() {
        let event = event(MemoryNodeKind::Fact, "Checkout uses DATABASE_URL.");
        let distiller = ProviderDistiller::new(FakeProvider {
            raw: Err("timeout".to_string()),
            repaired: None,
        });

        let outcome = distiller
            .distill(&event, &[])
            .unwrap_or_else(|err| panic!("{err}"));

        assert!(outcome.rejected);
        assert_eq!(outcome.metrics.provider_errors, 1);
        assert_eq!(outcome.metrics.rejected_outputs, 1);
    }

    #[test]
    fn provider_distiller_rejects_unknown_fields() {
        let event = event(MemoryNodeKind::Fact, "Checkout uses DATABASE_URL.");
        let raw = serde_json::json!({
            "memories": [{
                "op": "add",
                "node_kind": "fact",
                "text": "Checkout uses DATABASE_URL.",
                "target_node_id": null,
                "cited_spans": [event.cited_span()],
                "unexpected": true
            }]
        })
        .to_string();
        let distiller = ProviderDistiller::new(FakeProvider {
            raw: Ok(raw),
            repaired: None,
        });

        let outcome = distiller
            .distill(&event, &[])
            .unwrap_or_else(|err| panic!("{err}"));

        assert!(outcome.rejected);
        assert_eq!(outcome.metrics.schema_errors, 1);
        assert_eq!(outcome.metrics.repair_attempts, 1);
        assert_eq!(outcome.metrics.rejected_outputs, 1);
    }

    #[test]
    fn provider_distiller_rejects_empty_batches() {
        let event = event(MemoryNodeKind::Fact, "Checkout uses DATABASE_URL.");
        let distiller = ProviderDistiller::new(FakeProvider {
            raw: Ok(serde_json::json!({ "memories": [] }).to_string()),
            repaired: None,
        });

        let outcome = distiller
            .distill(&event, &[])
            .unwrap_or_else(|err| panic!("{err}"));

        assert!(outcome.rejected);
        assert_eq!(outcome.metrics.schema_errors, 1);
        assert_eq!(outcome.metrics.rejected_outputs, 1);
    }

    #[derive(Clone)]
    struct TwoRepairProvider;

    impl DistillationProvider for TwoRepairProvider {
        fn distill(&self, _prompt: DistillationPrompt<'_>) -> MemoryResult<String> {
            Ok("{\"memories\":".to_string())
        }

        fn repair(&self, prompt: DistillationRepairPrompt<'_>) -> MemoryResult<String> {
            if prompt.raw_output == "{\"memories\":" {
                Ok("{\"memories\":[]}".to_string())
            } else {
                Ok(provider_json(prompt.event, "Checkout uses DATABASE_URL."))
            }
        }
    }

    #[test]
    fn provider_distiller_honors_max_repairs() {
        let event = event(MemoryNodeKind::Fact, "Checkout uses DATABASE_URL.");
        let distiller = ProviderDistiller::new(TwoRepairProvider).with_max_repairs(2);

        let outcome = distiller
            .distill(&event, &[])
            .unwrap_or_else(|err| panic!("{err}"));

        assert!(!outcome.rejected);
        assert_eq!(outcome.metrics.provider_calls, 3);
        assert_eq!(outcome.metrics.schema_errors, 2);
        assert_eq!(outcome.metrics.repair_attempts, 2);
        assert_eq!(outcome.metrics.repair_successes, 1);
    }
}
