use crate::{
    model::{BeliefRevisionOp, DistilledMemory, MemoryNodeKind},
    store::{LedgerEvent, MemoryNode},
    text::{concise, overlap_score},
};

/// Offline/sleep-time distillation boundary.
///
/// A provider-backed implementation can be added here later with constrained
/// decoding. The engine only accepts this typed output, so malformed writes do
/// not leak into the graph.
pub trait Distiller {
    fn distill(&self, event: &LedgerEvent, neighbors: &[MemoryNode]) -> Vec<DistilledMemory>;
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
    fn distill(&self, event: &LedgerEvent, neighbors: &[MemoryNode]) -> Vec<DistilledMemory> {
        let body = concise(&event.text, self.max_memory_chars);
        if body.trim().is_empty() {
            return vec![DistilledMemory {
                op: BeliefRevisionOp::Noop,
                node_kind: MemoryNodeKind::Episode,
                text: String::new(),
                target_node_id: None,
                cited_spans: vec![event.cited_span()],
            }];
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
            best_target(&body, neighbors)
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
        out
    }
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

fn best_target(text: &str, neighbors: &[MemoryNode]) -> Option<String> {
    neighbors
        .iter()
        .filter(|node| node.valid_to_unix_ms.is_none())
        .map(|node| (overlap_score(text, &node.text), node))
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
        let memories = HeuristicDistiller::default().distill(
            &event(
                MemoryNodeKind::Gotcha,
                "Checkout failed with DATABASE_URL missing. Fix by setting it.",
            ),
            &[],
        );

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
            confidence: 0.7,
            token_estimate: 8,
            observation_count: 1,
        };
        let memories = HeuristicDistiller::default().distill(
            &event(
                MemoryNodeKind::Fact,
                "Do not use the old checkout token; it is deprecated.",
            ),
            &[neighbor],
        );

        assert_eq!(memories[1].op, BeliefRevisionOp::Invalidate);
        assert_eq!(memories[1].target_node_id.as_deref(), Some("node_old"));
    }
}
