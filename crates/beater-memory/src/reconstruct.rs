use serde::{Deserialize, Serialize};

use crate::model::MemoryNodeKind;

/// Candidate visible to a read-time active reconstruction policy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReconstructionCandidate {
    pub node_id: String,
    pub kind: MemoryNodeKind,
    pub text: String,
    pub score: f32,
    pub token_estimate: u32,
}

/// One bounded reconstruction step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReconstructionStep {
    pub question: String,
    pub step_index: u8,
    pub expanded_node_id: String,
    pub remaining_tokens: u32,
    pub candidates: Vec<ReconstructionCandidate>,
}

/// Validated action returned by an active reconstruction policy.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReconstructionDecision {
    Accept { node_id: String },
    Prune { node_id: String },
    Stop,
}

/// Provider-neutral hook for Tier 2 read-time graph exploration.
pub trait ActiveReconstructor: Clone + Send + Sync + 'static {
    fn decide(&self, step: &ReconstructionStep) -> ReconstructionDecision;
}

/// Deterministic, token-free reconstruction policy used by the local engine.
#[derive(Clone, Copy, Debug, Default)]
pub struct DeterministicReconstructor;

impl ActiveReconstructor for DeterministicReconstructor {
    fn decide(&self, step: &ReconstructionStep) -> ReconstructionDecision {
        step.candidates
            .iter()
            .filter(|candidate| candidate.token_estimate <= step.remaining_tokens)
            .max_by(|left, right| left.score.total_cmp(&right.score))
            .map(|candidate| ReconstructionDecision::Accept {
                node_id: candidate.node_id.clone(),
            })
            .unwrap_or(ReconstructionDecision::Stop)
    }
}
