use crate::model::{MemoryMode, MemoryNodeKind, RoutingReason, RoutingReport};

pub(crate) fn route_memory_query(
    question: &str,
    allowed_modes: &[MemoryMode],
    modes_explicit: bool,
) -> RoutingReport {
    let allowed_modes = normalized_modes(allowed_modes);
    let default_modes = MemoryNodeKind::default_modes();
    if modes_explicit || !same_mode_set(&allowed_modes, &default_modes) {
        return RoutingReport {
            allowed_modes: allowed_modes.clone(),
            routed_modes: allowed_modes,
            reconstruction_modes: None,
            reason: RoutingReason::AmbiguousFallback,
            confidence: 0.20,
        };
    }
    let lower = question.to_ascii_lowercase();
    let mut preferred = Vec::new();
    let mut matches = Vec::new();
    if contains_any(
        &lower,
        &[
            "current",
            "state",
            "config",
            "setting",
            "environment",
            "now",
        ],
    ) {
        push_mode(&mut preferred, MemoryMode::State);
        matches.push(RoutingReason::StateIntent);
    }
    if contains_any(
        &lower,
        &[
            "fix", "failure", "fails", "error", "bug", "gotcha", "blocked", "broken",
        ],
    ) {
        push_mode(&mut preferred, MemoryMode::Gotcha);
        push_mode(&mut preferred, MemoryMode::Procedural);
        matches.push(RoutingReason::GotchaIntent);
    }
    if contains_any(
        &lower,
        &[
            "procedure",
            "runbook",
            "workflow",
            "steps",
            "step-by-step",
            "how to",
        ],
    ) {
        push_mode(&mut preferred, MemoryMode::Procedural);
        matches.push(RoutingReason::ProceduralIntent);
    }
    if contains_any(
        &lower,
        &[
            "when", "timeline", "history", "episode", "trace", "span", "happened",
        ],
    ) {
        push_mode(&mut preferred, MemoryMode::Episodic);
        push_mode(&mut preferred, MemoryMode::Semantic);
        matches.push(RoutingReason::EpisodicIntent);
    }
    if contains_any(
        &lower,
        &[
            "what",
            "which",
            "where",
            "who",
            "fact",
            "token",
            "remember about",
        ],
    ) {
        push_mode(&mut preferred, MemoryMode::Semantic);
        matches.push(RoutingReason::SemanticIntent);
    }
    let (routed_modes, constraint_fallback) = constrain_modes(preferred, &allowed_modes);
    let (reason, confidence) = if matches.is_empty() || constraint_fallback {
        (RoutingReason::AmbiguousFallback, 0.20)
    } else if matches.len() > 1 {
        (RoutingReason::AmbiguousFallback, 0.55)
    } else {
        let reason = matches[0];
        let confidence = match reason {
            RoutingReason::StateIntent | RoutingReason::ProceduralIntent => 0.85,
            RoutingReason::GotchaIntent | RoutingReason::EpisodicIntent => 0.75,
            RoutingReason::SemanticIntent => 0.65,
            RoutingReason::AmbiguousFallback | RoutingReason::EmptyRouteFallback => 0.20,
        };
        (reason, confidence)
    };
    RoutingReport {
        allowed_modes,
        routed_modes,
        reconstruction_modes: None,
        reason,
        confidence,
    }
}

pub(crate) fn fallback_route_after_empty_evidence(report: &RoutingReport) -> RoutingReport {
    RoutingReport {
        allowed_modes: report.allowed_modes.clone(),
        routed_modes: report.allowed_modes.clone(),
        reconstruction_modes: report.reconstruction_modes.clone(),
        reason: RoutingReason::EmptyRouteFallback,
        confidence: 0.0,
    }
}

pub(crate) fn modes_accept_kind(modes: &[MemoryMode], kind: MemoryNodeKind) -> bool {
    modes.iter().any(|mode| mode.accepts(kind))
}

pub(crate) fn support_kinds_for_modes(modes: &[MemoryMode]) -> Vec<MemoryNodeKind> {
    let mut kinds = vec![MemoryNodeKind::EntityCue];
    for mode in modes {
        for kind in kinds_for_mode(*mode) {
            if !kinds.contains(&kind) {
                kinds.push(kind);
            }
        }
    }
    kinds
}

fn kinds_for_mode(mode: MemoryMode) -> Vec<MemoryNodeKind> {
    match mode {
        MemoryMode::Semantic => vec![
            MemoryNodeKind::Fact,
            MemoryNodeKind::Tag,
            MemoryNodeKind::Topic,
        ],
        MemoryMode::Episodic => vec![MemoryNodeKind::Episode],
        MemoryMode::Procedural => vec![MemoryNodeKind::Procedure],
        MemoryMode::Gotcha => vec![MemoryNodeKind::Gotcha, MemoryNodeKind::AntiMemory],
        MemoryMode::State => vec![MemoryNodeKind::State],
    }
}

fn normalized_modes(modes: &[MemoryMode]) -> Vec<MemoryMode> {
    let mut out = Vec::new();
    for mode in modes {
        if !out.contains(mode) {
            out.push(*mode);
        }
    }
    if out.is_empty() {
        MemoryNodeKind::default_modes()
    } else {
        out
    }
}

fn same_mode_set(left: &[MemoryMode], right: &[MemoryMode]) -> bool {
    left.len() == right.len() && left.iter().all(|mode| right.contains(mode))
}

fn constrain_modes(
    preferred: Vec<MemoryMode>,
    allowed_modes: &[MemoryMode],
) -> (Vec<MemoryMode>, bool) {
    if preferred.is_empty() {
        return (allowed_modes.to_vec(), false);
    }
    let routed = preferred
        .into_iter()
        .filter(|mode| allowed_modes.contains(mode))
        .fold(Vec::new(), |mut out, mode| {
            if !out.contains(&mode) {
                out.push(mode);
            }
            out
        });
    if routed.is_empty() {
        (allowed_modes.to_vec(), true)
    } else {
        (routed, false)
    }
}

fn push_mode(modes: &mut Vec<MemoryMode>, mode: MemoryMode) {
    if !modes.contains(&mode) {
        modes.push(mode);
    }
}

fn contains_any(value: &str, markers: &[&str]) -> bool {
    markers.iter().any(|marker| value.contains(marker))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn modes(question: &str) -> Vec<MemoryMode> {
        route_memory_query(question, &MemoryNodeKind::default_modes(), false).routed_modes
    }

    #[test]
    fn routes_failure_queries_to_gotchas_and_procedures() {
        assert_eq!(
            modes("how do we fix checkout failure?"),
            vec![MemoryMode::Gotcha, MemoryMode::Procedural]
        );
    }

    #[test]
    fn routes_workflow_queries_to_procedures() {
        assert_eq!(modes("deploy workflow steps"), vec![MemoryMode::Procedural]);
    }

    #[test]
    fn routes_current_config_queries_to_state() {
        assert_eq!(modes("current production config"), vec![MemoryMode::State]);
    }

    #[test]
    fn routes_timeline_queries_to_episodes_and_facts() {
        assert_eq!(
            modes("when did checkout fail in the timeline?"),
            vec![MemoryMode::Episodic, MemoryMode::Semantic]
        );
    }

    #[test]
    fn ambiguous_queries_fall_back_to_all_allowed_modes() {
        let report = route_memory_query(
            "alpha beta",
            &[MemoryMode::Semantic, MemoryMode::Gotcha],
            true,
        );

        assert_eq!(report.reason, RoutingReason::AmbiguousFallback);
        assert_eq!(
            report.routed_modes,
            vec![MemoryMode::Semantic, MemoryMode::Gotcha]
        );
    }

    #[test]
    fn explicit_modes_constrain_routing() {
        let report = route_memory_query("deploy workflow steps", &[MemoryMode::Semantic], true);

        assert_eq!(report.routed_modes, vec![MemoryMode::Semantic]);
        assert_eq!(report.reason, RoutingReason::AmbiguousFallback);
    }

    #[test]
    fn explicit_multi_modes_are_preserved() {
        let report = route_memory_query(
            "deploy workflow steps",
            &[MemoryMode::Semantic, MemoryMode::Procedural],
            true,
        );

        assert_eq!(
            report.routed_modes,
            vec![MemoryMode::Semantic, MemoryMode::Procedural]
        );
        assert_eq!(report.reason, RoutingReason::AmbiguousFallback);
    }

    #[test]
    fn explicit_all_modes_are_preserved() {
        let report = route_memory_query(
            "deploy workflow steps",
            &[
                MemoryMode::State,
                MemoryMode::Semantic,
                MemoryMode::Gotcha,
                MemoryMode::Episodic,
                MemoryMode::Procedural,
            ],
            true,
        );

        assert_eq!(
            report.routed_modes,
            vec![
                MemoryMode::State,
                MemoryMode::Semantic,
                MemoryMode::Gotcha,
                MemoryMode::Episodic,
                MemoryMode::Procedural,
            ]
        );
        assert_eq!(report.reason, RoutingReason::AmbiguousFallback);
    }

    #[test]
    fn support_kinds_keep_entity_cues_for_graph_traversal() {
        assert_eq!(
            support_kinds_for_modes(&[MemoryMode::Procedural]),
            vec![MemoryNodeKind::EntityCue, MemoryNodeKind::Procedure]
        );
    }
}
