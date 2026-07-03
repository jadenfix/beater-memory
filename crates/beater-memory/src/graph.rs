use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use crate::{
    error::MemoryResult,
    model::{
        ActivationWeights, Contradiction, Evidence, MemoryAnswer, MemoryEdgeKind, MemoryMode,
        MemoryNodeKind, MemoryQuery, MemoryTier, ReconstructionMode, ReconstructionReason,
        ReconstructionReport, RoutingReport, StaleAssumption, blend_activation, budget_evidence,
        estimate_tokens,
    },
    reconstruct::{
        ActiveReconstructor, ReconstructionCandidate, ReconstructionDecision, ReconstructionStep,
    },
    route::{
        fallback_route_after_empty_evidence, modes_accept_kind, route_memory_query,
        support_kinds_for_modes,
    },
    store::{MemoryEdge, MemoryNode, SqliteMemoryStore, StoreScope},
    text::{concise, now_unix_ms, overlap_score, terms, top_terms},
};

pub(crate) fn answer_query(
    store: &SqliteMemoryStore,
    query: &MemoryQuery,
    weights: ActivationWeights,
    reconstructor: &impl ActiveReconstructor,
) -> MemoryResult<MemoryAnswer> {
    let now = now_unix_ms();
    let effective_as_of = query.scope.as_of_unix_ms.unwrap_or(now);
    let effective_known_at = query.scope.known_at_unix_ms;
    let mut routing = route_memory_query(&query.question, &query.modes, query.modes_explicit);
    let mut context = retrieve_with_modes(store, query, weights, effective_as_of, &routing)?;
    if context.selected.is_empty() && routing.routed_modes != routing.allowed_modes {
        routing = fallback_route_after_empty_evidence(&routing);
        context = retrieve_with_modes(store, query, weights, effective_as_of, &routing)?;
    }

    let selected = std::mem::take(&mut context.selected);
    let escalation_reason = escalation_reason(query, &selected);
    let reconstruction_modes = if escalation_reason.is_some() {
        routing.allowed_modes.clone()
    } else {
        routing.routed_modes.clone()
    };
    if escalation_reason.is_some() && routing.routed_modes != routing.allowed_modes {
        let allowed_route = fallback_route_after_empty_evidence(&routing);
        let mut allowed_context =
            retrieve_with_modes(store, query, weights, effective_as_of, &allowed_route)?;
        allowed_context.selected.clear();
        context = allowed_context;
    }

    let RetrievalContext {
        nodes,
        visible_edges,
        ranking_edges,
        node_by_id,
        selected: _,
    } = context;
    let (selected, reconstruction) = if let Some(reason) = escalation_reason {
        reconstruct_active(
            ReconstructionRequest {
                store,
                query,
                routed_modes: &reconstruction_modes,
                selected,
                nodes: &nodes,
                edges: &ranking_edges,
                as_of_unix_ms: Some(effective_as_of),
                known_at_unix_ms: effective_known_at,
                reason,
            },
            reconstructor,
        )?
    } else {
        (selected, None)
    };
    if reconstruction.is_some() {
        routing.reconstruction_modes = Some(reconstruction_modes.clone());
    }
    let stale_modes = if reconstruction.is_some() {
        reconstruction_modes.as_slice()
    } else {
        routing.routed_modes.as_slice()
    };
    let cited_spans = selected
        .iter()
        .flat_map(|item| item.cited_spans.iter().cloned())
        .collect::<BTreeMapKeyedSpans>()
        .into_vec();
    let selected_ids: BTreeSet<_> = selected.iter().map(|item| item.node_id.clone()).collect();
    let contradictions = contradictions_for(&visible_edges, &selected_ids, &node_by_id);
    let stale_assumptions = stale_assumptions_for(
        store,
        &selected,
        &contradictions,
        &node_by_id,
        query,
        stale_modes,
        ReadWindow {
            as_of_unix_ms: Some(effective_as_of),
            known_at_unix_ms: effective_known_at,
        },
    )?;
    let suggested_next_queries = suggested_queries(&query.question, &selected);
    let answer = synthesize_answer(&selected, &contradictions);
    let token_estimate =
        estimate_tokens(&answer) + selected.iter().map(|item| item.token_estimate).sum::<u32>();
    let tier_used = if reconstruction.is_some() {
        MemoryTier::ActiveReconstruction
    } else {
        MemoryTier::Activation
    };

    Ok(MemoryAnswer {
        answer,
        evidence: selected,
        cited_spans,
        contradictions,
        stale_assumptions,
        suggested_next_queries,
        token_estimate,
        tier_used,
        routing: Some(routing),
        reconstruction,
    })
}

struct RetrievalContext {
    nodes: Vec<MemoryNode>,
    visible_edges: Vec<MemoryEdge>,
    ranking_edges: Vec<MemoryEdge>,
    node_by_id: HashMap<String, MemoryNode>,
    selected: Vec<Evidence>,
}

#[derive(Clone, Copy)]
struct ReadWindow {
    as_of_unix_ms: Option<i64>,
    known_at_unix_ms: Option<i64>,
}

fn retrieve_with_modes(
    store: &SqliteMemoryStore,
    query: &MemoryQuery,
    weights: ActivationWeights,
    effective_as_of: i64,
    routing: &RoutingReport,
) -> MemoryResult<RetrievalContext> {
    let env = query.scope.environment_id.as_deref();
    let query_terms = terms(&query.question);
    let support_kinds = support_kinds_for_modes(&routing.routed_modes);
    let known_at_unix_ms = query.scope.known_at_unix_ms;
    let seeds = store.seed_nodes_observed_known_by_kinds(
        StoreScope::new(&query.scope.tenant_id, &query.scope.project_id, env),
        &query_terms,
        64,
        Some(effective_as_of),
        known_at_unix_ms,
        &support_kinds,
    )?;
    let mut raw_history_nodes = store.all_nodes_observed_known_by_kinds(
        &query.scope.tenant_id,
        &query.scope.project_id,
        env,
        effective_as_of,
        known_at_unix_ms,
        &support_kinds,
    )?;
    merge_seed_nodes(&mut raw_history_nodes, seeds);
    let mut history_nodes = Vec::new();
    for node in &raw_history_nodes {
        if let Some(snapshot) =
            store.node_snapshot_known_at(node, Some(effective_as_of), known_at_unix_ms)?
        {
            history_nodes.push(snapshot);
        }
    }
    let mut nodes = Vec::new();
    for node in &history_nodes {
        if store.node_is_visible_at(node, Some(effective_as_of), known_at_unix_ms)? {
            nodes.push(node.clone());
        }
    }
    let active_ids: BTreeSet<_> = nodes.iter().map(|node| node.id.clone()).collect();
    let edges = store.edges_for_scope(&query.scope.tenant_id, &query.scope.project_id, env)?;
    let node_by_id: HashMap<_, _> = history_nodes
        .iter()
        .map(|node| (node.id.clone(), node.clone()))
        .collect();
    let visible_edges =
        visible_edges(store, edges, effective_as_of, known_at_unix_ms, &node_by_id)?;
    let ranking_edges: Vec<_> = visible_edges
        .iter()
        .filter(|edge| {
            active_ids.contains(&edge.from_node_id) && active_ids.contains(&edge.to_node_id)
        })
        .cloned()
        .collect();
    let ppr = personalized_pagerank(&query.question, &nodes, &ranking_edges);
    let edge_strength = edge_type_strength(&visible_edges);
    let mut evidence = Vec::new();
    for node in &nodes {
        if node.kind == crate::model::MemoryNodeKind::EntityCue {
            continue;
        }
        if !modes_accept_kind(&routing.routed_modes, node.kind) {
            continue;
        }
        if query.require_fresh
            && !store.node_is_visible_at(node, Some(effective_as_of), known_at_unix_ms)?
        {
            continue;
        }
        let ppr_score = ppr.get(&node.id).copied().unwrap_or(0.0);
        let lexical = overlap_score(&query.question, &node.text);
        if ppr_score <= 0.001 && lexical <= 0.001 && !query_terms.is_empty() {
            continue;
        }
        let base = base_level(node, effective_as_of);
        let freshness = freshness(node, effective_as_of, true);
        let edge_type = edge_strength.get(&node.id).copied().unwrap_or(0.0);
        let score = blend_activation(
            ppr_score.max(lexical * 0.65),
            base,
            edge_type,
            freshness,
            weights,
        );
        if score <= 0.02 {
            continue;
        }
        let cited_spans = store.cited_spans_for_node_read_at(
            &node.id,
            Some(effective_as_of),
            known_at_unix_ms,
        )?;
        evidence.push(Evidence {
            node_id: node.id.clone(),
            kind: node.kind,
            text: node.text.clone(),
            score,
            token_estimate: node.token_estimate,
            cited_spans,
        });
    }

    let selected = budget_evidence(evidence, query.max_tokens);
    Ok(RetrievalContext {
        nodes,
        visible_edges,
        ranking_edges,
        node_by_id,
        selected,
    })
}

fn escalation_reason(query: &MemoryQuery, selected: &[Evidence]) -> Option<ReconstructionReason> {
    match query.reconstruction.mode {
        ReconstructionMode::Off => None,
        ReconstructionMode::Force => Some(ReconstructionReason::Forced),
        ReconstructionMode::Auto => {
            let mut scores = selected.iter().map(|item| item.score).collect::<Vec<_>>();
            scores.sort_by(|left, right| right.total_cmp(left));
            let top_score = scores.first().copied().unwrap_or(0.0);
            let second_score = scores.get(1).copied().unwrap_or(0.0);
            if selected.is_empty() {
                Some(ReconstructionReason::EmptyEvidence)
            } else if top_score < 0.18 {
                Some(ReconstructionReason::LowConfidence)
            } else if second_score > 0.0 && (top_score - second_score) < 0.03 {
                Some(ReconstructionReason::AmbiguousEvidence)
            } else if is_compositional_query(&query.question) {
                Some(ReconstructionReason::CompositionalQuery)
            } else {
                None
            }
        }
    }
}

fn is_compositional_query(question: &str) -> bool {
    let lower = question.to_ascii_lowercase();
    [
        "because",
        "caused",
        "connected",
        "depends",
        "related",
        "why",
        "what led",
        "after",
        "before",
        "chain",
        "path",
        "multi-hop",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

struct ReconstructionRequest<'a> {
    store: &'a SqliteMemoryStore,
    query: &'a MemoryQuery,
    routed_modes: &'a [MemoryMode],
    selected: Vec<Evidence>,
    nodes: &'a [MemoryNode],
    edges: &'a [MemoryEdge],
    as_of_unix_ms: Option<i64>,
    known_at_unix_ms: Option<i64>,
    reason: ReconstructionReason,
}

fn reconstruct_active(
    request: ReconstructionRequest<'_>,
    reconstructor: &impl ActiveReconstructor,
) -> MemoryResult<(Vec<Evidence>, Option<ReconstructionReport>)> {
    let ReconstructionRequest {
        store,
        query,
        routed_modes,
        selected,
        nodes,
        edges,
        as_of_unix_ms,
        known_at_unix_ms,
        reason,
    } = request;
    let node_by_id: HashMap<_, _> = nodes
        .iter()
        .map(|node| (node.id.clone(), node.clone()))
        .collect();
    let mut selected = selected;
    let mut selected_ids: BTreeSet<_> = selected.iter().map(|item| item.node_id.clone()).collect();
    let mut frontier: VecDeque<String> = selected_ids.iter().cloned().collect();
    if frontier.is_empty()
        && let Some(seed) = nodes
            .iter()
            .filter(|node| node.kind != MemoryNodeKind::EntityCue)
            .filter(|node| modes_accept_kind(routed_modes, node.kind))
            .find(|node| overlap_score(&query.question, &node.text) > 0.0)
    {
        frontier.push_back(seed.id.clone());
    }
    let mut expanded = BTreeSet::new();
    let mut expanded_node_ids = Vec::new();
    let mut accepted_node_ids = Vec::new();
    let mut pruned_node_ids = Vec::new();
    let mut tokens_spent = 0_u32;
    let mut steps_used = 0_u8;

    for step_index in 0..query.reconstruction.max_steps {
        let Some(expanded_node_id) = pop_next_frontier(&mut frontier, &expanded) else {
            break;
        };
        expanded.insert(expanded_node_id.clone());
        expanded_node_ids.push(expanded_node_id.clone());
        steps_used = step_index + 1;

        let remaining_tokens = query.reconstruction.max_tokens.saturating_sub(tokens_spent);
        let candidates = reconstruction_candidates(CandidateRequest {
            store,
            query,
            routed_modes,
            expanded_node_id: &expanded_node_id,
            nodes: &node_by_id,
            edges,
            selected_ids: &selected_ids,
            expanded_ids: &expanded,
            as_of_unix_ms,
            known_at_unix_ms,
        })?;
        let (candidates, budget_pruned_node_ids) =
            budget_reconstruction_candidates(candidates, remaining_tokens);
        pruned_node_ids.extend(budget_pruned_node_ids);
        if remaining_tokens == 0 {
            break;
        }
        let step = ReconstructionStep {
            question: query.question.clone(),
            step_index,
            expanded_node_id,
            remaining_tokens,
            candidates,
        };
        match reconstructor.decide(&step) {
            ReconstructionDecision::Accept { node_id } => {
                if let Some(candidate) = step
                    .candidates
                    .iter()
                    .find(|candidate| candidate.node_id == node_id)
                    && let Some(node) = node_by_id.get(&candidate.node_id)
                    && candidate.token_estimate <= remaining_tokens
                {
                    let cited_spans = store.cited_spans_for_node_read_at(
                        &node.id,
                        as_of_unix_ms,
                        known_at_unix_ms,
                    )?;
                    tokens_spent += candidate.token_estimate;
                    selected_ids.insert(node.id.clone());
                    accepted_node_ids.push(node.id.clone());
                    frontier.push_front(node.id.clone());
                    selected.push(Evidence {
                        node_id: node.id.clone(),
                        kind: node.kind,
                        text: node.text.clone(),
                        score: candidate.score,
                        token_estimate: node.token_estimate,
                        cited_spans,
                    });
                } else if step
                    .candidates
                    .iter()
                    .any(|candidate| candidate.node_id == node_id)
                {
                    pruned_node_ids.push(node_id);
                }
            }
            ReconstructionDecision::Prune { node_id } => {
                if step
                    .candidates
                    .iter()
                    .any(|candidate| candidate.node_id == node_id)
                {
                    pruned_node_ids.push(node_id);
                }
            }
            ReconstructionDecision::Stop => break,
        }
    }

    selected = budget_evidence(selected, query.max_tokens);
    Ok((
        selected,
        Some(ReconstructionReport {
            mode: query.reconstruction.mode,
            reason,
            steps_used,
            tokens_spent,
            expanded_node_ids,
            accepted_node_ids,
            pruned_node_ids,
        }),
    ))
}

fn pop_next_frontier(
    frontier: &mut VecDeque<String>,
    expanded_ids: &BTreeSet<String>,
) -> Option<String> {
    while let Some(node_id) = frontier.pop_front() {
        if !expanded_ids.contains(&node_id) {
            return Some(node_id);
        }
    }
    None
}

fn budget_reconstruction_candidates(
    mut candidates: Vec<ReconstructionCandidate>,
    max_tokens: u32,
) -> (Vec<ReconstructionCandidate>, Vec<String>) {
    candidates.sort_by(|left, right| right.score.total_cmp(&left.score));
    let mut used = 0_u32;
    let mut selected = Vec::new();
    let mut pruned = Vec::new();
    for candidate in candidates {
        if candidate.token_estimate > max_tokens
            || used.saturating_add(candidate.token_estimate) > max_tokens
        {
            pruned.push(candidate.node_id);
            continue;
        }
        used += candidate.token_estimate;
        selected.push(candidate);
    }
    (selected, pruned)
}

struct CandidateRequest<'a> {
    store: &'a SqliteMemoryStore,
    query: &'a MemoryQuery,
    routed_modes: &'a [MemoryMode],
    expanded_node_id: &'a str,
    nodes: &'a HashMap<String, MemoryNode>,
    edges: &'a [MemoryEdge],
    selected_ids: &'a BTreeSet<String>,
    expanded_ids: &'a BTreeSet<String>,
    as_of_unix_ms: Option<i64>,
    known_at_unix_ms: Option<i64>,
}

fn reconstruction_candidates(
    request: CandidateRequest<'_>,
) -> MemoryResult<Vec<ReconstructionCandidate>> {
    let CandidateRequest {
        store,
        query,
        routed_modes,
        expanded_node_id,
        nodes,
        edges,
        selected_ids,
        expanded_ids,
        as_of_unix_ms,
        known_at_unix_ms,
    } = request;
    let mut candidates = BTreeMap::new();
    for edge in edges {
        let neighbor_id = if edge.from_node_id == expanded_node_id {
            Some(edge.to_node_id.as_str())
        } else if edge.to_node_id == expanded_node_id {
            Some(edge.from_node_id.as_str())
        } else {
            None
        };
        let Some(neighbor_id) = neighbor_id else {
            continue;
        };
        if selected_ids.contains(neighbor_id) || expanded_ids.contains(neighbor_id) {
            continue;
        }
        let Some(node) = nodes.get(neighbor_id) else {
            continue;
        };
        if node.kind == MemoryNodeKind::EntityCue || !modes_accept_kind(routed_modes, node.kind) {
            continue;
        }
        if query.require_fresh
            && !store.node_is_visible_at(node, as_of_unix_ms, known_at_unix_ms)?
        {
            continue;
        }
        let lexical = overlap_score(&query.question, &node.text);
        let edge_score = edge_kind_weight(edge.kind) * edge.weight;
        let score = (0.60 * edge_score + 0.40 * lexical).clamp(0.0, 1.0);
        if score <= 0.02 {
            continue;
        }
        candidates.insert(
            node.id.clone(),
            ReconstructionCandidate {
                node_id: node.id.clone(),
                kind: node.kind,
                text: node.text.clone(),
                score,
                token_estimate: node.token_estimate,
            },
        );
    }
    Ok(candidates.into_values().collect())
}

fn visible_edges(
    store: &SqliteMemoryStore,
    edges: Vec<MemoryEdge>,
    as_of_unix_ms: i64,
    known_at_unix_ms: Option<i64>,
    nodes: &HashMap<String, MemoryNode>,
) -> MemoryResult<Vec<MemoryEdge>> {
    let mut visible = Vec::new();
    for edge in edges {
        let edge_visible = if matches!(
            edge.kind,
            MemoryEdgeKind::Contradicts | MemoryEdgeKind::Supersedes
        ) {
            contradiction_edge_visible_at(store, &edge, nodes, as_of_unix_ms, known_at_unix_ms)?
        } else {
            ordinary_edge_visible_at(store, &edge, nodes, as_of_unix_ms, known_at_unix_ms)?
        };
        if edge_visible {
            visible.push(edge);
        }
    }
    Ok(visible)
}

fn ordinary_edge_visible_at(
    store: &SqliteMemoryStore,
    edge: &MemoryEdge,
    nodes: &HashMap<String, MemoryNode>,
    as_of_unix_ms: i64,
    known_at_unix_ms: Option<i64>,
) -> MemoryResult<bool> {
    if edge.created_at_unix_ms > as_of_unix_ms {
        return Ok(false);
    }
    if known_at_unix_ms.is_none() {
        return Ok(true);
    }
    // Ordinary edges currently have observed time but no source event id. For
    // known-time reads, hiding them is conservative until edge provenance is
    // added to the projection schema.
    if !matches!(
        edge.kind,
        MemoryEdgeKind::Contradicts | MemoryEdgeKind::Supersedes
    ) {
        return Ok(false);
    }
    let Some(from_node) = nodes.get(&edge.from_node_id) else {
        return Ok(false);
    };
    let Some(to_node) = nodes.get(&edge.to_node_id) else {
        return Ok(false);
    };
    Ok(
        store.node_is_visible_at(from_node, Some(as_of_unix_ms), known_at_unix_ms)?
            && store.node_is_visible_at(to_node, Some(as_of_unix_ms), known_at_unix_ms)?,
    )
}

fn contradiction_edge_visible_at(
    store: &SqliteMemoryStore,
    edge: &MemoryEdge,
    nodes: &HashMap<String, MemoryNode>,
    as_of_unix_ms: i64,
    known_at_unix_ms: Option<i64>,
) -> MemoryResult<bool> {
    if !matches!(
        edge.kind,
        MemoryEdgeKind::Contradicts | MemoryEdgeKind::Supersedes
    ) {
        return Ok(false);
    }
    let Some(newer) = nodes.get(&edge.from_node_id) else {
        return Ok(false);
    };
    let Some(older) = nodes.get(&edge.to_node_id) else {
        return Ok(false);
    };
    Ok(
        store.node_is_visible_at(newer, Some(as_of_unix_ms), known_at_unix_ms)?
            && store.node_is_stale_at(older, Some(as_of_unix_ms), known_at_unix_ms)?,
    )
}

fn merge_seed_nodes(nodes: &mut Vec<MemoryNode>, seeds: Vec<MemoryNode>) {
    let mut seen: BTreeSet<_> = nodes.iter().map(|node| node.id.clone()).collect();
    for seed in seeds {
        if seen.insert(seed.id.clone()) {
            nodes.push(seed);
        }
    }
}

fn personalized_pagerank(
    question: &str,
    nodes: &[MemoryNode],
    edges: &[MemoryEdge],
) -> HashMap<String, f32> {
    if nodes.is_empty() {
        return HashMap::new();
    }
    let mut adjacency: HashMap<String, Vec<(String, f32)>> = HashMap::new();
    for edge in edges {
        let typed = edge.weight * edge_kind_weight(edge.kind);
        adjacency
            .entry(edge.from_node_id.clone())
            .or_default()
            .push((edge.to_node_id.clone(), typed));
        adjacency
            .entry(edge.to_node_id.clone())
            .or_default()
            .push((edge.from_node_id.clone(), typed * 0.65));
    }

    let mut seed = HashMap::new();
    for node in nodes {
        let lexical = overlap_score(question, &node.text);
        if lexical > 0.0 {
            seed.insert(node.id.clone(), lexical.max(0.05));
        }
    }
    if seed.is_empty() {
        let fallback = 1.0 / nodes.len() as f32;
        for node in nodes.iter().take(32) {
            seed.insert(node.id.clone(), fallback);
        }
    }
    normalize(&mut seed);

    let mut rank = seed.clone();
    for _ in 0..12 {
        let mut next = seed
            .iter()
            .map(|(id, value)| (id.clone(), value * 0.18))
            .collect::<HashMap<_, _>>();
        for (from, neighbors) in &adjacency {
            let Some(from_rank) = rank.get(from) else {
                continue;
            };
            let total = neighbors
                .iter()
                .map(|(_, weight)| *weight)
                .sum::<f32>()
                .max(0.0001);
            for (to, weight) in neighbors {
                *next.entry(to.clone()).or_insert(0.0) += from_rank * 0.82 * (*weight / total);
            }
        }
        normalize(&mut next);
        rank = next;
    }
    rank
}

fn normalize(scores: &mut HashMap<String, f32>) {
    let total = scores.values().sum::<f32>();
    if total <= f32::EPSILON {
        return;
    }
    for value in scores.values_mut() {
        *value = (*value / total).clamp(0.0, 1.0);
    }
}

fn edge_kind_weight(kind: MemoryEdgeKind) -> f32 {
    match kind {
        MemoryEdgeKind::Mentions => 0.7,
        MemoryEdgeKind::CausedBy => 1.0,
        MemoryEdgeKind::Fixes => 1.0,
        MemoryEdgeKind::Contradicts => 0.9,
        MemoryEdgeKind::Supersedes => 0.95,
        MemoryEdgeKind::Before | MemoryEdgeKind::After => 0.55,
        MemoryEdgeKind::PartOf => 0.8,
        MemoryEdgeKind::DerivedFrom => 0.75,
        MemoryEdgeKind::Blocks | MemoryEdgeKind::Enables => 0.9,
        MemoryEdgeKind::ObservedIn => 0.6,
    }
}

fn edge_type_strength(edges: &[MemoryEdge]) -> HashMap<String, f32> {
    let mut out = HashMap::new();
    for edge in edges {
        let weight = edge_kind_weight(edge.kind) * edge.weight;
        out.entry(edge.from_node_id.clone())
            .and_modify(|value: &mut f32| *value = value.max(weight))
            .or_insert(weight);
        out.entry(edge.to_node_id.clone())
            .and_modify(|value: &mut f32| *value = value.max(weight * 0.8))
            .or_insert(weight * 0.8);
    }
    out
}

fn base_level(node: &MemoryNode, now_unix_ms: i64) -> f32 {
    let count = (node.observation_count as f32 + 1.0).ln() / 4.0;
    let age_days = ((now_unix_ms - node.updated_at_unix_ms).max(0) as f32) / 86_400_000.0;
    let recency = 1.0 / (1.0 + age_days).powf(0.28);
    (count * recency).clamp(0.0, 1.0)
}

fn freshness(node: &MemoryNode, now_unix_ms: i64, visible_at_read_time: bool) -> f32 {
    if !visible_at_read_time {
        return 0.05;
    }
    let age_days = ((now_unix_ms - node.updated_at_unix_ms).max(0) as f32) / 86_400_000.0;
    (1.0 / (1.0 + age_days / 45.0)).clamp(0.0, 1.0)
}

fn contradictions_for(
    edges: &[MemoryEdge],
    selected_ids: &BTreeSet<String>,
    nodes: &HashMap<String, MemoryNode>,
) -> Vec<Contradiction> {
    let mut out = Vec::new();
    for edge in edges {
        if !matches!(
            edge.kind,
            MemoryEdgeKind::Contradicts | MemoryEdgeKind::Supersedes
        ) {
            continue;
        }
        if !selected_ids.contains(&edge.from_node_id) && !selected_ids.contains(&edge.to_node_id) {
            continue;
        }
        let newer = nodes.get(&edge.from_node_id);
        let older = nodes.get(&edge.to_node_id);
        let summary = match (newer, older) {
            (Some(newer), Some(older)) => {
                format!(
                    "{} conflicts with {}",
                    concise(&newer.text, 120),
                    concise(&older.text, 120)
                )
            }
            _ => format!("{} conflicts with {}", edge.from_node_id, edge.to_node_id),
        };
        out.push(Contradiction {
            older_node_id: edge.to_node_id.clone(),
            newer_node_id: edge.from_node_id.clone(),
            summary,
        });
    }
    out.sort_by(|left, right| left.older_node_id.cmp(&right.older_node_id));
    out.dedup_by(|left, right| {
        left.older_node_id == right.older_node_id && left.newer_node_id == right.newer_node_id
    });
    out
}

fn stale_assumptions_for(
    store: &SqliteMemoryStore,
    evidence: &[Evidence],
    contradictions: &[Contradiction],
    nodes: &HashMap<String, MemoryNode>,
    query: &MemoryQuery,
    routed_modes: &[MemoryMode],
    window: ReadWindow,
) -> MemoryResult<Vec<StaleAssumption>> {
    let mut out = BTreeMap::new();
    for node in evidence
        .iter()
        .filter_map(|item| nodes.get(&item.node_id))
        .chain(
            contradictions
                .iter()
                .filter_map(|item| nodes.get(&item.older_node_id)),
        )
    {
        if store.node_is_stale_at(node, window.as_of_unix_ms, window.known_at_unix_ms)? {
            insert_stale_assumption(&mut out, node);
        }
    }
    for node in nodes
        .values()
        .filter(|node| node.kind != crate::model::MemoryNodeKind::EntityCue)
        .filter(|node| modes_accept_kind(routed_modes, node.kind))
        .filter(|node| overlap_score(&query.question, &node.text) > 0.001)
    {
        if store.node_is_stale_at(node, window.as_of_unix_ms, window.known_at_unix_ms)?
            && !has_visible_family_successor(
                store,
                node,
                nodes,
                window.as_of_unix_ms,
                window.known_at_unix_ms,
            )?
        {
            insert_stale_assumption(&mut out, node);
        }
    }
    Ok(out.into_values().collect())
}

fn has_visible_family_successor(
    store: &SqliteMemoryStore,
    node: &MemoryNode,
    nodes: &HashMap<String, MemoryNode>,
    as_of_unix_ms: Option<i64>,
    known_at_unix_ms: Option<i64>,
) -> MemoryResult<bool> {
    let Some(valid_to_unix_ms) = node.valid_to_unix_ms else {
        return Ok(false);
    };
    for candidate in nodes.values() {
        if candidate.id != node.id
            && candidate.tenant_id == node.tenant_id
            && candidate.project_id == node.project_id
            && candidate.environment_id == node.environment_id
            && candidate.kind == node.kind
            && canonical_family_key(&candidate.canonical_key)
                == canonical_family_key(&node.canonical_key)
            && candidate.valid_from_unix_ms >= valid_to_unix_ms
            && store.node_is_visible_at(candidate, as_of_unix_ms, known_at_unix_ms)?
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn canonical_family_key(canonical_key: &str) -> &str {
    canonical_key
        .split_once("|rev:")
        .map(|(family, _)| family)
        .unwrap_or(canonical_key)
}

fn insert_stale_assumption(out: &mut BTreeMap<String, StaleAssumption>, node: &MemoryNode) {
    out.entry(node.id.clone())
        .or_insert_with(|| StaleAssumption {
            node_id: node.id.clone(),
            summary: concise(&node.text, 180),
            invalidated_at_unix_ms: node.valid_to_unix_ms,
        });
}

fn synthesize_answer(evidence: &[Evidence], contradictions: &[Contradiction]) -> String {
    if evidence.is_empty() {
        return "No matching memory was found for this scope.".to_string();
    }
    let mut lines = vec!["Relevant memory:".to_string()];
    for item in evidence.iter().take(8) {
        lines.push(format!(
            "- [{} score {:.2}] {}",
            item.kind,
            item.score,
            concise(&item.text, 260)
        ));
    }
    if !contradictions.is_empty() {
        lines.push(format!(
            "Warning: {} contradiction(s) or superseded premise(s) were surfaced.",
            contradictions.len()
        ));
    }
    lines.join("\n")
}

fn suggested_queries(question: &str, evidence: &[Evidence]) -> Vec<String> {
    let mut terms = top_terms(question, 3);
    for item in evidence {
        for term in top_terms(&item.text, 2) {
            if !terms.contains(&term) {
                terms.push(term);
            }
            if terms.len() >= 5 {
                break;
            }
        }
        if terms.len() >= 5 {
            break;
        }
    }
    terms
        .into_iter()
        .take(3)
        .map(|term| format!("what should I remember about {term}?"))
        .collect()
}

#[derive(Default)]
struct BTreeMapKeyedSpans(BTreeMap<String, crate::model::CitedSpan>);

impl Extend<crate::model::CitedSpan> for BTreeMapKeyedSpans {
    fn extend<T: IntoIterator<Item = crate::model::CitedSpan>>(&mut self, iter: T) {
        for span in iter {
            let key = format!(
                "{}:{}:{}:{}:{}",
                span.tenant_id, span.project_id, span.trace_id, span.span_id, span.seq
            );
            self.0.insert(key, span);
        }
    }
}

impl FromIterator<crate::model::CitedSpan> for BTreeMapKeyedSpans {
    fn from_iter<T: IntoIterator<Item = crate::model::CitedSpan>>(iter: T) -> Self {
        let mut spans = Self::default();
        spans.extend(iter);
        spans
    }
}

impl BTreeMapKeyedSpans {
    fn into_vec(self) -> Vec<crate::model::CitedSpan> {
        self.0.into_values().collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        engine::MemoryEngine,
        model::{
            MemoryNodeKind, MemoryScope, ReconstructionMode, ReconstructionOptions,
            ReconstructionReason,
        },
        reconstruct::DeterministicReconstructor,
        store::LedgerEvent,
    };

    use super::*;

    #[test]
    fn auto_escalation_treats_exact_score_ties_as_ambiguous() {
        let query = MemoryQuery::new("incident alpha", MemoryScope::new("tenant", "project"))
            .with_reconstruction(ReconstructionOptions {
                mode: ReconstructionMode::Auto,
                ..ReconstructionOptions::default()
            });

        let reason = escalation_reason(
            &query,
            &[
                Evidence::new(
                    "left",
                    MemoryNodeKind::Fact,
                    "Incident alpha blocked.",
                    0.42,
                ),
                Evidence::new(
                    "right",
                    MemoryNodeKind::Fact,
                    "Incident alpha recovered.",
                    0.42,
                ),
            ],
        );

        assert_eq!(reason, Some(ReconstructionReason::AmbiguousEvidence));
    }

    #[test]
    fn accepted_frontier_nodes_are_expanded_before_remaining_seeds() {
        let mut frontier = VecDeque::from(["seed-a".to_string(), "seed-b".to_string()]);
        let mut expanded = BTreeSet::new();

        assert_eq!(
            pop_next_frontier(&mut frontier, &expanded),
            Some("seed-a".to_string())
        );
        expanded.insert("seed-a".to_string());
        frontier.push_front("accepted-hop".to_string());

        assert_eq!(
            pop_next_frontier(&mut frontier, &expanded),
            Some("accepted-hop".to_string())
        );
    }

    #[test]
    fn reconstruction_candidates_are_budgeted_before_policy_decision() {
        let (selected, pruned) = budget_reconstruction_candidates(
            vec![
                ReconstructionCandidate {
                    node_id: "large".to_string(),
                    kind: MemoryNodeKind::Fact,
                    text: "large".to_string(),
                    score: 0.95,
                    token_estimate: 8,
                },
                ReconstructionCandidate {
                    node_id: "medium".to_string(),
                    kind: MemoryNodeKind::Fact,
                    text: "medium".to_string(),
                    score: 0.90,
                    token_estimate: 5,
                },
                ReconstructionCandidate {
                    node_id: "small".to_string(),
                    kind: MemoryNodeKind::Fact,
                    text: "small".to_string(),
                    score: 0.70,
                    token_estimate: 2,
                },
            ],
            10,
        );

        assert_eq!(
            selected
                .iter()
                .map(|candidate| candidate.node_id.as_str())
                .collect::<Vec<_>>(),
            vec!["large", "small"]
        );
        assert_eq!(pruned, vec!["medium".to_string()]);
        assert!(
            selected
                .iter()
                .map(|candidate| candidate.token_estimate)
                .sum::<u32>()
                <= 10
        );
    }

    #[test]
    fn query_returns_cited_evidence() -> MemoryResult<()> {
        let engine = MemoryEngine::in_memory()?;
        engine.ingest_event(&LedgerEvent::direct_memory_write(
            "tenant",
            "project",
            crate::model::MemoryNodeKind::Gotcha,
            "Checkout fails with DATABASE_URL missing. Fix by setting DATABASE_URL.",
        ))?;
        engine.project_pending(100)?;

        let answer = answer_query(
            engine.store(),
            &MemoryQuery::new(
                "checkout database failure",
                MemoryScope::new("tenant", "project"),
            ),
            ActivationWeights::default(),
            &DeterministicReconstructor,
        )?;

        assert!(!answer.evidence.is_empty());
        assert!(!answer.cited_spans.is_empty());
        assert!(
            answer
                .evidence
                .iter()
                .all(|item| item.kind != crate::model::MemoryNodeKind::EntityCue)
        );
        assert!(answer.answer.contains("Relevant memory"));
        Ok(())
    }
}
