use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::{
    error::MemoryResult,
    model::{
        ActivationWeights, Contradiction, Evidence, MemoryAnswer, MemoryEdgeKind, MemoryQuery,
        MemoryTier, StaleAssumption, blend_activation, budget_evidence, estimate_tokens,
    },
    store::{MemoryEdge, MemoryNode, SqliteMemoryStore},
    text::{concise, now_unix_ms, overlap_score, terms, top_terms},
};

pub(crate) fn answer_query(
    store: &SqliteMemoryStore,
    query: &MemoryQuery,
    weights: ActivationWeights,
) -> MemoryResult<MemoryAnswer> {
    let env = query.scope.environment_id.as_deref();
    let query_terms = terms(&query.question);
    let seeds = store.seed_nodes(
        &query.scope.tenant_id,
        &query.scope.project_id,
        env,
        &query_terms,
        64,
    )?;
    let mut nodes = store.all_nodes(&query.scope.tenant_id, &query.scope.project_id, env)?;
    merge_seed_nodes(&mut nodes, seeds);
    let edges = store.edges_for_scope(&query.scope.tenant_id, &query.scope.project_id, env)?;
    let ppr = personalized_pagerank(&query.question, &nodes, &edges);
    let edge_strength = edge_type_strength(&edges);
    let now = now_unix_ms();

    let mut evidence = Vec::new();
    for node in &nodes {
        if node.kind == crate::model::MemoryNodeKind::EntityCue {
            continue;
        }
        if !query.accepts_kind(node.kind) {
            continue;
        }
        if query.require_fresh && !node.is_active_at(query.scope.as_of_unix_ms) {
            continue;
        }
        let ppr_score = ppr.get(&node.id).copied().unwrap_or(0.0);
        let lexical = overlap_score(&query.question, &node.text);
        if ppr_score <= 0.001 && lexical <= 0.001 && !query_terms.is_empty() {
            continue;
        }
        let base = base_level(node, now);
        let freshness = freshness(node, now, query.scope.as_of_unix_ms);
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
        let cited_spans = store.cited_spans_for_node(&node.id)?;
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
    let cited_spans = selected
        .iter()
        .flat_map(|item| item.cited_spans.iter().cloned())
        .collect::<BTreeMapKeyedSpans>()
        .into_vec();
    let selected_ids: BTreeSet<_> = selected.iter().map(|item| item.node_id.clone()).collect();
    let node_by_id: HashMap<_, _> = nodes
        .iter()
        .map(|node| (node.id.clone(), node.clone()))
        .collect();
    let contradictions = contradictions_for(&edges, &selected_ids, &node_by_id);
    let stale_assumptions = selected
        .iter()
        .filter_map(|item| node_by_id.get(&item.node_id))
        .filter(|node| node.valid_to_unix_ms.is_some())
        .map(|node| StaleAssumption {
            node_id: node.id.clone(),
            summary: concise(&node.text, 180),
            invalidated_at_unix_ms: node.valid_to_unix_ms,
        })
        .collect();
    let suggested_next_queries = suggested_queries(&query.question, &selected);
    let answer = synthesize_answer(&selected, &contradictions);
    let token_estimate =
        estimate_tokens(&answer) + selected.iter().map(|item| item.token_estimate).sum::<u32>();

    Ok(MemoryAnswer {
        answer,
        evidence: selected,
        cited_spans,
        contradictions,
        stale_assumptions,
        suggested_next_queries,
        token_estimate,
        tier_used: MemoryTier::Activation,
    })
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

fn freshness(node: &MemoryNode, now_unix_ms: i64, as_of_unix_ms: Option<i64>) -> f32 {
    if !node.is_active_at(as_of_unix_ms) {
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
    use crate::{engine::MemoryEngine, model::MemoryScope, store::LedgerEvent};

    use super::*;

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
