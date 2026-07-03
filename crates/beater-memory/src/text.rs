use std::collections::BTreeSet;

const STOP_WORDS: &[&str] = &[
    "a", "about", "after", "all", "also", "an", "and", "are", "as", "at", "be", "by", "can", "for",
    "from", "had", "has", "have", "if", "in", "into", "is", "it", "its", "of", "on", "or", "our",
    "should", "so", "that", "the", "their", "then", "there", "this", "to", "use", "was", "we",
    "when", "where", "with", "without", "you", "your",
];

pub(crate) fn now_unix_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

pub(crate) fn normalize_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last_space = true;
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' || ch == '.' || ch == '/' {
            out.push(ch.to_ascii_lowercase());
            last_space = false;
        } else if !last_space {
            out.push(' ');
            last_space = true;
        }
    }
    out.trim().to_string()
}

pub(crate) fn terms(text: &str) -> Vec<String> {
    let mut seen = BTreeSet::new();
    for term in normalize_text(text).split_whitespace() {
        let term = term.trim_matches(|ch| matches!(ch, '.' | '-' | '/'));
        if term.len() < 3 {
            continue;
        }
        if STOP_WORDS.binary_search(&term).is_ok() {
            continue;
        }
        if term.chars().all(|ch| ch.is_ascii_digit()) {
            continue;
        }
        seen.insert(term.to_string());
    }
    seen.into_iter().collect()
}

pub(crate) fn top_terms(text: &str, limit: usize) -> Vec<String> {
    let mut scored: Vec<(usize, String)> = terms(text)
        .into_iter()
        .map(|term| {
            let score = term.len()
                + usize::from(term.contains('.')) * 4
                + usize::from(term.contains('/')) * 4
                + usize::from(term.contains('_')) * 2
                + usize::from(term.contains('-')) * 2;
            (score, term)
        })
        .collect();
    scored.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
    scored
        .into_iter()
        .take(limit)
        .map(|(_, term)| term)
        .collect()
}

pub(crate) fn concise(text: &str, max_chars: usize) -> String {
    let squashed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if squashed.len() <= max_chars {
        return squashed;
    }
    let mut end = max_chars.min(squashed.len());
    while !squashed.is_char_boundary(end) {
        end -= 1;
    }
    let mut clipped = squashed[..end].trim_end().to_string();
    clipped.push_str("...");
    clipped
}

pub(crate) fn stable_id(prefix: &str, parts: &[&str]) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for part in parts {
        for byte in part.as_bytes().iter().chain([0xff].iter()) {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    format!("{prefix}_{hash:016x}")
}

pub(crate) fn canonical_key(kind: &str, text: &str) -> String {
    let normalized = normalize_text(text);
    let compact = top_terms(&normalized, 12).join(" ");
    if compact.is_empty() {
        format!("{kind}:{}", stable_id("empty", &[&normalized]))
    } else {
        format!("{kind}:{compact}")
    }
}

pub(crate) fn overlap_score(left: &str, right: &str) -> f32 {
    let left_terms: BTreeSet<_> = terms(left).into_iter().collect();
    let right_terms: BTreeSet<_> = terms(right).into_iter().collect();
    if left_terms.is_empty() || right_terms.is_empty() {
        return 0.0;
    }
    let overlap = left_terms.intersection(&right_terms).count() as f32;
    let union = left_terms.union(&right_terms).count() as f32;
    (overlap / union).clamp(0.0, 1.0)
}

pub(crate) fn json_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => String::new(),
        serde_json::Value::Bool(value) => value.to_string(),
        serde_json::Value::Number(value) => value.to_string(),
        serde_json::Value::String(value) => value.clone(),
        serde_json::Value::Array(values) => {
            values.iter().map(json_text).collect::<Vec<_>>().join(" ")
        }
        serde_json::Value::Object(values) => {
            let mut pieces = Vec::new();
            for (key, value) in values {
                if matches!(
                    key.as_str(),
                    "trace_id"
                        | "span_id"
                        | "parent_span_id"
                        | "tenant_id"
                        | "project_id"
                        | "environment_id"
                        | "schema_version"
                        | "normalizer_version"
                ) {
                    continue;
                }
                let text = json_text(value);
                if !text.is_empty() {
                    pieces.push(format!("{key}: {text}"));
                }
            }
            pieces.join(" ")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_ids_are_repeatable() {
        assert_eq!(
            stable_id("node", &["a", "b"]),
            stable_id("node", &["a", "b"])
        );
        assert_ne!(stable_id("node", &["a", "b"]), stable_id("node", &["ab"]));
    }

    #[test]
    fn terms_ignore_common_words() {
        assert_eq!(
            terms("The checkout route uses DATABASE_URL"),
            vec!["checkout", "database_url", "route", "uses"]
        );
    }
}
