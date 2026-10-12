//! Built-in lattice templates.

use crate::graph::{Delta, LatticeGraph};
use serde_json::json;

/// A text-statistics pipeline with a latent divide-by-zero in
/// `avg_word_len`, so a fresh install immediately demonstrates fault
/// isolation and gives the agent something to repair.
pub fn demo_lattice() -> LatticeGraph {
    let delta: Delta = serde_json::from_value(json!({
        "summary": "Seed: text statistics lattice",
        "ops": [
            {"op": "upsert_node", "id": "text", "kind": "input", "doc": "Raw text to analyse"},
            {"op": "upsert_node", "id": "tokenize", "kind": "function", "params": ["text"],
             "doc": "Split text into lowercase words",
             "body": "let words = [];\nfor w in text.split(\" \") {\n    let t = w;\n    t.trim();\n    if t.len() > 0 { words.push(t.to_lower()); }\n}\nwords"},
            {"op": "upsert_node", "id": "word_count", "kind": "function", "params": ["words"],
             "doc": "Number of words", "body": "words.len()"},
            {"op": "upsert_node", "id": "char_count", "kind": "function", "params": ["words"],
             "doc": "Total letters across all words",
             "body": "let n = 0;\nfor w in words { n += w.len(); }\nn"},
            {"op": "upsert_node", "id": "avg_word_len", "kind": "function", "params": ["chars", "count"],
             "doc": "Average word length (integer)", "body": "chars / count"},
            {"op": "upsert_node", "id": "top_word", "kind": "function", "params": ["words"],
             "doc": "Most frequent word (alphabetical tie-break)",
             "body": "let freq = #{};\nfor w in words {\n    if freq.contains(w) { freq[w] += 1; } else { freq[w] = 1; }\n}\nlet best = \"\";\nlet n = 0;\nfor k in freq.keys() {\n    let c = freq[k];\n    if c > n || (c == n && k < best) { best = k; n = c; }\n}\nbest"},
            {"op": "upsert_node", "id": "report", "kind": "output", "params": ["word_count", "avg_word_len", "top_word"],
             "doc": "Aggregated statistics"},
            {"op": "connect", "from": "text", "to": "tokenize", "port": "text"},
            {"op": "connect", "from": "tokenize", "to": "word_count", "port": "words"},
            {"op": "connect", "from": "tokenize", "to": "char_count", "port": "words"},
            {"op": "connect", "from": "tokenize", "to": "top_word", "port": "words"},
            {"op": "connect", "from": "char_count", "to": "avg_word_len", "port": "chars"},
            {"op": "connect", "from": "word_count", "to": "avg_word_len", "port": "count"},
            {"op": "connect", "from": "word_count", "to": "report", "port": "word_count"},
            {"op": "connect", "from": "avg_word_len", "to": "report", "port": "avg_word_len"},
            {"op": "connect", "from": "top_word", "to": "report", "port": "top_word"},
            {"op": "upsert_probe", "name": "pangram",
             "inputs": {"text": "The quick brown fox jumps over the lazy dog"},
             "expect": {"report": {"word_count": 9, "avg_word_len": 3, "top_word": "the"}}},
            {"op": "upsert_probe", "name": "empty",
             "inputs": {"text": ""},
             "expect": {"report": {"word_count": 0, "avg_word_len": 0, "top_word": ""}}}
        ]
    }))
    .expect("seed delta is valid JSON");
    LatticeGraph::default().apply(&delta).expect("seed lattice is valid").0
}

pub fn empty_lattice() -> LatticeGraph {
    LatticeGraph::default()
}

pub fn template(name: &str) -> Option<LatticeGraph> {
    match name {
        "demo" => Some(demo_lattice()),
        "empty" => Some(empty_lattice()),
        _ => None,
    }
}
