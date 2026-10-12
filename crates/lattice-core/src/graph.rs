//! The codebase tensor `T_C = (V, E, W)`.
//!
//! `V` are functional nodes (inputs, functions, type boundaries, outputs),
//! `E` are data-flow edges bound to named ports, and `W` is the runtime
//! telemetry the Fluidity Routing Engine writes back after every execution.
//! Generative changes never touch text files: they arrive as a [`Delta`] and
//! are projected onto the graph with [`LatticeGraph::apply`].

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use thiserror::Error;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    /// Execution boundary that receives an external value.
    Input,
    /// A closure evaluated over its incoming ports.
    #[default]
    Function,
    /// A validation boundary; semantically a function that must return its
    /// (possibly normalised) input or throw.
    Type,
    /// Execution boundary whose value is reported to the caller.
    Output,
}

/// `W`: execution-efficiency and velocity telemetry for a node.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Weights {
    /// Exponential moving average of wall-clock latency in microseconds.
    pub latency_us: f64,
    /// Script operations consumed by the last execution.
    pub ops: u64,
    /// `1 / (1 + ops / 500)`: 1.0 is free, approaching 0 is expensive.
    pub efficiency: f64,
    /// Operations per microsecond on the last execution.
    pub velocity: f64,
    /// Fraction of executions that succeeded.
    pub reliability: f64,
    pub runs: u64,
    pub failures: u64,
}

impl Weights {
    pub fn record(&mut self, latency_us: f64, ops: u64, ok: bool) {
        const ALPHA: f64 = 0.3;
        self.latency_us = if self.runs == 0 {
            latency_us
        } else {
            ALPHA * latency_us + (1.0 - ALPHA) * self.latency_us
        };
        self.ops = ops;
        self.efficiency = 1.0 / (1.0 + ops as f64 / 500.0);
        self.velocity = if latency_us > 0.0 { ops as f64 / latency_us } else { 0.0 };
        self.runs += 1;
        if !ok {
            self.failures += 1;
        }
        self.reliability = (self.runs - self.failures) as f64 / self.runs as f64;
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    #[serde(default)]
    pub kind: NodeKind,
    /// Port names; each is bound as a variable when the body executes.
    #[serde(default)]
    pub params: Vec<String>,
    /// Rhai source evaluated with `params` in scope. The value of the last
    /// expression is the node output.
    #[serde(default)]
    pub body: String,
    /// Optional expression evaluated (with the same scope) when the node is
    /// isolated by the Fluidity Routing Engine.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback: Option<String>,
    #[serde(default)]
    pub doc: String,
    #[serde(default)]
    pub weights: Weights,
}

impl Node {
    pub fn new(id: impl Into<String>, kind: NodeKind) -> Self {
        Self {
            id: id.into(),
            kind,
            params: Vec::new(),
            body: String::new(),
            fallback: None,
            doc: String::new(),
            weights: Weights::default(),
        }
    }

    /// Hash of everything that affects execution. Used as the compile cache
    /// key and to detect when an isolated node has been refactored.
    pub fn content_hash(&self) -> String {
        let mut h = Sha256::new();
        h.update(format!("{:?}", self.kind));
        h.update([0]);
        h.update(self.params.join(","));
        h.update([0]);
        h.update(normalize_source(&self.body));
        h.update([0]);
        h.update(self.fallback.as_deref().map(normalize_source).unwrap_or_default());
        hex::encode(h.finalize())
    }
}

/// Collapse whitespace so cosmetic edits do not produce new lattice states.
pub fn normalize_source(src: &str) -> String {
    src.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub from: String,
    pub to: String,
    pub port: String,
    /// EMA of the serialized payload size flowing along the edge, in bytes.
    #[serde(default)]
    pub weight: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Probe {
    pub name: String,
    /// Values for input nodes, keyed by node id.
    #[serde(default)]
    pub inputs: BTreeMap<String, Value>,
    /// Expected values for output nodes, keyed by node id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expect: Option<BTreeMap<String, Value>>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LatticeGraph {
    pub nodes: BTreeMap<String, Node>,
    pub edges: Vec<Edge>,
    #[serde(default)]
    pub probes: BTreeMap<String, Probe>,
}

/// `ΔT_agent`: a structural tensor projection proposed by the generation layer.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Delta {
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub ops: Vec<DeltaOp>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum DeltaOp {
    /// Create a node, or patch the provided fields of an existing one.
    UpsertNode {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        kind: Option<NodeKind>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        params: Option<Vec<String>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        body: Option<String>,
        /// `Some(None)` is not expressible in JSON; an empty string clears it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        fallback: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        doc: Option<String>,
    },
    RemoveNode {
        id: String,
    },
    /// Bind `from`'s output to `to`'s `port`, replacing any existing binding.
    Connect {
        from: String,
        to: String,
        port: String,
    },
    Disconnect {
        from: String,
        to: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        port: Option<String>,
    },
    UpsertProbe {
        name: String,
        #[serde(default)]
        inputs: BTreeMap<String, Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        expect: Option<BTreeMap<String, Value>>,
    },
    RemoveProbe {
        name: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum GraphError {
    #[error("invalid identifier `{0}` (use letters, digits and `_`, not starting with a digit)")]
    InvalidIdentifier(String),
    #[error("unknown node `{0}`")]
    UnknownNode(String),
    #[error("node `{node}` has no port `{port}` (declared params: {params:?})")]
    UnknownPort { node: String, port: String, params: Vec<String> },
    #[error("input node `{0}` cannot have incoming edges")]
    InputHasIncoming(String),
    #[error("output node `{0}` cannot have outgoing edges")]
    OutputHasOutgoing(String),
    #[error("self-loop on `{0}`")]
    SelfLoop(String),
    #[error("delta would introduce a cycle through: {0:?}")]
    Cycle(Vec<String>),
    #[error("duplicate param `{param}` on node `{node}`")]
    DuplicateParam { node: String, param: String },
    #[error("probe `{probe}` references unknown {what} node `{node}`")]
    ProbeReference { probe: String, what: &'static str, node: String },
}

/// Which parts of the graph a delta touched.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DeltaFootprint {
    pub nodes_added: Vec<String>,
    pub nodes_changed: Vec<String>,
    pub nodes_removed: Vec<String>,
    pub edges_changed: usize,
    pub probes_changed: Vec<String>,
}

impl DeltaFootprint {
    pub fn touched_nodes(&self) -> BTreeSet<String> {
        self.nodes_added
            .iter()
            .chain(&self.nodes_changed)
            .chain(&self.nodes_removed)
            .cloned()
            .collect()
    }
}

pub fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

impl LatticeGraph {
    /// `T_C' = T_C + ΔT`. Returns a new, validated graph and leaves `self`
    /// untouched so callers can reject the projection.
    pub fn apply(&self, delta: &Delta) -> Result<(LatticeGraph, DeltaFootprint), GraphError> {
        let mut g = self.clone();
        let mut fp = DeltaFootprint::default();

        for op in &delta.ops {
            match op {
                DeltaOp::UpsertNode { id, kind, params, body, fallback, doc } => {
                    if !is_identifier(id) {
                        return Err(GraphError::InvalidIdentifier(id.clone()));
                    }
                    let existed = g.nodes.contains_key(id);
                    let node = g
                        .nodes
                        .entry(id.clone())
                        .or_insert_with(|| Node::new(id.clone(), kind.unwrap_or_default()));
                    let before = node.content_hash();
                    if let Some(k) = kind {
                        node.kind = *k;
                    }
                    if let Some(p) = params {
                        node.params = p.clone();
                    }
                    if let Some(b) = body {
                        node.body = b.clone();
                    }
                    if let Some(f) = fallback {
                        node.fallback = if f.trim().is_empty() { None } else { Some(f.clone()) };
                    }
                    if let Some(d) = doc {
                        node.doc = d.clone();
                    }
                    if !existed {
                        push_unique(&mut fp.nodes_added, id);
                    } else if node.content_hash() != before {
                        node.weights = Weights::default();
                        push_unique(&mut fp.nodes_changed, id);
                    }
                }
                DeltaOp::RemoveNode { id } => {
                    if g.nodes.remove(id).is_none() {
                        return Err(GraphError::UnknownNode(id.clone()));
                    }
                    let before = g.edges.len();
                    g.edges.retain(|e| &e.from != id && &e.to != id);
                    fp.edges_changed += before - g.edges.len();
                    fp.nodes_added.retain(|n| n != id);
                    fp.nodes_changed.retain(|n| n != id);
                    push_unique(&mut fp.nodes_removed, id);
                }
                DeltaOp::Connect { from, to, port } => {
                    g.edges.retain(|e| !(&e.to == to && &e.port == port));
                    g.edges.push(Edge {
                        from: from.clone(),
                        to: to.clone(),
                        port: port.clone(),
                        weight: 0.0,
                    });
                    fp.edges_changed += 1;
                }
                DeltaOp::Disconnect { from, to, port } => {
                    let before = g.edges.len();
                    g.edges.retain(|e| {
                        !(&e.from == from
                            && &e.to == to
                            && port.as_ref().is_none_or(|p| &e.port == p))
                    });
                    fp.edges_changed += before - g.edges.len();
                }
                DeltaOp::UpsertProbe { name, inputs, expect } => {
                    g.probes.insert(
                        name.clone(),
                        Probe { name: name.clone(), inputs: inputs.clone(), expect: expect.clone() },
                    );
                    push_unique(&mut fp.probes_changed, name);
                }
                DeltaOp::RemoveProbe { name } => {
                    g.probes.remove(name);
                    push_unique(&mut fp.probes_changed, name);
                }
            }
        }

        g.validate()?;
        Ok((g, fp))
    }

    pub fn validate(&self) -> Result<(), GraphError> {
        for node in self.nodes.values() {
            let mut seen = BTreeSet::new();
            for p in &node.params {
                if !is_identifier(p) {
                    return Err(GraphError::InvalidIdentifier(p.clone()));
                }
                if !seen.insert(p) {
                    return Err(GraphError::DuplicateParam { node: node.id.clone(), param: p.clone() });
                }
            }
        }
        for e in &self.edges {
            if e.from == e.to {
                return Err(GraphError::SelfLoop(e.from.clone()));
            }
            let from = self.nodes.get(&e.from).ok_or_else(|| GraphError::UnknownNode(e.from.clone()))?;
            let to = self.nodes.get(&e.to).ok_or_else(|| GraphError::UnknownNode(e.to.clone()))?;
            if to.kind == NodeKind::Input {
                return Err(GraphError::InputHasIncoming(to.id.clone()));
            }
            if from.kind == NodeKind::Output {
                return Err(GraphError::OutputHasOutgoing(from.id.clone()));
            }
            if !to.params.iter().any(|p| p == &e.port) {
                return Err(GraphError::UnknownPort {
                    node: to.id.clone(),
                    port: e.port.clone(),
                    params: to.params.clone(),
                });
            }
        }
        for probe in self.probes.values() {
            for k in probe.inputs.keys() {
                match self.nodes.get(k) {
                    Some(n) if n.kind == NodeKind::Input => {}
                    _ => {
                        return Err(GraphError::ProbeReference {
                            probe: probe.name.clone(),
                            what: "input",
                            node: k.clone(),
                        })
                    }
                }
            }
            for k in probe.expect.iter().flat_map(|e| e.keys()) {
                match self.nodes.get(k) {
                    Some(n) if n.kind == NodeKind::Output => {}
                    _ => {
                        return Err(GraphError::ProbeReference {
                            probe: probe.name.clone(),
                            what: "output",
                            node: k.clone(),
                        })
                    }
                }
            }
        }
        self.topo_order().map(|_| ())
    }

    /// Kahn's algorithm. On failure returns the nodes left on a cycle.
    pub fn topo_order(&self) -> Result<Vec<String>, GraphError> {
        let mut indeg: BTreeMap<&str, usize> = self.nodes.keys().map(|k| (k.as_str(), 0)).collect();
        for e in &self.edges {
            if let Some(d) = indeg.get_mut(e.to.as_str()) {
                *d += 1;
            }
        }
        let mut queue: VecDeque<&str> = indeg.iter().filter(|(_, d)| **d == 0).map(|(k, _)| *k).collect();
        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(n) = queue.pop_front() {
            order.push(n.to_string());
            for e in self.edges.iter().filter(|e| e.from == n) {
                let d = indeg.get_mut(e.to.as_str()).expect("validated edge");
                *d -= 1;
                if *d == 0 {
                    queue.push_back(e.to.as_str());
                }
            }
        }
        if order.len() != self.nodes.len() {
            let done: BTreeSet<&str> = order.iter().map(String::as_str).collect();
            let cyclic = self.nodes.keys().filter(|k| !done.contains(k.as_str())).cloned().collect();
            return Err(GraphError::Cycle(cyclic));
        }
        Ok(order)
    }

    /// Longest-path depth of every node, used for layered layouts.
    pub fn layers(&self) -> BTreeMap<String, usize> {
        let mut depth: BTreeMap<String, usize> = self.nodes.keys().map(|k| (k.clone(), 0)).collect();
        if let Ok(order) = self.topo_order() {
            for n in order {
                let d = depth[&n];
                for e in self.edges.iter().filter(|e| e.from == n) {
                    let entry = depth.get_mut(&e.to).expect("validated edge");
                    *entry = (*entry).max(d + 1);
                }
            }
        }
        depth
    }

    pub fn incoming<'a>(&'a self, node: &'a str) -> impl Iterator<Item = &'a Edge> + 'a {
        self.edges.iter().filter(move |e| e.to == node)
    }

    /// Cryptographic identity of the structural state (telemetry and docs
    /// excluded). The Loop-Breaking Matrix tracks these to spot oscillation.
    pub fn state_hash(&self) -> String {
        let mut h = Sha256::new();
        for n in self.nodes.values() {
            h.update(n.id.as_bytes());
            h.update([0]);
            h.update(n.content_hash().as_bytes());
            h.update([1]);
        }
        let mut edges: Vec<_> = self.edges.iter().map(|e| (&e.from, &e.to, &e.port)).collect();
        edges.sort();
        for (f, t, p) in edges {
            h.update(format!("{f}->{t}.{p};"));
        }
        hex::encode(h.finalize())
    }
}

fn push_unique(v: &mut Vec<String>, s: &str) {
    if !v.iter().any(|x| x == s) {
        v.push(s.to_string());
    }
}

/// Canonical hash of a delta's operations (summary excluded).
pub fn delta_hash(delta: &Delta) -> String {
    let canonical = serde_json::to_string(&delta.ops).unwrap_or_default();
    hex::encode(Sha256::digest(normalize_source(&canonical)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn node(id: &str, kind: NodeKind, params: &[&str], body: &str) -> DeltaOp {
        DeltaOp::UpsertNode {
            id: id.into(),
            kind: Some(kind),
            params: Some(params.iter().map(|s| s.to_string()).collect()),
            body: Some(body.into()),
            fallback: None,
            doc: None,
        }
    }

    fn connect(from: &str, to: &str, port: &str) -> DeltaOp {
        DeltaOp::Connect { from: from.into(), to: to.into(), port: port.into() }
    }

    fn base() -> LatticeGraph {
        let delta = Delta {
            summary: "base".into(),
            ops: vec![
                node("x", NodeKind::Input, &[], ""),
                node("double", NodeKind::Function, &["v"], "v * 2"),
                node("out", NodeKind::Output, &["v"], ""),
                connect("x", "double", "v"),
                connect("double", "out", "v"),
            ],
        };
        LatticeGraph::default().apply(&delta).unwrap().0
    }

    #[test]
    fn apply_builds_dag_and_orders_it() {
        let g = base();
        assert_eq!(g.topo_order().unwrap(), vec!["x", "double", "out"]);
        assert_eq!(g.layers()["out"], 2);
    }

    #[test]
    fn cycles_are_rejected_and_original_is_untouched() {
        let g = base();
        let before = g.state_hash();
        let delta = Delta {
            summary: String::new(),
            ops: vec![
                node("double", NodeKind::Function, &["v", "back"], "v * 2"),
                node("tap", NodeKind::Function, &["v"], "v"),
                connect("double", "tap", "v"),
                connect("tap", "double", "back"),
            ],
        };
        assert!(matches!(g.apply(&delta), Err(GraphError::Cycle(_))));
        assert_eq!(g.state_hash(), before);
    }

    #[test]
    fn connect_replaces_existing_port_binding() {
        let g = base();
        let delta = Delta {
            summary: String::new(),
            ops: vec![node("y", NodeKind::Input, &[], ""), connect("y", "double", "v")],
        };
        let (g2, _) = g.apply(&delta).unwrap();
        let inc: Vec<_> = g2.incoming("double").map(|e| e.from.as_str()).collect();
        assert_eq!(inc, vec!["y"]);
    }

    #[test]
    fn unknown_port_and_probe_refs_are_rejected() {
        let g = base();
        let bad_port = Delta { summary: String::new(), ops: vec![connect("x", "double", "nope")] };
        assert!(matches!(g.apply(&bad_port), Err(GraphError::UnknownPort { .. })));

        let bad_probe = Delta {
            summary: String::new(),
            ops: vec![DeltaOp::UpsertProbe {
                name: "p".into(),
                inputs: [("double".to_string(), json!(1))].into(),
                expect: None,
            }],
        };
        assert!(matches!(g.apply(&bad_probe), Err(GraphError::ProbeReference { .. })));
    }

    #[test]
    fn state_hash_ignores_whitespace_and_docs() {
        let g = base();
        let delta = Delta {
            summary: String::new(),
            ops: vec![DeltaOp::UpsertNode {
                id: "double".into(),
                kind: None,
                params: None,
                body: Some("  v   *\n 2 ".into()),
                fallback: None,
                doc: Some("doubles".into()),
            }],
        };
        let (g2, fp) = g.apply(&delta).unwrap();
        assert_eq!(g.state_hash(), g2.state_hash());
        assert!(fp.nodes_changed.is_empty());
    }

    #[test]
    fn delta_ops_parse_from_json() {
        let raw = json!({
            "summary": "s",
            "ops": [
                {"op": "upsert_node", "id": "a", "kind": "input"},
                {"op": "remove_probe", "name": "p"},
                {"op": "disconnect", "from": "a", "to": "b"}
            ]
        });
        let d: Delta = serde_json::from_value(raw).unwrap();
        assert_eq!(d.ops.len(), 3);
    }
}
