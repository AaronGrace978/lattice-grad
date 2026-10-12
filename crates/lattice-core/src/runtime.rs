//! The live lattice runtime: owns `T_C`, its compiled assembly, the FRE and
//! LBM, and the revision history. Every mutation goes through [`Runtime::propose`].

use crate::cgc::{CompileStats, CompiledLattice, Compiler, Program};
use crate::embed;
use crate::fre::{Bubble, Fault, FluidityEngine, FreConfig, NodeTrace, RunMode, RunReport, TraceStatus};
use crate::graph::{delta_hash, Delta, DeltaFootprint, Edge, GraphError, LatticeGraph, Node, NodeKind, Probe};
use crate::lbm::{Intervention, LbmConfig, LoopBreaker, Verdict};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

const HISTORY_LIMIT: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RevisionSource {
    Seed,
    Agent,
    Manual,
    Rollback,
    Import,
}

#[derive(Debug, Clone, Serialize)]
pub struct Revision {
    pub revision: u64,
    pub state_hash: String,
    pub summary: String,
    pub source: RevisionSource,
    pub footprint: DeltaFootprint,
    pub at_ms: u64,
    #[serde(skip)]
    pub graph: LatticeGraph,
}

#[derive(Debug, Error)]
pub enum ProposeError {
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error("loop-breaking matrix rejected the delta")]
    Oscillation(Box<Intervention>),
    #[error("unknown revision {0}")]
    UnknownRevision(u64),
}

#[derive(Debug, Clone, Serialize)]
pub struct ApplyOutcome {
    pub revision: u64,
    pub state_hash: String,
    pub noop: bool,
    pub footprint: DeltaFootprint,
    pub compile: CompileStats,
    pub compile_errors: Vec<(String, String)>,
    pub delta_norm: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct SuiteReport {
    pub revision: u64,
    pub state_hash: String,
    pub reports: Vec<RunReport>,
    pub passed: usize,
    pub failed: usize,
    pub faults: usize,
    pub healthy: bool,
    pub total_us: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SuiteSummary {
    pub revision: u64,
    pub state_hash: String,
    pub probes: usize,
    pub passed: usize,
    pub failed: usize,
    pub faults: Vec<Fault>,
    pub healthy: bool,
    pub total_us: f64,
}

impl SuiteReport {
    pub fn summary(&self) -> SuiteSummary {
        SuiteSummary {
            revision: self.revision,
            state_hash: self.state_hash.clone(),
            probes: self.reports.len(),
            passed: self.passed,
            failed: self.failed,
            faults: self.reports.iter().flat_map(|r| r.faults.clone()).collect(),
            healthy: self.healthy,
            total_us: self.total_us,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct NodeView {
    #[serde(flatten)]
    pub node: Node,
    pub layer: usize,
    pub coords: [f32; 3],
    /// `healthy`, `isolated`, `broken`, `tainted` or `untested`.
    pub status: &'static str,
    pub bubble: Option<Bubble>,
    pub compile_error: Option<String>,
    pub trace: Option<NodeTrace>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProbeView {
    #[serde(flatten)]
    pub probe: Probe,
    pub passed: Option<bool>,
    pub healthy: Option<bool>,
    pub outputs: Option<BTreeMap<String, Value>>,
    pub faults: Vec<Fault>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TensorView {
    pub ids: Vec<String>,
    /// `W[i][j]` in `[0, 1]`: normalised data-flow weight of edge `i → j`.
    pub weights: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LatticeSnapshot {
    pub revision: u64,
    pub state_hash: String,
    pub nodes: Vec<NodeView>,
    pub edges: Vec<Edge>,
    pub probes: Vec<ProbeView>,
    pub history: Vec<Revision>,
    pub interventions: Vec<Intervention>,
    pub forbidden_states: Vec<String>,
    pub compile: CompileStats,
    pub last_suite: Option<SuiteSummary>,
    pub fre_config: FreConfig,
    pub tensor: TensorView,
}

pub struct Runtime {
    graph: LatticeGraph,
    revision: u64,
    history: Vec<Revision>,
    compiler: Compiler,
    compiled: CompiledLattice,
    pub fre: FluidityEngine,
    pub lbm: LoopBreaker,
    last_suite: Option<SuiteReport>,
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

impl Runtime {
    pub fn new(graph: LatticeGraph) -> Result<Self, GraphError> {
        graph.validate()?;
        let mut compiler = Compiler::new();
        let compiled = compiler.compile(&graph)?;
        let mut lbm = LoopBreaker::new(LbmConfig::default());
        lbm.anchor(0, graph.state_hash());
        let history = vec![Revision {
            revision: 0,
            state_hash: graph.state_hash(),
            summary: "Seed lattice".into(),
            source: RevisionSource::Seed,
            footprint: DeltaFootprint::default(),
            at_ms: now_ms(),
            graph: graph.clone(),
        }];
        Ok(Self {
            graph,
            revision: 0,
            history,
            compiler,
            compiled,
            fre: FluidityEngine::default(),
            lbm,
            last_suite: None,
        })
    }

    pub fn graph(&self) -> &LatticeGraph {
        &self.graph
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn state_hash(&self) -> &str {
        &self.compiled.state_hash
    }

    pub fn last_suite(&self) -> Option<&SuiteReport> {
        self.last_suite.as_ref()
    }

    pub fn set_fre_config(&mut self, cfg: FreConfig) {
        self.fre.config = cfg;
    }

    /// Project `delta` onto the lattice. With `check_loops`, the LBM may veto it.
    pub fn propose(
        &mut self,
        delta: &Delta,
        source: RevisionSource,
        check_loops: bool,
    ) -> Result<ApplyOutcome, ProposeError> {
        let (next, footprint) = self.graph.apply(delta)?;
        let next_hash = next.state_hash();
        let vector = embed::sub(&embed::embed_graph(&next), &embed::embed_graph(&self.graph));
        let delta_norm = embed::norm(&vector);

        if next_hash == self.compiled.state_hash {
            self.graph = carry_metadata(next, &self.graph);
            return Ok(self.outcome(true, footprint, delta_norm));
        }

        let touched: Vec<String> = footprint.touched_nodes().into_iter().collect();
        if check_loops {
            if let Verdict::Intervene(iv) = self.lbm.evaluate(self.revision + 1, &next_hash, &vector, &touched) {
                return Err(ProposeError::Oscillation(Box::new(iv)));
            }
        }

        let compiled = self.compiler.compile(&next)?;
        self.graph = next;
        self.compiled = compiled;
        self.revision += 1;
        self.lbm.commit(self.revision, next_hash.clone(), delta_hash(delta), vector, touched);
        self.push_history(delta.summary.clone(), source, footprint.clone());
        Ok(self.outcome(false, footprint, delta_norm))
    }

    fn outcome(&self, noop: bool, footprint: DeltaFootprint, delta_norm: f32) -> ApplyOutcome {
        ApplyOutcome {
            revision: self.revision,
            state_hash: self.compiled.state_hash.clone(),
            noop,
            footprint,
            compile: self.compiled.stats.clone(),
            compile_errors: self.compiled.compile_errors(),
            delta_norm,
        }
    }

    fn push_history(&mut self, summary: String, source: RevisionSource, footprint: DeltaFootprint) {
        self.history.push(Revision {
            revision: self.revision,
            state_hash: self.compiled.state_hash.clone(),
            summary: if summary.trim().is_empty() { "(no summary)".into() } else { summary },
            source,
            footprint,
            at_ms: now_ms(),
            graph: self.graph.clone(),
        });
        if self.history.len() > HISTORY_LIMIT {
            let excess = self.history.len() - HISTORY_LIMIT;
            self.history.drain(0..excess);
        }
    }

    /// Restore an earlier revision as a new revision. Manual restores re-anchor
    /// the LBM instead of counting as oscillation.
    pub fn rollback(&mut self, revision: u64) -> Result<ApplyOutcome, ProposeError> {
        let target = self
            .history
            .iter()
            .find(|r| r.revision == revision)
            .ok_or(ProposeError::UnknownRevision(revision))?
            .graph
            .clone();
        let compiled = self.compiler.compile(&target)?;
        self.graph = target;
        self.compiled = compiled;
        self.revision += 1;
        self.lbm.anchor(self.revision, self.compiled.state_hash.clone());
        self.push_history(format!("Rollback to r{revision}"), RevisionSource::Rollback, DeltaFootprint::default());
        Ok(self.outcome(false, DeltaFootprint::default(), 0.0))
    }

    /// Replace the whole lattice (import / reset). Clears history and LBM state.
    pub fn replace(&mut self, graph: LatticeGraph, summary: &str) -> Result<(), GraphError> {
        let fre_config = self.fre.config.clone();
        *self = Runtime::new(graph)?;
        self.fre.config = fre_config;
        self.history[0].summary = summary.to_string();
        self.history[0].source = RevisionSource::Import;
        Ok(())
    }

    /// Run every probe in diagnostic mode, write telemetry back into `W`, and
    /// reconcile isolation bubbles.
    pub fn run_suite(&mut self) -> SuiteReport {
        let probes: Vec<Probe> = self.graph.probes.values().cloned().collect();
        let mut reports = Vec::with_capacity(probes.len());
        for p in &probes {
            reports.push(self.fre.run(&self.compiled, &p.name, &p.inputs, p.expect.as_ref(), RunMode::Diagnostic));
        }
        for r in &reports {
            self.absorb_telemetry(r);
        }
        self.fre.update_bubbles(&self.compiled, &reports, self.revision);
        let passed = reports.iter().filter(|r| r.healthy()).count();
        let suite = SuiteReport {
            revision: self.revision,
            state_hash: self.compiled.state_hash.clone(),
            passed,
            failed: reports.len() - passed,
            faults: reports.iter().map(|r| r.faults.len()).sum(),
            healthy: !reports.is_empty() && passed == reports.len(),
            total_us: reports.iter().map(|r| r.total_us).sum(),
            reports,
        };
        self.last_suite = Some(suite.clone());
        suite
    }

    /// Live invocation: isolated nodes are routed around via their proxies.
    pub fn invoke(&mut self, inputs: &BTreeMap<String, Value>) -> RunReport {
        let report = self.fre.run(&self.compiled, "live", inputs, None, RunMode::Routed);
        self.absorb_telemetry(&report);
        report
    }

    fn absorb_telemetry(&mut self, report: &RunReport) {
        for t in &report.traces {
            let executed = t.status != TraceStatus::Bubbled
                && !matches!(self.compiled.nodes.get(&t.node).map(|n| &n.program), Some(Program::Boundary));
            if let Some(n) = self.graph.nodes.get_mut(&t.node) {
                if executed {
                    n.weights.record(t.latency_us, t.ops, t.status == TraceStatus::Ok);
                }
            }
            for e in self.graph.edges.iter_mut().filter(|e| e.from == t.node) {
                e.weight = if e.weight == 0.0 { t.output_bytes as f64 } else { 0.3 * t.output_bytes as f64 + 0.7 * e.weight };
            }
        }
    }

    pub fn snapshot(&self) -> LatticeSnapshot {
        let layers = self.graph.layers();
        let ids: Vec<String> = self.graph.nodes.keys().cloned().collect();
        let vectors: Vec<embed::Vector> = self.graph.nodes.values().map(embed::embed_node).collect();
        let coords = embed::pca3(&vectors);
        let errors: BTreeMap<String, String> = self.compiled.compile_errors().into_iter().collect();

        let mut traces: BTreeMap<&str, &NodeTrace> = BTreeMap::new();
        if let Some(s) = &self.last_suite {
            if s.state_hash == self.compiled.state_hash {
                for t in s.reports.iter().flat_map(|r| &r.traces) {
                    let entry = traces.entry(t.node.as_str()).or_insert(t);
                    if entry.status == TraceStatus::Ok && t.status != TraceStatus::Ok
                        || (!entry.tainted && t.tainted && entry.status == TraceStatus::Ok)
                    {
                        *entry = t;
                    }
                }
            }
        }

        let nodes = self
            .graph
            .nodes
            .values()
            .enumerate()
            .map(|(i, n)| {
                let bubble = self.fre.bubbles.get(&n.id).cloned();
                let trace = traces.get(n.id.as_str()).map(|t| (*t).clone());
                let status = if errors.contains_key(&n.id) {
                    "broken"
                } else if bubble.is_some() {
                    "isolated"
                } else {
                    match &trace {
                        Some(t) if t.status != TraceStatus::Ok => "isolated",
                        Some(t) if t.tainted => "tainted",
                        Some(_) => "healthy",
                        None => "untested",
                    }
                };
                NodeView {
                    node: n.clone(),
                    layer: layers.get(&n.id).copied().unwrap_or(0),
                    coords: coords.get(i).copied().unwrap_or([0.0; 3]),
                    status,
                    bubble,
                    compile_error: errors.get(&n.id).cloned(),
                    trace,
                }
            })
            .collect();

        let current_suite = self.last_suite.as_ref().filter(|s| s.state_hash == self.compiled.state_hash);
        let probes = self
            .graph
            .probes
            .values()
            .map(|p| {
                let report = current_suite.and_then(|s| s.reports.iter().find(|r| r.probe == p.name));
                ProbeView {
                    probe: p.clone(),
                    passed: report.and_then(|r| r.passed),
                    healthy: report.map(|r| r.healthy()),
                    outputs: report.map(|r| r.outputs.clone()),
                    faults: report.map(|r| r.faults.clone()).unwrap_or_default(),
                }
            })
            .collect();

        let index: BTreeMap<&str, usize> = ids.iter().enumerate().map(|(i, id)| (id.as_str(), i)).collect();
        let max_w = self.graph.edges.iter().map(|e| e.weight).fold(0.0f64, f64::max);
        let mut weights = vec![vec![0.0f32; ids.len()]; ids.len()];
        for e in &self.graph.edges {
            let (Some(&i), Some(&j)) = (index.get(e.from.as_str()), index.get(e.to.as_str())) else { continue };
            let w = if max_w > 0.0 { (e.weight / max_w) as f32 } else { 0.0 };
            weights[i][j] = 0.15 + 0.85 * w;
        }

        LatticeSnapshot {
            revision: self.revision,
            state_hash: self.compiled.state_hash.clone(),
            nodes,
            edges: self.graph.edges.clone(),
            probes,
            history: self.history.iter().rev().take(100).cloned().collect(),
            interventions: self.lbm.interventions.iter().rev().take(50).cloned().collect(),
            forbidden_states: self.lbm.forbidden.iter().cloned().collect(),
            compile: self.compiled.stats.clone(),
            last_suite: self.last_suite.as_ref().map(|s| s.summary()),
            fre_config: self.fre.config.clone(),
            tensor: TensorView { ids, weights },
        }
    }

    /// Compact textual projection of the lattice for the generation layer.
    pub fn digest(&self) -> String {
        let snap = self.snapshot();
        let mut out = format!(
            "LATTICE r{} state={} nodes={} edges={} probes={}\n",
            snap.revision,
            &snap.state_hash[..12],
            snap.nodes.len(),
            snap.edges.len(),
            snap.probes.len()
        );
        out.push_str("NODES (topological order):\n");
        let order = self.compiled.order.clone();
        for id in &order {
            let Some(v) = snap.nodes.iter().find(|n| &n.node.id == id) else { continue };
            let n = &v.node;
            out.push_str(&format!(
                "- {} [{}]({}) status={}",
                n.id,
                kind_name(n.kind),
                n.params.join(", "),
                v.status
            ));
            if n.weights.runs > 0 {
                out.push_str(&format!(" reliability={:.2} ops={}", n.weights.reliability, n.weights.ops));
            }
            out.push('\n');
            if !n.doc.trim().is_empty() {
                out.push_str(&format!("  doc: {}\n", n.doc.trim()));
            }
            if !n.body.trim().is_empty() {
                out.push_str("  body: |\n");
                for line in n.body.lines() {
                    out.push_str(&format!("    {line}\n"));
                }
            }
            if let Some(f) = &n.fallback {
                out.push_str(&format!("  fallback: {}\n", f.trim()));
            }
            if let Some(e) = &v.compile_error {
                out.push_str(&format!("  COMPILE ERROR: {e}\n"));
            }
        }
        out.push_str("EDGES (from -> to.port):\n");
        for e in &snap.edges {
            out.push_str(&format!("- {} -> {}.{}\n", e.from, e.to, e.port));
        }
        out.push_str("PROBES:\n");
        if snap.probes.is_empty() {
            out.push_str("- (none)\n");
        }
        for p in &snap.probes {
            let verdict = match p.healthy {
                Some(true) => "PASS".to_string(),
                Some(false) => format!("FAIL ({} faults)", p.faults.len()),
                None => "not run".to_string(),
            };
            out.push_str(&format!(
                "- {} inputs={} expect={} => {}\n",
                p.probe.name,
                truncate(&serde_json::to_string(&p.probe.inputs).unwrap_or_default(), 300),
                p.probe
                    .expect
                    .as_ref()
                    .map(|e| truncate(&serde_json::to_string(e).unwrap_or_default(), 300))
                    .unwrap_or_else(|| "none".into()),
                verdict
            ));
            if let Some(outs) = &p.outputs {
                out.push_str(&format!("    actual={}\n", truncate(&serde_json::to_string(outs).unwrap_or_default(), 300)));
            }
        }
        let bubbles: Vec<&Bubble> = self.fre.bubbles.values().collect();
        if !bubbles.is_empty() {
            out.push_str("ISOLATION BUBBLES (routed through proxies):\n");
            for b in bubbles {
                out.push_str(&format!("- {} {:?}: {} (proxy={:?})\n", b.node, b.kind, truncate(&b.reason, 200), b.proxy));
            }
        }
        out
    }
}

/// Keep telemetry from `prev` when a no-op delta only touched docs.
fn carry_metadata(next: LatticeGraph, prev: &LatticeGraph) -> LatticeGraph {
    let mut next = next;
    for (id, n) in next.nodes.iter_mut() {
        if let Some(p) = prev.nodes.get(id) {
            n.weights = p.weights.clone();
        }
    }
    next
}

pub fn kind_name(k: NodeKind) -> &'static str {
    match k {
        NodeKind::Input => "input",
        NodeKind::Function => "function",
        NodeKind::Type => "type",
        NodeKind::Output => "output",
    }
}

pub fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let t: String = s.chars().take(max).collect();
        format!("{t}…")
    }
}

pub fn format_faults(faults: &[Fault], limit: usize) -> String {
    let mut out = String::new();
    for f in faults.iter().take(limit) {
        out.push_str(&format!("- [{:?}] node `{}` on probe `{}`: {}", f.kind, f.node, f.probe, truncate(&f.message, 400)));
        if !f.inputs.is_empty() {
            out.push_str(&format!(" | inputs={}", truncate(&serde_json::to_string(&f.inputs).unwrap_or_default(), 300)));
        }
        if let Some(p) = f.proxy {
            out.push_str(&format!(" | routed via {p:?} proxy"));
        }
        out.push('\n');
    }
    if faults.len() > limit {
        out.push_str(&format!("- … and {} more\n", faults.len() - limit));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seed;

    #[test]
    fn demo_lattice_has_an_isolated_node_after_suite() {
        let mut rt = Runtime::new(seed::demo_lattice()).unwrap();
        let suite = rt.run_suite();
        assert!(!suite.healthy);
        assert!(rt.fre.bubbles.contains_key("avg_word_len"));
        let snap = rt.snapshot();
        let node = snap.nodes.iter().find(|n| n.node.id == "avg_word_len").unwrap();
        assert_eq!(node.status, "isolated");
        assert!(rt.digest().contains("ISOLATION BUBBLES"));
    }

    #[test]
    fn rollback_restores_state_as_new_revision() {
        let mut rt = Runtime::new(seed::demo_lattice()).unwrap();
        let original = rt.state_hash().to_string();
        let delta: Delta = serde_json::from_value(serde_json::json!({
            "summary": "tweak",
            "ops": [{"op": "upsert_node", "id": "word_count", "body": "words.len() + 0"}]
        }))
        .unwrap();
        rt.propose(&delta, RevisionSource::Manual, true).unwrap();
        assert_ne!(rt.state_hash(), original);
        rt.rollback(0).unwrap();
        assert_eq!(rt.state_hash(), original);
        assert_eq!(rt.revision(), 2);
    }

    #[test]
    fn telemetry_flows_into_weights() {
        let mut rt = Runtime::new(seed::demo_lattice()).unwrap();
        rt.run_suite();
        let n = &rt.graph().nodes["tokenize"];
        assert!(n.weights.runs > 0 && n.weights.ops > 0);
        assert!(rt.graph().edges.iter().any(|e| e.weight > 0.0));
    }
}
