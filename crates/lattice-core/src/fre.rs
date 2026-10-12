//! Fluidity Routing Engine (FRE).
//!
//! Executes a [`CompiledLattice`] inside a sandboxed, resource-bounded Rhai
//! engine (no filesystem, network or `eval`). When a node faults — compile
//! error, runtime exception, or exhausted operation budget — the FRE wraps it
//! in an isolation *bubble*, synthesises a fallback proxy for its output and
//! keeps routing data downstream, so one broken node never takes the lattice
//! down. Every fault becomes a [`Fault`] vector for the generation layer.

use crate::cgc::{CompiledLattice, CompiledNode, Program};
use rhai::{Dynamic, Engine, EvalAltResult, Scope, AST};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FreConfig {
    /// Operation budget per node execution; exceeding it is a `budget` fault.
    pub max_operations: u64,
    pub max_call_levels: usize,
    pub max_string_size: usize,
    pub max_array_size: usize,
    pub max_map_size: usize,
}

impl Default for FreConfig {
    fn default() -> Self {
        Self {
            max_operations: 250_000,
            max_call_levels: 48,
            max_string_size: 1 << 20,
            max_array_size: 100_000,
            max_map_size: 100_000,
        }
    }
}

pub fn sandbox_engine(cfg: &FreConfig) -> Engine {
    let mut e = Engine::new();
    e.set_max_operations(cfg.max_operations);
    e.set_max_call_levels(cfg.max_call_levels);
    e.set_max_string_size(cfg.max_string_size);
    e.set_max_array_size(cfg.max_array_size);
    e.set_max_map_size(cfg.max_map_size);
    e.set_max_expr_depths(96, 48);
    e.disable_symbol("eval");
    e
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FaultKind {
    Compile,
    Runtime,
    Budget,
    /// An output node disagreed with a probe's expectation.
    Expectation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProxyKind {
    /// Memoised output from an earlier healthy execution with identical inputs.
    LastKnownGood,
    /// The node's declared `fallback` expression.
    DeclaredFallback,
    /// Forward the first input port unchanged.
    PassThrough,
    Null,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fault {
    pub probe: String,
    pub node: String,
    pub kind: FaultKind,
    pub message: String,
    /// Values bound to the node's ports when it faulted.
    pub inputs: BTreeMap<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proxy: Option<ProxyKind>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actual: Option<Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceStatus {
    Ok,
    Faulted,
    /// Skipped because the node is isolated; its proxy value was used.
    Bubbled,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeTrace {
    pub node: String,
    pub status: TraceStatus,
    pub output: Value,
    pub latency_us: f64,
    pub ops: u64,
    pub output_bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proxy: Option<ProxyKind>,
    /// True when any upstream value came from a proxy.
    pub tainted: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub logs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunReport {
    pub probe: String,
    pub state_hash: String,
    pub outputs: BTreeMap<String, Value>,
    pub traces: Vec<NodeTrace>,
    pub faults: Vec<Fault>,
    /// `None` when the probe declares no expectations.
    pub passed: Option<bool>,
    pub total_us: f64,
}

impl RunReport {
    pub fn healthy(&self) -> bool {
        self.faults.is_empty() && self.passed != Some(false)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bubble {
    pub node: String,
    /// Content hash at isolation time; a refactor changes it and lifts routing.
    pub hash: String,
    pub kind: FaultKind,
    pub reason: String,
    pub proxy: ProxyKind,
    pub faults: u64,
    pub since_revision: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunMode {
    /// Execute every node (used for probe suites) to surface all faults.
    Diagnostic,
    /// Route around isolated nodes whose code has not changed.
    Routed,
}

const MEMO_LIMIT: usize = 20_000;

pub struct FluidityEngine {
    pub config: FreConfig,
    pub bubbles: BTreeMap<String, Bubble>,
    memo: HashMap<(String, String), Value>,
}

impl Default for FluidityEngine {
    fn default() -> Self {
        Self::new(FreConfig::default())
    }
}

struct Instrumented {
    engine: Engine,
    ops: Arc<AtomicU64>,
    logs: Arc<Mutex<Vec<String>>>,
}

impl Instrumented {
    fn new(cfg: &FreConfig) -> Self {
        let mut engine = sandbox_engine(cfg);
        let ops = Arc::new(AtomicU64::new(0));
        let logs = Arc::new(Mutex::new(Vec::new()));
        let ops_cb = ops.clone();
        engine.on_progress(move |n| {
            ops_cb.store(n, Ordering::Relaxed);
            None
        });
        let print_logs = logs.clone();
        engine.on_print(move |s| print_logs.lock().unwrap().push(s.to_string()));
        let debug_logs = logs.clone();
        engine.on_debug(move |s, _, pos| debug_logs.lock().unwrap().push(format!("[debug {pos}] {s}")));
        Self { engine, ops, logs }
    }

    fn eval(&self, ast: &AST, scope: &mut Scope) -> (Result<Dynamic, Box<EvalAltResult>>, u64, Vec<String>) {
        self.ops.store(0, Ordering::Relaxed);
        let r = self.engine.eval_ast_with_scope::<Dynamic>(scope, ast);
        let logs = std::mem::take(&mut *self.logs.lock().unwrap());
        (r, self.ops.load(Ordering::Relaxed), logs)
    }
}

pub fn to_json(d: &Dynamic) -> Value {
    rhai::serde::from_dynamic::<Value>(d).unwrap_or_else(|_| Value::String(d.to_string()))
}

fn to_dynamic(v: &Value) -> Dynamic {
    rhai::serde::to_dynamic(v).unwrap_or(Dynamic::UNIT)
}

fn hash_inputs(inputs: &BTreeMap<String, Value>) -> String {
    hex::encode(Sha256::digest(serde_json::to_string(inputs).unwrap_or_default()))
}

/// Structural equality with numeric tolerance (`3 == 3.0`).
pub fn values_match(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => match (x.as_f64(), y.as_f64()) {
            (Some(x), Some(y)) => (x - y).abs() <= 1e-9 * x.abs().max(y.abs()).max(1.0),
            _ => x == y,
        },
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(a, b)| values_match(a, b)),
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| values_match(v, w)))
        }
        _ => a == b,
    }
}

fn classify(err: &EvalAltResult) -> FaultKind {
    match err {
        EvalAltResult::ErrorTooManyOperations(_)
        | EvalAltResult::ErrorStackOverflow(_)
        | EvalAltResult::ErrorDataTooLarge(..) => FaultKind::Budget,
        _ => FaultKind::Runtime,
    }
}

impl FluidityEngine {
    pub fn new(config: FreConfig) -> Self {
        Self { config, bubbles: BTreeMap::new(), memo: HashMap::new() }
    }

    pub fn run(
        &mut self,
        compiled: &CompiledLattice,
        probe: &str,
        inputs: &BTreeMap<String, Value>,
        expect: Option<&BTreeMap<String, Value>>,
        mode: RunMode,
    ) -> RunReport {
        let started = Instant::now();
        let rt = Instrumented::new(&self.config);
        let mut values: HashMap<String, Value> = HashMap::new();
        let mut tainted: BTreeSet<String> = BTreeSet::new();
        let mut traces = Vec::with_capacity(compiled.order.len());
        let mut faults = Vec::new();

        for id in &compiled.order {
            let node = &compiled.nodes[id];
            let bindings = &compiled.bindings[id];
            let mut bound: BTreeMap<String, Value> = BTreeMap::new();
            let mut is_tainted = false;
            for p in &node.params {
                let v = bindings
                    .iter()
                    .find(|(port, _)| port == p)
                    .map(|(_, from)| {
                        is_tainted |= tainted.contains(from);
                        values.get(from).cloned().unwrap_or(Value::Null)
                    })
                    .unwrap_or(Value::Null);
                bound.insert(p.clone(), v);
            }

            let t0 = Instant::now();
            let (outcome, ops, logs) = match &node.program {
                Program::Boundary => (Ok(inputs.get(id).cloned().unwrap_or(Value::Null)), 0, Vec::new()),
                Program::Identity => (Ok(identity(node, &bound)), 0, Vec::new()),
                Program::Broken(e) => (Err((FaultKind::Compile, e.clone(), false)), 0, Vec::new()),
                Program::Script(ast) => {
                    let bubbled = mode == RunMode::Routed
                        && self.bubbles.get(id).is_some_and(|b| b.hash == node.hash);
                    if bubbled {
                        (Err((FaultKind::Runtime, "routed around isolation bubble".into(), true)), 0, Vec::new())
                    } else {
                        let mut scope = scope_for(&bound);
                        let (r, ops, logs) = rt.eval(ast, &mut scope);
                        let r = r.map(|d| to_json(&d)).map_err(|e| (classify(&e), e.to_string(), false));
                        (r, ops, logs)
                    }
                }
            };
            let latency_us = t0.elapsed().as_secs_f64() * 1e6;
            let input_key = (id.clone(), hash_inputs(&bound));

            let (status, output, error, proxy) = match outcome {
                Ok(v) => {
                    if !matches!(node.program, Program::Boundary) && !is_tainted {
                        if self.memo.len() >= MEMO_LIMIT {
                            self.memo.clear();
                        }
                        self.memo.insert(input_key, v.clone());
                    }
                    (TraceStatus::Ok, v, None, None)
                }
                Err((kind, message, routed)) => {
                    let (proxy, value) = self.proxy_for(&rt, node, &bound, &input_key);
                    tainted.insert(id.clone());
                    if !routed {
                        faults.push(Fault {
                            probe: probe.to_string(),
                            node: id.clone(),
                            kind,
                            message: message.clone(),
                            inputs: bound.clone(),
                            proxy: Some(proxy),
                            expected: None,
                            actual: None,
                        });
                    }
                    let status = if routed { TraceStatus::Bubbled } else { TraceStatus::Faulted };
                    (status, value, Some(message), Some(proxy))
                }
            };
            if is_tainted {
                tainted.insert(id.clone());
            }
            let output_bytes = serde_json::to_string(&output).map(|s| s.len()).unwrap_or(0);
            values.insert(id.clone(), output.clone());
            traces.push(NodeTrace {
                node: id.clone(),
                status,
                output,
                latency_us,
                ops,
                output_bytes,
                error,
                proxy,
                tainted: is_tainted,
                logs,
            });
        }

        let mut outputs = BTreeMap::new();
        for (id, n) in &compiled.nodes {
            if n.kind == crate::graph::NodeKind::Output {
                outputs.insert(id.clone(), values.get(id).cloned().unwrap_or(Value::Null));
            }
        }

        let passed = expect.filter(|e| !e.is_empty()).map(|expect| {
            let mut ok = true;
            for (out, want) in expect {
                let got = outputs.get(out).cloned().unwrap_or(Value::Null);
                if !values_match(&got, want) {
                    ok = false;
                    faults.push(Fault {
                        probe: probe.to_string(),
                        node: out.clone(),
                        kind: FaultKind::Expectation,
                        message: format!("expected {want}, got {got}"),
                        inputs: BTreeMap::new(),
                        proxy: None,
                        expected: Some(want.clone()),
                        actual: Some(got),
                    });
                }
            }
            ok
        });

        RunReport {
            probe: probe.to_string(),
            state_hash: compiled.state_hash.clone(),
            outputs,
            traces,
            faults,
            passed,
            total_us: started.elapsed().as_secs_f64() * 1e6,
        }
    }

    fn proxy_for(
        &self,
        rt: &Instrumented,
        node: &CompiledNode,
        bound: &BTreeMap<String, Value>,
        key: &(String, String),
    ) -> (ProxyKind, Value) {
        if let Some(v) = self.memo.get(key) {
            return (ProxyKind::LastKnownGood, v.clone());
        }
        if let Some(Ok(ast)) = &node.fallback {
            let mut scope = scope_for(bound);
            if let (Ok(v), _, _) = rt.eval(ast, &mut scope) {
                return (ProxyKind::DeclaredFallback, to_json(&v));
            }
        }
        if let Some(first) = node.params.first() {
            return (ProxyKind::PassThrough, bound.get(first).cloned().unwrap_or(Value::Null));
        }
        (ProxyKind::Null, Value::Null)
    }

    /// Reconcile isolation bubbles after a diagnostic suite: faulting nodes are
    /// isolated, nodes that ran cleanly everywhere are released.
    pub fn update_bubbles(&mut self, compiled: &CompiledLattice, reports: &[RunReport], revision: u64) {
        self.bubbles.retain(|id, _| compiled.nodes.contains_key(id));
        let mut faulted: BTreeMap<&str, &Fault> = BTreeMap::new();
        for f in reports.iter().flat_map(|r| &r.faults) {
            if f.kind != FaultKind::Expectation {
                faulted.entry(f.node.as_str()).or_insert(f);
            }
        }
        for (id, fault) in &faulted {
            let hash = compiled.nodes[*id].hash.clone();
            let entry = self.bubbles.entry(id.to_string()).or_insert_with(|| Bubble {
                node: id.to_string(),
                hash: hash.clone(),
                kind: fault.kind,
                reason: fault.message.clone(),
                proxy: fault.proxy.unwrap_or(ProxyKind::Null),
                faults: 0,
                since_revision: revision,
            });
            if entry.hash != hash {
                entry.since_revision = revision;
            }
            entry.hash = hash;
            entry.kind = fault.kind;
            entry.reason = fault.message.clone();
            entry.proxy = fault.proxy.unwrap_or(ProxyKind::Null);
            entry.faults += reports.iter().flat_map(|r| &r.faults).filter(|f| f.node == *id).count() as u64;
        }
        if !reports.is_empty() {
            self.bubbles.retain(|id, _| faulted.contains_key(id.as_str()));
        }
    }
}

fn scope_for(bound: &BTreeMap<String, Value>) -> Scope<'static> {
    let mut scope = Scope::new();
    for (k, v) in bound {
        scope.push_dynamic(k.clone(), to_dynamic(v));
    }
    scope
}

fn identity(node: &CompiledNode, bound: &BTreeMap<String, Value>) -> Value {
    match node.params.len() {
        0 => Value::Null,
        1 => bound.get(&node.params[0]).cloned().unwrap_or(Value::Null),
        _ => Value::Object(bound.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cgc::Compiler;
    use crate::graph::{Delta, DeltaOp, LatticeGraph, NodeKind};
    use serde_json::json;

    fn upsert(id: &str, kind: NodeKind, params: &[&str], body: &str, fallback: Option<&str>) -> DeltaOp {
        DeltaOp::UpsertNode {
            id: id.into(),
            kind: Some(kind),
            params: Some(params.iter().map(|s| s.to_string()).collect()),
            body: Some(body.into()),
            fallback: fallback.map(String::from),
            doc: None,
        }
    }

    fn connect(from: &str, to: &str, port: &str) -> DeltaOp {
        DeltaOp::Connect { from: from.into(), to: to.into(), port: port.into() }
    }

    fn pipeline(divide_body: &str, fallback: Option<&str>) -> LatticeGraph {
        let d = Delta {
            summary: String::new(),
            ops: vec![
                upsert("items", NodeKind::Input, &[], "", None),
                upsert("total", NodeKind::Function, &["xs"], "let t = 0; for x in xs { t += x; } t", None),
                upsert("count", NodeKind::Function, &["xs"], "xs.len()", None),
                upsert("mean", NodeKind::Function, &["t", "n"], divide_body, fallback),
                upsert("out", NodeKind::Output, &["v"], "", None),
                connect("items", "total", "xs"),
                connect("items", "count", "xs"),
                connect("total", "mean", "t"),
                connect("count", "mean", "n"),
                connect("mean", "out", "v"),
            ],
        };
        LatticeGraph::default().apply(&d).unwrap().0
    }

    fn run(g: &LatticeGraph, fre: &mut FluidityEngine, items: Value, expect: Option<Value>) -> RunReport {
        let compiled = Compiler::new().compile(g).unwrap();
        let inputs = [("items".to_string(), items)].into();
        let expect: Option<BTreeMap<String, Value>> = expect.map(|e| [("out".to_string(), e)].into());
        fre.run(&compiled, "p", &inputs, expect.as_ref(), RunMode::Diagnostic)
    }

    #[test]
    fn healthy_pipeline_runs_and_records_ops() {
        let g = pipeline("t / n", None);
        let mut fre = FluidityEngine::default();
        let r = run(&g, &mut fre, json!([2, 4, 6]), Some(json!(4)));
        assert!(r.healthy(), "{:?}", r.faults);
        assert_eq!(r.outputs["out"], json!(4));
        assert!(r.traces.iter().find(|t| t.node == "total").unwrap().ops > 0);
    }

    #[test]
    fn runtime_fault_is_isolated_and_routed_through_declared_fallback() {
        let g = pipeline("t / n", Some("0"));
        let mut fre = FluidityEngine::default();
        let r = run(&g, &mut fre, json!([]), None);
        assert_eq!(r.faults.len(), 1);
        let f = &r.faults[0];
        assert_eq!((f.node.as_str(), f.kind), ("mean", FaultKind::Runtime));
        assert_eq!(f.proxy, Some(ProxyKind::DeclaredFallback));
        assert_eq!(r.outputs["out"], json!(0));
        assert!(r.traces.iter().find(|t| t.node == "out").unwrap().tainted);
    }

    #[test]
    fn last_known_good_wins_over_fallback_for_seen_inputs() {
        let mut fre = FluidityEngine::default();
        let healthy = pipeline("t / n", None);
        run(&healthy, &mut fre, json!([3, 3]), None);
        let broken = pipeline("t / (n - n)", Some("-1"));
        let r = run(&broken, &mut fre, json!([3, 3]), None);
        assert_eq!(r.faults[0].proxy, Some(ProxyKind::LastKnownGood));
        assert_eq!(r.outputs["out"], json!(3));
    }

    #[test]
    fn infinite_loops_hit_the_budget() {
        let g = pipeline("loop { }", None);
        let mut fre = FluidityEngine::new(FreConfig { max_operations: 5_000, ..Default::default() });
        let r = run(&g, &mut fre, json!([1]), None);
        assert_eq!(r.faults[0].kind, FaultKind::Budget);
        assert_eq!(r.faults[0].proxy, Some(ProxyKind::PassThrough));
    }

    #[test]
    fn bubbles_route_in_routed_mode_and_heal_after_refactor() {
        let mut fre = FluidityEngine::default();
        let broken = pipeline("throw \"boom\"", None);
        let compiled = Compiler::new().compile(&broken).unwrap();
        let inputs: BTreeMap<String, Value> = [("items".to_string(), json!([1, 2]))].into();
        let diag = fre.run(&compiled, "p", &inputs, None, RunMode::Diagnostic);
        fre.update_bubbles(&compiled, &[diag], 1);
        assert!(fre.bubbles.contains_key("mean"));

        let routed = fre.run(&compiled, "live", &inputs, None, RunMode::Routed);
        assert!(routed.faults.is_empty());
        let t = routed.traces.iter().find(|t| t.node == "mean").unwrap();
        assert_eq!(t.status, TraceStatus::Bubbled);

        let fixed = pipeline("t / n", None);
        let compiled = Compiler::new().compile(&fixed).unwrap();
        let diag = fre.run(&compiled, "p", &inputs, None, RunMode::Diagnostic);
        fre.update_bubbles(&compiled, &[diag], 2);
        assert!(fre.bubbles.is_empty());
    }

    #[test]
    fn expectation_mismatch_is_reported() {
        let g = pipeline("t / n", None);
        let mut fre = FluidityEngine::default();
        let r = run(&g, &mut fre, json!([1, 3]), Some(json!(3)));
        assert_eq!(r.passed, Some(false));
        assert_eq!(r.faults[0].kind, FaultKind::Expectation);
    }

    #[test]
    fn numeric_match_is_tolerant() {
        assert!(values_match(&json!(3), &json!(3.0)));
        assert!(values_match(&json!({"a": [1, 2.0]}), &json!({"a": [1.0, 2]})));
        assert!(!values_match(&json!("3"), &json!(3)));
    }
}
