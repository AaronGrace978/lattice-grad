//! Continuous Graph Compiler (CGC).
//!
//! Turns the lattice graph into a live in-memory execution plan: a topological
//! order, port bindings, and a compiled Rhai AST per node. Nothing touches the
//! disk. ASTs are cached by source hash, so a delta only recompiles the nodes
//! it actually changed and everything else is reused from the lattice cache.

use crate::graph::{GraphError, LatticeGraph, NodeKind};
use rhai::{Engine, AST};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone)]
pub enum Program {
    /// Input boundary: value comes from the caller.
    Boundary,
    /// Output (or empty-bodied type) node that forwards its inputs.
    Identity,
    Script(Arc<AST>),
    /// The node failed to compile; the FRE routes around it.
    Broken(String),
}

#[derive(Clone)]
pub struct CompiledNode {
    pub id: String,
    pub kind: NodeKind,
    pub hash: String,
    pub params: Vec<String>,
    pub program: Program,
    pub fallback: Option<Result<Arc<AST>, String>>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CompileStats {
    pub compiled: usize,
    pub reused: usize,
    pub failed: usize,
    pub micros: u64,
}

#[derive(Clone)]
pub struct CompiledLattice {
    pub state_hash: String,
    pub order: Vec<String>,
    pub nodes: BTreeMap<String, CompiledNode>,
    /// For each node: `(port, upstream node)` pairs.
    pub bindings: BTreeMap<String, Vec<(String, String)>>,
    pub stats: CompileStats,
}

impl CompiledLattice {
    pub fn compile_errors(&self) -> Vec<(String, String)> {
        self.nodes
            .values()
            .filter_map(|n| match &n.program {
                Program::Broken(e) => Some((n.id.clone(), e.clone())),
                _ => None,
            })
            .collect()
    }
}

pub struct Compiler {
    engine: Engine,
    cache: HashMap<String, Result<Arc<AST>, String>>,
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}

const CACHE_LIMIT: usize = 4096;

impl Compiler {
    pub fn new() -> Self {
        Self { engine: crate::fre::sandbox_engine(&Default::default()), cache: HashMap::new() }
    }

    fn compile_source(&mut self, src: &str, stats: &mut CompileStats) -> Result<Arc<AST>, String> {
        let key = hex::encode(Sha256::digest(src.as_bytes()));
        if let Some(hit) = self.cache.get(&key) {
            stats.reused += 1;
            return hit.clone();
        }
        if self.cache.len() >= CACHE_LIMIT {
            self.cache.clear();
        }
        stats.compiled += 1;
        let result = self.engine.compile(src).map(Arc::new).map_err(|e| e.to_string());
        self.cache.insert(key, result.clone());
        result
    }

    pub fn compile(&mut self, graph: &LatticeGraph) -> Result<CompiledLattice, GraphError> {
        let started = Instant::now();
        let order = graph.topo_order()?;
        let mut stats = CompileStats::default();
        let mut nodes = BTreeMap::new();
        let mut bindings = BTreeMap::new();

        for id in &order {
            let node = &graph.nodes[id];
            let program = match node.kind {
                NodeKind::Input => Program::Boundary,
                _ if node.body.trim().is_empty() => match node.kind {
                    NodeKind::Function => Program::Broken("function node has an empty body".into()),
                    _ => Program::Identity,
                },
                _ => match self.compile_source(&node.body, &mut stats) {
                    Ok(ast) => Program::Script(ast),
                    Err(e) => Program::Broken(e),
                },
            };
            if matches!(program, Program::Broken(_)) {
                stats.failed += 1;
            }
            let fallback = node
                .fallback
                .as_ref()
                .filter(|f| !f.trim().is_empty())
                .map(|f| self.compile_source(f, &mut stats));

            let mut ports: Vec<(String, String)> =
                graph.incoming(id).map(|e| (e.port.clone(), e.from.clone())).collect();
            ports.sort();
            bindings.insert(id.clone(), ports);

            nodes.insert(
                id.clone(),
                CompiledNode {
                    id: id.clone(),
                    kind: node.kind,
                    hash: node.content_hash(),
                    params: node.params.clone(),
                    program,
                    fallback,
                },
            );
        }

        stats.micros = started.elapsed().as_micros() as u64;
        Ok(CompiledLattice { state_hash: graph.state_hash(), order, nodes, bindings, stats })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Delta, DeltaOp};

    fn upsert(id: &str, kind: NodeKind, params: &[&str], body: &str) -> DeltaOp {
        DeltaOp::UpsertNode {
            id: id.into(),
            kind: Some(kind),
            params: Some(params.iter().map(|s| s.to_string()).collect()),
            body: Some(body.into()),
            fallback: None,
            doc: None,
        }
    }

    #[test]
    fn incremental_compile_reuses_unchanged_nodes() {
        let d = Delta {
            summary: String::new(),
            ops: vec![
                upsert("x", NodeKind::Input, &[], ""),
                upsert("a", NodeKind::Function, &["v"], "v + 1"),
                upsert("b", NodeKind::Function, &["v"], "v * 3"),
                upsert("out", NodeKind::Output, &["v"], ""),
            ],
        };
        let (g, _) = LatticeGraph::default().apply(&d).unwrap();
        let mut c = Compiler::new();
        let first = c.compile(&g).unwrap();
        assert_eq!(first.stats.compiled, 2);

        let d2 = Delta { summary: String::new(), ops: vec![upsert("b", NodeKind::Function, &["v"], "v * 4")] };
        let (g2, _) = g.apply(&d2).unwrap();
        let second = c.compile(&g2).unwrap();
        assert_eq!(second.stats.compiled, 1);
        assert_eq!(second.stats.reused, 1);
    }

    #[test]
    fn syntax_errors_mark_node_broken_without_failing_the_lattice() {
        let d = Delta {
            summary: String::new(),
            ops: vec![upsert("x", NodeKind::Input, &[], ""), upsert("bad", NodeKind::Function, &["v"], "let = ;")],
        };
        let (g, _) = LatticeGraph::default().apply(&d).unwrap();
        let compiled = Compiler::new().compile(&g).unwrap();
        assert_eq!(compiled.stats.failed, 1);
        assert_eq!(compiled.compile_errors()[0].0, "bad");
    }
}
