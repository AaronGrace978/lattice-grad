//! Loop-Breaking Matrix (LBM).
//!
//! Every lattice state has a SHA-256 identity and every delta a direction in
//! embedding space. The LBM watches the trajectory of `ΔT_agent`:
//!
//! * **State revisit** — a proposed state hash already appeared in the recent
//!   window (`A → B → A`), or was explicitly forbidden earlier.
//! * **Anti-parallel swing** — the proposed delta points back along the
//!   previous one (`cos(Δ_t, Δ_{t-1}) < threshold`), i.e. the agent is undoing
//!   itself without reaching an identical state.
//!
//! On oscillation the LBM rejects the delta, forbids the oscillating states,
//! computes the oscillation axis `u = normalize(Δ_{t-1} - Δ_t)` and picks the
//! repair strategy whose embedding has the largest component orthogonal to
//! `u` (Gram–Schmidt rejection). That strategy is injected into the next
//! generation prompt as the alternative logical pathway.

use crate::embed::{self, Vector};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, VecDeque};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LbmConfig {
    pub window: usize,
    pub anti_parallel_threshold: f32,
    /// Deltas smaller than this (in embedding norm) are ignored for the
    /// anti-parallel test; tiny tweaks are not oscillation.
    pub min_delta_norm: f32,
}

impl Default for LbmConfig {
    fn default() -> Self {
        Self { window: 8, anti_parallel_threshold: -0.8, min_delta_norm: 0.05 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    pub revision: u64,
    pub state_hash: String,
    pub delta_hash: String,
    pub vector: Vector,
    pub touched: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Oscillation {
    StateRevisit { cycle_len: usize },
    ForbiddenState,
    AntiParallel { cosine: f32 },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Strategy {
    pub id: &'static str,
    pub title: &'static str,
    pub directive: &'static str,
    pub keywords: &'static str,
}

pub const STRATEGIES: &[Strategy] = &[
    Strategy {
        id: "decompose",
        title: "Decompose",
        directive: "Split the oscillating node(s) into smaller helper nodes with single responsibilities, connected by new edges. Do not rewrite the original node in place.",
        keywords: "split helper node edge connect decompose upsert_node port",
    },
    Strategy {
        id: "guard",
        title: "Guard boundaries",
        directive: "Insert a `type` node in front of the faulting node that validates and normalises its inputs (empty collections, zero divisors, wrong types via type_of) and throws a descriptive error or returns a safe value.",
        keywords: "type_of is_empty len == 0 if else throw guard validate type",
    },
    Strategy {
        id: "functional",
        title: "Functional pipeline",
        directive: "Replace explicit loops and mutable accumulators with functional array pipelines (map, filter, reduce, all, some, sort).",
        keywords: "map filter reduce all some sort closure |x| fn",
    },
    Strategy {
        id: "imperative",
        title: "Imperative rewrite",
        directive: "Replace functional pipelines, closures and recursion with explicit `for`/`while` loops and local accumulators.",
        keywords: "for while loop let += push break continue index",
    },
    Strategy {
        id: "representation",
        title: "Change representation",
        directive: "Change the shape of the data flowing along the oscillating edges (e.g. pass an object map `#{...}` with precomputed fields instead of a bare array or string) and update consumers accordingly.",
        keywords: "#{ } map object field key value keys values to_json parse",
    },
    Strategy {
        id: "reroute",
        title: "Re-route data flow",
        directive: "Change the topology instead of the code: connect the consumer to a different upstream node, add a parallel path, or bypass the faulting node entirely.",
        keywords: "connect disconnect route edge port bypass parallel remove_node",
    },
    Strategy {
        id: "fallback",
        title: "Embrace isolation",
        directive: "Keep the node isolated: give it a precise `fallback` expression that produces a correct value for the failing probe inputs and make downstream nodes robust to it.",
        keywords: "fallback default () null proxy safe",
    },
    Strategy {
        id: "restart",
        title: "Clean-room rewrite",
        directive: "Discard the previous attempts entirely. Re-derive the algorithm from the probe expectations and write the simplest possible implementation with a different identifier set.",
        keywords: "rewrite simple minimal scratch new algorithm",
    },
];

fn strategy_vector(s: &Strategy) -> Vector {
    embed::embed_text(&format!("{} {} {}", s.title, s.directive, s.keywords))
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Intervention {
    pub revision: u64,
    pub oscillation: Oscillation,
    pub rejected_state: String,
    pub forbidden_states: Vec<String>,
    pub oscillating_nodes: Vec<String>,
    pub strategy: Strategy,
    /// `|s⊥| / |s|`: how orthogonal the chosen strategy is to the oscillation axis.
    pub orthogonality: f32,
    /// First components of the injected orthogonal unit vector (for display).
    pub orthogonal_preview: Vec<f32>,
}

impl Intervention {
    pub fn prompt(&self) -> String {
        let short: Vec<String> = self.forbidden_states.iter().map(|h| h[..12.min(h.len())].to_string()).collect();
        format!(
            "LOOP-BREAKING MATRIX INTERVENTION: your last delta oscillated ({}) and was rejected.\n\
             Oscillating nodes: {}.\n\
             Forbidden lattice states (any delta producing them is rejected): {}.\n\
             Mandatory alternative pathway — {}: {}",
            match &self.oscillation {
                Oscillation::StateRevisit { cycle_len } => format!("returned to a state from {cycle_len} revisions ago"),
                Oscillation::ForbiddenState => "re-entered a forbidden state".to_string(),
                Oscillation::AntiParallel { cosine } => format!("undid the previous change, cos = {cosine:.2}"),
            },
            if self.oscillating_nodes.is_empty() { "(topology only)".to_string() } else { self.oscillating_nodes.join(", ") },
            short.join(", "),
            self.strategy.title,
            self.strategy.directive
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    Clear,
    Intervene(Intervention),
}

#[derive(Debug, Clone, Default)]
pub struct LoopBreaker {
    pub config: LbmConfig,
    pub history: VecDeque<Observation>,
    pub forbidden: BTreeSet<String>,
    pub interventions: Vec<Intervention>,
    used: BTreeSet<String>,
}

impl LoopBreaker {
    pub fn new(config: LbmConfig) -> Self {
        Self { config, ..Default::default() }
    }

    /// Record the initial (or a manually restored) state.
    pub fn anchor(&mut self, revision: u64, state_hash: String) {
        self.push(Observation {
            revision,
            state_hash,
            delta_hash: String::new(),
            vector: vec![0.0; embed::DIM],
            touched: Vec::new(),
        });
    }

    fn push(&mut self, obs: Observation) {
        self.history.push_back(obs);
        while self.history.len() > self.config.window + 1 {
            self.history.pop_front();
        }
    }

    /// Judge a prospective transition before it is applied.
    pub fn evaluate(
        &mut self,
        revision: u64,
        proposed_state: &str,
        delta_vector: &[f32],
        touched: &[String],
    ) -> Verdict {
        let current = self.history.back().map(|o| o.state_hash.clone()).unwrap_or_default();
        if proposed_state == current {
            return Verdict::Clear;
        }

        let oscillation = if self.forbidden.contains(proposed_state) {
            Some(Oscillation::ForbiddenState)
        } else if let Some(pos) = self.history.iter().rposition(|o| o.state_hash == proposed_state) {
            Some(Oscillation::StateRevisit { cycle_len: self.history.len() - pos })
        } else {
            self.history.back().and_then(|prev| {
                let (a, b) = (embed::norm(&prev.vector), embed::norm(delta_vector));
                if a < self.config.min_delta_norm || b < self.config.min_delta_norm {
                    return None;
                }
                let c = embed::cosine(&prev.vector, delta_vector);
                (c < self.config.anti_parallel_threshold).then_some(Oscillation::AntiParallel { cosine: c })
            })
        };

        let Some(oscillation) = oscillation else { return Verdict::Clear };

        let prev_vec = self.history.back().map(|o| o.vector.clone()).unwrap_or_else(|| vec![0.0; embed::DIM]);
        let mut axis = embed::sub(&prev_vec, delta_vector);
        if embed::norm(&axis) < 1e-6 {
            axis = delta_vector.to_vec();
        }
        embed::normalize(&mut axis);

        let (strategy, orthogonality, orthogonal) = self.most_orthogonal(&axis);
        self.used.insert(strategy.id.to_string());

        self.forbidden.insert(proposed_state.to_string());
        if !current.is_empty() && matches!(oscillation, Oscillation::AntiParallel { .. }) {
            if let Some(prev) = self.history.iter().rev().nth(1) {
                self.forbidden.insert(prev.state_hash.clone());
            }
        }

        let mut nodes: BTreeSet<String> = touched.iter().cloned().collect();
        if let Some(prev) = self.history.back() {
            let prev_touched: BTreeSet<String> = prev.touched.iter().cloned().collect();
            let overlap: BTreeSet<String> = nodes.intersection(&prev_touched).cloned().collect();
            if !overlap.is_empty() {
                nodes = overlap;
            }
        }

        let intervention = Intervention {
            revision,
            oscillation,
            rejected_state: proposed_state.to_string(),
            forbidden_states: self.forbidden.iter().cloned().collect(),
            oscillating_nodes: nodes.into_iter().collect(),
            strategy,
            orthogonality,
            orthogonal_preview: orthogonal.iter().take(8).copied().collect(),
        };
        self.interventions.push(intervention.clone());
        Verdict::Intervene(intervention)
    }

    fn most_orthogonal(&self, axis: &[f32]) -> (Strategy, f32, Vector) {
        let fresh: Vec<&Strategy> = STRATEGIES.iter().filter(|s| !self.used.contains(s.id)).collect();
        let pool: Vec<&Strategy> = if fresh.is_empty() { STRATEGIES.iter().collect() } else { fresh };
        let mut best: Option<(Strategy, f32, Vector)> = None;
        for s in pool {
            let v = strategy_vector(s);
            let mut perp = embed::reject(&v, axis);
            let score = embed::norm(&perp) / embed::norm(&v).max(1e-9);
            embed::normalize(&mut perp);
            if best.as_ref().is_none_or(|(_, b, _)| score > *b + 1e-6) {
                best = Some((s.clone(), score, perp));
            }
        }
        best.expect("strategy library is non-empty")
    }

    /// Record an accepted transition.
    pub fn commit(&mut self, revision: u64, state_hash: String, delta_hash: String, vector: Vector, touched: Vec<String>) {
        self.push(Observation { revision, state_hash, delta_hash, vector, touched });
    }

    pub fn is_forbidden(&self, state: &str) -> bool {
        self.forbidden.contains(state)
    }

    pub fn reset(&mut self) {
        let config = self.config.clone();
        *self = Self::new(config);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(text: &str) -> Vector {
        embed::embed_text(text)
    }

    #[test]
    fn detects_a_b_a_revisit_and_forbids_state() {
        let mut lbm = LoopBreaker::default();
        lbm.anchor(0, "A".into());
        let d1 = vec_of("for loop accumulate");
        assert_eq!(lbm.evaluate(1, "B", &d1, &["n".into()]), Verdict::Clear);
        lbm.commit(1, "B".into(), "d1".into(), d1.clone(), vec!["n".into()]);

        let back: Vector = d1.iter().map(|x| -x).collect();
        match lbm.evaluate(2, "A", &back, &["n".into()]) {
            Verdict::Intervene(i) => {
                assert_eq!(i.oscillation, Oscillation::StateRevisit { cycle_len: 2 });
                assert!(i.forbidden_states.contains(&"A".to_string()));
                assert_eq!(i.oscillating_nodes, vec!["n".to_string()]);
                assert!(i.orthogonality > 0.5);
                assert!(i.prompt().contains("Mandatory alternative pathway"));
            }
            Verdict::Clear => panic!("oscillation not detected"),
        }
        assert!(lbm.is_forbidden("A"));
    }

    #[test]
    fn detects_anti_parallel_swing_without_exact_revisit() {
        let mut lbm = LoopBreaker::default();
        lbm.anchor(0, "A".into());
        let d1 = vec_of("map filter reduce closure");
        lbm.commit(1, "B".into(), "d1".into(), d1.clone(), vec![]);
        let swing: Vector = d1.iter().map(|x| -0.9 * x).collect();
        assert!(matches!(
            lbm.evaluate(2, "C", &swing, &[]),
            Verdict::Intervene(Intervention { oscillation: Oscillation::AntiParallel { .. }, .. })
        ));
    }

    #[test]
    fn successive_interventions_rotate_strategies() {
        let mut lbm = LoopBreaker::default();
        lbm.anchor(0, "A".into());
        let d = vec_of("x");
        lbm.commit(1, "B".into(), "d".into(), d.clone(), vec![]);
        let mut seen = BTreeSet::new();
        for i in 0..3 {
            if let Verdict::Intervene(iv) = lbm.evaluate(2 + i, "A", &d, &[]) {
                seen.insert(iv.strategy.id);
            }
        }
        assert_eq!(seen.len(), 3);
    }

    #[test]
    fn forward_progress_is_clear() {
        let mut lbm = LoopBreaker::default();
        lbm.anchor(0, "A".into());
        let d1 = vec_of("tokenize split words");
        lbm.commit(1, "B".into(), "d1".into(), d1, vec![]);
        assert_eq!(lbm.evaluate(2, "C", &vec_of("count frequency map"), &[]), Verdict::Clear);
    }
}
