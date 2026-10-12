//! Lattice-OS core engine.
//!
//! | Subsystem | Module |
//! |---|---|
//! | Graph tensor `T_C = (V, E, W)` and `ΔT` projections | [`graph`] |
//! | Geometric embedding of nodes, graphs and deltas | [`embed`] |
//! | Continuous Graph Compiler | [`cgc`] |
//! | Fluidity Routing Engine | [`fre`] |
//! | Loop-Breaking Matrix | [`lbm`] |
//! | Live runtime, revisions, snapshots | [`runtime`] |
//! | Generation loop | [`agent`] |
//! | Anthropic / OpenAI / Gemini / Ollama Cloud | [`providers`] |

pub mod agent;
pub mod cgc;
pub mod embed;
pub mod fre;
pub mod graph;
pub mod lbm;
pub mod providers;
pub mod runtime;
pub mod seed;

pub use agent::{run_agent, AgentConfig, AgentEvent, AgentOutcome, EventSink, Generator};
pub use graph::{Delta, DeltaOp, LatticeGraph, Node, NodeKind, Probe};
pub use providers::{ChatMessage, ModelInfo, ProviderClient, ProviderConfig, ProviderGenerator, ProviderKind};
pub use runtime::{LatticeSnapshot, RevisionSource, Runtime};
