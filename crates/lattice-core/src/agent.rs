//! The generation loop that closes the execution topology:
//!
//! ```text
//! [LLM tokens] -> [CGC] -> [Live Assembly Lattice]
//!                                  |
//! [LBM anti-loop] <---- [FRE] <----+
//! ```
//!
//! Each iteration streams a completion, extracts a `ΔT` projection, lets the
//! LBM veto oscillation, compiles and executes the probe suite, and feeds the
//! resulting fault vectors (or LBM directive) back into the next prompt.

use crate::graph::Delta;
use crate::lbm::Intervention;
use crate::providers::ChatMessage;
use crate::runtime::{format_faults, ApplyOutcome, ProposeError, RevisionSource, Runtime, SuiteSummary};
use async_trait::async_trait;
use serde::Serialize;
use serde_json::Value;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use thiserror::Error;

#[derive(Debug, Error, Clone)]
#[error("{0}")]
pub struct GenError(pub String);

#[async_trait]
pub trait Generator: Send + Sync {
    async fn generate(
        &self,
        system: &str,
        messages: &[ChatMessage],
        on_token: &(dyn for<'t> Fn(&'t str) + Send + Sync),
    ) -> Result<String, GenError>;

    fn label(&self) -> String;
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    Started { goal: String, generator: String, max_iterations: u32 },
    Iteration { iteration: u32 },
    Token { iteration: u32, text: String },
    DeltaParsed { iteration: u32, summary: String, ops: usize },
    DeltaRejected { iteration: u32, reason: String },
    Intervention { iteration: u32, intervention: Intervention },
    Applied { iteration: u32, outcome: ApplyOutcome },
    Suite { iteration: u32, summary: SuiteSummary },
    LatticeChanged { revision: u64 },
    Finished { success: bool, iterations: u32, reason: String },
    Error { message: String },
}

pub trait EventSink: Send + Sync {
    fn emit(&self, event: AgentEvent);
}

impl<F: Fn(AgentEvent) + Send + Sync> EventSink for F {
    fn emit(&self, event: AgentEvent) {
        self(event)
    }
}

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub max_iterations: u32,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self { max_iterations: 6 }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentOutcome {
    pub success: bool,
    pub iterations: u32,
    pub reason: String,
}

pub const SYSTEM_PROMPT: &str = r#"You are the generation layer of Lattice-OS, a geometric AI runtime.
The codebase is NOT text files. It is a live directed acyclic graph tensor T = (V, E, W):
- V: nodes. kind is one of `input` (external value, no body), `function` (Rhai closure),
  `type` (validation boundary: returns its normalised input or throws), `output` (reported value).
- E: data-flow edges. An edge binds the output of `from` to the named port `port` of `to`.
  Every port must be listed in the target node's `params`. One edge per (to, port).
- W: execution telemetry measured by the runtime (you only read it).

You change the system ONLY by emitting a tensor projection ΔT as JSON inside exactly one
fenced block tagged `lattice`:

```lattice
{
  "summary": "one-line description of the change",
  "ops": [
    {"op": "upsert_node", "id": "parse", "kind": "function", "params": ["text"],
     "body": "text.split(\",\")", "fallback": "[]", "doc": "split CSV"},
    {"op": "connect", "from": "raw", "to": "parse", "port": "text"},
    {"op": "disconnect", "from": "a", "to": "b", "port": "x"},
    {"op": "remove_node", "id": "old"},
    {"op": "upsert_probe", "name": "basic", "inputs": {"raw": "a,b"}, "expect": {"out": ["a", "b"]}},
    {"op": "remove_probe", "name": "stale"}
  ]
}
```
`upsert_node` patches only the fields you provide on an existing node. Ids and params are
identifiers ([A-Za-z_][A-Za-z0-9_]*). The graph must stay acyclic. Probes feed input nodes
by id and assert output nodes by id; expectations compare JSON structurally (3 == 3.0).
An output node with an empty body forwards its single param, or an object of all params.

Node bodies are Rhai scripts. The node's params are pre-bound variables; the value of the
final expression (no trailing semicolon) is the node output. Rhai essentials:
- `let x = 1;`  `if c { a } else { b }` is an expression.  `for x in arr { }`  `for i in 0..n { }`  `while c { }`
- Integers are i64; `/` on integers truncates and throws on division by zero. Use `x.to_float()`,
  `f.round()`, `f.floor()`, `f.to_int()`. Round to 2 decimals: `(f * 100.0).round() / 100.0`.
- Arrays: `[1, 2]`, `a.len()`, `a.push(v)`, `a.map(|x| x * 2)`, `a.filter(|x| x > 1)`,
  `a.reduce(|acc, x| acc + x, 0)`, `a.contains(v)`, `a.sort()` (in place), `a + b`.
- Maps: `#{ key: 1, "two words": 2 }`, `m[k] = v`, `m.contains(k)`, `m.keys()`, `m.values()`, missing key -> `()`.
- Strings: `s.len()`, `s.split(",")`, `s.contains("x")`, `s.to_upper()`, `s.to_lower()`, `parse_int(s)`,
  `parse_float(s)`, `` `interp ${x}` ``, `s.sub_string(start, len)`, `s.index_of("x")`.
  CAUTION: `s.trim()`, `s.replace(a, b)`, `s.make_upper()` mutate in place and return `()`:
  write `let t = s; t.trim(); t`.
- `type_of(x)` returns "i64", "f64", "string", "array", "map", "bool", "()".  `throw "msg"` raises a fault.
- Helper functions may be declared inside a body with `fn name(a) { ... }`; they cannot see outer
  variables, so pass everything as arguments. `print(x)` writes to the execution trace.
- The sandbox has no I/O, no imports and no `eval`, and an operation budget per node.

Runtime behaviour you should exploit:
- The Continuous Graph Compiler recompiles only nodes whose code changed.
- The Fluidity Routing Engine executes every probe. A faulting node is isolated in a bubble and its
  output is replaced by a proxy (last-known-good, then its `fallback`, then pass-through of the
  first port). You receive those faults as repair vectors.
- The Loop-Breaking Matrix rejects deltas that return to earlier states or undo your previous change.
  When it intervenes you MUST follow its alternative pathway.

Rules:
1. Respond with brief reasoning (at most a few sentences) followed by exactly one ```lattice block.
2. Keep deltas minimal and reference existing ids exactly. Do not resend unchanged nodes.
3. Always maintain probes with concrete expectations that prove the requested behaviour, including
   edge cases (empty input, zero, missing fields).
4. If the lattice already satisfies the goal and every probe passes, emit an empty ops list."#;

#[derive(Debug, Error, PartialEq)]
pub enum ParseError {
    #[error("no ```lattice block or JSON object with an `ops` array was found")]
    NoDelta,
    #[error("the lattice block is not a valid delta: {0}")]
    Invalid(String),
}

/// Extract `ΔT` from a model response. Prefers a ```lattice fence, then a
/// ```json fence, then the first balanced JSON object containing `"ops"`.
pub fn parse_delta(response: &str) -> Result<Delta, ParseError> {
    let mut candidates: Vec<String> = Vec::new();
    for tag in ["```lattice", "```json", "```"] {
        let mut rest = response;
        while let Some(start) = rest.find(tag) {
            let after = &rest[start + tag.len()..];
            let body_start = after.find('\n').map(|i| i + 1).unwrap_or(0);
            let body = &after[body_start..];
            match body.find("```") {
                Some(end) => {
                    candidates.push(body[..end].trim().to_string());
                    rest = &body[end + 3..];
                }
                None => {
                    candidates.push(body.trim().to_string());
                    break;
                }
            }
        }
        if !candidates.is_empty() {
            break;
        }
    }
    candidates.extend(balanced_objects(response));

    let mut last_err = None;
    for c in candidates {
        let Ok(value) = serde_json::from_str::<Value>(&c) else {
            if c.contains("\"ops\"") {
                last_err = Some(ParseError::Invalid("malformed JSON".into()));
            }
            continue;
        };
        let value = match value {
            Value::Array(ops) => serde_json::json!({ "ops": ops }),
            v if v.get("ops").is_some() => v,
            _ => continue,
        };
        match serde_json::from_value::<Delta>(value) {
            Ok(d) => return Ok(d),
            Err(e) => last_err = Some(ParseError::Invalid(e.to_string())),
        }
    }
    Err(last_err.unwrap_or(ParseError::NoDelta))
}

fn balanced_objects(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'{' {
            i += 1;
            continue;
        }
        let (mut depth, mut in_str, mut esc) = (0i32, false, false);
        let mut end = None;
        for (j, &b) in bytes.iter().enumerate().skip(i) {
            if in_str {
                match (esc, b) {
                    (true, _) => esc = false,
                    (false, b'\\') => esc = true,
                    (false, b'"') => in_str = false,
                    _ => {}
                }
                continue;
            }
            match b {
                b'"' => in_str = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(j);
                        break;
                    }
                }
                _ => {}
            }
        }
        match end {
            Some(j) => {
                let s = &text[i..=j];
                if s.contains("\"ops\"") {
                    out.push(s.to_string());
                }
                i = j + 1;
            }
            None => break,
        }
    }
    out
}

fn prune(messages: &[ChatMessage]) -> Vec<ChatMessage> {
    const TAIL: usize = 4;
    if messages.len() <= TAIL + 1 {
        return messages.to_vec();
    }
    let mut out = vec![messages[0].clone()];
    out.extend_from_slice(&messages[messages.len() - TAIL..]);
    out
}

fn lock(rt: &Mutex<Runtime>) -> std::sync::MutexGuard<'_, Runtime> {
    rt.lock().unwrap_or_else(|p| p.into_inner())
}

pub async fn run_agent(
    runtime: Arc<Mutex<Runtime>>,
    generator: &dyn Generator,
    goal: &str,
    config: &AgentConfig,
    sink: &dyn EventSink,
    cancel: &AtomicBool,
) -> AgentOutcome {
    sink.emit(AgentEvent::Started {
        goal: goal.to_string(),
        generator: generator.label(),
        max_iterations: config.max_iterations,
    });

    let opening = {
        let mut rt = lock(&runtime);
        let suite = rt.run_suite();
        let mut msg = format!("GOAL:\n{goal}\n\nCURRENT LATTICE:\n{}", rt.digest());
        let faults = suite.summary().faults;
        if !faults.is_empty() {
            msg.push_str(&format!("\nCURRENT FAULTS:\n{}", format_faults(&faults, 12)));
        }
        msg
    };
    sink.emit(AgentEvent::LatticeChanged { revision: lock(&runtime).revision() });
    let mut messages = vec![ChatMessage::user(opening)];

    let finish = |success: bool, iterations: u32, reason: String| {
        sink.emit(AgentEvent::Finished { success, iterations, reason: reason.clone() });
        AgentOutcome { success, iterations, reason }
    };

    for iteration in 1..=config.max_iterations {
        if cancel.load(Ordering::Relaxed) {
            return finish(false, iteration - 1, "cancelled".into());
        }
        sink.emit(AgentEvent::Iteration { iteration });

        let on_token = |t: &str| sink.emit(AgentEvent::Token { iteration, text: t.to_string() });
        let response = match generator.generate(SYSTEM_PROMPT, &prune(&messages), &on_token).await {
            Ok(r) => r,
            Err(e) => {
                sink.emit(AgentEvent::Error { message: e.0.clone() });
                return finish(false, iteration, format!("generation failed: {}", e.0));
            }
        };
        if cancel.load(Ordering::Relaxed) {
            return finish(false, iteration, "cancelled".into());
        }
        messages.push(ChatMessage::assistant(response.clone()));

        let feedback = match parse_delta(&response) {
            Err(e) => {
                sink.emit(AgentEvent::DeltaRejected { iteration, reason: e.to_string() });
                format!("Your response was not applied: {e}. Reply with exactly one ```lattice block.")
            }
            Ok(delta) => {
                sink.emit(AgentEvent::DeltaParsed { iteration, summary: delta.summary.clone(), ops: delta.ops.len() });
                let proposed = lock(&runtime).propose(&delta, RevisionSource::Agent, true);
                match proposed {
                    Err(ProposeError::Oscillation(iv)) => {
                        let prompt = iv.prompt();
                        sink.emit(AgentEvent::Intervention { iteration, intervention: *iv });
                        sink.emit(AgentEvent::LatticeChanged { revision: lock(&runtime).revision() });
                        format!("{prompt}\nThe lattice is unchanged.")
                    }
                    Err(e) => {
                        sink.emit(AgentEvent::DeltaRejected { iteration, reason: e.to_string() });
                        format!("The CGC validator rejected your delta: {e}. The lattice is unchanged.")
                    }
                    Ok(outcome) => {
                        let noop = outcome.noop;
                        let revision = outcome.revision;
                        sink.emit(AgentEvent::Applied { iteration, outcome });
                        let summary = lock(&runtime).run_suite().summary();
                        sink.emit(AgentEvent::Suite { iteration, summary: summary.clone() });
                        sink.emit(AgentEvent::LatticeChanged { revision });
                        if summary.healthy {
                            return finish(true, iteration, format!("all {} probes pass at r{revision}", summary.probes));
                        }
                        if summary.probes == 0 {
                            "The delta was applied but the lattice has no probes. Add probes with concrete expectations that prove the goal.".to_string()
                        } else {
                            format!(
                                "{}Execution report for r{revision}: {}/{} probes healthy.\nFAULT VECTORS:\n{}",
                                if noop { "Your delta did not change the lattice state. " } else { "" },
                                summary.passed,
                                summary.probes,
                                format_faults(&summary.faults, 12)
                            )
                        }
                    }
                }
            }
        };
        let digest = lock(&runtime).digest();
        messages.push(ChatMessage::user(format!("{feedback}\n\nCURRENT LATTICE:\n{digest}")));
    }

    finish(false, config.max_iterations, "iteration budget exhausted".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seed;
    use std::sync::Mutex as StdMutex;

    struct Scripted {
        replies: StdMutex<Vec<String>>,
        seen: StdMutex<Vec<Vec<ChatMessage>>>,
    }

    impl Scripted {
        fn new(replies: &[&str]) -> Self {
            Self {
                replies: StdMutex::new(replies.iter().rev().map(|s| s.to_string()).collect()),
                seen: StdMutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl Generator for Scripted {
        async fn generate(&self, _: &str, messages: &[ChatMessage], on_token: &(dyn for<'t> Fn(&'t str) + Send + Sync)) -> Result<String, GenError> {
            self.seen.lock().unwrap().push(messages.to_vec());
            let r = self.replies.lock().unwrap().pop().ok_or_else(|| GenError("script exhausted".into()))?;
            on_token(&r);
            Ok(r)
        }
        fn label(&self) -> String {
            "scripted".into()
        }
    }

    fn fence(json: &str) -> String {
        format!("Reasoning.\n```lattice\n{json}\n```")
    }

    #[test]
    fn parse_delta_variants() {
        let d = parse_delta(&fence(r#"{"summary":"x","ops":[{"op":"remove_node","id":"a"}]}"#)).unwrap();
        assert_eq!(d.ops.len(), 1);
        let d = parse_delta("sure {\"ops\": [{\"op\": \"remove_probe\", \"name\": \"p\"}], \"summary\": \"has } brace\"}").unwrap();
        assert_eq!(d.summary, "has } brace");
        let d = parse_delta("```json\n[{\"op\":\"remove_node\",\"id\":\"z\"}]\n```").unwrap();
        assert_eq!(d.ops.len(), 1);
        assert_eq!(parse_delta("no delta here"), Err(ParseError::NoDelta));
        assert!(matches!(parse_delta(&fence(r#"{"ops":[{"op":"explode"}]}"#)), Err(ParseError::Invalid(_))));
    }

    #[tokio::test]
    async fn agent_repairs_the_demo_lattice() {
        let rt = Arc::new(Mutex::new(Runtime::new(seed::demo_lattice()).unwrap()));
        let fix = fence(r#"{"summary":"guard empty input","ops":[{"op":"upsert_node","id":"avg_word_len","body":"if count == 0 { 0 } else { chars / count }"}]}"#);
        let gen = Scripted::new(&[&fix]);
        let events = StdMutex::new(Vec::new());
        let sink = |e: AgentEvent| events.lock().unwrap().push(e);
        let out = run_agent(rt.clone(), &gen, "make every probe pass", &AgentConfig::default(), &sink, &AtomicBool::new(false)).await;
        assert!(out.success, "{out:?}");
        assert_eq!(out.iterations, 1);
        assert!(lock(&rt).fre.bubbles.is_empty());
        let opening = &gen.seen.lock().unwrap()[0][0].content;
        assert!(opening.contains("CURRENT FAULTS") && opening.contains("avg_word_len"));
        assert!(events.lock().unwrap().iter().any(|e| matches!(e, AgentEvent::Suite { .. })));
    }

    #[tokio::test]
    async fn lbm_breaks_an_oscillating_agent() {
        let rt = Arc::new(Mutex::new(Runtime::new(seed::demo_lattice()).unwrap()));
        let seed_state = lock(&rt).state_hash().to_string();
        let a = fence(r#"{"summary":"try float","ops":[{"op":"upsert_node","id":"avg_word_len","body":"chars.to_float() / count"}]}"#);
        let b = fence(r#"{"summary":"back to int","ops":[{"op":"upsert_node","id":"avg_word_len","body":"chars / count"}]}"#);
        let fixed = fence(r#"{"summary":"guard","ops":[{"op":"upsert_node","id":"avg_word_len","body":"if count == 0 { 0 } else { chars / count }"}]}"#);
        let gen = Scripted::new(&[&a, &b, &fixed]);
        let events = StdMutex::new(Vec::new());
        let sink = |e: AgentEvent| events.lock().unwrap().push(e);
        let out = run_agent(rt.clone(), &gen, "fix", &AgentConfig::default(), &sink, &AtomicBool::new(false)).await;
        assert!(out.success, "{out:?}");
        let evs = events.lock().unwrap();
        let iv = evs.iter().find_map(|e| match e {
            AgentEvent::Intervention { intervention, .. } => Some(intervention.clone()),
            _ => None,
        });
        let iv = iv.expect("LBM should intervene on A -> B -> A");
        assert_eq!(iv.oscillating_nodes, vec!["avg_word_len".to_string()]);
        let third_prompt = gen.seen.lock().unwrap()[2].last().unwrap().content.clone();
        assert!(third_prompt.contains("LOOP-BREAKING MATRIX INTERVENTION"));
        assert!(lock(&rt).lbm.is_forbidden(&seed_state));
    }

    #[tokio::test]
    async fn invalid_responses_are_fed_back_and_budget_is_respected() {
        let rt = Arc::new(Mutex::new(Runtime::new(seed::demo_lattice()).unwrap()));
        let gen = Scripted::new(&["I refuse to emit JSON", &fence(r#"{"ops":[{"op":"connect","from":"report","to":"tokenize","port":"text"}]}"#)]);
        let sink = |_: AgentEvent| {};
        let out = run_agent(rt, &gen, "x", &AgentConfig { max_iterations: 2 }, &sink, &AtomicBool::new(false)).await;
        assert!(!out.success);
        let seen = gen.seen.lock().unwrap();
        assert!(seen[1].last().unwrap().content.contains("not applied"));
    }

    #[tokio::test]
    async fn cancellation_stops_before_generation() {
        let rt = Arc::new(Mutex::new(Runtime::new(seed::demo_lattice()).unwrap()));
        let gen = Scripted::new(&[]);
        let sink = |_: AgentEvent| {};
        let out = run_agent(rt, &gen, "x", &AgentConfig::default(), &sink, &AtomicBool::new(true)).await;
        assert_eq!(out.reason, "cancelled");
    }
}
