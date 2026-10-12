import { useEffect, useRef, useState } from "react";
import type { AgentEvent } from "../types";
import { short } from "../api";

export interface IterationLog {
  iteration: number;
  text: string;
  events: AgentEvent[];
}

export interface AgentRun {
  id: number;
  goal: string;
  generator: string;
  maxIterations: number;
  iterations: IterationLog[];
  finished?: { success: boolean; reason: string; iterations: number };
  error?: string;
}

export function reduceRun(run: AgentRun, e: AgentEvent): AgentRun {
  switch (e.type) {
    case "started":
      return { ...run, goal: e.goal, generator: e.generator, maxIterations: e.max_iterations };
    case "iteration":
      return { ...run, iterations: [...run.iterations, { iteration: e.iteration, text: "", events: [] }] };
    case "token": {
      const its = run.iterations.slice();
      const last = its[its.length - 1];
      if (last) its[its.length - 1] = { ...last, text: last.text + e.text };
      return { ...run, iterations: its };
    }
    case "finished":
      return { ...run, finished: { success: e.success, reason: e.reason, iterations: e.iterations } };
    case "error":
      return { ...run, error: e.message };
    case "lattice_changed":
      return run;
    default: {
      const its = run.iterations.slice();
      const last = its[its.length - 1];
      if (last) its[its.length - 1] = { ...last, events: [...last.events, e] };
      return { ...run, iterations: its };
    }
  }
}

const PRESETS = [
  "Repair every isolated node so all probes pass, including edge cases.",
  "Add a `unique_words` metric to the report, with probes proving it.",
  "Change avg_word_len to a float rounded to 2 decimals and update the probes.",
  "Replace the lattice with a CSV → JSON pipeline: parse header + rows into objects, with probes.",
];

interface Props {
  runs: AgentRun[];
  running: boolean;
  canRun: boolean;
  modelLabel: string;
  onStart: (goal: string) => void;
  onCancel: () => void;
  onClear: () => void;
}

export function AgentConsole({ runs, running, canRun, modelLabel, onStart, onCancel, onClear }: Props) {
  const [goal, setGoal] = useState("");
  const scroller = useRef<HTMLDivElement>(null);
  const lastText = runs.at(-1)?.iterations.at(-1)?.text.length ?? 0;

  useEffect(() => {
    const el = scroller.current;
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 260) el.scrollTop = el.scrollHeight;
  }, [runs.length, lastText, runs.at(-1)?.iterations.length]);

  const submit = () => {
    if (!goal.trim() || running) return;
    onStart(goal.trim());
  };

  return (
    <div className="agent-console">
      <div className="panel-head">
        <h3>Generation layer</h3>
        {runs.length > 0 && !running && (
          <button className="ghost small" onClick={onClear}>
            clear
          </button>
        )}
      </div>
      <div className="runs" ref={scroller}>
        {runs.length === 0 && (
          <div className="console-empty">
            <p>
              Describe a feature or a repair. The model emits <code>ΔT</code> tensor projections, the CGC compiles them into
              the live lattice, the FRE executes every probe and isolates faults, and the LBM breaks oscillating loops.
            </p>
            <div className="presets">
              {PRESETS.map((p) => (
                <button key={p} className="preset" onClick={() => setGoal(p)}>
                  {p}
                </button>
              ))}
            </div>
          </div>
        )}
        {runs.map((run, idx) => (
          <RunCard key={run.id} run={run} defaultOpen={idx === runs.length - 1} />
        ))}
      </div>
      <div className="composer">
        <textarea
          value={goal}
          placeholder="e.g. Add a sentiment score node fed by tokenize, expose it in the report, prove it with probes…"
          onChange={(e) => setGoal(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
          }}
          rows={3}
        />
        <div className="composer-row">
          <span className="muted small">{modelLabel}</span>
          {running ? (
            <button className="danger" onClick={onCancel}>
              ■ Stop
            </button>
          ) : (
            <button className="primary" disabled={!canRun || !goal.trim()} onClick={submit} title="Ctrl/⌘ + Enter">
              ▶ Run agent
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function RunCard({ run, defaultOpen }: { run: AgentRun; defaultOpen: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  useEffect(() => setOpen(defaultOpen), [defaultOpen]);
  const status = run.finished ? (run.finished.success ? "success" : "failed") : "running";
  return (
    <div className={`run-card ${status}`}>
      <button className="run-head" onClick={() => setOpen(!open)}>
        <span className={`run-dot ${status}`} />
        <span className="run-goal">{run.goal}</span>
        <span className="muted small">{run.generator}</span>
      </button>
      {open && (
        <div className="run-body">
          {run.iterations.map((it) => (
            <div key={it.iteration} className="iteration">
              <div className="it-head">
                iteration {it.iteration}
                <span className="muted"> / {run.maxIterations}</span>
              </div>
              {it.text && <pre className="tokens">{it.text}</pre>}
              <div className="chips">
                {it.events.map((e, i) => (
                  <EventChip key={i} e={e} />
                ))}
              </div>
            </div>
          ))}
          {run.error && <div className="chip bad">✕ {run.error}</div>}
          {run.finished && (
            <div className={`finish ${run.finished.success ? "ok" : "bad"}`}>
              {run.finished.success ? "✓ Converged" : "■ Stopped"} — {run.finished.reason}
            </div>
          )}
          {!run.finished && (
            <div className="thinking">
              <span />
              <span />
              <span />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function EventChip({ e }: { e: AgentEvent }) {
  switch (e.type) {
    case "delta_parsed":
      return (
        <div className="chip info">
          ΔT parsed · {e.ops} ops · <em>{e.summary || "no summary"}</em>
        </div>
      );
    case "delta_rejected":
      return <div className="chip bad">✕ rejected · {e.reason}</div>;
    case "applied":
      return (
        <div className="chip cgc">
          CGC · r{e.outcome.revision} {short(e.outcome.state_hash, 8)} · compiled {e.outcome.compile.compiled}, reused{" "}
          {e.outcome.compile.reused}
          {e.outcome.compile.failed > 0 && `, ${e.outcome.compile.failed} broken`}
          {e.outcome.noop && " · no-op"} · ‖Δ‖={e.outcome.delta_norm.toFixed(2)}
        </div>
      );
    case "suite":
      return (
        <div className={`chip ${e.summary.healthy ? "ok" : "warn"}`}>
          FRE · {e.summary.passed}/{e.summary.probes} probes healthy
          {e.summary.faults.length > 0 && ` · ${e.summary.faults.length} fault vectors`}
        </div>
      );
    case "intervention":
      return (
        <div className="chip lbm">
          LBM · {e.intervention.oscillation.type.replace("_", " ")} on{" "}
          {e.intervention.oscillating_nodes.join(", ") || "topology"} → orthogonal pathway{" "}
          <strong>{e.intervention.strategy.title}</strong> (⊥ {e.intervention.orthogonality.toFixed(2)})
        </div>
      );
    default:
      return null;
  }
}
