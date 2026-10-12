import type { LatticeSnapshot } from "../types";
import { short } from "../api";

const SOURCE_ICON: Record<string, string> = { seed: "◎", agent: "✦", manual: "✎", rollback: "↺", import: "⇪" };

interface Props {
  snapshot: LatticeSnapshot;
  onRollback: (revision: number) => void;
  locked: boolean;
}

export function Timeline({ snapshot, onRollback, locked }: Props) {
  const forbidden = new Set(snapshot.forbidden_states);
  return (
    <div className="timeline">
      <div className="revisions">
        <div className="strip-title">Tensor history</div>
        <div className="rev-strip">
          {snapshot.history.map((r) => {
            const current = r.revision === snapshot.revision;
            return (
              <button
                key={r.revision}
                className={`rev ${current ? "current" : ""} ${forbidden.has(r.state_hash) ? "forbidden" : ""} src-${r.source}`}
                title={`${r.summary}\nstate ${r.state_hash}${forbidden.has(r.state_hash) ? "\nforbidden by LBM" : ""}${current ? "" : "\nclick to restore"}`}
                disabled={current || locked}
                onClick={() => onRollback(r.revision)}
              >
                <span className="rev-id">
                  {SOURCE_ICON[r.source]} r{r.revision}
                </span>
                <span className="rev-hash">{short(r.state_hash, 8)}</span>
                <span className="rev-sum">{r.summary}</span>
              </button>
            );
          })}
        </div>
      </div>
      <div className="lbm">
        <div className="strip-title">
          Loop-Breaking Matrix <span className="muted">· {snapshot.forbidden_states.length} forbidden states</span>
        </div>
        {snapshot.interventions.length === 0 ? (
          <div className="muted small lbm-empty">No oscillation detected. Every state is SHA-256 tracked; ΔT directions are compared in ℝ⁶⁴.</div>
        ) : (
          <div className="lbm-list">
            {snapshot.interventions.map((iv, i) => (
              <div key={i} className="lbm-item" title={iv.strategy.directive}>
                <div className="row between">
                  <strong>{iv.strategy.title}</strong>
                  <span className="tag lbm">⊥ {iv.orthogonality.toFixed(2)}</span>
                </div>
                <div className="small muted">
                  r{iv.revision} · {iv.oscillation.type.replace("_", " ")}
                  {iv.oscillation.type === "anti_parallel" && ` cos ${iv.oscillation.cosine.toFixed(2)}`}
                  {iv.oscillation.type === "state_revisit" && ` (cycle ${iv.oscillation.cycle_len})`} ·{" "}
                  {iv.oscillating_nodes.join(", ") || "topology"}
                </div>
                <div className="vec">
                  {iv.orthogonal_preview.map((v, j) => (
                    <i key={j} style={{ height: `${Math.min(100, Math.abs(v) * 260)}%`, background: v >= 0 ? "#7cf3ff" : "#ff6ad5" }} />
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
