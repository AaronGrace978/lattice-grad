import { useEffect, useState } from "react";
import type { LatticeSnapshot } from "../types";
import { api, errorText } from "../api";

interface Props {
  onClose: () => void;
  onLoaded: (s: LatticeSnapshot) => void;
  locked: boolean;
}

export function LatticeIOModal({ onClose, onLoaded, locked }: Props) {
  const [json, setJson] = useState("");
  const [msg, setMsg] = useState<string | null>(null);

  useEffect(() => {
    api.exportLattice().then(setJson).catch((e) => setMsg(errorText(e)));
  }, []);

  const run = async (fn: () => Promise<LatticeSnapshot>, label: string) => {
    try {
      onLoaded(await fn());
      setMsg(label);
      setJson(await api.exportLattice());
    } catch (e) {
      setMsg(errorText(e));
    }
  };

  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <div className="modal wide" onMouseDown={(e) => e.stopPropagation()}>
        <div className="panel-head">
          <h3>Lattice state</h3>
          <button className="ghost" onClick={onClose}>
            ✕
          </button>
        </div>
        <p className="muted small">
          The lattice lives in memory; this is its serialized tensor. Copy it out to persist it, or paste a lattice to replace the
          live one.
        </p>
        <textarea className="code" rows={20} value={json} spellCheck={false} onChange={(e) => setJson(e.target.value)} />
        <div className="row wrap">
          <button onClick={() => navigator.clipboard.writeText(json).then(() => setMsg("Copied to clipboard"))}>Copy</button>
          <button className="primary" disabled={locked} onClick={() => run(() => api.importLattice(json), "Imported")}>
            Import JSON
          </button>
          <span className="spacer" />
          <button disabled={locked} onClick={() => run(() => api.loadTemplate("demo"), "Loaded demo lattice")}>
            Load demo lattice
          </button>
          <button disabled={locked} className="danger ghost" onClick={() => run(() => api.loadTemplate("empty"), "Cleared")}>
            Empty lattice
          </button>
        </div>
        {msg && <div className="muted small">{msg}</div>}
      </div>
    </div>
  );
}
