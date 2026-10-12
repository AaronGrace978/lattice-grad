import { useState } from "react";
import type { LatticeSnapshot } from "../types";
import { fmtMicros } from "../api";

interface Props {
  snapshot: LatticeSnapshot;
  selected: string | null;
  onSelect: (id: string | null) => void;
}

/** The raw tensor: adjacency weights W[i][j] and per-node execution weights. */
export function TensorView({ snapshot, selected, onSelect }: Props) {
  const { ids, weights } = snapshot.tensor;
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null);
  const cell = Math.max(18, Math.min(38, Math.floor(420 / Math.max(1, ids.length))));
  const byId = new Map(snapshot.nodes.map((n) => [n.id, n]));

  return (
    <div className="tensor-view">
      <section>
        <h4>
          Adjacency tensor <span className="muted">W ∈ ℝ^{ids.length}×{ids.length}</span>
        </h4>
        <div className="matrix" style={{ gridTemplateColumns: `120px repeat(${ids.length}, ${cell}px)` }}>
          <div />
          {ids.map((id, j) => (
            <div key={id} className={`col-label ${hover?.j === j ? "hl" : ""}`} title={id}>
              <span>{id}</span>
            </div>
          ))}
          {ids.map((row, i) => (
            <Row
              key={row}
              row={row}
              i={i}
              ids={ids}
              weights={weights[i]}
              cell={cell}
              hover={hover}
              setHover={setHover}
              selected={selected}
              onSelect={onSelect}
            />
          ))}
        </div>
        <div className="muted small">
          {hover && weights[hover.i][hover.j] > 0
            ? `${ids[hover.i]} → ${ids[hover.j]} · normalised flow ${weights[hover.i][hover.j].toFixed(2)}`
            : "Cell intensity = EMA of bytes flowing along the edge, normalised to the heaviest edge."}
        </div>
      </section>
      <section>
        <h4>Execution weights</h4>
        <table className="weights-table">
          <thead>
            <tr>
              <th>node</th>
              <th>efficiency</th>
              <th>reliability</th>
              <th>latency</th>
              <th>ops</th>
              <th>velocity</th>
              <th>runs</th>
            </tr>
          </thead>
          <tbody>
            {ids.map((id) => {
              const w = byId.get(id)!.weights;
              return (
                <tr key={id} className={selected === id ? "sel" : ""} onClick={() => onSelect(id)}>
                  <td>{id}</td>
                  <td>
                    <Bar v={w.runs ? w.efficiency : 0} />
                  </td>
                  <td>
                    <Bar v={w.runs ? w.reliability : 0} bad={w.runs > 0 && w.reliability < 1} />
                  </td>
                  <td>{w.runs ? fmtMicros(w.latency_us) : "—"}</td>
                  <td>{w.runs ? w.ops.toLocaleString() : "—"}</td>
                  <td>{w.runs ? `${w.velocity.toFixed(2)} op/µs` : "—"}</td>
                  <td>{w.runs}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>
    </div>
  );
}

function Row(props: {
  row: string;
  i: number;
  ids: string[];
  weights: number[];
  cell: number;
  hover: { i: number; j: number } | null;
  setHover: (h: { i: number; j: number } | null) => void;
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  const { row, i, ids, weights, cell, hover, setHover, selected, onSelect } = props;
  return (
    <>
      <div className={`row-label ${hover?.i === i || selected === row ? "hl" : ""}`} onClick={() => onSelect(row)}>
        {row}
      </div>
      {ids.map((col, j) => {
        const v = weights[j];
        return (
          <div
            key={col}
            className="cell"
            style={{
              width: cell,
              height: cell,
              background: v > 0 ? `rgba(124, 200, 255, ${0.15 + v * 0.85})` : undefined,
              boxShadow: v > 0.6 ? `0 0 ${8 * v}px rgba(124,200,255,${v * 0.6})` : undefined,
            }}
            onMouseEnter={() => setHover({ i, j })}
            onMouseLeave={() => setHover(null)}
          />
        );
      })}
    </>
  );
}

function Bar({ v, bad }: { v: number; bad?: boolean }) {
  return (
    <div className={`mini-bar ${bad ? "bad" : ""}`}>
      <i style={{ width: `${Math.round(v * 100)}%` }} />
      <span>{v ? v.toFixed(2) : "—"}</span>
    </div>
  );
}
