import { useEffect, useMemo, useRef, useState } from "react";
import type { LatticeSnapshot, NodeView } from "../types";

const NODE_W = 184;
const NODE_H = 66;
const COL_GAP = 96;
const ROW_GAP = 34;

interface Props {
  snapshot: LatticeSnapshot;
  selected: string | null;
  onSelect: (id: string | null) => void;
  flowing: boolean;
  pulse: Set<string>;
}

interface Placed {
  node: NodeView;
  x: number;
  y: number;
}

function layout(snapshot: LatticeSnapshot): Map<string, Placed> {
  const byLayer = new Map<number, NodeView[]>();
  for (const n of snapshot.nodes) {
    const list = byLayer.get(n.layer) ?? [];
    list.push(n);
    byLayer.set(n.layer, list);
  }
  const layers = [...byLayer.keys()].sort((a, b) => a - b);
  const placed = new Map<string, Placed>();
  const preds = new Map<string, string[]>();
  for (const e of snapshot.edges) {
    preds.set(e.to, [...(preds.get(e.to) ?? []), e.from]);
  }
  const tallest = Math.max(1, ...[...byLayer.values()].map((l) => l.length));
  const totalH = tallest * (NODE_H + ROW_GAP);

  for (const layer of layers) {
    const nodes = byLayer.get(layer)!;
    const bary = (n: NodeView) => {
      const ys = (preds.get(n.id) ?? []).map((p) => placed.get(p)?.y).filter((y): y is number => y !== undefined);
      return ys.length ? ys.reduce((a, b) => a + b, 0) / ys.length : Number.MAX_SAFE_INTEGER;
    };
    nodes.sort((a, b) => bary(a) - bary(b) || a.id.localeCompare(b.id));
    const colH = nodes.length * (NODE_H + ROW_GAP);
    const offset = (totalH - colH) / 2;
    nodes.forEach((n, i) => {
      placed.set(n.id, { node: n, x: layer * (NODE_W + COL_GAP), y: offset + i * (NODE_H + ROW_GAP) });
    });
  }
  return placed;
}

const KIND_GLYPH: Record<string, string> = { input: "◇", function: "λ", type: "τ", output: "◆" };

export function LatticeView({ snapshot, selected, onSelect, flowing, pulse }: Props) {
  const placed = useMemo(() => layout(snapshot), [snapshot]);
  const svgRef = useRef<SVGSVGElement>(null);
  const [view, setView] = useState({ x: 0, y: 0, k: 1 });
  const drag = useRef<{ x: number; y: number; vx: number; vy: number } | null>(null);
  const nodeCount = snapshot.nodes.length;

  const bounds = useMemo(() => {
    let maxX = 0;
    let maxY = 0;
    for (const p of placed.values()) {
      maxX = Math.max(maxX, p.x + NODE_W);
      maxY = Math.max(maxY, p.y + NODE_H);
    }
    return { w: maxX, h: maxY };
  }, [placed]);

  const fit = () => {
    const el = svgRef.current;
    if (!el) return;
    const { width, height } = el.getBoundingClientRect();
    if (!bounds.w || !width) return;
    const k = Math.min(1.25, (width - 80) / bounds.w, (height - 80) / bounds.h);
    setView({ k, x: (width - bounds.w * k) / 2, y: (height - bounds.h * k) / 2 });
  };

  useEffect(fit, [nodeCount, bounds.w, bounds.h]);
  useEffect(() => {
    const onResize = () => fit();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  });

  const maxWeight = Math.max(1, ...snapshot.edges.map((e) => e.weight));

  const onWheel = (e: React.WheelEvent) => {
    const rect = svgRef.current!.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const factor = Math.exp(-e.deltaY * 0.0015);
    setView((v) => {
      const k = Math.min(3, Math.max(0.25, v.k * factor));
      const r = k / v.k;
      return { k, x: mx - (mx - v.x) * r, y: my - (my - v.y) * r };
    });
  };

  return (
    <div className="lattice-view">
      <svg
        ref={svgRef}
        className={flowing ? "flowing" : ""}
        onWheel={onWheel}
        onMouseDown={(e) => {
          if ((e.target as Element).closest(".lnode")) return;
          drag.current = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y };
        }}
        onMouseMove={(e) => {
          const d = drag.current;
          if (d) setView((v) => ({ ...v, x: d.vx + e.clientX - d.x, y: d.vy + e.clientY - d.y }));
        }}
        onMouseUp={(e) => {
          const d = drag.current;
          drag.current = null;
          if (d && Math.abs(e.clientX - d.x) + Math.abs(e.clientY - d.y) < 4) onSelect(null);
        }}
        onMouseLeave={() => (drag.current = null)}
      >
        <defs>
          <pattern id="grid" width="28" height="28" patternUnits="userSpaceOnUse">
            <circle cx="1" cy="1" r="1" fill="rgba(140,160,255,0.12)" />
          </pattern>
          <linearGradient id="edge-grad" x1="0" x2="1">
            <stop offset="0" stopColor="#6fe7ff" stopOpacity="0.9" />
            <stop offset="1" stopColor="#9c8bff" stopOpacity="0.9" />
          </linearGradient>
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="6" result="b" />
            <feMerge>
              <feMergeNode in="b" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />
        <g transform={`translate(${view.x},${view.y}) scale(${view.k})`}>
          {snapshot.edges.map((e) => {
            const a = placed.get(e.from);
            const b = placed.get(e.to);
            if (!a || !b) return null;
            const portIdx = Math.max(0, b.node.params.indexOf(e.port));
            const ports = Math.max(1, b.node.params.length);
            const x1 = a.x + NODE_W;
            const y1 = a.y + NODE_H / 2;
            const x2 = b.x;
            const y2 = b.y + ((portIdx + 1) * NODE_H) / (ports + 1);
            const dx = Math.max(40, (x2 - x1) / 2);
            const w = 1.4 + 4 * (e.weight / maxWeight);
            const hot = selected === e.from || selected === e.to;
            const faulty = a.node.status === "isolated" || a.node.status === "broken";
            return (
              <g key={`${e.from}-${e.to}-${e.port}`} className={`ledge ${hot ? "hot" : ""} ${faulty ? "faulty" : ""}`}>
                <path d={`M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`} strokeWidth={w} />
                <path className="flow" d={`M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`} strokeWidth={Math.max(1, w - 1)} />
                <text x={x2 - 6} y={y2 - 5} textAnchor="end" className="port-label">
                  {e.port}
                </text>
              </g>
            );
          })}
          {[...placed.values()].map(({ node, x, y }) => {
            const isSel = selected === node.id;
            const isolated = node.status === "isolated" || node.status === "broken";
            return (
              <g
                key={node.id}
                className={`lnode status-${node.status} kind-${node.kind} ${isSel ? "selected" : ""} ${pulse.has(node.id) ? "pulse" : ""}`}
                transform={`translate(${x},${y})`}
                onMouseDown={(e) => e.stopPropagation()}
                onClick={() => onSelect(node.id)}
              >
                {isolated && (
                  <polygon
                    className="bubble"
                    points={hexagon(NODE_W / 2, NODE_H / 2, NODE_W / 2 + 26, NODE_H / 2 + 22)}
                    filter="url(#glow)"
                  />
                )}
                <rect className="card" width={NODE_W} height={NODE_H} rx="12" />
                <rect className="kind-bar" width="5" height={NODE_H - 16} x="8" y="8" rx="2.5" />
                <text className="glyph" x="24" y="27">
                  {KIND_GLYPH[node.kind]}
                </text>
                <text className="title" x="42" y="27">
                  {node.id.length > 17 ? node.id.slice(0, 16) + "…" : node.id}
                </text>
                <text className="sub" x="24" y="48">
                  {node.kind}
                  {node.params.length ? ` (${node.params.join(", ")})`.slice(0, 26) : ""}
                </text>
                <circle className="status-dot" cx={NODE_W - 16} cy="18" r="5" />
                {node.weights.runs > 0 && (
                  <rect
                    className="eff-bar"
                    x="24"
                    y={NODE_H - 9}
                    height="3"
                    rx="1.5"
                    width={(NODE_W - 48) * Math.max(0.04, node.weights.efficiency)}
                  />
                )}
                {node.params.map((p, i) => (
                  <circle key={p} className="port" cx="0" cy={((i + 1) * NODE_H) / (node.params.length + 1)} r="3.5" />
                ))}
                {node.kind !== "output" && <circle className="port out" cx={NODE_W} cy={NODE_H / 2} r="3.5" />}
                {isolated && node.bubble && (
                  <text className="proxy-label" x={NODE_W / 2} y={NODE_H + 18} textAnchor="middle">
                    proxy · {node.bubble.proxy.replace(/_/g, " ")}
                  </text>
                )}
              </g>
            );
          })}
        </g>
      </svg>
      <div className="view-controls">
        <button onClick={fit} title="Fit lattice">⤢</button>
        <button onClick={() => setView((v) => ({ ...v, k: Math.min(3, v.k * 1.2) }))}>＋</button>
        <button onClick={() => setView((v) => ({ ...v, k: Math.max(0.25, v.k / 1.2) }))}>－</button>
      </div>
      {snapshot.nodes.length === 0 && (
        <div className="empty-hint">
          <h3>Empty lattice</h3>
          <p>Describe a feature in the agent console, or add a node from the inspector.</p>
        </div>
      )}
      <div className="legend">
        {(["healthy", "tainted", "isolated", "broken", "untested"] as const).map((s) => (
          <span key={s} className={`legend-item status-${s}`}>
            <i /> {s}
          </span>
        ))}
      </div>
    </div>
  );
}

function hexagon(cx: number, cy: number, rx: number, ry: number) {
  return Array.from({ length: 6 }, (_, i) => {
    const a = (Math.PI / 3) * i;
    return `${cx + rx * Math.cos(a)},${cy + ry * Math.sin(a)}`;
  }).join(" ");
}
