import { useEffect, useRef } from "react";
import type { LatticeSnapshot } from "../types";

const STATUS_COLOR: Record<string, string> = {
  healthy: "#3ee8a5",
  tainted: "#ffd166",
  isolated: "#ff7a59",
  broken: "#ff4d6d",
  untested: "#8a93b8",
};

interface Props {
  snapshot: LatticeSnapshot;
  selected: string | null;
  onSelect: (id: string | null) => void;
}

/** Nodes projected from R^64 onto their top three principal components. */
export function EmbeddingView({ snapshot, selected, onSelect }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const state = useRef({ yaw: 0.6, pitch: 0.35, auto: true, drag: null as null | { x: number; y: number } });
  const hits = useRef<{ id: string; x: number; y: number; r: number }[]>([]);
  const latest = useRef({ snapshot, selected });
  latest.current = { snapshot, selected };

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    let raf = 0;

    const draw = () => {
      const { snapshot, selected } = latest.current;
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr;
        canvas.height = h * dpr;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const s = state.current;
      if (s.auto) s.yaw += 0.0035;
      const scale = Math.min(w, h) * 0.36;
      const cy = Math.cos(s.yaw), sy = Math.sin(s.yaw), cp = Math.cos(s.pitch), sp = Math.sin(s.pitch);
      const project = ([x, y, z]: [number, number, number]) => {
        const x1 = x * cy - z * sy;
        const z1 = x * sy + z * cy;
        const y1 = y * cp - z1 * sp;
        const z2 = y * sp + z1 * cp;
        const persp = 2.6 / (2.6 + z2);
        return { x: w / 2 + x1 * scale * persp, y: h / 2 + y1 * scale * persp, z: z2, p: persp };
      };

      ctx.lineWidth = 1;
      for (let i = -1; i <= 1; i += 0.5) {
        for (const axis of [0, 2] as const) {
          const a: [number, number, number] = axis === 0 ? [i, 1.05, -1] : [-1, 1.05, i];
          const b: [number, number, number] = axis === 0 ? [i, 1.05, 1] : [1, 1.05, i];
          const pa = project(a), pb = project(b);
          ctx.strokeStyle = "rgba(120,140,255,0.08)";
          ctx.beginPath();
          ctx.moveTo(pa.x, pa.y);
          ctx.lineTo(pb.x, pb.y);
          ctx.stroke();
        }
      }
      const axes: [[number, number, number], string][] = [
        [[1.15, 0, 0], "PC1"],
        [[0, -1.15, 0], "PC2"],
        [[0, 0, 1.15], "PC3"],
      ];
      const o = project([0, 0, 0]);
      for (const [v, label] of axes) {
        const p = project(v);
        ctx.strokeStyle = "rgba(160,170,255,0.25)";
        ctx.beginPath();
        ctx.moveTo(o.x, o.y);
        ctx.lineTo(p.x, p.y);
        ctx.stroke();
        ctx.fillStyle = "rgba(170,180,255,0.5)";
        ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillText(label, p.x + 4, p.y);
      }

      const pts = new Map(snapshot.nodes.map((n) => [n.id, { n, ...project(n.coords) }]));
      for (const e of snapshot.edges) {
        const a = pts.get(e.from), b = pts.get(e.to);
        if (!a || !b) continue;
        const grad = ctx.createLinearGradient(a.x, a.y, b.x, b.y);
        grad.addColorStop(0, "rgba(111,231,255,0.55)");
        grad.addColorStop(1, "rgba(156,139,255,0.55)");
        ctx.strokeStyle = grad;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }

      const sorted = [...pts.values()].sort((a, b) => b.z - a.z);
      hits.current = [];
      for (const p of sorted) {
        const r = (p.n.kind === "function" ? 8 : 6.5) * p.p * (selected === p.n.id ? 1.5 : 1);
        const color = STATUS_COLOR[p.n.status] ?? "#fff";
        const glow = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, r * 3.2);
        glow.addColorStop(0, color + "66");
        glow.addColorStop(1, color + "00");
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r * 3.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fill();
        if (p.n.status === "isolated" || p.n.status === "broken") {
          ctx.strokeStyle = color;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.arc(p.x, p.y, r + 7, 0, Math.PI * 2);
          ctx.stroke();
          ctx.setLineDash([]);
        }
        ctx.fillStyle = selected === p.n.id ? "#fff" : "rgba(220,226,255,0.82)";
        ctx.font = `${selected === p.n.id ? 600 : 500} 11px Inter, system-ui, sans-serif`;
        ctx.fillText(p.n.id, p.x + r + 6, p.y + 4);
        hits.current.push({ id: p.n.id, x: p.x, y: p.y, r: r + 6 });
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <div className="embedding-view">
      <canvas
        ref={canvasRef}
        onMouseDown={(e) => {
          state.current.drag = { x: e.clientX, y: e.clientY };
          state.current.auto = false;
        }}
        onMouseMove={(e) => {
          const d = state.current.drag;
          if (!d) return;
          state.current.yaw += (e.clientX - d.x) * 0.008;
          state.current.pitch = Math.max(-1.3, Math.min(1.3, state.current.pitch + (e.clientY - d.y) * 0.008));
          state.current.drag = { x: e.clientX, y: e.clientY };
        }}
        onMouseUp={(e) => {
          state.current.drag = null;
          const rect = canvasRef.current!.getBoundingClientRect();
          const x = e.clientX - rect.left, y = e.clientY - rect.top;
          const hit = hits.current.find((h) => (h.x - x) ** 2 + (h.y - y) ** 2 < h.r ** 2);
          if (hit) onSelect(hit.id);
        }}
        onDoubleClick={() => (state.current.auto = !state.current.auto)}
      />
      <div className="overlay-note">
        Feature-hashed code embeddings (ℝ⁶⁴) projected onto principal components · drag to rotate · double-click to toggle spin
      </div>
    </div>
  );
}
