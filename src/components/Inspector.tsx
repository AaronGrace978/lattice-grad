import { useEffect, useMemo, useState } from "react";
import type { Delta, LatticeSnapshot, NodeKind, NodeView, ProbeView, RunReport } from "../types";
import { fmtJson, fmtMicros } from "../api";

type Tab = "node" | "probes" | "invoke";

interface Props {
  snapshot: LatticeSnapshot;
  selected: string | null;
  onSelect: (id: string | null) => void;
  onDelta: (delta: Delta) => Promise<boolean>;
  onInvoke: (inputs: Record<string, unknown>) => Promise<RunReport | null>;
  locked: boolean;
}

export function Inspector(props: Props) {
  const [tab, setTab] = useState<Tab>("node");
  useEffect(() => {
    if (props.selected) setTab("node");
  }, [props.selected]);
  return (
    <div className="inspector">
      <div className="tabs">
        {(["node", "probes", "invoke"] as Tab[]).map((t) => (
          <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
            {t === "node" ? "Node" : t === "probes" ? `Probes (${props.snapshot.probes.length})` : "Invoke"}
          </button>
        ))}
      </div>
      <div className="tab-body">
        {tab === "node" && <NodeTab {...props} />}
        {tab === "probes" && <ProbesTab {...props} />}
        {tab === "invoke" && <InvokeTab {...props} />}
      </div>
    </div>
  );
}

function NodeTab({ snapshot, selected, onSelect, onDelta, locked }: Props) {
  const node = snapshot.nodes.find((n) => n.id === selected) ?? null;
  if (!node) return <NewNode snapshot={snapshot} onDelta={onDelta} onSelect={onSelect} locked={locked} />;
  return <NodeEditor key={node.id} node={node} snapshot={snapshot} onDelta={onDelta} onSelect={onSelect} locked={locked} />;
}

function NewNode({
  snapshot,
  onDelta,
  onSelect,
  locked,
}: Pick<Props, "snapshot" | "onDelta" | "onSelect" | "locked">) {
  const [id, setId] = useState("");
  const [kind, setKind] = useState<NodeKind>("function");
  const valid = /^[A-Za-z_][A-Za-z0-9_]*$/.test(id) && !snapshot.nodes.some((n) => n.id === id);
  const create = async () => {
    const ok = await onDelta({
      summary: `Add ${kind} node ${id}`,
      ops: [{ op: "upsert_node", id, kind, params: kind === "input" ? [] : ["value"], body: kind === "function" ? "value" : "" }],
    });
    if (ok) {
      onSelect(id);
      setId("");
    }
  };
  return (
    <div className="stack">
      <p className="muted">Select a node in the lattice to inspect its code, telemetry and isolation state.</p>
      <h4>New node</h4>
      <div className="row">
        <input placeholder="node_id" value={id} onChange={(e) => setId(e.target.value)} />
        <select value={kind} onChange={(e) => setKind(e.target.value as NodeKind)}>
          <option value="input">input</option>
          <option value="function">function</option>
          <option value="type">type</option>
          <option value="output">output</option>
        </select>
        <button className="primary" disabled={!valid || locked} onClick={create}>
          Add
        </button>
      </div>
      <SuiteFaults snapshot={snapshot} onSelect={onSelect} />
    </div>
  );
}

function SuiteFaults({ snapshot, onSelect }: Pick<Props, "snapshot" | "onSelect">) {
  const faults = snapshot.last_suite?.faults ?? [];
  if (!faults.length) return null;
  return (
    <>
      <h4>Fault vectors</h4>
      <div className="fault-list">
        {faults.map((f, i) => (
          <button key={i} className={`fault ${f.kind}`} onClick={() => onSelect(f.node)}>
            <strong>{f.node}</strong> <span className="tag">{f.kind}</span> <span className="muted">@{f.probe}</span>
            <div className="small">{f.message}</div>
            {f.proxy && <div className="small muted">routed via {f.proxy.replace(/_/g, " ")} proxy</div>}
          </button>
        ))}
      </div>
    </>
  );
}

function NodeEditor({
  node,
  snapshot,
  onDelta,
  onSelect,
  locked,
}: { node: NodeView } & Pick<Props, "snapshot" | "onDelta" | "onSelect" | "locked">) {
  const [params, setParams] = useState(node.params.join(", "));
  const [body, setBody] = useState(node.body);
  const [fallback, setFallback] = useState(node.fallback ?? "");
  const [doc, setDoc] = useState(node.doc);
  const [kind, setKind] = useState<NodeKind>(node.kind);
  const [connFrom, setConnFrom] = useState("");
  const [connPort, setConnPort] = useState(node.params[0] ?? "");

  useEffect(() => {
    setParams(node.params.join(", "));
    setBody(node.body);
    setFallback(node.fallback ?? "");
    setDoc(node.doc);
    setKind(node.kind);
    setConnPort(node.params[0] ?? "");
  }, [snapshot.revision, node.id]);

  const parsedParams = params
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  const dirty =
    parsedParams.join(",") !== node.params.join(",") ||
    body !== node.body ||
    fallback !== (node.fallback ?? "") ||
    doc !== node.doc ||
    kind !== node.kind;

  const incoming = snapshot.edges.filter((e) => e.to === node.id);
  const outgoing = snapshot.edges.filter((e) => e.from === node.id);
  const sources = snapshot.nodes.filter((n) => n.id !== node.id && n.kind !== "output");

  const apply = () =>
    onDelta({
      summary: `Edit ${node.id}`,
      ops: [{ op: "upsert_node", id: node.id, kind, params: parsedParams, body, fallback, doc }],
    });

  return (
    <div className="stack">
      <div className="node-head">
        <div>
          <h3>{node.id}</h3>
          <span className={`status-pill status-${node.status}`}>{node.status}</span>
          <span className="tag">{node.kind}</span>
        </div>
        <button className="ghost small" onClick={() => onSelect(null)}>
          ✕
        </button>
      </div>

      {node.compile_error && (
        <div className="callout bad">
          <strong>CGC compile error</strong>
          <pre>{node.compile_error}</pre>
        </div>
      )}
      {node.bubble && (
        <div className="callout warn">
          <strong>Isolation bubble</strong> since r{node.bubble.since_revision} · {node.bubble.faults} faults
          <div className="small">{node.bubble.reason}</div>
          <div className="small muted">
            Downstream is routed through the <b>{node.bubble.proxy.replace(/_/g, " ")}</b> proxy until the node is refactored.
          </div>
        </div>
      )}

      <label>
        kind
        <select value={kind} onChange={(e) => setKind(e.target.value as NodeKind)}>
          <option value="input">input</option>
          <option value="function">function</option>
          <option value="type">type</option>
          <option value="output">output</option>
        </select>
      </label>
      {kind !== "input" && (
        <label>
          params (ports)
          <input value={params} onChange={(e) => setParams(e.target.value)} placeholder="a, b" />
        </label>
      )}
      {kind !== "input" && (
        <label>
          body <span className="muted small">(Rhai · last expression is the output)</span>
          <textarea className="code" value={body} spellCheck={false} rows={Math.min(16, Math.max(4, body.split("\n").length + 1))} onChange={(e) => setBody(e.target.value)} />
        </label>
      )}
      {kind !== "input" && (
        <label>
          fallback <span className="muted small">(proxy expression when isolated)</span>
          <input className="code" value={fallback} onChange={(e) => setFallback(e.target.value)} placeholder="e.g. 0 or []" />
        </label>
      )}
      <label>
        doc
        <input value={doc} onChange={(e) => setDoc(e.target.value)} />
      </label>
      <div className="row">
        <button className="primary" disabled={!dirty || locked} onClick={apply}>
          Project ΔT
        </button>
        <button
          className="danger ghost"
          disabled={locked}
          onClick={async () => {
            if (await onDelta({ summary: `Remove ${node.id}`, ops: [{ op: "remove_node", id: node.id }] })) onSelect(null);
          }}
        >
          Remove node
        </button>
      </div>

      <h4>Data flow</h4>
      <div className="edge-list">
        {incoming.map((e) => (
          <div key={`${e.from}-${e.port}`} className="edge-row">
            <span>
              <button className="link" onClick={() => onSelect(e.from)}>
                {e.from}
              </button>{" "}
              → <code>{e.port}</code>
            </span>
            <button
              className="ghost small"
              disabled={locked}
              onClick={() => onDelta({ summary: `Disconnect ${e.from} → ${node.id}.${e.port}`, ops: [{ op: "disconnect", from: e.from, to: node.id, port: e.port }] })}
            >
              ✕
            </button>
          </div>
        ))}
        {outgoing.map((e) => (
          <div key={`${e.to}-${e.port}`} className="edge-row">
            <span>
              → <button className="link" onClick={() => onSelect(e.to)}>{e.to}</button>.<code>{e.port}</code>
            </span>
          </div>
        ))}
        {!incoming.length && !outgoing.length && <div className="muted small">No edges.</div>}
      </div>
      {node.kind !== "input" && node.params.length > 0 && (
        <div className="row">
          <select value={connFrom} onChange={(e) => setConnFrom(e.target.value)}>
            <option value="">upstream…</option>
            {sources.map((s) => (
              <option key={s.id} value={s.id}>
                {s.id}
              </option>
            ))}
          </select>
          <select value={connPort} onChange={(e) => setConnPort(e.target.value)}>
            {node.params.map((p) => (
              <option key={p}>{p}</option>
            ))}
          </select>
          <button
            disabled={!connFrom || !connPort || locked}
            onClick={() => onDelta({ summary: `Connect ${connFrom} → ${node.id}.${connPort}`, ops: [{ op: "connect", from: connFrom, to: node.id, port: connPort }] })}
          >
            Connect
          </button>
        </div>
      )}

      {node.trace && (
        <>
          <h4>Last execution</h4>
          <div className="trace-box">
            <div className="row small">
              <span className={`tag ${node.trace.status}`}>{node.trace.status}</span>
              <span>{fmtMicros(node.trace.latency_us)}</span>
              <span>{node.trace.ops.toLocaleString()} ops</span>
              {node.trace.tainted && <span className="tag tainted">tainted input</span>}
            </div>
            {node.trace.error && <pre className="err">{node.trace.error}</pre>}
            <pre>{fmtJson(node.trace.output, 2)}</pre>
            {node.trace.logs?.map((l, i) => (
              <div key={i} className="log small">
                › {l}
              </div>
            ))}
          </div>
        </>
      )}

      <h4>Weights W</h4>
      <div className="kv">
        <span>efficiency</span>
        <b>{node.weights.runs ? node.weights.efficiency.toFixed(3) : "—"}</b>
        <span>reliability</span>
        <b>{node.weights.runs ? node.weights.reliability.toFixed(3) : "—"}</b>
        <span>latency (EMA)</span>
        <b>{node.weights.runs ? fmtMicros(node.weights.latency_us) : "—"}</b>
        <span>ops</span>
        <b>{node.weights.ops.toLocaleString()}</b>
        <span>runs / failures</span>
        <b>
          {node.weights.runs} / {node.weights.failures}
        </b>
        <span>embedding (PC1-3)</span>
        <b>{node.coords.map((c) => c.toFixed(2)).join(", ")}</b>
      </div>
    </div>
  );
}

function probeTemplate(snapshot: LatticeSnapshot) {
  const inputs: Record<string, unknown> = {};
  const expect: Record<string, unknown> = {};
  for (const n of snapshot.nodes) {
    if (n.kind === "input") inputs[n.id] = "";
    if (n.kind === "output") expect[n.id] = null;
  }
  return { inputs, expect };
}

function ProbesTab({ snapshot, onDelta, locked, onSelect }: Props) {
  const [editing, setEditing] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [json, setJson] = useState("");
  const [err, setErr] = useState<string | null>(null);

  const open = (p: ProbeView | null) => {
    setErr(null);
    if (p) {
      setEditing(p.name);
      setName(p.name);
      setJson(fmtJson({ inputs: p.inputs, expect: p.expect ?? {} }, 2));
    } else {
      setEditing("");
      setName(`probe_${snapshot.probes.length + 1}`);
      setJson(fmtJson(probeTemplate(snapshot), 2));
    }
  };

  const save = async () => {
    try {
      const parsed = JSON.parse(json) as { inputs?: Record<string, unknown>; expect?: Record<string, unknown> };
      const ops: Delta["ops"] = [];
      if (editing && editing !== name) ops.push({ op: "remove_probe", name: editing });
      const expect = parsed.expect && Object.keys(parsed.expect).length ? parsed.expect : undefined;
      ops.push({ op: "upsert_probe", name, inputs: parsed.inputs ?? {}, expect });
      if (await onDelta({ summary: `Probe ${name}`, ops })) setEditing(null);
    } catch (e) {
      setErr(String(e));
    }
  };

  return (
    <div className="stack">
      <div className="row between">
        <span className="muted small">Probes feed input nodes and assert output nodes. The FRE runs all of them after every ΔT.</span>
        <button className="small" disabled={locked} onClick={() => open(null)}>
          + probe
        </button>
      </div>
      {snapshot.probes.map((p) => (
        <div key={p.name} className={`probe ${p.healthy === true ? "ok" : p.healthy === false ? "bad" : ""}`}>
          <div className="row between">
            <strong>{p.name}</strong>
            <span className="row">
              <span className={`tag ${p.healthy === true ? "ok" : p.healthy === false ? "faulted" : ""}`}>
                {p.healthy === null ? "not run" : p.healthy ? "healthy" : `${p.faults.length} faults`}
              </span>
              <button className="ghost small" disabled={locked} onClick={() => open(p)}>
                edit
              </button>
              <button className="ghost small" disabled={locked} onClick={() => onDelta({ summary: `Remove probe ${p.name}`, ops: [{ op: "remove_probe", name: p.name }] })}>
                ✕
              </button>
            </span>
          </div>
          <div className="probe-io">
            <span className="muted">in</span>
            <code>{fmtJson(p.inputs)}</code>
            {p.expect && (
              <>
                <span className="muted">expect</span>
                <code>{fmtJson(p.expect)}</code>
              </>
            )}
            {p.outputs && (
              <>
                <span className="muted">actual</span>
                <code className={p.passed === false ? "bad" : ""}>{fmtJson(p.outputs)}</code>
              </>
            )}
          </div>
          {p.faults.map((f, i) => (
            <button key={i} className={`fault ${f.kind}`} onClick={() => onSelect(f.node)}>
              <strong>{f.node}</strong> <span className="tag">{f.kind}</span>
              <div className="small">{f.message}</div>
            </button>
          ))}
        </div>
      ))}
      {editing !== null && (
        <div className="probe-editor">
          <label>
            name
            <input value={name} onChange={(e) => setName(e.target.value)} />
          </label>
          <label>
            inputs / expect
            <textarea className="code" rows={10} value={json} spellCheck={false} onChange={(e) => setJson(e.target.value)} />
          </label>
          {err && <div className="callout bad small">{err}</div>}
          <div className="row">
            <button className="primary" disabled={locked || !name} onClick={save}>
              Save probe
            </button>
            <button className="ghost" onClick={() => setEditing(null)}>
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function InvokeTab({ snapshot, onInvoke, onSelect }: Props) {
  const inputs = useMemo(() => snapshot.nodes.filter((n) => n.kind === "input"), [snapshot]);
  const [json, setJson] = useState(() => fmtJson(Object.fromEntries(inputs.map((n) => [n.id, ""])), 2));
  const [report, setReport] = useState<RunReport | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const run = async () => {
    try {
      setErr(null);
      setReport(await onInvoke(JSON.parse(json)));
    } catch (e) {
      setErr(String(e));
    }
  };

  return (
    <div className="stack">
      <p className="muted small">
        Live invocation runs in <b>routed mode</b>: isolated nodes are not executed; their proxies carry the data so the lattice
        stays up while the generation layer refactors them.
      </p>
      <textarea className="code" rows={6} value={json} spellCheck={false} onChange={(e) => setJson(e.target.value)} />
      {err && <div className="callout bad small">{err}</div>}
      <button className="primary" onClick={run}>
        Invoke lattice
      </button>
      {report && (
        <>
          <h4>Outputs · {fmtMicros(report.total_us)}</h4>
          <pre className="trace-box">{fmtJson(report.outputs, 2)}</pre>
          <h4>Route</h4>
          <table className="weights-table">
            <thead>
              <tr>
                <th>node</th>
                <th>status</th>
                <th>ops</th>
                <th>time</th>
                <th>output</th>
              </tr>
            </thead>
            <tbody>
              {report.traces.map((t) => (
                <tr key={t.node} onClick={() => onSelect(t.node)}>
                  <td>{t.node}</td>
                  <td>
                    <span className={`tag ${t.status}`}>{t.proxy ? `${t.status} · ${t.proxy.replace(/_/g, " ")}` : t.status}</span>
                  </td>
                  <td>{t.ops}</td>
                  <td>{fmtMicros(t.latency_us)}</td>
                  <td className="mono ellipsis">{fmtJson(t.output)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
