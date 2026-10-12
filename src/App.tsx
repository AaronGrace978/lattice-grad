import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, errorText, inTauri, PROVIDER_ORDER, PROVIDER_SHORT, short } from "./api";
import type { AgentEvent, Delta, LatticeSnapshot, ModelInfo, ModelListing, ProviderKind, PublicSettings } from "./types";
import { LatticeView } from "./components/LatticeView";
import { EmbeddingView } from "./components/EmbeddingView";
import { TensorView } from "./components/TensorView";
import { AgentConsole, reduceRun, type AgentRun } from "./components/AgentConsole";
import { Inspector } from "./components/Inspector";
import { Timeline } from "./components/Timeline";
import { SettingsModal } from "./components/SettingsModal";
import { LatticeIOModal } from "./components/LatticeIOModal";

type View = "lattice" | "embedding" | "tensor";
type Toast = { kind: "ok" | "bad" | "info"; text: string };

const TIER_ORDER = ["frontier", "flagship", "reasoning", "balanced", "code", "fast", "legacy"];

export default function App() {
  const [snapshot, setSnapshot] = useState<LatticeSnapshot | null>(null);
  const [settings, setSettings] = useState<PublicSettings | null>(null);
  const [models, setModels] = useState<Partial<Record<ProviderKind, ModelListing>>>({});
  const [provider, setProvider] = useState<ProviderKind>("anthropic");
  const [model, setModel] = useState("");
  const [customModel, setCustomModel] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [view, setView] = useState<View>("lattice");
  const [runs, setRuns] = useState<AgentRun[]>([]);
  const [running, setRunning] = useState(false);
  const [toast, setToast] = useState<Toast | null>(null);
  const [modal, setModal] = useState<"settings" | "io" | null>(null);
  const [pulse, setPulse] = useState<Set<string>>(new Set());
  const toastTimer = useRef<number>(0);

  const notify = useCallback((kind: Toast["kind"], text: string) => {
    setToast({ kind, text });
    window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setToast(null), 5200);
  }, []);

  const refresh = useCallback(async () => {
    try {
      setSnapshot(await api.snapshot());
    } catch (e) {
      notify("bad", errorText(e));
    }
  }, [notify]);

  const loadModels = useCallback(async (kind: ProviderKind) => {
    try {
      const listing = await api.listModels(kind);
      setModels((m) => ({ ...m, [kind]: listing }));
    } catch (e) {
      notify("bad", errorText(e));
    }
  }, [notify]);

  useEffect(() => {
    if (!inTauri()) return;
    refresh();
    api.settings().then((s) => {
      setSettings(s);
      setProvider(s.preferences.provider);
      setModel(s.preferences.model);
      loadModels(s.preferences.provider);
    });
    api.agentRunning().then(setRunning);
    const unlisten = api.onAgentEvent(handleEvent);
    return () => {
      unlisten.then((f) => f());
    };
  }, []);

  const handleEvent = (e: AgentEvent) => {
    if (e.type === "started") {
      setRuns((rs) => [
        ...rs,
        { id: Date.now(), goal: e.goal, generator: e.generator, maxIterations: e.max_iterations, iterations: [] },
      ]);
      return;
    }
    setRuns((rs) => (rs.length ? [...rs.slice(0, -1), reduceRun(rs[rs.length - 1], e)] : rs));
    if (e.type === "lattice_changed") refresh();
    if (e.type === "applied") {
      const fp = e.outcome.footprint;
      setPulse(new Set([...fp.nodes_added, ...fp.nodes_changed]));
      window.setTimeout(() => setPulse(new Set()), 1800);
    }
    if (e.type === "finished") {
      setRunning(false);
      refresh();
      notify(e.success ? "ok" : "info", e.success ? `Converged in ${e.iterations} iteration(s)` : `Agent stopped: ${e.reason}`);
    }
    if (e.type === "error") notify("bad", e.message);
  };

  const persistChoice = async (p: ProviderKind, m: string) => {
    if (!settings) return;
    try {
      setSettings(await api.savePreferences({ ...settings.preferences, provider: p, model: m }));
    } catch (e) {
      notify("bad", errorText(e));
    }
  };

  const chooseProvider = (p: ProviderKind) => {
    setProvider(p);
    setCustomModel(false);
    const def = settings?.providers.find((x) => x.kind === p)?.default_model ?? "";
    setModel(def);
    persistChoice(p, def);
    if (!models[p]) loadModels(p);
  };

  const chooseModel = (m: string) => {
    setModel(m);
    persistChoice(provider, m);
  };

  const onDelta = async (delta: Delta) => {
    try {
      const r = await api.applyDelta(delta);
      await refresh();
      const errs = r.outcome.compile_errors.length;
      notify(
        r.suite.healthy ? "ok" : "info",
        `${r.outcome.noop ? "No-op" : `r${r.outcome.revision}`} · ${r.suite.passed}/${r.suite.probes} probes healthy${errs ? ` · ${errs} compile errors` : ""}`,
      );
      return true;
    } catch (e) {
      notify("bad", errorText(e));
      return false;
    }
  };

  const runSuite = async () => {
    try {
      const s = await api.runSuite();
      await refresh();
      notify(s.healthy ? "ok" : "info", `FRE · ${s.passed}/${s.reports.length} probes healthy · ${s.faults} faults`);
    } catch (e) {
      notify("bad", errorText(e));
    }
  };

  const startAgent = async (goal: string) => {
    try {
      setRunning(true);
      await api.startAgent(goal, provider, model);
    } catch (e) {
      setRunning(false);
      notify("bad", errorText(e));
    }
  };

  const providerInfo = settings?.providers.find((p) => p.kind === provider);
  const listing = models[provider];
  const grouped = useMemo(() => {
    const groups = new Map<string, ModelInfo[]>();
    for (const m of listing?.models ?? []) {
      const tier = m.tier ?? "other";
      groups.set(tier, [...(groups.get(tier) ?? []), m]);
    }
    const rank = (tier: string) => (TIER_ORDER.includes(tier) ? TIER_ORDER.indexOf(tier) : TIER_ORDER.length);
    return [...groups.entries()].sort((a, b) => rank(a[0]) - rank(b[0]));
  }, [listing]);
  const modelKnown = !!listing?.models.some((m) => m.id === model);

  if (!inTauri()) {
    return (
      <div className="boot">
        <img src="/lattice.svg" width={72} />
        <h1>Lattice-OS</h1>
        <p>
          This UI talks to the Rust runtime over Tauri IPC. Start it with <code>npm run tauri dev</code>.
        </p>
      </div>
    );
  }

  if (!snapshot || !settings) {
    return (
      <div className="boot">
        <img src="/lattice.svg" width={72} className="spin" />
        <p>Compiling lattice…</p>
      </div>
    );
  }

  const suite = snapshot.last_suite;
  const bubbles = snapshot.nodes.filter((n) => n.bubble).length;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <img src="/lattice.svg" width={30} height={30} />
          <div>
            <h1>Lattice-OS</h1>
            <span>geometric AI runtime</span>
          </div>
        </div>
        <div className="provider-switch">
          {PROVIDER_ORDER.map((p) => {
            const info = settings.providers.find((x) => x.kind === p);
            return (
              <button key={p} className={`prov prov-${p} ${provider === p ? "active" : ""}`} onClick={() => chooseProvider(p)}>
                <i className={info?.has_key || !info?.requires_key ? "on" : "off"} />
                {PROVIDER_SHORT[p]}
              </button>
            );
          })}
        </div>
        <div className="model-pick">
          {customModel || (!modelKnown && model && listing) ? (
            <input
              value={model}
              onChange={(e) => setModel(e.target.value)}
              onBlur={() => persistChoice(provider, model)}
              placeholder="model id"
            />
          ) : (
            <select value={model} onChange={(e) => (e.target.value === "__custom" ? setCustomModel(true) : chooseModel(e.target.value))}>
              {!listing && <option value={model}>{model || "loading…"}</option>}
              {grouped.map(([tier, ms]) => (
                <optgroup key={tier} label={tier}>
                  {ms.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.live ? "● " : ""}
                      {m.label}
                      {m.label !== m.id ? `  —  ${m.id}` : ""}
                    </option>
                  ))}
                </optgroup>
              ))}
              <option value="__custom">custom model id…</option>
            </select>
          )}
          <button className="ghost" title="Refresh live model list" onClick={() => loadModels(provider)}>
            ⟳
          </button>
          {customModel && (
            <button className="ghost small" onClick={() => setCustomModel(false)}>
              list
            </button>
          )}
        </div>
        <div className="status">
          <span className="stat" title={snapshot.state_hash}>
            r{snapshot.revision} · <code>{short(snapshot.state_hash, 10)}</code>
          </span>
          <span className={`stat ${suite ? (suite.healthy ? "ok" : "warn") : ""}`}>
            {suite ? `${suite.passed}/${suite.probes} probes` : "no suite"}
          </span>
          <span className={`stat ${bubbles ? "warn" : ""}`}>{bubbles} bubbles</span>
          <span className={`stat ${snapshot.interventions.length ? "lbm" : ""}`}>{snapshot.interventions.length} LBM</span>
        </div>
        <div className="actions">
          <button onClick={runSuite} disabled={running}>
            ▶ Run probes
          </button>
          <button onClick={() => setModal("io")}>⇄ Lattice</button>
          <button onClick={() => setModal("settings")}>⚙ Settings</button>
        </div>
      </header>

      <main className="workspace">
        <aside className="left">
          <AgentConsole
            runs={runs}
            running={running}
            canRun={!!model && (!!providerInfo?.has_key || !providerInfo?.requires_key)}
            modelLabel={
              providerInfo?.has_key || !providerInfo?.requires_key
                ? `${PROVIDER_SHORT[provider]} · ${model}`
                : `Add a ${PROVIDER_SHORT[provider]} key in Settings to run`
            }
            onStart={startAgent}
            onCancel={() => api.cancelAgent()}
            onClear={() => setRuns([])}
          />
        </aside>
        <section className="center">
          <div className="view-tabs">
            {(
              [
                ["lattice", "Live assembly lattice"],
                ["embedding", "Embedding space"],
                ["tensor", "Tensor W"],
              ] as [View, string][]
            ).map(([v, label]) => (
              <button key={v} className={view === v ? "active" : ""} onClick={() => setView(v)}>
                {label}
              </button>
            ))}
            <span className="spacer" />
            <span className="muted small">
              CGC: {snapshot.compile.compiled} compiled · {snapshot.compile.reused} cached · {snapshot.compile.failed} broken
            </span>
          </div>
          <div className="view-body">
            {view === "lattice" && (
              <LatticeView snapshot={snapshot} selected={selected} onSelect={setSelected} flowing={running} pulse={pulse} />
            )}
            {view === "embedding" && <EmbeddingView snapshot={snapshot} selected={selected} onSelect={setSelected} />}
            {view === "tensor" && <TensorView snapshot={snapshot} selected={selected} onSelect={setSelected} />}
          </div>
          <Timeline
            snapshot={snapshot}
            locked={running}
            onRollback={async (rev) => {
              try {
                const r = await api.rollback(rev);
                await refresh();
                notify("info", `Restored r${rev} as r${r.outcome.revision}`);
              } catch (e) {
                notify("bad", errorText(e));
              }
            }}
          />
        </section>
        <aside className="right">
          <Inspector
            snapshot={snapshot}
            selected={selected}
            onSelect={setSelected}
            onDelta={onDelta}
            locked={running}
            onInvoke={async (inputs) => {
              try {
                const r = await api.invokeLattice(inputs);
                await refresh();
                return r;
              } catch (e) {
                notify("bad", errorText(e));
                return null;
              }
            }}
          />
        </aside>
      </main>

      {toast && <div className={`toast ${toast.kind}`}>{toast.text}</div>}
      {modal === "settings" && (
        <SettingsModal settings={settings} onChange={setSettings} onClose={() => setModal(null)} onModelsChanged={loadModels} />
      )}
      {modal === "io" && (
        <LatticeIOModal
          locked={running}
          onClose={() => setModal(null)}
          onLoaded={(s) => {
            setSnapshot(s);
            setSelected(null);
          }}
        />
      )}
    </div>
  );
}
