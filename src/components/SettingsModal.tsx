import { useState } from "react";
import type { Preferences, ProviderKind, PublicProvider, PublicSettings } from "../types";
import { api, errorText, PROVIDER_ORDER } from "../api";

interface Props {
  settings: PublicSettings;
  onChange: (s: PublicSettings) => void;
  onClose: () => void;
  onModelsChanged: (kind: ProviderKind) => void;
}

const KEY_URL: Record<ProviderKind, string> = {
  anthropic: "platform.claude.com",
  openai: "platform.openai.com/api-keys",
  gemini: "aistudio.google.com/apikey",
  ollama_cloud: "ollama.com/settings/keys",
};

export function SettingsModal({ settings, onChange, onClose, onModelsChanged }: Props) {
  const [prefs, setPrefs] = useState<Preferences>(settings.preferences);
  const [prefMsg, setPrefMsg] = useState<string | null>(null);
  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <div className="modal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="panel-head">
          <h3>Settings</h3>
          <button className="ghost" onClick={onClose}>
            ✕
          </button>
        </div>
        <p className="muted small">
          Keys are stored on this machine in <code>{settings.settings_path}</code> (mode 600) and are only used by the Rust core;
          they are never sent back to the UI. Environment variables are used when no key is saved.
        </p>
        <div className="provider-grid">
          {PROVIDER_ORDER.map((kind) => {
            const p = settings.providers.find((x) => x.kind === kind)!;
            return <ProviderCard key={kind} p={p} onChange={onChange} onModelsChanged={onModelsChanged} />;
          })}
        </div>
        <h4>Runtime</h4>
        <div className="row wrap">
          <label>
            Max agent iterations
            <input
              type="number"
              min={1}
              max={50}
              value={prefs.max_iterations}
              onChange={(e) => setPrefs({ ...prefs, max_iterations: Number(e.target.value) })}
            />
          </label>
          <label>
            Operation budget per node (FRE)
            <input
              type="number"
              min={1000}
              step={1000}
              value={prefs.max_operations}
              onChange={(e) => setPrefs({ ...prefs, max_operations: Number(e.target.value) })}
            />
          </label>
          <button
            className="primary"
            onClick={async () => {
              try {
                onChange(await api.savePreferences(prefs));
                setPrefMsg("Saved");
              } catch (e) {
                setPrefMsg(errorText(e));
              }
            }}
          >
            Save runtime settings
          </button>
          {prefMsg && <span className="muted small">{prefMsg}</span>}
        </div>
      </div>
    </div>
  );
}

function ProviderCard({
  p,
  onChange,
  onModelsChanged,
}: {
  p: PublicProvider;
  onChange: (s: PublicSettings) => void;
  onModelsChanged: (k: ProviderKind) => void;
}) {
  const [key, setKey] = useState("");
  const [base, setBase] = useState(p.base_url ?? "");
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  const save = async (clear = false) => {
    setBusy(true);
    try {
      const s = await api.saveProvider(p.kind, clear ? "" : key ? key : null, base || null);
      onChange(s);
      setKey("");
      setMsg({ ok: true, text: clear ? "Key removed" : "Saved" });
      onModelsChanged(p.kind);
    } catch (e) {
      setMsg({ ok: false, text: errorText(e) });
    } finally {
      setBusy(false);
    }
  };

  const test = async () => {
    setBusy(true);
    try {
      const listing = await api.listModels(p.kind);
      const live = listing.models.filter((m) => m.live).length;
      setMsg(
        listing.live_error
          ? { ok: false, text: listing.live_error }
          : { ok: true, text: `Connected · ${live} models available` },
      );
      onModelsChanged(p.kind);
    } catch (e) {
      setMsg({ ok: false, text: errorText(e) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className={`provider-card prov-${p.kind}`}>
      <div className="row between">
        <strong>{p.label}</strong>
        <span className={`tag ${p.has_key ? "ok" : p.requires_key ? "faulted" : ""}`}>
          {p.has_key ? `${p.key_source === "env" ? p.env_var : "saved"} · ${p.key_hint}` : p.requires_key ? "no key" : "key optional"}
        </span>
      </div>
      <label>
        API key <span className="muted small">({KEY_URL[p.kind]})</span>
        <input type="password" autoComplete="off" placeholder={p.has_key ? "•••••• (unchanged)" : p.env_var} value={key} onChange={(e) => setKey(e.target.value)} />
      </label>
      <label>
        Base URL
        <input placeholder={p.default_base_url} value={base} onChange={(e) => setBase(e.target.value)} />
      </label>
      {p.kind === "ollama_cloud" && (
        <div className="muted small">Point at http://localhost:11434 to use a local Ollama server without a key.</div>
      )}
      <div className="row">
        <button className="primary small" disabled={busy} onClick={() => save()}>
          Save
        </button>
        <button className="small" disabled={busy} onClick={test}>
          Test connection
        </button>
        {p.key_source === "saved" && (
          <button className="ghost small danger" disabled={busy} onClick={() => save(true)}>
            Remove key
          </button>
        )}
      </div>
      {msg && <div className={`small ${msg.ok ? "ok-text" : "bad-text"}`}>{msg.text}</div>}
    </div>
  );
}
