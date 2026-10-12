import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type {
  AgentEvent,
  ChangeResult,
  Delta,
  LatticeSnapshot,
  ModelListing,
  Preferences,
  ProviderKind,
  PublicSettings,
  RunReport,
  SuiteReport,
} from "./types";

export const api = {
  snapshot: () => invoke<LatticeSnapshot>("get_snapshot"),
  settings: () => invoke<PublicSettings>("get_settings"),
  saveProvider: (kind: ProviderKind, apiKey: string | null, baseUrl: string | null) =>
    invoke<PublicSettings>("save_provider", { kind, apiKey, baseUrl }),
  savePreferences: (preferences: Preferences) => invoke<PublicSettings>("save_preferences", { preferences }),
  listModels: (kind: ProviderKind) => invoke<ModelListing>("list_models", { kind }),
  startAgent: (goal: string, provider: ProviderKind, model: string) =>
    invoke<void>("start_agent", { goal, provider, model }),
  cancelAgent: () => invoke<void>("cancel_agent"),
  agentRunning: () => invoke<boolean>("agent_running"),
  applyDelta: (delta: Delta) => invoke<ChangeResult>("apply_delta", { delta }),
  runSuite: () => invoke<SuiteReport>("run_suite"),
  invokeLattice: (inputs: Record<string, unknown>) => invoke<RunReport>("invoke_lattice", { inputs }),
  rollback: (revision: number) => invoke<ChangeResult>("rollback", { revision }),
  loadTemplate: (name: string) => invoke<LatticeSnapshot>("load_template", { name }),
  exportLattice: () => invoke<string>("export_lattice"),
  importLattice: (json: string) => invoke<LatticeSnapshot>("import_lattice", { json }),
  systemPrompt: () => invoke<string>("system_prompt"),
  onAgentEvent: (cb: (e: AgentEvent) => void): Promise<UnlistenFn> =>
    listen<AgentEvent>("agent-event", (e) => cb(e.payload)),
};

export const inTauri = () => typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

export function errorText(e: unknown): string {
  if (typeof e === "string") return e;
  if (e instanceof Error) return e.message;
  return JSON.stringify(e);
}

export const short = (hash: string, n = 10) => hash.slice(0, n);

export function fmtJson(v: unknown, indent = 0): string {
  try {
    return JSON.stringify(v, null, indent) ?? "null";
  } catch {
    return String(v);
  }
}

export function fmtMicros(us: number): string {
  if (us < 1000) return `${us.toFixed(0)}µs`;
  if (us < 1_000_000) return `${(us / 1000).toFixed(1)}ms`;
  return `${(us / 1_000_000).toFixed(2)}s`;
}

export const PROVIDER_ORDER: ProviderKind[] = ["anthropic", "openai", "gemini", "ollama_cloud"];

export const PROVIDER_SHORT: Record<ProviderKind, string> = {
  anthropic: "Anthropic",
  openai: "OpenAI",
  gemini: "Gemini",
  ollama_cloud: "Ollama Cloud",
};
