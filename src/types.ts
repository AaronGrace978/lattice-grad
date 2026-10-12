export type ProviderKind = "anthropic" | "openai" | "gemini" | "ollama_cloud";
export type NodeKind = "input" | "function" | "type" | "output";
export type NodeStatus = "healthy" | "isolated" | "broken" | "tainted" | "untested";
export type FaultKind = "compile" | "runtime" | "budget" | "expectation";
export type ProxyKind = "last_known_good" | "declared_fallback" | "pass_through" | "null";

export interface Weights {
  latency_us: number;
  ops: number;
  efficiency: number;
  velocity: number;
  reliability: number;
  runs: number;
  failures: number;
}

export interface NodeTrace {
  node: string;
  status: "ok" | "faulted" | "bubbled";
  output: unknown;
  latency_us: number;
  ops: number;
  output_bytes: number;
  error?: string;
  proxy?: ProxyKind;
  tainted: boolean;
  logs?: string[];
}

export interface Bubble {
  node: string;
  hash: string;
  kind: FaultKind;
  reason: string;
  proxy: ProxyKind;
  faults: number;
  since_revision: number;
}

export interface NodeView {
  id: string;
  kind: NodeKind;
  params: string[];
  body: string;
  fallback?: string;
  doc: string;
  weights: Weights;
  layer: number;
  coords: [number, number, number];
  status: NodeStatus;
  bubble: Bubble | null;
  compile_error: string | null;
  trace: NodeTrace | null;
}

export interface Edge {
  from: string;
  to: string;
  port: string;
  weight: number;
}

export interface Fault {
  probe: string;
  node: string;
  kind: FaultKind;
  message: string;
  inputs: Record<string, unknown>;
  proxy?: ProxyKind;
  expected?: unknown;
  actual?: unknown;
}

export interface ProbeView {
  name: string;
  inputs: Record<string, unknown>;
  expect?: Record<string, unknown>;
  passed: boolean | null;
  healthy: boolean | null;
  outputs: Record<string, unknown> | null;
  faults: Fault[];
}

export interface DeltaFootprint {
  nodes_added: string[];
  nodes_changed: string[];
  nodes_removed: string[];
  edges_changed: number;
  probes_changed: string[];
}

export interface Revision {
  revision: number;
  state_hash: string;
  summary: string;
  source: "seed" | "agent" | "manual" | "rollback" | "import";
  footprint: DeltaFootprint;
  at_ms: number;
}

export interface Strategy {
  id: string;
  title: string;
  directive: string;
  keywords: string;
}

export type Oscillation =
  | { type: "state_revisit"; cycle_len: number }
  | { type: "forbidden_state" }
  | { type: "anti_parallel"; cosine: number };

export interface Intervention {
  revision: number;
  oscillation: Oscillation;
  rejected_state: string;
  forbidden_states: string[];
  oscillating_nodes: string[];
  strategy: Strategy;
  orthogonality: number;
  orthogonal_preview: number[];
}

export interface CompileStats {
  compiled: number;
  reused: number;
  failed: number;
  micros: number;
}

export interface SuiteSummary {
  revision: number;
  state_hash: string;
  probes: number;
  passed: number;
  failed: number;
  faults: Fault[];
  healthy: boolean;
  total_us: number;
}

export interface FreConfig {
  max_operations: number;
  max_call_levels: number;
  max_string_size: number;
  max_array_size: number;
  max_map_size: number;
}

export interface LatticeSnapshot {
  revision: number;
  state_hash: string;
  nodes: NodeView[];
  edges: Edge[];
  probes: ProbeView[];
  history: Revision[];
  interventions: Intervention[];
  forbidden_states: string[];
  compile: CompileStats;
  last_suite: SuiteSummary | null;
  fre_config: FreConfig;
  tensor: { ids: string[]; weights: number[][] };
}

export interface ApplyOutcome {
  revision: number;
  state_hash: string;
  noop: boolean;
  footprint: DeltaFootprint;
  compile: CompileStats;
  compile_errors: [string, string][];
  delta_norm: number;
}

export interface ChangeResult {
  outcome: ApplyOutcome;
  suite: SuiteSummary;
}

export interface RunReport {
  probe: string;
  state_hash: string;
  outputs: Record<string, unknown>;
  traces: NodeTrace[];
  faults: Fault[];
  passed: boolean | null;
  total_us: number;
}

export interface SuiteReport {
  revision: number;
  state_hash: string;
  reports: RunReport[];
  passed: number;
  failed: number;
  faults: number;
  healthy: boolean;
  total_us: number;
}

export type AgentEvent =
  | { type: "started"; goal: string; generator: string; max_iterations: number }
  | { type: "iteration"; iteration: number }
  | { type: "token"; iteration: number; text: string }
  | { type: "delta_parsed"; iteration: number; summary: string; ops: number }
  | { type: "delta_rejected"; iteration: number; reason: string }
  | { type: "intervention"; iteration: number; intervention: Intervention }
  | { type: "applied"; iteration: number; outcome: ApplyOutcome }
  | { type: "suite"; iteration: number; summary: SuiteSummary }
  | { type: "lattice_changed"; revision: number }
  | { type: "finished"; success: boolean; iterations: number; reason: string }
  | { type: "error"; message: string };

export interface ModelInfo {
  id: string;
  label: string;
  provider: ProviderKind;
  tier?: string;
  live: boolean;
}

export interface ModelListing {
  provider: ProviderKind;
  models: ModelInfo[];
  live_error: string | null;
}

export interface PublicProvider {
  kind: ProviderKind;
  label: string;
  has_key: boolean;
  key_source: "saved" | "env" | "none";
  key_hint: string | null;
  env_var: string;
  base_url: string | null;
  default_base_url: string;
  requires_key: boolean;
  default_model: string;
}

export interface Preferences {
  provider: ProviderKind;
  model: string;
  max_iterations: number;
  max_operations: number;
}

export interface PublicSettings {
  providers: PublicProvider[];
  preferences: Preferences;
  settings_path: string;
}

export type DeltaOp =
  | {
      op: "upsert_node";
      id: string;
      kind?: NodeKind;
      params?: string[];
      body?: string;
      fallback?: string;
      doc?: string;
    }
  | { op: "remove_node"; id: string }
  | { op: "connect"; from: string; to: string; port: string }
  | { op: "disconnect"; from: string; to: string; port?: string }
  | { op: "upsert_probe"; name: string; inputs: Record<string, unknown>; expect?: Record<string, unknown> }
  | { op: "remove_probe"; name: string };

export interface Delta {
  summary: string;
  ops: DeltaOp[];
}
