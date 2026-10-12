mod settings;

use lattice_core::fre::{FreConfig, RunReport};
use lattice_core::providers::ModelListing;
use lattice_core::runtime::{ApplyOutcome, SuiteReport, SuiteSummary};
use lattice_core::{
    run_agent, seed, AgentConfig, AgentEvent, Delta, LatticeGraph, LatticeSnapshot, ProviderClient, ProviderGenerator,
    ProviderKind, RevisionSource, Runtime,
};
use serde::Serialize;
use serde_json::Value;
use settings::{Preferences, PublicSettings, Settings};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use tauri::{AppHandle, Emitter, Manager, State};

pub const AGENT_EVENT: &str = "agent-event";

struct AppState {
    runtime: Arc<Mutex<Runtime>>,
    settings: Mutex<Settings>,
    settings_path: PathBuf,
    client: ProviderClient,
    running: Arc<AtomicBool>,
    cancel: Arc<AtomicBool>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

type CmdResult<T> = Result<T, String>;

#[derive(Serialize)]
struct ChangeResult {
    outcome: ApplyOutcome,
    suite: SuiteSummary,
}

fn fre_config(prefs: &Preferences) -> FreConfig {
    FreConfig { max_operations: prefs.max_operations, ..FreConfig::default() }
}

#[tauri::command]
fn get_snapshot(state: State<'_, AppState>) -> LatticeSnapshot {
    lock(&state.runtime).snapshot()
}

#[tauri::command]
fn get_settings(state: State<'_, AppState>) -> PublicSettings {
    lock(&state.settings).public(&state.settings_path)
}

/// `api_key`: `None` keeps the stored key, `Some("")` clears it.
#[tauri::command]
fn save_provider(
    state: State<'_, AppState>,
    kind: ProviderKind,
    api_key: Option<String>,
    base_url: Option<String>,
) -> CmdResult<PublicSettings> {
    let mut s = lock(&state.settings);
    let entry = s.providers.entry(kind).or_default();
    if let Some(k) = api_key {
        entry.api_key = Some(k.trim().to_string()).filter(|k| !k.is_empty());
    }
    entry.base_url = base_url.map(|b| b.trim().to_string()).filter(|b| !b.is_empty());
    s.save(&state.settings_path)?;
    Ok(s.public(&state.settings_path))
}

#[tauri::command]
fn save_preferences(state: State<'_, AppState>, preferences: Preferences) -> CmdResult<PublicSettings> {
    let mut s = lock(&state.settings);
    s.preferences = Preferences {
        max_iterations: preferences.max_iterations.clamp(1, 50),
        max_operations: preferences.max_operations.clamp(1_000, 50_000_000),
        ..preferences
    };
    s.save(&state.settings_path)?;
    lock(&state.runtime).set_fre_config(fre_config(&s.preferences));
    Ok(s.public(&state.settings_path))
}

#[tauri::command]
async fn list_models(state: State<'_, AppState>, kind: ProviderKind) -> CmdResult<ModelListing> {
    let cfg = lock(&state.settings).effective(kind);
    Ok(state.client.models(kind, &cfg).await)
}

#[tauri::command]
fn start_agent(
    app: AppHandle,
    state: State<'_, AppState>,
    goal: String,
    provider: ProviderKind,
    model: String,
) -> CmdResult<()> {
    if goal.trim().is_empty() {
        return Err("describe what the agent should build or repair".into());
    }
    if model.trim().is_empty() {
        return Err("select a model".into());
    }
    let cfg = lock(&state.settings).effective(provider);
    if cfg.key().is_none() && cfg.requires_key(provider) {
        return Err(format!(
            "{} has no API key. Add one in Settings or set {}.",
            provider.label(),
            provider.env_key()
        ));
    }
    if state.running.swap(true, Ordering::SeqCst) {
        return Err("an agent run is already in progress".into());
    }
    state.cancel.store(false, Ordering::SeqCst);

    let max_iterations = lock(&state.settings).preferences.max_iterations;
    let generator = ProviderGenerator { client: state.client.clone(), kind: provider, config: cfg, model };
    let runtime = state.runtime.clone();
    let running = state.running.clone();
    let cancel = state.cancel.clone();

    tauri::async_runtime::spawn(async move {
        let sink = |e: AgentEvent| {
            let _ = app.emit(AGENT_EVENT, &e);
        };
        run_agent(runtime, &generator, &goal, &AgentConfig { max_iterations }, &sink, &cancel).await;
        running.store(false, Ordering::SeqCst);
    });
    Ok(())
}

#[tauri::command]
fn cancel_agent(state: State<'_, AppState>) {
    state.cancel.store(true, Ordering::SeqCst);
}

#[tauri::command]
fn agent_running(state: State<'_, AppState>) -> bool {
    state.running.load(Ordering::SeqCst)
}

fn guard_idle(state: &AppState) -> CmdResult<()> {
    if state.running.load(Ordering::SeqCst) {
        Err("the agent is mutating the lattice; stop it first".into())
    } else {
        Ok(())
    }
}

#[tauri::command]
fn apply_delta(state: State<'_, AppState>, delta: Delta) -> CmdResult<ChangeResult> {
    guard_idle(&state)?;
    let mut rt = lock(&state.runtime);
    let outcome = rt.propose(&delta, RevisionSource::Manual, false).map_err(|e| e.to_string())?;
    let suite = rt.run_suite().summary();
    Ok(ChangeResult { outcome, suite })
}

#[tauri::command]
fn run_suite(state: State<'_, AppState>) -> SuiteReport {
    lock(&state.runtime).run_suite()
}

#[tauri::command]
fn invoke_lattice(state: State<'_, AppState>, inputs: BTreeMap<String, Value>) -> RunReport {
    lock(&state.runtime).invoke(&inputs)
}

#[tauri::command]
fn rollback(state: State<'_, AppState>, revision: u64) -> CmdResult<ChangeResult> {
    guard_idle(&state)?;
    let mut rt = lock(&state.runtime);
    let outcome = rt.rollback(revision).map_err(|e| e.to_string())?;
    let suite = rt.run_suite().summary();
    Ok(ChangeResult { outcome, suite })
}

#[tauri::command]
fn load_template(state: State<'_, AppState>, name: String) -> CmdResult<LatticeSnapshot> {
    guard_idle(&state)?;
    let graph = seed::template(&name).ok_or_else(|| format!("unknown template `{name}`"))?;
    let mut rt = lock(&state.runtime);
    rt.replace(graph, &format!("Template: {name}")).map_err(|e| e.to_string())?;
    rt.run_suite();
    Ok(rt.snapshot())
}

#[tauri::command]
fn export_lattice(state: State<'_, AppState>) -> CmdResult<String> {
    serde_json::to_string_pretty(lock(&state.runtime).graph()).map_err(|e| e.to_string())
}

#[tauri::command]
fn import_lattice(state: State<'_, AppState>, json: String) -> CmdResult<LatticeSnapshot> {
    guard_idle(&state)?;
    let graph: LatticeGraph = serde_json::from_str(&json).map_err(|e| format!("invalid lattice JSON: {e}"))?;
    let mut rt = lock(&state.runtime);
    rt.replace(graph, "Imported lattice").map_err(|e| e.to_string())?;
    rt.run_suite();
    Ok(rt.snapshot())
}

#[tauri::command]
fn system_prompt() -> &'static str {
    lattice_core::agent::SYSTEM_PROMPT
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            let settings_path = app
                .path()
                .app_config_dir()
                .unwrap_or_else(|_| std::env::temp_dir().join("lattice-os"))
                .join("settings.json");
            let settings = Settings::load(&settings_path);
            let mut runtime = Runtime::new(seed::demo_lattice()).expect("seed lattice is valid");
            runtime.set_fre_config(fre_config(&settings.preferences));
            runtime.run_suite();
            app.manage(AppState {
                runtime: Arc::new(Mutex::new(runtime)),
                settings: Mutex::new(settings),
                settings_path,
                client: ProviderClient::new(),
                running: Arc::new(AtomicBool::new(false)),
                cancel: Arc::new(AtomicBool::new(false)),
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_snapshot,
            get_settings,
            save_provider,
            save_preferences,
            list_models,
            start_agent,
            cancel_agent,
            agent_running,
            apply_delta,
            run_suite,
            invoke_lattice,
            rollback,
            load_template,
            export_lattice,
            import_lattice,
            system_prompt,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Lattice-OS");
}
