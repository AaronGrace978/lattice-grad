//! Provider credentials and preferences, persisted as JSON in the app config
//! directory. API keys never cross the IPC boundary back to the webview; the
//! UI only receives a masked hint and where the key came from.

use lattice_core::providers::catalog;
use lattice_core::{ProviderConfig, ProviderKind};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preferences {
    pub provider: ProviderKind,
    pub model: String,
    pub max_iterations: u32,
    pub max_operations: u64,
}

impl Default for Preferences {
    fn default() -> Self {
        Self {
            provider: ProviderKind::Anthropic,
            model: catalog::default_model(ProviderKind::Anthropic).to_string(),
            max_iterations: 6,
            max_operations: 250_000,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Settings {
    #[serde(default)]
    pub providers: BTreeMap<ProviderKind, ProviderConfig>,
    #[serde(default)]
    pub preferences: Preferences,
}

#[derive(Debug, Clone, Serialize)]
pub struct PublicProvider {
    pub kind: ProviderKind,
    pub label: &'static str,
    pub has_key: bool,
    /// `saved`, `env` or `none`.
    pub key_source: &'static str,
    pub key_hint: Option<String>,
    pub env_var: &'static str,
    pub base_url: Option<String>,
    pub default_base_url: &'static str,
    pub requires_key: bool,
    pub default_model: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct PublicSettings {
    pub providers: Vec<PublicProvider>,
    pub preferences: Preferences,
    pub settings_path: String,
}

fn hint(key: &str) -> String {
    let chars: Vec<char> = key.chars().collect();
    if chars.len() <= 8 {
        return "••••".into();
    }
    let head: String = chars[..4].iter().collect();
    let tail: String = chars[chars.len() - 4..].iter().collect();
    format!("{head}…{tail}")
}

impl Settings {
    pub fn load(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| e.to_string())?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
        }
        Ok(())
    }

    /// Saved values, with the provider's environment variable as key fallback.
    pub fn effective(&self, kind: ProviderKind) -> ProviderConfig {
        let mut cfg = self.providers.get(&kind).cloned().unwrap_or_default();
        if cfg.key().is_none() {
            cfg.api_key = std::env::var(kind.env_key()).ok().filter(|k| !k.trim().is_empty());
        }
        cfg
    }

    pub fn public(&self, path: &Path) -> PublicSettings {
        let providers = ProviderKind::ALL
            .iter()
            .map(|&kind| {
                let saved = self.providers.get(&kind).cloned().unwrap_or_default();
                let effective = self.effective(kind);
                let key_source = if saved.key().is_some() {
                    "saved"
                } else if effective.key().is_some() {
                    "env"
                } else {
                    "none"
                };
                PublicProvider {
                    kind,
                    label: kind.label(),
                    has_key: effective.key().is_some(),
                    key_source,
                    key_hint: effective.key().map(hint),
                    env_var: kind.env_key(),
                    base_url: saved.base_url.clone(),
                    default_base_url: kind.default_base_url(),
                    requires_key: effective.requires_key(kind),
                    default_model: catalog::default_model(kind),
                }
            })
            .collect();
        PublicSettings {
            providers,
            preferences: self.preferences.clone(),
            settings_path: path.display().to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hints_never_reveal_the_key() {
        assert_eq!(hint("sk-ant-1234567890abcd"), "sk-a…abcd");
        assert_eq!(hint("short"), "••••");
    }

    #[test]
    fn settings_round_trip_with_enum_keys() {
        let mut s = Settings::default();
        s.providers.insert(
            ProviderKind::OllamaCloud,
            ProviderConfig { api_key: Some("k".into()), base_url: None },
        );
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("ollama_cloud"));
        let back: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(back.providers[&ProviderKind::OllamaCloud].api_key.as_deref(), Some("k"));
    }
}
