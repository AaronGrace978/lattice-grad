//! Curated model catalogue (October 2026). The UI merges this with each
//! provider's live model listing, so newly released models show up without an
//! app update; the curated entries supply ordering, labels and tiers.

use super::{ModelInfo, ProviderKind};

const ANTHROPIC: &[(&str, &str, &str)] = &[
    ("claude-fable-5-1", "Claude Fable 5.1", "frontier"),
    ("claude-opus-5-5", "Claude Opus 5.5", "flagship"),
    ("claude-sonnet-5-5", "Claude Sonnet 5.5", "balanced"),
    ("claude-haiku-5-5", "Claude Haiku 5.5", "fast"),
    ("claude-fable-5", "Claude Fable 5", "legacy"),
    ("claude-opus-5", "Claude Opus 5", "legacy"),
    ("claude-sonnet-5", "Claude Sonnet 5", "legacy"),
    ("claude-opus-4-8", "Claude Opus 4.8", "legacy"),
];

const OPENAI: &[(&str, &str, &str)] = &[
    ("gpt-6-astra", "GPT-6 Astra", "flagship"),
    ("gpt-6.1-sol", "GPT-6.1 Sol", "balanced"),
    ("gpt-6-luna", "GPT-6 Luna", "fast"),
    ("gpt-5.6-sol", "GPT-5.6 Sol", "legacy"),
    ("gpt-5.6-luna", "GPT-5.6 Luna", "legacy"),
    ("gpt-5.5", "GPT-5.5", "legacy"),
];

const GEMINI: &[(&str, &str, &str)] = &[
    ("gemini-3.8-flash", "Gemini 3.8 Flash", "flagship"),
    ("gemini-3.1-pro-preview", "Gemini 3.1 Pro (preview)", "reasoning"),
    ("gemini-3.7-flash", "Gemini 3.7 Flash", "balanced"),
    ("gemini-3.6-flash", "Gemini 3.6 Flash", "balanced"),
    ("gemini-3.5-flash", "Gemini 3.5 Flash", "balanced"),
    ("gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite", "fast"),
    ("gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite", "fast"),
    ("gemini-3-flash-preview", "Gemini 3 Flash (preview)", "legacy"),
];

const OLLAMA_CLOUD: &[(&str, &str, &str)] = &[
    ("kimi-k3", "Kimi K3", "flagship"),
    ("glm-5.3", "GLM-5.3", "flagship"),
    ("deepseek-v4-pro:0813", "DeepSeek V4 Pro", "flagship"),
    ("kimi-k2.7-code", "Kimi K2.7 Code", "code"),
    ("minimax-m3", "MiniMax M3", "balanced"),
    ("mistral-large-4", "Mistral Large 4", "balanced"),
    ("nemotron-3-ultra", "Nemotron 3 Ultra", "balanced"),
    ("deepseek-v4.1-flash", "DeepSeek V4.1 Flash", "fast"),
    ("glm-5.3-flash", "GLM-5.3 Flash", "fast"),
    ("gpt-oss:120b", "gpt-oss 120B", "balanced"),
    ("nemotron-3-super", "Nemotron 3 Super", "fast"),
    ("gemma4:31b", "Gemma 4 31B", "fast"),
    ("gpt-oss:20b", "gpt-oss 20B", "fast"),
    ("nemotron-3-nano:30b", "Nemotron 3 Nano 30B", "fast"),
    ("kimi-k2.6", "Kimi K2.6", "legacy"),
    ("glm-5.2", "GLM-5.2", "legacy"),
    ("minimax-m2.7", "MiniMax M2.7", "legacy"),
    ("mistral-large-3:675b", "Mistral Large 3 675B", "legacy"),
];

pub fn curated(kind: ProviderKind) -> Vec<ModelInfo> {
    let table = match kind {
        ProviderKind::Anthropic => ANTHROPIC,
        ProviderKind::Openai => OPENAI,
        ProviderKind::Gemini => GEMINI,
        ProviderKind::OllamaCloud => OLLAMA_CLOUD,
    };
    table
        .iter()
        .map(|(id, label, tier)| ModelInfo {
            id: id.to_string(),
            label: label.to_string(),
            provider: kind,
            tier: Some(tier.to_string()),
            live: false,
        })
        .collect()
}

pub fn default_model(kind: ProviderKind) -> &'static str {
    match kind {
        ProviderKind::Anthropic => "claude-opus-5-5",
        ProviderKind::Openai => "gpt-6-astra",
        ProviderKind::Gemini => "gemini-3.8-flash",
        ProviderKind::OllamaCloud => "kimi-k3",
    }
}

const OPENAI_EXCLUDE: &[&str] = &[
    "audio", "realtime", "tts", "transcribe", "image", "embedding", "moderation", "search", "whisper", "dall-e",
    "live", "instruct", "cyber",
];

const GEMINI_EXCLUDE: &[&str] = &[
    "tts", "image", "embedding", "live", "banana", "transcribe", "aqa", "imagen", "veo", "omni", "cyber",
];

pub fn openai_is_chat_model(id: &str) -> bool {
    let chatty = id.starts_with("gpt-")
        || id.starts_with("chatgpt")
        || id.starts_with("codex")
        || (id.starts_with('o') && id.chars().nth(1).is_some_and(|c| c.is_ascii_digit()));
    chatty && !OPENAI_EXCLUDE.iter().any(|x| id.contains(x))
}

pub fn gemini_is_chat_model(id: &str, methods: &[String]) -> bool {
    methods.iter().any(|m| m == "generateContent") && !GEMINI_EXCLUDE.iter().any(|x| id.contains(x))
}
