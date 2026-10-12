//! Generation layer: streaming chat over Anthropic (Messages API), OpenAI
//! (Responses API), Google Gemini (`streamGenerateContent`) and Ollama Cloud
//! (`/api/chat`), plus live model discovery for each.

pub mod catalog;

use crate::agent::{GenError, Generator};
use async_trait::async_trait;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Anthropic,
    Openai,
    Gemini,
    OllamaCloud,
}

impl ProviderKind {
    pub const ALL: [ProviderKind; 4] =
        [ProviderKind::Anthropic, ProviderKind::Openai, ProviderKind::Gemini, ProviderKind::OllamaCloud];

    pub fn label(self) -> &'static str {
        match self {
            ProviderKind::Anthropic => "Anthropic",
            ProviderKind::Openai => "OpenAI",
            ProviderKind::Gemini => "Google Gemini",
            ProviderKind::OllamaCloud => "Ollama Cloud",
        }
    }

    pub fn default_base_url(self) -> &'static str {
        match self {
            ProviderKind::Anthropic => "https://api.anthropic.com",
            ProviderKind::Openai => "https://api.openai.com",
            ProviderKind::Gemini => "https://generativelanguage.googleapis.com",
            ProviderKind::OllamaCloud => "https://ollama.com",
        }
    }

    /// Environment variable consulted when no key is saved in settings.
    pub fn env_key(self) -> &'static str {
        match self {
            ProviderKind::Anthropic => "ANTHROPIC_API_KEY",
            ProviderKind::Openai => "OPENAI_API_KEY",
            ProviderKind::Gemini => "GEMINI_API_KEY",
            ProviderKind::OllamaCloud => "OLLAMA_API_KEY",
        }
    }

    fn api_suffix(self) -> &'static str {
        match self {
            ProviderKind::Anthropic | ProviderKind::Openai => "/v1",
            ProviderKind::Gemini => "/v1beta",
            ProviderKind::OllamaCloud => "/api",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProviderConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
}

impl ProviderConfig {
    pub fn base(&self, kind: ProviderKind) -> String {
        let raw = self
            .base_url
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(kind.default_base_url());
        let trimmed = raw.trim_end_matches('/');
        trimmed.strip_suffix(kind.api_suffix()).unwrap_or(trimmed).to_string()
    }

    pub fn key(&self) -> Option<&str> {
        self.api_key.as_deref().map(str::trim).filter(|k| !k.is_empty())
    }

    /// Local Ollama servers do not need a key; everything else does.
    pub fn requires_key(&self, kind: ProviderKind) -> bool {
        match kind {
            ProviderKind::OllamaCloud => {
                let base = self.base(kind);
                !(base.contains("localhost") || base.contains("127.0.0.1") || base.contains("[::1]"))
            }
            _ => true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub label: String,
    pub provider: ProviderKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tier: Option<String>,
    /// Confirmed by the provider's live model listing.
    pub live: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelListing {
    pub provider: ProviderKind,
    pub models: Vec<ModelInfo>,
    pub live_error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into() }
    }
}

#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("{0} API key is not configured (Settings → Providers, or set {1})")]
    MissingKey(&'static str, &'static str),
    #[error("{provider} returned HTTP {status}: {body}")]
    Http { provider: &'static str, status: u16, body: String },
    #[error("network error: {0}")]
    Transport(String),
    #[error("{provider} stream error: {message}")]
    Upstream { provider: &'static str, message: String },
    #[error("unexpected response from {provider}: {message}")]
    Protocol { provider: &'static str, message: String },
}

impl From<reqwest::Error> for ProviderError {
    fn from(e: reqwest::Error) -> Self {
        ProviderError::Transport(e.to_string())
    }
}

pub const MAX_OUTPUT_TOKENS: u32 = 16_000;

#[derive(Clone)]
pub struct ProviderClient {
    http: reqwest::Client,
}

impl Default for ProviderClient {
    fn default() -> Self {
        Self::new()
    }
}

fn upstream(kind: ProviderKind, message: impl Into<String>) -> ProviderError {
    ProviderError::Upstream { provider: kind.label(), message: message.into() }
}

fn error_message(v: &Value) -> Option<String> {
    let e = v.get("error")?;
    Some(
        e.get("message")
            .and_then(Value::as_str)
            .map(String::from)
            .or_else(|| e.as_str().map(String::from))
            .unwrap_or_else(|| e.to_string()),
    )
}

/// Extract streamed text from one decoded event. `Ok(None)` means the event
/// carried no visible text (pings, thinking, metadata).
pub fn extract_text(kind: ProviderKind, v: &Value) -> Result<Option<String>, ProviderError> {
    match kind {
        ProviderKind::Anthropic => match v.get("type").and_then(Value::as_str) {
            Some("content_block_delta") => {
                let d = &v["delta"];
                if d.get("type").and_then(Value::as_str) == Some("text_delta") {
                    Ok(d.get("text").and_then(Value::as_str).map(String::from))
                } else {
                    Ok(None)
                }
            }
            Some("error") => Err(upstream(kind, error_message(v).unwrap_or_else(|| v.to_string()))),
            _ => Ok(None),
        },
        ProviderKind::Openai => match v.get("type").and_then(Value::as_str) {
            Some("response.output_text.delta") => Ok(v.get("delta").and_then(Value::as_str).map(String::from)),
            Some("error") => Err(upstream(
                kind,
                v.get("message").and_then(Value::as_str).map(String::from).or_else(|| error_message(v)).unwrap_or_else(|| v.to_string()),
            )),
            Some("response.failed") => Err(upstream(
                kind,
                v.pointer("/response/error/message").and_then(Value::as_str).unwrap_or("response failed").to_string(),
            )),
            _ => Ok(None),
        },
        ProviderKind::Gemini => {
            if let Some(msg) = error_message(v) {
                return Err(upstream(kind, msg));
            }
            if let Some(reason) = v.pointer("/promptFeedback/blockReason").and_then(Value::as_str) {
                return Err(upstream(kind, format!("prompt blocked: {reason}")));
            }
            let mut out = String::new();
            if let Some(parts) = v.pointer("/candidates/0/content/parts").and_then(Value::as_array) {
                for p in parts {
                    if p.get("thought").and_then(Value::as_bool) == Some(true) {
                        continue;
                    }
                    if let Some(t) = p.get("text").and_then(Value::as_str) {
                        out.push_str(t);
                    }
                }
            }
            Ok((!out.is_empty()).then_some(out))
        }
        ProviderKind::OllamaCloud => {
            if let Some(msg) = error_message(v) {
                return Err(upstream(kind, msg));
            }
            Ok(v.pointer("/message/content").and_then(Value::as_str).filter(|s| !s.is_empty()).map(String::from))
        }
    }
}

/// Incremental line splitter shared by SSE and NDJSON streams.
#[derive(Default)]
pub struct LineBuffer {
    buf: Vec<u8>,
}

impl LineBuffer {
    pub fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(bytes);
        let mut lines = Vec::new();
        while let Some(pos) = self.buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=pos).collect();
            lines.push(String::from_utf8_lossy(&line).trim_end_matches(['\r', '\n']).to_string());
        }
        lines
    }

    pub fn finish(&mut self) -> Option<String> {
        if self.buf.is_empty() {
            return None;
        }
        let rest = String::from_utf8_lossy(&std::mem::take(&mut self.buf)).trim().to_string();
        (!rest.is_empty()).then_some(rest)
    }
}

/// Decode one stream line into a JSON event, if it carries one.
pub fn decode_line(kind: ProviderKind, line: &str) -> Result<Option<Value>, ProviderError> {
    let payload = match kind {
        ProviderKind::OllamaCloud => line.trim(),
        _ => match line.strip_prefix("data:") {
            Some(d) => d.trim(),
            None => return Ok(None),
        },
    };
    if payload.is_empty() || payload == "[DONE]" {
        return Ok(None);
    }
    serde_json::from_str(payload)
        .map(Some)
        .map_err(|e| ProviderError::Protocol { provider: kind.label(), message: format!("{e}: {payload}") })
}

impl ProviderClient {
    pub fn new() -> Self {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(20))
            .timeout(Duration::from_secs(900))
            .user_agent(concat!("Lattice-OS/", env!("CARGO_PKG_VERSION")))
            .build()
            .expect("http client");
        Self { http }
    }

    fn authed(&self, kind: ProviderKind, cfg: &ProviderConfig, rb: reqwest::RequestBuilder) -> Result<reqwest::RequestBuilder, ProviderError> {
        let key = cfg.key();
        if key.is_none() && cfg.requires_key(kind) {
            return Err(ProviderError::MissingKey(kind.label(), kind.env_key()));
        }
        Ok(match (kind, key) {
            (_, None) => rb,
            (ProviderKind::Anthropic, Some(k)) => rb.header("x-api-key", k).header("anthropic-version", "2023-06-01"),
            (ProviderKind::Gemini, Some(k)) => rb.header("x-goog-api-key", k),
            (_, Some(k)) => rb.bearer_auth(k),
        })
    }

    async fn checked(kind: ProviderKind, resp: reqwest::Response) -> Result<reqwest::Response, ProviderError> {
        if resp.status().is_success() {
            return Ok(resp);
        }
        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();
        let body = serde_json::from_str::<Value>(&body).ok().and_then(|v| error_message(&v)).unwrap_or(body);
        Err(ProviderError::Http { provider: kind.label(), status, body: crate::runtime::truncate(&body, 600) })
    }

    pub async fn list_live(&self, kind: ProviderKind, cfg: &ProviderConfig) -> Result<Vec<ModelInfo>, ProviderError> {
        let base = cfg.base(kind);
        let url = match kind {
            ProviderKind::Anthropic => format!("{base}/v1/models?limit=1000"),
            ProviderKind::Openai => format!("{base}/v1/models"),
            ProviderKind::Gemini => format!("{base}/v1beta/models?pageSize=1000"),
            ProviderKind::OllamaCloud => format!("{base}/api/tags"),
        };
        let rb = if kind == ProviderKind::OllamaCloud && cfg.key().is_none() {
            self.http.get(url)
        } else {
            self.authed(kind, cfg, self.http.get(url))?
        };
        let resp = Self::checked(kind, rb.send().await?).await?;
        let v: Value = resp.json().await?;
        Ok(parse_model_list(kind, &v))
    }

    /// Curated catalogue merged with the live listing (when reachable).
    pub async fn models(&self, kind: ProviderKind, cfg: &ProviderConfig) -> ModelListing {
        let mut models = catalog::curated(kind);
        let live = self.list_live(kind, cfg).await;
        let live_error = match live {
            Ok(list) => {
                for m in &mut models {
                    m.live = list.iter().any(|l| l.id == m.id);
                }
                for l in list {
                    if !models.iter().any(|m| m.id == l.id) {
                        models.push(l);
                    }
                }
                None
            }
            Err(e) => Some(e.to_string()),
        };
        ModelListing { provider: kind, models, live_error }
    }

    pub fn build_request(
        &self,
        kind: ProviderKind,
        cfg: &ProviderConfig,
        model: &str,
        system: &str,
        messages: &[ChatMessage],
    ) -> Result<reqwest::RequestBuilder, ProviderError> {
        let base = cfg.base(kind);
        let (url, body) = request_body(kind, &base, model, system, messages);
        self.authed(kind, cfg, self.http.post(url).json(&body))
    }

    pub async fn stream_chat<F: Fn(&str) + Send + Sync + ?Sized>(
        &self,
        kind: ProviderKind,
        cfg: &ProviderConfig,
        model: &str,
        system: &str,
        messages: &[ChatMessage],
        on_token: &F,
    ) -> Result<String, ProviderError> {
        let rb = self.build_request(kind, cfg, model, system, messages)?;
        let resp = Self::checked(kind, rb.send().await?).await?;
        let mut stream = resp.bytes_stream();
        let mut lines = LineBuffer::default();
        let mut full = String::new();
        let handle = |line: &str, full: &mut String| -> Result<(), ProviderError> {
            if let Some(ev) = decode_line(kind, line)? {
                if let Some(text) = extract_text(kind, &ev)? {
                    on_token(&text);
                    full.push_str(&text);
                }
            }
            Ok(())
        };
        while let Some(chunk) = stream.next().await {
            for line in lines.push(&chunk?) {
                handle(&line, &mut full)?;
            }
        }
        if let Some(rest) = lines.finish() {
            handle(&rest, &mut full)?;
        }
        if full.trim().is_empty() {
            return Err(ProviderError::Protocol { provider: kind.label(), message: "empty completion".into() });
        }
        Ok(full)
    }
}

pub fn request_body(kind: ProviderKind, base: &str, model: &str, system: &str, messages: &[ChatMessage]) -> (String, Value) {
    let role = |r: Role, assistant: &str| match r {
        Role::User => "user".to_string(),
        Role::Assistant => assistant.to_string(),
    };
    match kind {
        ProviderKind::Anthropic => (
            format!("{base}/v1/messages"),
            json!({
                "model": model,
                "max_tokens": MAX_OUTPUT_TOKENS,
                "system": system,
                "stream": true,
                "messages": messages.iter().map(|m| json!({"role": role(m.role, "assistant"), "content": m.content})).collect::<Vec<_>>(),
            }),
        ),
        ProviderKind::Openai => (
            format!("{base}/v1/responses"),
            json!({
                "model": model,
                "instructions": system,
                "stream": true,
                "store": false,
                "input": messages.iter().map(|m| json!({"role": role(m.role, "assistant"), "content": m.content})).collect::<Vec<_>>(),
            }),
        ),
        ProviderKind::Gemini => (
            format!("{base}/v1beta/models/{model}:streamGenerateContent?alt=sse"),
            json!({
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": messages.iter().map(|m| json!({"role": role(m.role, "model"), "parts": [{"text": m.content}]})).collect::<Vec<_>>(),
            }),
        ),
        ProviderKind::OllamaCloud => {
            let mut msgs = vec![json!({"role": "system", "content": system})];
            msgs.extend(messages.iter().map(|m| json!({"role": role(m.role, "assistant"), "content": m.content})));
            (format!("{base}/api/chat"), json!({"model": model, "stream": true, "messages": msgs}))
        }
    }
}

pub fn parse_model_list(kind: ProviderKind, v: &Value) -> Vec<ModelInfo> {
    let mk = |id: String, label: String| ModelInfo { id, label, provider: kind, tier: None, live: true };
    let arr = |key: &str| v.get(key).and_then(Value::as_array).cloned().unwrap_or_default();
    match kind {
        ProviderKind::Anthropic => arr("data")
            .into_iter()
            .filter_map(|m| {
                let id = m.get("id")?.as_str()?.to_string();
                let label = m.get("display_name").and_then(Value::as_str).unwrap_or(&id).to_string();
                Some(mk(id, label))
            })
            .collect(),
        ProviderKind::Openai => {
            let mut ids: Vec<String> = arr("data")
                .into_iter()
                .filter_map(|m| m.get("id")?.as_str().map(String::from))
                .filter(|id| catalog::openai_is_chat_model(id))
                .collect();
            ids.sort_by(|a, b| b.cmp(a));
            ids.into_iter().map(|id| mk(id.clone(), id)).collect()
        }
        ProviderKind::Gemini => arr("models")
            .into_iter()
            .filter_map(|m| {
                let id = m.get("name")?.as_str()?.trim_start_matches("models/").to_string();
                let methods: Vec<String> = m
                    .get("supportedGenerationMethods")
                    .and_then(Value::as_array)
                    .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                if !catalog::gemini_is_chat_model(&id, &methods) {
                    return None;
                }
                let label = m.get("displayName").and_then(Value::as_str).unwrap_or(&id).to_string();
                Some(mk(id, label))
            })
            .collect(),
        ProviderKind::OllamaCloud => arr("models")
            .into_iter()
            .filter_map(|m| {
                let id = m.get("name").or_else(|| m.get("model"))?.as_str()?.to_string();
                Some(mk(id.clone(), id))
            })
            .collect(),
    }
}

/// A [`Generator`] backed by one provider/model pair.
pub struct ProviderGenerator {
    pub client: ProviderClient,
    pub kind: ProviderKind,
    pub config: ProviderConfig,
    pub model: String,
}

#[async_trait]
impl Generator for ProviderGenerator {
    async fn generate(
        &self,
        system: &str,
        messages: &[ChatMessage],
        on_token: &(dyn for<'t> Fn(&'t str) + Send + Sync),
    ) -> Result<String, GenError> {
        self.client
            .stream_chat(self.kind, &self.config, &self.model, system, messages, on_token)
            .await
            .map_err(|e| GenError(e.to_string()))
    }

    fn label(&self) -> String {
        format!("{} · {}", self.kind.label(), self.model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(kind: ProviderKind, raw: &str) -> Result<String, ProviderError> {
        let mut lb = LineBuffer::default();
        let mut out = String::new();
        let mid = raw.len() / 2;
        let mut lines = lb.push(&raw.as_bytes()[..mid]);
        lines.extend(lb.push(&raw.as_bytes()[mid..]));
        lines.extend(lb.finish());
        for l in lines {
            if let Some(ev) = decode_line(kind, &l)? {
                if let Some(t) = extract_text(kind, &ev)? {
                    out.push_str(&t);
                }
            }
        }
        Ok(out)
    }

    #[test]
    fn anthropic_sse() {
        let raw = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{}}\n\n\
event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"hmm\"}}\n\n\
event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hel\"}}\n\n\
event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n\
event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        assert_eq!(collect(ProviderKind::Anthropic, raw).unwrap(), "Hello");
        let err = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n";
        assert!(collect(ProviderKind::Anthropic, err).unwrap_err().to_string().contains("Overloaded"));
    }

    #[test]
    fn openai_responses_sse() {
        let raw = "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{}}\n\n\
event: response.output_text.delta\r\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"Lat\"}\r\n\r\n\
event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"tice\"}\n\n\
event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{}}\n\n";
        assert_eq!(collect(ProviderKind::Openai, raw).unwrap(), "Lattice");
        let failed = "data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"quota\"}}}\n";
        assert!(collect(ProviderKind::Openai, failed).unwrap_err().to_string().contains("quota"));
    }

    #[test]
    fn gemini_sse_skips_thoughts() {
        let raw = "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"plan\",\"thought\":true}]}}]}\n\n\
data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Geo\"}]}}]}\n\n\
data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"metric\"}]},\"finishReason\":\"STOP\"}]}\n\n";
        assert_eq!(collect(ProviderKind::Gemini, raw).unwrap(), "Geometric");
    }

    #[test]
    fn ollama_ndjson() {
        let raw = "{\"model\":\"kimi-k3\",\"message\":{\"role\":\"assistant\",\"content\":\"Run\"},\"done\":false}\n\
{\"model\":\"kimi-k3\",\"message\":{\"role\":\"assistant\",\"content\":\"time\"},\"done\":false}\n\
{\"model\":\"kimi-k3\",\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true}";
        assert_eq!(collect(ProviderKind::OllamaCloud, raw).unwrap(), "Runtime");
        assert!(collect(ProviderKind::OllamaCloud, "{\"error\":\"model not found\"}\n").is_err());
    }

    #[test]
    fn request_shapes() {
        let msgs = vec![ChatMessage::user("hi"), ChatMessage::assistant("yo"), ChatMessage::user("go")];
        let (url, body) = request_body(ProviderKind::Gemini, "https://g", "gemini-3.8-flash", "sys", &msgs);
        assert_eq!(url, "https://g/v1beta/models/gemini-3.8-flash:streamGenerateContent?alt=sse");
        assert_eq!(body["contents"][1]["role"], "model");
        let (url, body) = request_body(ProviderKind::Openai, "https://o", "gpt-6-astra", "sys", &msgs);
        assert_eq!(url, "https://o/v1/responses");
        assert_eq!(body["instructions"], "sys");
        let (_, body) = request_body(ProviderKind::Anthropic, "https://a", "claude-opus-5-5", "sys", &msgs);
        assert_eq!(body["system"], "sys");
        assert_eq!(body["max_tokens"], MAX_OUTPUT_TOKENS);
        let (url, body) = request_body(ProviderKind::OllamaCloud, "https://ollama.com", "kimi-k3", "sys", &msgs);
        assert_eq!(url, "https://ollama.com/api/chat");
        assert_eq!(body["messages"][0]["role"], "system");
    }

    #[test]
    fn base_url_normalisation_and_key_rules() {
        let cfg = ProviderConfig { api_key: None, base_url: Some("https://api.openai.com/v1/".into()) };
        assert_eq!(cfg.base(ProviderKind::Openai), "https://api.openai.com");
        let local = ProviderConfig { api_key: None, base_url: Some("http://localhost:11434".into()) };
        assert!(!local.requires_key(ProviderKind::OllamaCloud));
        assert!(ProviderConfig::default().requires_key(ProviderKind::OllamaCloud));
    }

    #[test]
    fn model_list_parsing_filters_non_chat_models() {
        let openai = json!({"data": [{"id": "gpt-6-astra"}, {"id": "text-embedding-3-large"}, {"id": "gpt-realtime-2.1"}, {"id": "o4-mini"}]});
        let ids: Vec<_> = parse_model_list(ProviderKind::Openai, &openai).into_iter().map(|m| m.id).collect();
        assert_eq!(ids, vec!["o4-mini", "gpt-6-astra"]);

        let gemini = json!({"models": [
            {"name": "models/gemini-3.8-flash", "displayName": "Gemini 3.8 Flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-3.8-flash-tts", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/text-embedding-005", "supportedGenerationMethods": ["embedContent"]}
        ]});
        let ids: Vec<_> = parse_model_list(ProviderKind::Gemini, &gemini).into_iter().map(|m| m.id).collect();
        assert_eq!(ids, vec!["gemini-3.8-flash"]);

        let ollama = json!({"models": [{"name": "kimi-k3"}, {"name": "glm-5.3"}]});
        assert_eq!(parse_model_list(ProviderKind::OllamaCloud, &ollama).len(), 2);
    }

    #[test]
    fn every_provider_has_a_curated_default() {
        for k in ProviderKind::ALL {
            let models = catalog::curated(k);
            assert!(models.iter().any(|m| m.id == catalog::default_model(k)), "{k:?}");
        }
    }
}
