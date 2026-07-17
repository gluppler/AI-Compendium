use std::sync::atomic::AtomicI64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::{Duration, SystemTime};

use reqwest::blocking::Client;
use serde::Deserialize;
use serde::Serialize;

/// Free models — rotate through them to spread rate-limit burden.
/// Tested 2026-06-10: only `openai/gpt-oss-120b:free` works on this key.
/// Add more if they become available (test with `curl` first).
const FREE_MODELS: &[&str] = &[
    "openai/gpt-oss-120b:free",
];

static MODEL_COOLDOWNS: [AtomicI64; 1] = [
    AtomicI64::new(0),
];

const COOLDOWN_SECS: i64 = 5; // openai/gpt-oss-120b:free has no visible rate limit

static MODEL_INDEX: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Debug)]
pub struct LLMConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: f32,
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "https://openrouter.ai/api/v1".into(),
            model: "nvidia/nemotron-3-super-120b-a12b:free".into(),
            max_tokens: 2048,
            temperature: 0.7,
        }
    }
}

impl LLMConfig {
    pub fn from_env() -> Self {
        Self {
            api_key: std::env::var("OPENROUTER_API_KEY").unwrap_or_default(),
            base_url: std::env::var("OPENROUTER_BASE_URL")
                .unwrap_or_else(|_| "https://openrouter.ai/api/v1".into()),
            model: std::env::var("ATTACK_MODEL")
                .unwrap_or_else(|_| "openai/gpt-oss-120b:free".into()),
            max_tokens: std::env::var("MAX_TOKENS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2048),
            temperature: std::env::var("TEMPERATURE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.7),
        }
    }

    pub fn is_configured(&self) -> bool {
        !self.api_key.is_empty()
    }
}

#[derive(Debug)]
pub struct LLMClient {
    config: LLMConfig,
    http: Client,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    error: Option<ApiError>,
}

#[derive(Deserialize)]
struct Choice {
    message: ChoiceMessage,
}

#[derive(Deserialize)]
struct ChoiceMessage {
    content: String,
}

#[derive(Deserialize, Debug)]
struct ApiError {
    message: Option<String>,
}

impl LLMClient {
    pub fn new(config: LLMConfig) -> Self {
        let http = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("reqwest blocking client");
        Self { config, http }
    }

    pub fn config(&self) -> &LLMConfig {
        &self.config
    }

    /// Pick the next model whose cooldown has expired, waiting if all are on cooldown.
    fn next_available_model(&self) -> (String, usize) {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let n = FREE_MODELS.len();
        let start = MODEL_INDEX.fetch_add(1, Ordering::Relaxed) % n;

        for offset in 0..n {
            let idx = (start + offset) % n;
            let cooldown_until = MODEL_COOLDOWNS[idx].load(Ordering::Relaxed);
            if now >= cooldown_until {
                // Model is available
                return (FREE_MODELS[idx].to_string(), idx);
            }
        }

        // All models on cooldown — wait for the one that expires soonest
        let mut min_wait = COOLDOWN_SECS;
        for idx in 0..n {
            let cooldown_until = MODEL_COOLDOWNS[idx].load(Ordering::Relaxed);
            let wait = cooldown_until - now;
            if wait < min_wait { min_wait = wait; }
        }
        if min_wait > 0 {
            log::debug!("All models on cooldown, waiting {}s...", min_wait);
            std::thread::sleep(Duration::from_secs(min_wait as u64));
        }
        // After wait, pick next model
        let idx = MODEL_INDEX.fetch_add(1, Ordering::Relaxed) % n;
        (FREE_MODELS[idx].to_string(), idx)
    }

    /// Mark a model as used — set its cooldown to `now + COOLDOWN_SECS`.
    fn mark_used(&self, idx: usize) {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        MODEL_COOLDOWNS[idx].store(now + COOLDOWN_SECS, Ordering::Relaxed);
    }

    pub fn chat_completion(&self, system_prompt: &str, user_prompt: &str) -> Result<String, String> {
        let url = format!("{}/chat/completions", self.config.base_url.trim_end_matches('/'));

        let (model, idx) = self.next_available_model();
        let body = ChatRequest {
            model: model.clone(),
            messages: vec![
                Message { role: "system".into(), content: system_prompt.into() },
                Message { role: "user".into(), content: user_prompt.into() },
            ],
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
        };

        let resp = self
            .http
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send();

        match resp {
            Ok(r) => {
                let status = r.status();
                if status.is_success() {
                    self.mark_used(idx);
                    let parsed: ChatResponse = r.json().map_err(|e| format!("JSON parse error: {e}"))?;
                    if let Some(err) = parsed.error {
                        return Err(format!("API error: {:?}", err.message));
                    }
                    if let Some(choice) = parsed.choices.into_iter().next() {
                        return Ok(choice.message.content);
                    }
                    return Err("Empty choices in response".into());
                } else if status.as_u16() == 429 {
                    // Still rate-limited: set cooldown and return error
                    self.mark_used(idx);
                    return Err(format!("Rate limited on {model}"));
                } else {
                    let text = r.text().unwrap_or_default();
                    return Err(format!("HTTP {status} ({model}): {text}"));
                }
            }
            Err(e) => {
                return Err(format!("Request failed ({model}): {e}"));
            }
        }
    }

    pub fn chat_completion_raw(&self, messages: Vec<(String, String)>) -> Result<String, String> {
        let body = ChatRequest {
            model: self.config.model.clone(),
            messages: messages
                .into_iter()
                .map(|(role, content)| Message { role, content })
                .collect(),
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
        };

        let url = format!("{}/chat/completions", self.config.base_url.trim_end_matches('/'));

        let resp = self
            .http
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send();

        match resp {
            Ok(r) => {
                let status = r.status();
                if status.is_success() {
                    let parsed: ChatResponse = r.json().map_err(|e| format!("JSON parse error: {e}"))?;
                    if let Some(err) = parsed.error {
                        return Err(format!("API error: {:?}", err.message));
                    }
                    if let Some(choice) = parsed.choices.into_iter().next() {
                        return Ok(choice.message.content);
                    }
                    Err("Empty choices in response".into())
                } else {
                    let text = r.text().unwrap_or_default();
                    Err(format!("HTTP {status}: {text}"))
                }
            }
            Err(e) => Err(format!("Request failed: {e}")),
        }
    }
}
