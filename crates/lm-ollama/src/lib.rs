//! Thin async client for a local Ollama server.
//!
//! Only loopback base URLs are accepted (default `http://127.0.0.1:11434`):
//! LocalMotive is offline and never talks to a remote model host.
//!
//! Endpoints: `/api/version`, `/api/tags`, `/api/chat` (non-streaming, with an
//! optional JSON-schema `format`, `options.num_ctx` and `keep_alive`) and
//! `/api/embed`.

use std::time::Duration;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

pub const DEFAULT_BASE_URL: &str = "http://127.0.0.1:11434";

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("base URL {0:?} is not a loopback http address")]
    NotLoopback(String),
    #[error("cannot reach Ollama: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Ollama returned HTTP {status}: {body}")]
    Status { status: u16, body: String },
    #[error("could not decode model output as JSON: {0}")]
    Decode(#[from] serde_json::Error),
    #[error("embedding count mismatch: sent {sent}, got {got}")]
    EmbedCount { sent: usize, got: usize },
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Options {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_thread: Option<u32>,
    /// Cap on generated tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<i32>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    /// JSON schema the reply must follow (Ollama structured outputs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<Options>,
    /// e.g. "10m", or "0" to unload right after.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
    /// Thinking models (qwen3 hybrid): false disables the reasoning phase.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub think: Option<bool>,
    pub stream: bool,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            format: None,
            options: None,
            keep_alive: None,
            think: None,
            stream: false,
        }
    }
}

fn empty_message() -> Message {
    Message {
        role: "assistant".into(),
        content: String::new(),
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ChatResponse {
    #[serde(default)]
    pub model: String,
    #[serde(default = "empty_message")]
    pub message: Message,
    #[serde(default)]
    pub done: bool,
    #[serde(default)]
    pub total_duration: u64,
    #[serde(default)]
    pub prompt_eval_count: u32,
    #[serde(default)]
    pub eval_count: u32,
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<&'a str>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct VersionResponse {
    version: String,
}

#[derive(Deserialize)]
struct TagsResponse {
    #[serde(default)]
    models: Vec<ModelInfo>,
}

#[derive(Debug, Clone)]
pub struct Client {
    base: String,
    http: reqwest::Client,
}

fn is_loopback(url: &str) -> bool {
    let Some(rest) = url.strip_prefix("http://") else {
        return false;
    };
    let host = rest.split('/').next().unwrap_or_default();
    let host = if let Some(h) = host.strip_prefix('[') {
        h.split(']').next().unwrap_or_default()
    } else {
        host.split(':').next().unwrap_or_default()
    };
    matches!(host, "127.0.0.1" | "localhost" | "::1")
}

impl Client {
    /// Client for `http://127.0.0.1:11434`.
    pub fn new() -> Result<Self> {
        Self::with_base_url(DEFAULT_BASE_URL)
    }

    /// Client for another loopback address (tests, non-default port).
    pub fn with_base_url(base: &str) -> Result<Self> {
        let base = base.trim_end_matches('/').to_string();
        if !is_loopback(&base) {
            return Err(Error::NotLoopback(base));
        }
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(3))
            .timeout(Duration::from_secs(900))
            .no_proxy()
            .build()?;
        Ok(Self { base, http })
    }

    async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let r = self.http.get(format!("{}{path}", self.base)).send().await?;
        Self::decode(r).await
    }

    async fn post<B: Serialize + ?Sized, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T> {
        let r = self
            .http
            .post(format!("{}{path}", self.base))
            .json(body)
            .send()
            .await?;
        Self::decode(r).await
    }

    async fn decode<T: DeserializeOwned>(r: reqwest::Response) -> Result<T> {
        let status = r.status();
        if !status.is_success() {
            let body = r.text().await.unwrap_or_default();
            return Err(Error::Status {
                status: status.as_u16(),
                body,
            });
        }
        Ok(r.json().await?)
    }

    pub async fn version(&self) -> Result<String> {
        Ok(self.get::<VersionResponse>("/api/version").await?.version)
    }

    pub async fn tags(&self) -> Result<Vec<ModelInfo>> {
        Ok(self.get::<TagsResponse>("/api/tags").await?.models)
    }

    /// True when `model` (with or without the `:latest` tag) is pulled.
    pub async fn has_model(&self, model: &str) -> Result<bool> {
        let want = if model.contains(':') {
            model.to_string()
        } else {
            format!("{model}:latest")
        };
        Ok(self.tags().await?.iter().any(|m| m.name == want))
    }

    pub async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse> {
        let mut req = req.clone();
        req.stream = false;
        self.post("/api/chat", &req).await
    }

    /// Streaming chat: `on_token` gets each content piece as it is generated;
    /// returns the full reply (content concatenated, counts from the final line).
    pub async fn chat_stream(
        &self,
        req: &ChatRequest,
        on_token: &mut (dyn FnMut(&str) + Send),
    ) -> Result<ChatResponse> {
        let mut req = req.clone();
        req.stream = true;
        let mut r = self
            .http
            .post(format!("{}/api/chat", self.base))
            .json(&req)
            .send()
            .await?;
        let status = r.status();
        if !status.is_success() {
            let body = r.text().await.unwrap_or_default();
            return Err(Error::Status {
                status: status.as_u16(),
                body,
            });
        }
        let mut buf: Vec<u8> = Vec::new();
        let mut content = String::new();
        let mut last: Option<ChatResponse> = None;
        let mut handle_line =
            |line: &[u8], content: &mut String, last: &mut Option<ChatResponse>| -> Result<()> {
                if line.iter().all(u8::is_ascii_whitespace) {
                    return Ok(());
                }
                let part: ChatResponse = serde_json::from_slice(line)?;
                if !part.message.content.is_empty() {
                    on_token(&part.message.content);
                    content.push_str(&part.message.content);
                }
                *last = Some(part);
                Ok(())
            };
        while let Some(bytes) = r.chunk().await? {
            buf.extend_from_slice(&bytes);
            while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buf.drain(..=nl).collect();
                handle_line(&line, &mut content, &mut last)?;
            }
        }
        handle_line(&buf, &mut content, &mut last)?;
        let mut resp = last.ok_or(Error::Status {
            status: 200,
            body: "empty stream".into(),
        })?;
        resp.message.content = content;
        Ok(resp)
    }

    /// Chat with a JSON-schema `format` and parse the reply into `T`.
    pub async fn chat_json<T: DeserializeOwned>(&self, req: &ChatRequest) -> Result<T> {
        let resp = self.chat(req).await?;
        Ok(serde_json::from_str(&resp.message.content)?)
    }

    /// Embed a batch of texts; returns one vector per input, in order.
    pub async fn embed(
        &self,
        model: &str,
        input: &[String],
        keep_alive: Option<&str>,
    ) -> Result<Vec<Vec<f32>>> {
        let resp: EmbedResponse = self
            .post(
                "/api/embed",
                &EmbedRequest {
                    model,
                    input,
                    keep_alive,
                },
            )
            .await?;
        if resp.embeddings.len() != input.len() {
            return Err(Error::EmbedCount {
                sent: input.len(),
                got: resp.embeddings.len(),
            });
        }
        Ok(resp.embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn only_loopback_urls() {
        assert!(Client::with_base_url("http://127.0.0.1:11434").is_ok());
        assert!(Client::with_base_url("http://localhost:11434/").is_ok());
        assert!(Client::with_base_url("http://[::1]:11434").is_ok());
        for bad in [
            "http://10.0.0.5:11434",
            "https://127.0.0.1:11434",
            "http://ollama.example.com",
            "http://127.0.0.1.evil.com:80",
        ] {
            assert!(
                matches!(Client::with_base_url(bad), Err(Error::NotLoopback(_))),
                "{bad}"
            );
        }
    }

    async fn client(server: &MockServer) -> Client {
        Client::with_base_url(&server.uri()).unwrap()
    }

    #[tokio::test]
    async fn version_and_tags() {
        let s = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/version"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"version":"0.40.6"})))
            .mount(&s)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                json!({"models":[{"name":"qwen3-embedding:0.6b","size":639,"digest":"abc"}]}),
            ))
            .mount(&s)
            .await;
        let c = client(&s).await;
        assert_eq!(c.version().await.unwrap(), "0.40.6");
        assert_eq!(c.tags().await.unwrap()[0].name, "qwen3-embedding:0.6b");
        assert!(c.has_model("qwen3-embedding:0.6b").await.unwrap());
        assert!(!c.has_model("qwen3:4b").await.unwrap());
    }

    #[tokio::test]
    async fn chat_sends_format_options_keep_alive_and_parses_json() {
        let s = MockServer::start().await;
        let schema = json!({"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]});
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .and(body_partial_json(json!({"model":"qwen3:4b","stream":false,"format":schema,"options":{"num_ctx":8192},"keep_alive":"10m"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "model":"qwen3:4b","message":{"role":"assistant","content":"{\"answer\":\"1.52 mm\"}"},"done":true,"eval_count":7
            })))
            .expect(2)
            .mount(&s)
            .await;
        let c = client(&s).await;
        let mut req = ChatRequest::new(
            "qwen3:4b",
            vec![Message::system("s"), Message::user("gap?")],
        );
        req.format = Some(schema.clone());
        req.options = Some(Options {
            num_ctx: Some(8192),
            ..Options::default()
        });
        req.keep_alive = Some("10m".into());
        let resp = c.chat(&req).await.unwrap();
        assert_eq!(resp.eval_count, 7);
        #[derive(Deserialize)]
        struct A {
            answer: String,
        }
        let a: A = c.chat_json(&req).await.unwrap();
        assert_eq!(a.answer, "1.52 mm");
    }

    #[tokio::test]
    async fn chat_stream_concatenates_tokens() {
        let s = MockServer::start().await;
        let body = concat!(
            "{\"model\":\"m\",\"message\":{\"role\":\"assistant\",\"content\":\"{\\\"a\\\":\"},\"done\":false}\n",
            "{\"model\":\"m\",\"message\":{\"role\":\"assistant\",\"content\":\"1}\"},\"done\":false}\n",
            "{\"model\":\"m\",\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"prompt_eval_count\":12,\"eval_count\":3}\n"
        );
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .and(body_partial_json(json!({"stream": true, "think": false})))
            .respond_with(ResponseTemplate::new(200).set_body_raw(body, "application/x-ndjson"))
            .mount(&s)
            .await;
        let mut req = ChatRequest::new("m", vec![Message::user("x")]);
        req.think = Some(false);
        let mut tokens = Vec::new();
        let resp = client(&s)
            .await
            .chat_stream(&req, &mut |t| tokens.push(t.to_string()))
            .await
            .unwrap();
        assert_eq!(tokens, vec!["{\"a\":", "1}"]);
        assert_eq!(resp.message.content, "{\"a\":1}");
        assert_eq!(
            (resp.prompt_eval_count, resp.eval_count, resp.done),
            (12, 3, true)
        );
    }

    #[tokio::test]
    async fn embed_batches_and_checks_count() {
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .and(body_partial_json(
                json!({"model":"qwen3-embedding:0.6b","input":["a","b"]}),
            ))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({"embeddings":[[0.1,0.2],[0.3,0.4]]})),
            )
            .mount(&s)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .and(body_partial_json(json!({"input":["x"]})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"embeddings":[]})))
            .mount(&s)
            .await;
        let c = client(&s).await;
        let v = c
            .embed("qwen3-embedding:0.6b", &["a".into(), "b".into()], None)
            .await
            .unwrap();
        assert_eq!(v, vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
        assert!(matches!(
            c.embed("m", &["x".into()], None).await,
            Err(Error::EmbedCount { sent: 1, got: 0 })
        ));
    }

    #[tokio::test]
    async fn http_errors_surface_status_and_body() {
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(
                ResponseTemplate::new(404).set_body_string(r#"{"error":"model 'nope' not found"}"#),
            )
            .mount(&s)
            .await;
        let err = client(&s)
            .await
            .chat(&ChatRequest::new("nope", vec![]))
            .await
            .unwrap_err();
        match err {
            Error::Status { status, body } => {
                assert_eq!(status, 404);
                assert!(body.contains("not found"));
            }
            e => panic!("unexpected {e}"),
        }
    }

    #[tokio::test]
    async fn unreachable_server_is_http_error() {
        let c = Client::with_base_url("http://127.0.0.1:9").unwrap();
        assert!(matches!(c.version().await, Err(Error::Http(_))));
    }
}
