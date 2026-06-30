use std::time::{Duration, Instant};

use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::Mutex;

const OAUTH_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
const GMAIL_API_BASE: &str = "https://gmail.googleapis.com/gmail/v1";

#[derive(Clone, Serialize, Deserialize)]
pub struct GmailConfig {
    pub client_id: String,
    pub client_secret: String,
    pub refresh_token: String,
}

/// Holds Google OAuth credentials (one refresh token) and a cached access
/// token. Mirrors [`GDriveAccessor`](super::GDriveAccessor)'s OAuth flow; both
/// speak Google APIs with the same `client_id`/`secret`/`refresh_token`, but a
/// Gmail mount needs a token whose scope covers Gmail (e.g.
/// `https://www.googleapis.com/auth/gmail.modify`).
pub struct GmailAccessor {
    client: reqwest::Client,
    config: GmailConfig,
    /// Cached OAuth access token + its expiry. Refreshed proactively before
    /// expiry and on a 401 (see [`Self::send_with_refresh`]).
    access_token: Mutex<Option<(String, Instant)>>,
}

impl GmailAccessor {
    pub fn new(config: &GmailConfig) -> anyhow::Result<Self> {
        Ok(Self {
            // Bound every request: a hung upstream call run behind the FUSE
            // forward server would otherwise wedge the guest FUSE op (and any
            // process touching the mount) forever. A timeout makes it recoverable.
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .connect_timeout(Duration::from_secs(10))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            config: config.clone(),
            access_token: Mutex::new(None),
        })
    }

    async fn token(&self) -> anyhow::Result<String> {
        let mut guard = self.access_token.lock().await;
        // Reuse a cached token until it's within 60s of expiry (proactive refresh
        // avoids the "everything 401s after ~1h" failure).
        if let Some((t, exp)) = guard.as_ref()
            && *exp > Instant::now() + Duration::from_secs(60)
        {
            return Ok(t.clone());
        }
        let resp = self
            .client
            .post(OAUTH_TOKEN_URL)
            .form(&[
                ("client_id", self.config.client_id.as_str()),
                ("client_secret", self.config.client_secret.as_str()),
                ("refresh_token", self.config.refresh_token.as_str()),
                ("grant_type", "refresh_token"),
            ])
            .send()
            .await?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            anyhow::bail!("google token exchange {status}: {body}");
        }
        let v: Value = serde_json::from_str(&body)?;
        let token = v
            .get("access_token")
            .and_then(|t| t.as_str())
            .ok_or_else(|| anyhow::anyhow!("no access_token in response"))?
            .to_string();
        let expires_in = v.get("expires_in").and_then(|e| e.as_u64()).unwrap_or(3600);
        *guard = Some((
            token.clone(),
            Instant::now() + Duration::from_secs(expires_in),
        ));
        Ok(token)
    }

    /// Send a request built from the current access token; on 401 (expired or
    /// revoked despite proactive refresh) drop the token, refresh, and retry once.
    async fn send_with_refresh(
        &self,
        build: impl Fn(&str) -> reqwest::RequestBuilder,
    ) -> anyhow::Result<reqwest::Response> {
        let token = self.token().await?;
        let resp = build(&token).send().await?;
        if resp.status() != reqwest::StatusCode::UNAUTHORIZED {
            return Ok(resp);
        }
        *self.access_token.lock().await = None;
        let token = self.token().await?;
        Ok(build(&token).send().await?)
    }

    async fn get_json(&self, url: &str) -> anyhow::Result<Value> {
        let resp = self
            .send_with_refresh(|t| self.client.get(url).bearer_auth(t))
            .await?
            .error_for_status()?;
        Ok(resp.json().await?)
    }

    /// All Gmail labels (system + user). Each is `{id, name, type}`.
    pub async fn list_labels(&self) -> anyhow::Result<Vec<Value>> {
        let url = format!("{GMAIL_API_BASE}/users/me/labels");
        let v = self.get_json(&url).await?;
        Ok(v.get("labels")
            .and_then(|l| l.as_array())
            .cloned()
            .unwrap_or_default())
    }

    /// Message stubs (`{id, threadId}`) for a label and/or query.
    pub async fn list_messages(
        &self,
        label_id: Option<&str>,
        query: Option<&str>,
        max_results: u32,
    ) -> anyhow::Result<Vec<Value>> {
        let mut params: Vec<(&str, String)> = vec![("maxResults", max_results.to_string())];
        if let Some(l) = label_id {
            params.push(("labelIds", l.to_string()));
        }
        if let Some(q) = query {
            params.push(("q", q.to_string()));
        }
        let url = reqwest::Url::parse_with_params(
            &format!("{GMAIL_API_BASE}/users/me/messages"),
            &params,
        )?;
        let v = self
            .send_with_refresh(|t| self.client.get(url.clone()).bearer_auth(t))
            .await?
            .error_for_status()?
            .json::<Value>()
            .await?;
        Ok(v.get("messages")
            .and_then(|m| m.as_array())
            .cloned()
            .unwrap_or_default())
    }

    /// Full message resource (`format=full`): headers, payload parts, labels,
    /// `internalDate`, `sizeEstimate`.
    pub async fn get_message_full(&self, id: &str) -> anyhow::Result<Value> {
        let url = format!("{GMAIL_API_BASE}/users/me/messages/{id}?format=full");
        self.get_json(&url).await
    }

    /// Minimal message resource (`format=minimal`): `internalDate` + labels but
    /// no headers/body. Used to date-bucket a label listing cheaply.
    pub async fn get_message_minimal(&self, id: &str) -> anyhow::Result<Value> {
        let url = format!("{GMAIL_API_BASE}/users/me/messages/{id}?format=minimal");
        self.get_json(&url).await
    }

    /// Fetch and base64url-decode an attachment's bytes.
    pub async fn get_attachment(
        &self,
        message_id: &str,
        attachment_id: &str,
    ) -> anyhow::Result<Vec<u8>> {
        let url =
            format!("{GMAIL_API_BASE}/users/me/messages/{message_id}/attachments/{attachment_id}");
        let v = self.get_json(&url).await?;
        let data = v.get("data").and_then(|d| d.as_str()).unwrap_or("");
        Ok(decode_b64url(data))
    }

    /// Move a message to Trash (the `rm` of a `.gmail.json`).
    pub async fn trash(&self, message_id: &str) -> anyhow::Result<()> {
        let url = format!("{GMAIL_API_BASE}/users/me/messages/{message_id}/trash");
        self.send_with_refresh(|t| {
            self.client
                .post(&url)
                .bearer_auth(t)
                .json(&serde_json::json!({}))
        })
        .await?
        .error_for_status()?;
        Ok(())
    }

    /// Send a raw (base64url) RFC-2822 message, optionally within a thread.
    pub async fn send_raw(&self, raw_b64: &str, thread_id: Option<&str>) -> anyhow::Result<Value> {
        let mut body = serde_json::json!({ "raw": raw_b64 });
        if let Some(tid) = thread_id {
            body["threadId"] = Value::from(tid);
        }
        let url = format!("{GMAIL_API_BASE}/users/me/messages/send");
        let resp = self
            .send_with_refresh(|t| self.client.post(&url).bearer_auth(t).json(&body))
            .await?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            anyhow::bail!("gmail send {status}: {text}");
        }
        Ok(serde_json::from_str(&text).unwrap_or(Value::Null))
    }
}

/// Decode Gmail's base64url payload data, tolerating missing padding.
fn decode_b64url(s: &str) -> Vec<u8> {
    let trimmed = s.trim_end_matches('=');
    base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(trimmed)
        .unwrap_or_default()
}

/// Base64url-encode raw MIME bytes for the Gmail `send` API.
pub fn encode_b64url(bytes: &[u8]) -> String {
    base64::engine::general_purpose::URL_SAFE.encode(bytes)
}
