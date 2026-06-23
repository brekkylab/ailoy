use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Mutex;

const OAUTH_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
const DRIVE_FILES: &str = "https://www.googleapis.com/drive/v3/files";
const DOCS_API: &str = "https://docs.googleapis.com/v1/documents";
const LIST_TTL: Duration = Duration::from_secs(10);

#[derive(Clone, Serialize, Deserialize)]
pub struct GDriveConfig {
    pub client_id: String,
    pub client_secret: String,
    pub refresh_token: String,
}

/// Holds Google OAuth credentials (one refresh token) and a cached access
/// token. Mirrors mirage `accessor/gdrive.py` + `core/google/config.py`.
pub struct GDriveAccessor {
    client: reqwest::Client,
    config: GDriveConfig,
    access_token: Mutex<Option<String>>,
    /// Short-TTL cache of the Drive file listing, shared by readdir/stat/resolve
    /// so an `ls` (one readdir + a getattr per entry) costs one Drive call, not one per entry.
    list_cache: Mutex<Option<(Instant, Vec<Value>)>>,
    /// Cache of exported Google Doc text (keyed by file id), shared by stat and read.
    export_cache: Mutex<HashMap<String, Vec<u8>>>,
}

impl GDriveAccessor {
    pub fn new(config: &GDriveConfig) -> anyhow::Result<Self> {
        Ok(Self {
            client: reqwest::Client::new(),
            config: config.clone(),
            access_token: Mutex::new(None),
            list_cache: Mutex::new(None),
            export_cache: Mutex::new(HashMap::new()),
        })
    }

    async fn token(&self) -> anyhow::Result<String> {
        let mut guard = self.access_token.lock().await;
        if let Some(t) = guard.as_ref() {
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
        *guard = Some(token.clone());
        Ok(token)
    }

    pub async fn list_files(&self) -> anyhow::Result<Vec<Value>> {
        {
            let cache = self.list_cache.lock().await;
            if let Some((at, files)) = cache.as_ref()
                && at.elapsed() < LIST_TTL
            {
                return Ok(files.clone());
            }
        }
        let token = self.token().await?;
        let url = format!(
            "{DRIVE_FILES}?fields=files(id,name,mimeType,size)&pageSize=200&q=trashed%3Dfalse"
        );
        let resp = self
            .client
            .get(url)
            .bearer_auth(token)
            .send()
            .await?
            .error_for_status()?;
        let v: Value = resp.json().await?;
        let files = v
            .get("files")
            .and_then(|f| f.as_array())
            .cloned()
            .unwrap_or_default();
        *self.list_cache.lock().await = Some((Instant::now(), files.clone()));
        Ok(files)
    }

    pub async fn download(&self, id: &str) -> anyhow::Result<Vec<u8>> {
        let token = self.token().await?;
        let resp = self
            .client
            .get(format!("{DRIVE_FILES}/{id}?alt=media"))
            .bearer_auth(token)
            .send()
            .await?
            .error_for_status()?;
        Ok(resp.bytes().await?.to_vec())
    }

    /// Export a Google Doc as text/plain, caching the result so stat (size) and
    /// a subsequent read share one network round trip.
    pub async fn export_text(&self, id: &str) -> anyhow::Result<Vec<u8>> {
        if let Some(cached) = self.export_cache.lock().await.get(id) {
            return Ok(cached.clone());
        }
        let token = self.token().await?;
        let resp = self
            .client
            .get(format!("{DRIVE_FILES}/{id}/export?mimeType=text/plain"))
            .bearer_auth(token)
            .send()
            .await?
            .error_for_status()?;
        let data = resp.bytes().await?.to_vec();
        self.export_cache
            .lock()
            .await
            .insert(id.to_string(), data.clone());
        Ok(data)
    }

    pub async fn docs_append(&self, document_id: &str, text: &str) -> anyhow::Result<Value> {
        let token = self.token().await?;
        let body = json!({
            "requests": [{
                "insertText": {"endOfSegmentLocation": {}, "text": text}
            }]
        });
        let resp = self
            .client
            .post(format!("{DOCS_API}/{document_id}:batchUpdate"))
            .bearer_auth(token)
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            anyhow::bail!("google docs batchUpdate {status}: {body}");
        }
        Ok(serde_json::from_str(&body).unwrap_or(Value::Null))
    }
}
