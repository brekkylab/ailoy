use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Mutex;

const OAUTH_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
const DRIVE_FILES: &str = "https://www.googleapis.com/drive/v3/files";
const DRIVE_DRIVES: &str = "https://www.googleapis.com/drive/v3/drives";
const DOCS_API: &str = "https://docs.googleapis.com/v1/documents";
const FILE_FIELDS: &str =
    "nextPageToken,files(id,name,mimeType,driveId,size,quotaBytesUsed,modifiedTime,parents)";

#[derive(Clone, Serialize, Deserialize)]
pub struct GDriveConfig {
    pub client_id: String,
    pub client_secret: String,
    pub refresh_token: String,
}

/// Holds Google OAuth credentials (one refresh token), a cached access token,
/// and a cache of exported Workspace-doc text.
pub struct GDriveAccessor {
    client: reqwest::Client,
    config: GDriveConfig,
    access_token: Mutex<Option<String>>,
    /// Exported Google Doc/Sheet/Slide text (keyed by file id), shared by stat
    /// (size) and read so they share one network round trip.
    export_cache: Mutex<HashMap<String, Vec<u8>>>,
}

impl GDriveAccessor {
    pub fn new(config: &GDriveConfig) -> anyhow::Result<Self> {
        Ok(Self {
            client: reqwest::Client::new(),
            config: config.clone(),
            access_token: Mutex::new(None),
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

    /// List the immediate children of `folder_id` ("root" for My Drive root).
    /// `drive_id` is set when listing inside a shared drive.
    pub async fn list_files(
        &self,
        folder_id: &str,
        drive_id: Option<&str>,
    ) -> anyhow::Result<Vec<Value>> {
        let token = self.token().await?;
        let q = format!("'{folder_id}' in parents and trashed=false");
        let mut files = Vec::new();
        let mut page_token: Option<String> = None;
        loop {
            let mut params: Vec<(&str, String)> = vec![
                ("q", q.clone()),
                ("fields", FILE_FIELDS.to_string()),
                ("pageSize", "1000".to_string()),
                ("orderBy", "modifiedTime desc".to_string()),
            ];
            if let Some(d) = drive_id {
                params.push(("corpora", "drive".to_string()));
                params.push(("driveId", d.to_string()));
                params.push(("includeItemsFromAllDrives", "true".to_string()));
                params.push(("supportsAllDrives", "true".to_string()));
            }
            if let Some(pt) = &page_token {
                params.push(("pageToken", pt.clone()));
            }
            let url = reqwest::Url::parse_with_params(DRIVE_FILES, &params)?;
            let resp = self
                .client
                .get(url)
                .bearer_auth(&token)
                .send()
                .await?
                .error_for_status()?;
            let v: Value = resp.json().await?;
            if let Some(arr) = v.get("files").and_then(|f| f.as_array()) {
                files.extend(arr.iter().cloned());
            }
            page_token = v
                .get("nextPageToken")
                .and_then(|t| t.as_str())
                .map(|s| s.to_string());
            if page_token.is_none() {
                break;
            }
        }
        Ok(files)
    }

    /// Shared drives visible to the account (best-effort; needs scope).
    pub async fn list_shared_drives(&self) -> anyhow::Result<Vec<Value>> {
        let token = self.token().await?;
        let mut drives = Vec::new();
        let mut page_token: Option<String> = None;
        loop {
            let mut params: Vec<(&str, String)> = vec![
                ("fields", "nextPageToken,drives(id,name)".to_string()),
                ("pageSize", "100".to_string()),
            ];
            if let Some(pt) = &page_token {
                params.push(("pageToken", pt.clone()));
            }
            let url = reqwest::Url::parse_with_params(DRIVE_DRIVES, &params)?;
            let resp = self
                .client
                .get(url)
                .bearer_auth(&token)
                .send()
                .await?
                .error_for_status()?;
            let v: Value = resp.json().await?;
            if let Some(arr) = v.get("drives").and_then(|d| d.as_array()) {
                drives.extend(arr.iter().cloned());
            }
            page_token = v
                .get("nextPageToken")
                .and_then(|t| t.as_str())
                .map(|s| s.to_string());
            if page_token.is_none() {
                break;
            }
        }
        Ok(drives)
    }

    pub async fn download(&self, id: &str) -> anyhow::Result<Vec<u8>> {
        let token = self.token().await?;
        let resp = self
            .client
            .get(format!(
                "{DRIVE_FILES}/{id}?alt=media&supportsAllDrives=true"
            ))
            .bearer_auth(token)
            .send()
            .await?
            .error_for_status()?;
        Ok(resp.bytes().await?.to_vec())
    }

    /// Export a Workspace doc as `mime` text (Docs/Slides use `text/plain`,
    /// Sheets use `text/csv` — Google rejects `text/plain` for spreadsheets),
    /// caching the result so stat (size) and a subsequent read share one round
    /// trip. Keyed by id+mime.
    pub async fn export_text(&self, id: &str, mime: &str) -> anyhow::Result<Vec<u8>> {
        let key = format!("{id}|{mime}");
        if let Some(cached) = self.export_cache.lock().await.get(&key) {
            return Ok(cached.clone());
        }
        let token = self.token().await?;
        let url = reqwest::Url::parse_with_params(
            &format!("{DRIVE_FILES}/{id}/export"),
            &[("mimeType", mime), ("supportsAllDrives", "true")],
        )?;
        let resp = self
            .client
            .get(url)
            .bearer_auth(token)
            .send()
            .await?
            .error_for_status()?;
        let data = resp.bytes().await?.to_vec();
        self.export_cache.lock().await.insert(key, data.clone());
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
