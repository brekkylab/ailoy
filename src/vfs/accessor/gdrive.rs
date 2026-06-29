use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Mutex;

const OAUTH_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
const DRIVE_FILES: &str = "https://www.googleapis.com/drive/v3/files";
const DRIVE_DRIVES: &str = "https://www.googleapis.com/drive/v3/drives";
const DOCS_API: &str = "https://docs.googleapis.com/v1/documents";
const SHEETS_API: &str = "https://sheets.googleapis.com/v4/spreadsheets";
const SLIDES_API: &str = "https://slides.googleapis.com/v1/presentations";
const FILE_FIELDS: &str =
    "nextPageToken,files(id,name,mimeType,driveId,size,quotaBytesUsed,modifiedTime,parents)";
/// Hard cap on listing pages (1000 files/page for files, 100 drives/page) so a
/// duplicate/looping `nextPageToken` can't spin forever.
const MAX_PAGES: usize = 50;

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
    /// Cached OAuth access token + its expiry. Refreshed proactively before
    /// expiry and on a 401 (see [`Self::send_with_refresh`]).
    access_token: Mutex<Option<(String, Instant)>>,
    /// Exported Google Doc/Sheet/Slide text (keyed by file id), shared by stat
    /// (size) and read so they share one network round trip.
    export_cache: Mutex<HashMap<String, Vec<u8>>>,
}

impl GDriveAccessor {
    pub fn new(config: &GDriveConfig) -> anyhow::Result<Self> {
        Ok(Self {
            // Bound every request: a hung upstream call run behind the FUSE
            // forward server would otherwise wedge the guest FUSE op (and any
            // process touching the mount) forever. A timeout makes it recoverable.
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            config: config.clone(),
            access_token: Mutex::new(None),
            export_cache: Mutex::new(HashMap::new()),
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

    /// Drop any cached exported text for `id` (all mimes) — call after a write
    /// (docs-append) so a subsequent read/stat re-exports the new content.
    pub async fn invalidate_export(&self, id: &str) {
        let pfx = format!("{id}|");
        self.export_cache
            .lock()
            .await
            .retain(|k, _| !k.starts_with(&pfx));
    }

    /// List the immediate children of `folder_id` ("root" for My Drive root).
    /// `drive_id` is set when listing inside a shared drive.
    pub async fn list_files(
        &self,
        folder_id: &str,
        drive_id: Option<&str>,
    ) -> anyhow::Result<Vec<Value>> {
        let q = format!("'{folder_id}' in parents and trashed=false");
        let mut files = Vec::new();
        let mut page_token: Option<String> = None;
        // Bound the pagination: a buggy/duplicate `nextPageToken` (a known Drive
        // API pathology with some query/corpora combos) must not loop forever.
        let mut pages = 0usize;
        loop {
            pages += 1;
            if pages > MAX_PAGES {
                break;
            }
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
                .send_with_refresh(|t| self.client.get(url.clone()).bearer_auth(t))
                .await?
                .error_for_status()?;
            let v: Value = resp.json().await?;
            if let Some(arr) = v.get("files").and_then(|f| f.as_array()) {
                files.extend(arr.iter().cloned());
            }
            let next = v
                .get("nextPageToken")
                .and_then(|t| t.as_str())
                .map(|s| s.to_string());
            // Stop on no token, or a token identical to the one we just used
            // (would otherwise re-fetch the same page forever).
            if next.is_none() || next == page_token {
                break;
            }
            page_token = next;
        }
        Ok(files)
    }

    /// Shared drives visible to the account (best-effort; needs scope).
    pub async fn list_shared_drives(&self) -> anyhow::Result<Vec<Value>> {
        let mut drives = Vec::new();
        let mut page_token: Option<String> = None;
        let mut pages = 0usize;
        loop {
            pages += 1;
            if pages > MAX_PAGES {
                break;
            }
            let mut params: Vec<(&str, String)> = vec![
                ("fields", "nextPageToken,drives(id,name)".to_string()),
                ("pageSize", "100".to_string()),
            ];
            if let Some(pt) = &page_token {
                params.push(("pageToken", pt.clone()));
            }
            let url = reqwest::Url::parse_with_params(DRIVE_DRIVES, &params)?;
            let resp = self
                .send_with_refresh(|t| self.client.get(url.clone()).bearer_auth(t))
                .await?
                .error_for_status()?;
            let v: Value = resp.json().await?;
            if let Some(arr) = v.get("drives").and_then(|d| d.as_array()) {
                drives.extend(arr.iter().cloned());
            }
            let next = v
                .get("nextPageToken")
                .and_then(|t| t.as_str())
                .map(|s| s.to_string());
            if next.is_none() || next == page_token {
                break;
            }
            page_token = next;
        }
        Ok(drives)
    }

    pub async fn download(&self, id: &str) -> anyhow::Result<Vec<u8>> {
        let url = format!("{DRIVE_FILES}/{id}?alt=media&supportsAllDrives=true");
        let resp = self
            .send_with_refresh(|t| self.client.get(&url).bearer_auth(t))
            .await?
            .error_for_status()?;
        Ok(resp.bytes().await?.to_vec())
    }

    /// A Workspace doc's native API JSON (full document), mirroring mirage:
    /// Docs `documents.get`, Sheets `spreadsheets.get`, Slides
    /// `presentations.get`. Pretty-printed bytes are cached by id (shared by
    /// stat-size and read, and across the kernel's chunked reads of one file).
    /// `kind` is "sheet" / "slide" / anything-else (= doc).
    pub async fn workspace_json(&self, id: &str, kind: &str) -> anyhow::Result<Vec<u8>> {
        let key = format!("{id}|json");
        if let Some(cached) = self.export_cache.lock().await.get(&key) {
            return Ok(cached.clone());
        }
        let base = match kind {
            "sheet" => SHEETS_API,
            "slide" => SLIDES_API,
            _ => DOCS_API,
        };
        let url = format!("{base}/{id}");
        let resp = self
            .send_with_refresh(|t| self.client.get(&url).bearer_auth(t))
            .await?
            .error_for_status()?;
        let v: Value = resp.json().await?;
        let bytes = serde_json::to_vec_pretty(&v)?;
        self.export_cache.lock().await.insert(key, bytes.clone());
        Ok(bytes)
    }

    /// Byte length of an already-exported Workspace doc if it's cached — WITHOUT
    /// triggering a network export. `stat` uses this so sizing a directory of
    /// Workspace docs (tab-completion / `ls -l`) doesn't fetch every document.
    pub async fn cached_workspace_len(&self, id: &str) -> Option<u64> {
        let key = format!("{id}|json");
        self.export_cache
            .lock()
            .await
            .get(&key)
            .map(|b| b.len() as u64)
    }

    pub async fn docs_append(&self, document_id: &str, text: &str) -> anyhow::Result<Value> {
        let url = format!("{DOCS_API}/{document_id}:batchUpdate");
        let body = json!({
            "requests": [{
                "insertText": {"endOfSegmentLocation": {}, "text": text}
            }]
        });
        let resp = self
            .send_with_refresh(|t| self.client.post(&url).bearer_auth(t).json(&body))
            .await?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            anyhow::bail!("google docs batchUpdate {status}: {text}");
        }
        Ok(serde_json::from_str(&text).unwrap_or(Value::Null))
    }
}
