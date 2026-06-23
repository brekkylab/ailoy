use async_trait::async_trait;
use serde_json::{Value, json};

use crate::vfs::{
    accessor::{GDriveAccessor, GDriveConfig},
    path::VPath,
    resource::{DirEntry, FileKind, FileStat, Resource},
};

const DOC_MIME: &str = "application/vnd.google-apps.document";
const GDOC_SUFFIX: &str = ".gdoc.json";

const GDRIVE_PROMPT: &str = "\
Google Drive (read + GDocs append). Files are listed at the mount root.
Google Docs appear as `<name>.gdoc.json`; cat returns JSON with `documentId` and text.
Append to a Google Doc by writing JSON to the control path:
  echo '{\"document_id\":\"DOC_ID\",\"text\":\"hello\"}' > .cmd/docs-append";

pub struct GDriveResource {
    accessor: GDriveAccessor,
}

impl GDriveResource {
    pub fn new(config: &GDriveConfig) -> anyhow::Result<Self> {
        Ok(Self {
            accessor: GDriveAccessor::new(config)?,
        })
    }

    fn display_name(file: &Value) -> Option<String> {
        let name = file.get("name")?.as_str()?;
        let mime = file.get("mimeType").and_then(|m| m.as_str()).unwrap_or("");
        if mime == DOC_MIME {
            Some(format!("{name}{GDOC_SUFFIX}"))
        } else if mime.starts_with("application/vnd.google-apps.") {
            None
        } else {
            Some(name.to_string())
        }
    }

    /// Resolve a display name to `(file id, is_google_doc, metadata size)`. The
    /// size comes from the Drive listing (no download); `None` for Google Docs,
    /// which carry no byte size in metadata.
    async fn resolve(&self, display: &str) -> anyhow::Result<(String, bool, Option<u64>)> {
        let files = self.accessor.list_files().await?;
        let (base, want_doc) = match display.strip_suffix(GDOC_SUFFIX) {
            Some(b) => (b, true),
            None => (display, false),
        };
        for f in &files {
            let name = f.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let mime = f.get("mimeType").and_then(|m| m.as_str()).unwrap_or("");
            let is_doc = mime == DOC_MIME;
            if name == base && is_doc == want_doc {
                let id = f
                    .get("id")
                    .and_then(|i| i.as_str())
                    .ok_or_else(|| anyhow::anyhow!("file has no id"))?;
                let size = f
                    .get("size")
                    .and_then(|s| s.as_str())
                    .and_then(|s| s.parse::<u64>().ok());
                return Ok((id.to_string(), is_doc, size));
            }
        }
        anyhow::bail!("gdrive file not found: {display}")
    }
}

#[async_trait]
impl Resource for GDriveResource {
    async fn read_bytes(
        &self,
        path: &VPath,
        range: Option<std::ops::Range<u64>>,
    ) -> anyhow::Result<Vec<u8>> {
        let name = path.as_str().trim_start_matches('/');
        if name.is_empty() {
            anyhow::bail!("is a directory: /");
        }
        let (id, is_doc, _size) = self.resolve(name).await?;
        let data = if is_doc {
            let text = self.accessor.export_text(&id).await?;
            serde_json::to_vec_pretty(&json!({
                "documentId": id,
                "text": String::from_utf8_lossy(&text),
            }))?
        } else {
            self.accessor.download(&id).await?
        };
        match range {
            Some(r) => {
                let start = (r.start as usize).min(data.len());
                let end = (r.end as usize).min(data.len());
                Ok(data[start..end].to_vec())
            }
            None => Ok(data),
        }
    }

    async fn write_bytes(&self, path: &VPath, _data: Vec<u8>) -> anyhow::Result<()> {
        anyhow::bail!(
            "gdrive file writes not supported; use .cmd/docs-append (path was {})",
            path.as_str()
        )
    }

    async fn readdir(&self, path: &VPath) -> anyhow::Result<Vec<DirEntry>> {
        if !path.is_root() {
            anyhow::bail!("gdrive only lists the mount root");
        }
        let files = self.accessor.list_files().await?;
        Ok(files
            .iter()
            .filter_map(|f| {
                let name = Self::display_name(f)?;
                let size = f
                    .get("size")
                    .and_then(|s| s.as_str())
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0);
                Some(DirEntry {
                    name,
                    kind: FileKind::File,
                    size,
                })
            })
            .collect())
    }

    async fn stat(&self, path: &VPath) -> anyhow::Result<FileStat> {
        if path.is_root() {
            return Ok(FileStat {
                kind: FileKind::Dir,
                size: 0,
            });
        }
        let name = path.as_str().trim_start_matches('/');
        let (id, is_doc, meta_size) = self.resolve(name).await?;
        // Never download to stat: binary files take their size from Drive
        // metadata; Google Docs (no metadata size) use the cached export.
        let size = match meta_size {
            Some(s) => s,
            None if is_doc => self.accessor.export_text(&id).await?.len() as u64,
            None => 0,
        };
        Ok(FileStat {
            kind: FileKind::File,
            size,
        })
    }

    async fn command(&self, name: &str, body: &[u8]) -> anyhow::Result<Vec<u8>> {
        match name {
            "docs-append" => {
                let v: Value = serde_json::from_slice(body)
                    .map_err(|e| anyhow::anyhow!("docs-append: invalid JSON: {e}"))?;
                let doc_id = v
                    .get("document_id")
                    .and_then(|x| x.as_str())
                    .ok_or_else(|| anyhow::anyhow!("docs-append: missing document_id"))?;
                let text = v
                    .get("text")
                    .and_then(|x| x.as_str())
                    .ok_or_else(|| anyhow::anyhow!("docs-append: missing text"))?;
                let result = self.accessor.docs_append(doc_id, text).await?;
                Ok(serde_json::to_vec(&result)?)
            }
            other => anyhow::bail!("unknown gdrive command: {other}"),
        }
    }

    fn prompt(&self) -> &str {
        GDRIVE_PROMPT
    }
}
