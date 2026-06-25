use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::vfs::{
    accessor::{GDriveAccessor, GDriveConfig},
    path::VPath,
    resource::{DirEntry, FileKind, FileStat, Resource},
};

const FOLDER_MIME: &str = "application/vnd.google-apps.folder";
const DOC_MIME: &str = "application/vnd.google-apps.document";
const SHEET_MIME: &str = "application/vnd.google-apps.spreadsheet";
const SLIDE_MIME: &str = "application/vnd.google-apps.presentation";
const DIR_TTL: Duration = Duration::from_secs(10);

const GDRIVE_PROMPT: &str = "\
Google Drive (read + GDocs append). Mirrors the Drive folder hierarchy:
folders are directories you can descend into; Google Docs/Sheets/Slides appear
as `<name>.gdoc.json` / `.gsheet.json` / `.gslide.json` (cat returns JSON with
the file's `documentId` and exported text); other files are downloaded as-is.
Append to a Google Doc by writing JSON to the control path:
  echo '{\"document_id\":\"DOC_ID\",\"text\":\"hello\"}' > .cmd/docs-append";

#[derive(Clone, Copy, PartialEq, Eq)]
enum GKind {
    Folder,
    SharedDrive,
    Doc,
    Sheet,
    Slide,
    File,
}

impl GKind {
    fn is_dir(self) -> bool {
        matches!(self, GKind::Folder | GKind::SharedDrive)
    }
    fn is_workspace_doc(self) -> bool {
        matches!(self, GKind::Doc | GKind::Sheet | GKind::Slide)
    }
}

#[derive(Clone)]
struct Child {
    vfs_name: String,
    id: String,
    drive_id: Option<String>,
    size: Option<u64>,
    kind: GKind,
}

pub struct GDriveResource {
    accessor: GDriveAccessor,
    /// Per-directory listing cache (folder path -> children), short TTL. Lets a
    /// path resolution / `ls` reuse a parent listing instead of re-querying.
    dir_cache: Mutex<HashMap<String, (Instant, Vec<Child>)>>,
}

impl GDriveResource {
    pub fn new(config: &GDriveConfig) -> anyhow::Result<Self> {
        Ok(Self {
            accessor: GDriveAccessor::new(config)?,
            dir_cache: Mutex::new(HashMap::new()),
        })
    }

    /// List a directory's immediate children (cached). At the root, shared
    /// drives are appended as top-level directories.
    async fn list_dir(&self, folder: &str) -> anyhow::Result<Vec<Child>> {
        {
            let cache = self.dir_cache.lock().await;
            if let Some((at, children)) = cache.get(folder)
                && at.elapsed() < DIR_TTL
            {
                return Ok(children.clone());
            }
        }
        let (folder_id, drive_id) = self.folder_id_of(folder).await?;
        let files = self
            .accessor
            .list_files(&folder_id, drive_id.as_deref())
            .await?;
        let mut children: Vec<Child> = files.iter().filter_map(child_from_file).collect();

        if folder == "/" {
            // Best-effort: enumerate shared drives as extra top-level dirs.
            if let Ok(drives) = self.accessor.list_shared_drives().await {
                let mut existing: HashSet<String> =
                    children.iter().map(|c| c.vfs_name.clone()).collect();
                for d in &drives {
                    if let (Some(id), Some(name)) = (
                        d.get("id").and_then(|x| x.as_str()),
                        d.get("name").and_then(|x| x.as_str()),
                    ) {
                        let vfs_name = unique_name(name, &existing);
                        existing.insert(vfs_name.clone());
                        children.push(Child {
                            vfs_name,
                            id: id.to_string(),
                            drive_id: Some(id.to_string()),
                            size: None,
                            kind: GKind::SharedDrive,
                        });
                    }
                }
            }
        }

        self.dir_cache
            .lock()
            .await
            .insert(folder.to_string(), (Instant::now(), children.clone()));
        Ok(children)
    }

    /// Resolve a folder path to its Drive id (+ drive id), walking from root.
    async fn folder_id_of(&self, folder: &str) -> anyhow::Result<(String, Option<String>)> {
        if folder == "/" {
            return Ok(("root".to_string(), None));
        }
        let (parent, name) = split_last(folder);
        let children = Box::pin(self.list_dir(&parent)).await?;
        let entry = children
            .iter()
            .find(|c| c.vfs_name == name && c.kind.is_dir())
            .ok_or_else(|| anyhow::anyhow!("gdrive folder not found: {folder}"))?;
        Ok((entry.id.clone(), entry.drive_id.clone()))
    }

    /// Resolve any path (file or folder) to its child entry via its parent dir.
    async fn resolve(&self, path: &str) -> anyhow::Result<Child> {
        let (parent, name) = split_last(path);
        let children = self.list_dir(&parent).await?;
        children
            .into_iter()
            .find(|c| c.vfs_name == name)
            .ok_or_else(|| anyhow::anyhow!("gdrive path not found: {path}"))
    }

    /// The exact bytes a workspace doc (Doc/Sheet/Slide) reads as: a JSON
    /// envelope with the file id and its exported text. `read` and `stat` both
    /// go through this so the reported size always matches the content.
    async fn workspace_doc_bytes(&self, child: &Child) -> anyhow::Result<Vec<u8>> {
        let text = self
            .accessor
            .export_text(&child.id, export_mime(child.kind))
            .await?;
        Ok(serde_json::to_vec_pretty(&json!({
            "documentId": child.id,
            "text": String::from_utf8_lossy(&text),
        }))?)
    }

    async fn doc_size(&self, child: &Child) -> u64 {
        match child.kind {
            k if k.is_workspace_doc() => self
                .workspace_doc_bytes(child)
                .await
                .map(|d| d.len() as u64)
                .unwrap_or(0),
            _ => child.size.unwrap_or(0),
        }
    }
}

#[async_trait]
impl Resource for GDriveResource {
    async fn read_bytes(
        &self,
        path: &VPath,
        range: Option<std::ops::Range<u64>>,
    ) -> anyhow::Result<Vec<u8>> {
        if path.is_root() {
            anyhow::bail!("is a directory: /");
        }
        let child = self.resolve(path.as_str()).await?;
        let data = match child.kind {
            k if k.is_dir() => anyhow::bail!("is a directory: {}", path.as_str()),
            k if k.is_workspace_doc() => self.workspace_doc_bytes(&child).await?,
            _ => self.accessor.download(&child.id).await?,
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
        let children = self.list_dir(path.as_str()).await?;
        Ok(children
            .into_iter()
            .map(|c| DirEntry {
                name: c.vfs_name,
                kind: if c.kind.is_dir() {
                    FileKind::Dir
                } else {
                    FileKind::File
                },
                size: c.size.unwrap_or(0),
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
        let child = self.resolve(path.as_str()).await?;
        if child.kind.is_dir() {
            return Ok(FileStat {
                kind: FileKind::Dir,
                size: 0,
            });
        }
        Ok(FileStat {
            kind: FileKind::File,
            size: self.doc_size(&child).await,
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

/// Export MIME for a workspace doc: Sheets must use `text/csv` (Google rejects
/// `text/plain` for spreadsheets); Docs and Slides use `text/plain`.
fn export_mime(kind: GKind) -> &'static str {
    match kind {
        GKind::Sheet => "text/csv",
        _ => "text/plain",
    }
}

fn child_from_file(f: &Value) -> Option<Child> {
    let name = f.get("name")?.as_str()?;
    let id = f.get("id")?.as_str()?;
    let mime = f.get("mimeType").and_then(|m| m.as_str()).unwrap_or("");
    let size = f
        .get("size")
        .and_then(|s| s.as_str())
        .and_then(|s| s.parse::<u64>().ok())
        .or_else(|| {
            f.get("quotaBytesUsed")
                .and_then(|s| s.as_str())
                .and_then(|s| s.parse::<u64>().ok())
        });
    let drive_id = f
        .get("driveId")
        .and_then(|d| d.as_str())
        .map(|s| s.to_string());
    let (vfs_name, kind) = match mime {
        FOLDER_MIME => (name.to_string(), GKind::Folder),
        DOC_MIME => (format!("{name}.gdoc.json"), GKind::Doc),
        SHEET_MIME => (format!("{name}.gsheet.json"), GKind::Sheet),
        SLIDE_MIME => (format!("{name}.gslide.json"), GKind::Slide),
        _ => (name.to_string(), GKind::File),
    };
    Some(Child {
        vfs_name,
        id: id.to_string(),
        drive_id,
        // Workspace docs read as a JSON envelope of exported text, so their
        // Drive `quotaBytesUsed` is not the byte length `read` produces. Leave
        // the size unknown so `stat` computes the real content length instead
        // of caching a mismatching value (which would truncate/zero-fill cat).
        size: if kind.is_workspace_doc() { None } else { size },
        kind,
    })
}

/// Split a path into `(parent_dir, last_segment)`. `/a/b` -> (`/a`, `b`);
/// `/a` -> (`/`, `a`).
fn split_last(path: &str) -> (String, String) {
    let p = path.trim_end_matches('/');
    match p.rsplit_once('/') {
        Some((parent, name)) => {
            let parent = if parent.is_empty() {
                "/".to_string()
            } else {
                parent.to_string()
            };
            (parent, name.to_string())
        }
        None => ("/".to_string(), p.to_string()),
    }
}

/// Disambiguate a shared-drive name that collides with a My Drive entry.
fn unique_name(name: &str, existing: &HashSet<String>) -> String {
    if !existing.contains(name) {
        return name.to_string();
    }
    let mut candidate = format!("{name} [Shared Drive]");
    let mut suffix = 2;
    while existing.contains(&candidate) {
        candidate = format!("{name} [Shared Drive {suffix}]");
        suffix += 1;
    }
    candidate
}
