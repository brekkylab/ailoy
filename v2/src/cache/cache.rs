use std::{
    collections::HashMap,
    env::var,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, bail};
use async_stream::try_stream;
use futures::{Stream, StreamExt as _, stream::FuturesUnordered};
use tokio::sync::RwLock;
use url::Url;

use super::{
    filesystem::{read, write},
    manifest::{Manifest, ManifestDirectory},
};
use crate::{
    cache::{CacheContents, CacheEntry, TryFromCache},
    utils::MaybeSend,
};

async fn download(url: Url) -> anyhow::Result<Vec<u8>> {
    let req = reqwest::get(url);
    let resp = req.await?;
    if !resp.status().is_success() {
        bail!("HTTP error: {}", resp.status());
    }
    let bytes = resp
        .bytes()
        .await
        .context("reqwest::Response::bytes failed")?;
    Ok(bytes.to_vec())
}

/// [`super::FromCache`] or [`super::TryFromCache`] results with it's progress.
///
/// This method works similarly to [`Cache::try_create`], but instead of waiting
/// until all required files are fetched and the object is initialized, it yields
/// incremental progress events as an asynchronous stream.
///
/// # Fields
/// - `comment`: Human-readable description of the current step (e.g., a file name
///   that finished downloading, or `"Initialized"` at the end).
/// - `current_task`: Number of completed steps so far.
/// - `total_task`: Total number of steps (all downloads + final initialization).
/// - `result`: `Some(T)` **only** on the final event; `None` otherwise.
///
/// # Guarantees
/// - The result will be provided(`result.is_some() == true`) if and only if on the final event(`current_task == total_task`).
#[derive(Debug)]
pub struct CacheProgress<T> {
    pub comment: String,
    pub current_task: usize,
    pub total_task: usize,
    pub result: Option<T>,
}

/// A content-addressed, remote-backed cache for Ailoy assets.
///
/// # Terminology
///
/// ## Cache Root
///
/// The **cache root** is the local directory used by Ailoy to store cached files.
/// By default, it is located at:
/// - **Linux / macOS**: `${HOME}/.cache/ailoy`
/// - **Windows**: `%LOCALAPPDATA%\ailoy`
///
/// You can override this location via the `AILOY_CACHE_ROOT` environment variable.
/// The active root directory can be retrieved with [`Cache::get_root`].
///
/// ## Remote URL
///
/// On first run, the local cache is empty, and files must be downloaded from a remote server
/// following the Ailoy cache protocol.
///
/// You can override the default remote URL via the `AILOY_CACHE_REMOTE_URL` environment variable.
/// The active remote URL can be retrieved with [`Cache::get_remote_url`].
///
/// ## Entry
///
/// A **cache entry** is represented by a [`CacheEntry`], which consists of two parts:
/// - **dirname** — the directory containing the file
/// - **filename** — the name of the file
///
/// Each `dirname` must have a corresponding manifest file (`_manifest.json`).
/// This manifest maps logical filenames to their current content hash (SHA-1)
/// and determines whether the cache can use an existing local file or must
/// fetch a new copy from the remote.
///
/// # Workflow
///
/// To resolve a file, the cache follows these steps:
///
/// 1. Load the per-directory `_manifest.json` from the remote.
/// 2. Check for the local file at `<root>/<dirname>/<filename>`.
/// 3. If the local file’s SHA-1 matches the manifest entry, return the local bytes.
/// 4. Otherwise, download the file by its content hash from the remote,
///    save it locally, and return the new bytes.
///
///
/// See the module docs and the method docs for end-to-end behavior. This type
/// is `Clone` and intended to be shared across async tasks. Manifests are
/// memoized per directory for the lifetime of a `Cache` instance.
#[derive(Debug, Clone)]
pub struct Cache {
    root: PathBuf,
    remote_url: Url,
    manifests: Arc<RwLock<HashMap<String, ManifestDirectory>>>,
}

impl Cache {
    /// Create a new cache instance using environment defaults.
    /// You can control the root or remote url with environment variable:
    /// - root: AILOY_CACHE_ROOT
    /// - remote url: AILOY_CACHE_REMOTE_URL
    /// Falls back to a built-in default if unset or invalid.
    pub fn new() -> Self {
        let root = match var("AILOY_CACHE_ROOT") {
            Ok(env_path) => PathBuf::from(env_path),
            Err(_) => {
                #[cfg(any(target_family = "unix"))]
                {
                    PathBuf::from(var("HOME").unwrap())
                        .join(".cache")
                        .join("ailoy")
                }
                #[cfg(target_family = "windows")]
                {
                    PathBuf::from(var("LOCALAPPDATA").unwrap()).join("ailoy")
                }
                #[cfg(target_family = "wasm")]
                {
                    PathBuf::from("/").join("ailoy")
                }
            }
        };
        let remote_url = {
            let default = Url::parse("https://pub-9bacbd05eeb6446c9bf8285fe54c9f9e.r2.dev");
            if let Ok(env_value) = var("AILOY_CACHE_REMOTE_URL") {
                if let Ok(value) = Url::parse(&env_value) {
                    value
                } else {
                    eprintln!("Invalid AILOY_CACHE_REMOTE_URL value: {}", env_value);
                    default.unwrap()
                }
            } else {
                default.unwrap()
            }
        };
        Self {
            root,
            remote_url,
            manifests: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Return the local cache root directory.
    pub fn root(&self) -> &Path {
        return &self.root;
    }

    /// Return the configured remote base URL.
    pub fn remote_url(&self) -> &Url {
        return &self.remote_url;
    }

    /// Compute the local on-disk path for a logical entry.
    ///
    /// This is `<root>/<dirname>/<filename>`.
    pub fn path(&self, elem: impl AsRef<CacheEntry>) -> PathBuf {
        self.root
            .join(&elem.as_ref().dirname())
            .join(&elem.as_ref().filename())
    }

    /// Compute the remote URL for a logical entry.
    ///
    /// This is `<remote>/<dirname>/<filename>`. Note that for actual downloads
    /// of file content the filename is usually a content hash taken from the
    /// manifest; see [`Cache::get`].
    pub fn get_url(&self, elem: impl AsRef<CacheEntry>) -> Url {
        self.remote_url
            .join(&format!(
                "{}/{}",
                elem.as_ref().dirname(),
                elem.as_ref().filename()
            ))
            .unwrap()
    }

    async fn get_manifest(&self, entry: &CacheEntry) -> anyhow::Result<Manifest> {
        if !self.manifests.read().await.contains_key(entry.dirname()) {
            let entry_manifest = CacheEntry::new(entry.dirname(), "_manifest.json");
            let bytes = download(self.get_url(&entry_manifest)).await?;
            let text = std::str::from_utf8(&bytes).context("std::str::from_utf_8 failed")?;
            let value = serde_json::from_str::<ManifestDirectory>(text)
                .context("`serde_json::from_str` failed")?;
            self.manifests
                .write()
                .await
                .insert(entry.dirname().to_string(), value);
        };
        let manifests_lock = self.manifests.read().await;
        let files = &manifests_lock.get(entry.dirname()).unwrap().files;
        if files.contains_key(entry.filename()) {
            Ok(files.get(entry.filename()).unwrap().to_owned())
        } else {
            bail!("The file {} not exists in manifest", entry.filename())
        }
    }

    /// Fetch bytes for a logical file using the cache entry acquirement process.
    ///
    /// Steps:
    /// 1. Read the directory manifest to find the content hash for `entry`.
    /// 2. If a local file exists, compute its content hash (via [`Manifest::from_u8`])
    ///    and compare; if it matches, return the local bytes.
    /// 3. Otherwise, download the blob named by the content hash and write it back
    ///    to the logical filename, then return the bytes.
    ///
    /// Errors bubble up from manifest lookup, IO, and network.
    pub async fn get(&self, entry: impl AsRef<CacheEntry>) -> anyhow::Result<Vec<u8>> {
        let entry = entry.as_ref();

        // Get manifest
        let manifest = self.get_manifest(entry).await?;

        // Get local
        let (local_manifest, local_bytes) = match read(&self.path(entry)).await {
            Ok(v) => (Some(Manifest::from_u8(&v)), Some(v)),
            Err(_) => (None, None),
        };

        // Return local if sha matches, or download
        if local_manifest.is_some() && local_manifest.unwrap().sha1() == manifest.sha1() {
            Ok(local_bytes.unwrap())
        } else {
            let remote_entry = CacheEntry::new(entry.dirname(), manifest.sha1());
            let url = self.get_url(&remote_entry);
            let bytes = download(url).await?;
            // Write back
            write(&self.path(entry), &bytes, true).await?;
            Ok(bytes)
        }
    }

    /// Builds a typed value from the cache.
    ///
    /// Note that it returns the value and it's **streaming progress updates**.
    /// Initialization can be slow because it may download files and initialize hardwares.
    /// Instead of returning the result immediately, this method yields a stream of
    /// [`CacheProgress`] events. These events provide real-time feedback so users
    /// don’t assume the process has stalled.
    ///
    /// See [`CacheProgress`] for details.
    ///
    /// # Type bounds
    /// `T: TryFromCache + Send + 'static`
    ///
    /// # Progress semantics
    /// - The final event satisfies `current_task == total_task` and `result.is_some()`.
    /// - All preceding events have `result == None`.
    ///
    /// # Example
    /// ```rust,ignore
    /// use futures::StreamExt;
    ///
    /// let cache = Cache::new();
    /// let mut s = cache.try_create_stream::<MyType>("Qwen/Qwen3-0.6B");
    /// while let Some(evt) = s.next().await {
    ///     let evt = evt?;
    ///     println!("[{}/{}] {}", evt.current_task(), evt.total_task(), evt.comment());
    ///     if let Some(obj) = evt.take() {
    ///         println!("Done: {:?}", obj);
    ///     }
    /// }
    /// # Ok::<(), String>(())
    /// ```
    pub fn try_create<T>(
        self,
        key: impl Into<String>,
    ) -> impl Stream<Item = anyhow::Result<CacheProgress<T>>> + 'static
    where
        T: TryFromCache + MaybeSend + 'static,
    {
        let key = key.into();
        let this = self.clone();

        try_stream! {
            // Claim files to be processed
            let claim = T::claim_files(this.clone(), key.clone()).await?;

            // Number of tasks => downloading all files + initializing T
            let total_task: usize = claim.entries.len() + 1;

            // Current processed task
            let mut current_task = 0;

            // Main tasks
            let mut tasks: FuturesUnordered<_> = claim.entries
                .into_iter()
                .map(|entry| {
                    let this = this.clone();
                    async move {
                        let res = this.get(&entry).await;
                        (entry, res)
                    }
                })
                .collect();

            let mut pairs: Vec<(CacheEntry, Vec<u8>)> = Vec::with_capacity(total_task);
            while let Some((entry, res)) = tasks.next().await {
                let bytes = res?;
                current_task += 1;
                pairs.push((entry.clone(), bytes));
                yield CacheProgress::<T> {
                    comment: format!("Downloaded {}", entry.filename()),
                    current_task,
                    total_task,
                    result: None,
                };
            }

            // Assemble cache contents
            let contents = CacheContents {
                root: self.root.clone(),
                entries: pairs.into_iter().collect(),
                ctx: claim.ctx,
            };

            // Final creation
            let value = T::try_from_contents(contents).await?;
            current_task += 1;
            yield CacheProgress::<T> {
                comment: "Intialized".to_owned(),
                current_task,
                total_task,
                result: Some(value),
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr as _;

    use super::*;

    #[tokio::test]
    async fn prepare_files() {
        let src_dir = PathBuf::from_str(
            "/Users/ijaehwan/.cache/ailoy/Qwen--Qwen3-0.6B--wasm32-unknown-unknown--webgpu",
        )
        .unwrap();
        let dst_dir = PathBuf::from_str(
            "/Users/ijaehwan/Workspace/ailoy/out/Qwen--Qwen3-0.6B--wasm32-unknown-unknown--webgpu",
        )
        .unwrap();
        if dst_dir.exists() {
            std::fs::remove_dir_all(&dst_dir).unwrap();
        }
        std::fs::create_dir(&dst_dir).unwrap();
        let mut manifests = ManifestDirectory::new();

        for entry in std::fs::read_dir(src_dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();

            if path.is_file() {
                if path.file_name().unwrap() == "_manifest.json" {
                    continue;
                }
                let content = std::fs::read(&path).unwrap();
                let manifest = Manifest::from_u8(&content);
                std::fs::write(dst_dir.join(manifest.sha1()), content).unwrap();
                manifests.insert_file(
                    path.file_name().unwrap().to_str().unwrap().to_owned(),
                    manifest,
                );
            }
        }
        let contents = serde_json::to_string(&manifests).unwrap();
        std::fs::write(dst_dir.join("_manifest.json"), contents).unwrap();
    }

    #[tokio::test]
    async fn test1() {
        let cache = Cache::new();
        let bytes = cache
            .get(&CacheEntry::new("Qwen--Qwen3-0.6B", "tokenizer.json"))
            .await
            .unwrap();
        println!("Downloaded {}", bytes.len());
    }
}
