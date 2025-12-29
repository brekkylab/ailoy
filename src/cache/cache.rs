use std::{
    collections::HashMap,
    env::var,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, bail};
use async_stream::try_stream;
use futures::StreamExt;
use tokio::sync::RwLock;
use url::Url;

use super::{
    filesystem,
    manifest::{Manifest, ManifestDirectory},
};
use crate::{
    boxed,
    cache::{ByteSource, CacheContents, CacheEntry, TryFromCache},
    constants::AILOY_VERSION,
    utils::{BoxStream, MaybeSend},
    value::Value,
};

async fn download_attempt(url: &Url) -> anyhow::Result<Vec<u8>> {
    let client = reqwest::Client::builder().build()?;

    let resp = client.get(url.clone()).send().await?;

    if !resp.status().is_success() {
        bail!("HTTP error: {}", resp.status());
    }

    let bytes = resp
        .bytes()
        .await
        .context("reqwest::Response::bytes failed")?;

    Ok(bytes.to_vec())
}

async fn download(url: Url) -> anyhow::Result<Vec<u8>> {
    let max_retries = 3;
    let mut last_error = None;

    for attempt in 0..max_retries {
        match download_attempt(&url).await {
            Ok(bytes) => return Ok(bytes),
            Err(e) => {
                last_error = Some(e);
                if attempt < max_retries - 1 {
                    let delay_ms = 1000 * (attempt as i32 + 1);
                    crate::utils::sleep(delay_ms).await;
                }
            }
        }
    }

    Err(last_error.unwrap())
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
            let default = Url::parse("https://cache.ailoy.co");
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

            // Try to download manifest from remote first
            let bytes = match download(self.get_url(&entry_manifest)).await {
                Ok(data) => {
                    // Save the manifest to local filesystem
                    filesystem::write(&self.path(&entry_manifest), &data, true).await?;
                    data
                }
                Err(_) => {
                    crate::warn!(
                        "Failed to download manifest. Try to read from local filesystem..."
                    );
                    // Fallback to getting from local filesystem
                    match filesystem::read(&self.path(&entry_manifest)).await {
                        Ok(data) => data,
                        Err(_) => {
                            bail!("Failed to get the manifest for this entry.");
                        }
                    }
                }
            };

            let text = std::str::from_utf8(&bytes).context("std::str::from_utf_8 failed")?;
            let value = serde_json::from_str::<ManifestDirectory>(text)
                .context("`serde_json::from_str` failed")?;
            self.manifests
                .write()
                .await
                .insert(entry.dirname().to_string(), value);
        };
        let manifests_lock = self.manifests.read().await;
        let manifest_dir = &manifests_lock.get(entry.dirname()).unwrap();
        match manifest_dir.get_file_manifest(entry.filename(), AILOY_VERSION) {
            Some(manifest) => Ok(manifest),
            None => bail!(
                "The file \"{}\" does not exist in manifest",
                entry.filename()
            ),
        }
    }

    /// Fetch bytes for a logical file using the cache entry acquirement process.
    ///
    /// Steps:
    /// 1. Read the directory manifest to find the content hash for `entry`.
    /// 2. If a local file exists,
    ///   - If `validate_checksum` is enabled, compute its content hash (via [`Manifest::from_u8`]) and compare; if it matches, return the local bytes.
    ///   - Otherwise, just return the local bytes immediately.
    /// 3. Otherwise, download the blob named by the content hash and write it back
    ///    to the logical filename, then return the bytes.
    ///
    /// Errors bubble up from manifest lookup, IO, and network.
    pub async fn get(
        &self,
        entry: impl AsRef<CacheEntry>,
        validate_checksum: Option<bool>,
    ) -> anyhow::Result<Vec<u8>> {
        let entry = entry.as_ref();

        // Get manifest
        let manifest = self.get_manifest(entry).await?;

        // Get local data
        let local_bytes = filesystem::read(&self.path(entry)).await.ok();

        if let Some(local_bytes) = local_bytes {
            // Validating checksum is enabled by default.
            let validate_checksum = validate_checksum.unwrap_or(true);

            if validate_checksum {
                // If validation is enabled, validate sha1 checksum
                let local_manifest = Manifest::from_u8(&local_bytes);
                if local_manifest.sha1() == manifest.sha1() {
                    return Ok(local_bytes);
                }
            } else {
                // If validation is disabled, just return the bytes immediately
                return Ok(local_bytes);
            }
        }

        // Download otherwise
        let remote_entry = CacheEntry::new(entry.dirname(), manifest.sha1());
        let url = self.get_url(&remote_entry);
        let bytes = download(url).await?;
        // Write back
        filesystem::write(&self.path(entry), &bytes, true).await?;
        Ok(bytes)
    }

    pub fn prepare_files<T>(
        &self,
        key: impl Into<String>,
        validate_checksum: Option<bool>,
    ) -> BoxStream<'static, anyhow::Result<(CacheEntry, usize, usize, Vec<u8>)>>
    where
        T: TryFromCache + MaybeSend + 'static,
    {
        let key = key.into();
        let this = self.clone();
        let mut context = HashMap::new();

        boxed!(try_stream! {
            // Claim files to be processed
            let claim = T::claim_files(this.clone(), key.clone(), &mut context).await?;

            // Number of tasks
            let total_task: usize = claim.entries.len();

            // Current processed task
            let mut current_task: usize = 0;

            let tasks = claim.entries.into_iter().map(|entry| {
                let this = this.clone();
                async move {
                    let res = this.get(&entry, validate_checksum).await;
                    (entry, res)
                }
            }).collect::<Vec<_>>();
            let mut futures_strm = futures::stream::iter(tasks).buffer_unordered(10);
            while let Some((entry, res)) = futures_strm.next().await {
                let bytes = res?;
                current_task += 1;
                yield (entry, current_task, total_task, bytes);
            }
        })
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
    /// let mut s = cache.try_create::<MyType>("Qwen/Qwen3-0.6B");
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
        &self,
        key: impl Into<String>,
        context: Option<HashMap<String, Value>>,
        validate_checksum: Option<bool>,
    ) -> BoxStream<'static, anyhow::Result<CacheProgress<T>>>
    where
        T: TryFromCache + MaybeSend + 'static,
    {
        let root = self.root.clone();
        let cache_clone = self.clone();
        let mut strm = self.prepare_files::<T>(key, validate_checksum);
        boxed!(try_stream! {
            let mut pairs: Vec<(CacheEntry, ByteSource)> = Vec::new();
            let mut total_task: usize = 0;

            // Threshold for eager loading: 10MB
            // Files smaller than this (like tokenizer.json, chat-template.j2, configs)
            // are kept in memory for fast access. Larger files (model weights) are
            // kept as lazy references to avoid OOM.
            const EAGER_LOAD_THRESHOLD: usize = 10 * 1024 * 1024; // 10MB

            while let Some(res) = strm.next().await {
                let (entry, current_task, _total_task, bytes) = res?;
                // Number of total tasks: preparing all files + initialization
                total_task = _total_task + 1;

                // Decide whether to eager load or lazy load based on file size
                let source = if bytes.len() < EAGER_LOAD_THRESHOLD {
                    // Small file: keep in memory
                    ByteSource::Eager(bytes)
                } else {
                    // Large file: drop bytes and keep just the path reference
                    // The file is already on disk from prepare_files/Cache::get
                    ByteSource::Lazy(cache_clone.path(&entry))
                };

                pairs.push((entry.clone(), source));
                yield CacheProgress::<T> {
                    comment: format!("{} ready", entry.filename()),
                    current_task,
                    total_task,
                    result: None,
                };
            }

            // Assemble cache contents
            let mut contents = CacheContents {
                root,
                entries: pairs.into_iter().collect(),
            };

            // Final creation
            let context = context.clone().unwrap_or_default();
            let value = T::try_from_contents(&mut contents, &context).await?;
            yield CacheProgress::<T> {
                comment: "Intialized".to_owned(),
                current_task: total_task,
                total_task,
                result: Some(value),
            };
        })
    }

    pub async fn remove(&self, elem: &CacheEntry) -> anyhow::Result<()> {
        filesystem::remove(self.path(elem)).await
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
            .get(&CacheEntry::new("Qwen--Qwen3-0.6B", "tokenizer.json"), None)
            .await
            .unwrap();
        println!("Downloaded {}", bytes.len());
    }
}
