use std::{
    collections::HashMap,
    env::var,
    path::{Path, PathBuf},
    sync::Arc,
};

use tokio::sync::RwLock;
use url::Url;

use crate::cache::{CacheEntry, TryFromCache};

use super::{
    filesystem::{read, write},
    manifest::{Manifest, ManifestDirectory},
};

async fn download(url: Url) -> Result<Vec<u8>, String> {
    let req = reqwest::get(url);
    let resp = req.await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        Err(format!("HTTP error: {}", resp.status()))?;
    }
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("reqwest::Response::bytes failed: {}", e.to_string()))?;
    Ok(bytes.to_vec())
}

#[derive(Debug, Clone)]
pub struct Cache {
    root: PathBuf,
    remote_url: Url,
    manifests: Arc<RwLock<HashMap<String, ManifestDirectory>>>,
}

impl Cache {
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

    pub fn get_root(&self) -> &Path {
        return &self.root;
    }

    pub fn get_remote_url(&self) -> &Url {
        return &self.remote_url;
    }

    pub fn get_path(&self, elem: impl AsRef<CacheEntry>) -> PathBuf {
        self.root
            .join(&elem.as_ref().dirname())
            .join(&elem.as_ref().filename())
    }

    pub fn get_url(&self, elem: impl AsRef<CacheEntry>) -> Url {
        self.remote_url
            .join(&format!(
                "{}/{}",
                elem.as_ref().dirname(),
                elem.as_ref().filename()
            ))
            .unwrap()
    }

    async fn get_manifest(&self, elem: &CacheEntry) -> Result<Manifest, String> {
        if !self.manifests.read().await.contains_key(elem.dirname()) {
            let elem_manifest = CacheEntry::new(elem.dirname(), "_manifest.json");
            let bytes = download(self.get_url(&elem_manifest)).await?;
            let text = std::str::from_utf8(&bytes)
                .map_err(|e| format!("std::str::from_utf_8 failed: {}", e.to_string()))?;
            let value = serde_json::from_str::<ManifestDirectory>(text)
                .map_err(|e| format!("`serde_json::from_str` failed: {}", e.to_string()))?;
            self.manifests
                .write()
                .await
                .insert(elem.dirname().to_string(), value);
        };
        let manifests_lock = self.manifests.read().await;
        let files = &manifests_lock.get(elem.dirname()).unwrap().files;
        if files.contains_key(elem.filename()) {
            Ok(files.get(elem.filename()).unwrap().to_owned())
        } else {
            Err(format!(
                "The file {} not exists in manifest",
                elem.filename()
            ))
        }
    }

    pub async fn get(&self, elem: impl AsRef<CacheEntry>) -> Result<Vec<u8>, String> {
        let elem = elem.as_ref();

        // Get manifest
        let manifest = self.get_manifest(elem).await?;

        // Get local
        let (local_manifest, local_bytes) = match read(&self.get_path(elem)).await {
            Ok(v) => (Some(Manifest::from_u8(&v)), Some(v)),
            Err(_) => (None, None),
        };

        // Return local if sha matches, or download
        if local_manifest.is_some() && local_manifest.unwrap().sha1() == manifest.sha1() {
            Ok(local_bytes.unwrap())
        } else {
            let remote_elem = CacheEntry::new(elem.dirname(), manifest.sha1());
            let url = self.get_url(&remote_elem);
            let bytes = download(url).await?;
            // Write back
            write(&self.get_path(elem), &bytes).await?;
            Ok(bytes)
        }
    }

    pub async fn try_create<T: TryFromCache>(&self, key: impl AsRef<str>) -> Result<T, String> {
        use futures::future::join_all;

        let key = key.as_ref();
        let files = T::claim_files(self.clone(), key.to_owned()).await?;
        let futures = files.into_iter().map(|elem| {
            let this = self.clone();
            async move {
                let bytes = this.get(&elem).await.map(|v| (elem, v))?;
                Ok::<_, String>(bytes)
            }
        });
        let file_and_bytes: Vec<(CacheEntry, Vec<u8>)> = join_all(futures)
            .await
            .into_iter()
            .collect::<Result<_, _>>()?;
        T::try_from_files(self, file_and_bytes)
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr as _;

    use super::*;

    #[tokio::test]
    async fn prepare_files() {
        let src_dir = PathBuf::from_str("/Users/ijaehwan/.cache/ailoy/Qwen--Qwen3-0.6B").unwrap();
        let dst_dir =
            PathBuf::from_str("/Users/ijaehwan/Workspace/ailoy/out/Qwen--Qwen3-0.6B").unwrap();
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
