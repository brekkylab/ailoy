use std::{collections::HashMap, env::var, path::PathBuf, str::FromStr, sync::Arc};

use tokio::sync::RwLock;
use url::Url;

use crate::compat::filesystem::{read, write};

use super::manifest::{Manifest, ManifestDirectory};

fn get_cache_root() -> PathBuf {
    match var("AILOY_CACHE_ROOT") {
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
    }
}

fn get_remote_cache_url() -> Url {
    if let Ok(env_value) = var("AILOY_REMOTE_CACHE_URL") {
        if let Ok(value) = Url::parse(&env_value) {
            return value;
        } else {
            panic!("Invalid AILOY_REMOTE_CACHE_URL value: {}", env_value)
        }
    };
    Url::parse("https://pub-9bacbd05eeb6446c9bf8285fe54c9f9e.r2.dev").unwrap()
}

fn build_url(url: &Url, dir: &str, name: &str) -> Result<Url, String> {
    let path = format!("{}/{}", dir, name);
    url.join(&path)
        .map_err(|_| format!("Invalid URL: {} path: {}", url, path))
}

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
    base_url: Url,
    manifests: Arc<RwLock<HashMap<String, ManifestDirectory>>>,
}

impl Cache {
    pub fn new() -> Self {
        Self {
            base_url: get_remote_cache_url(),
            manifests: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    async fn get_manifest(
        &self,
        dir: impl AsRef<str>,
        name: impl AsRef<str>,
    ) -> Result<Manifest, String> {
        let dir = dir.as_ref();
        let name = name.as_ref();

        if !self.manifests.read().await.contains_key(dir) {
            let url = build_url(&self.base_url, dir, "_manifest.json")?;
            let bytes = download(url).await?;
            let text = String::from_utf8_lossy(&bytes);
            let elem: ManifestDirectory = serde_json::from_str(&text)
                .map_err(|e| format!("`serde_json::from_str` failed: {}", e.to_string()))?;
            self.manifests.write().await.insert(dir.to_owned(), elem);
        };
        let manifests_lock = self.manifests.read().await;
        let files = &manifests_lock.get(dir).unwrap().files;
        if files.contains_key(name) {
            Ok(files.get(name).unwrap().to_owned())
        } else {
            Err(format!("The file {} not exists in manifest", name))
        }
    }

    pub async fn get(
        &self,
        dir: impl AsRef<str>,
        name: impl AsRef<str>,
    ) -> Result<Vec<u8>, String> {
        let dir = dir.as_ref();
        let name = name.as_ref();

        // Get manifest
        let manifest = self.get_manifest(dir, name).await?;

        // Get local
        let (local_manifest, local_bytes) =
            match read(&PathBuf::from_str(dir).unwrap().join(name)).await {
                Ok(v) => (Some(Manifest::from_u8(&v)), Some(v)),
                Err(_) => (None, None),
            };

        // Return local if sha matches, or download
        if local_manifest.is_some() && local_manifest.unwrap().sha1() == manifest.sha1() {
            Ok(local_bytes.unwrap())
        } else {
            let url = build_url(&self.base_url, dir, manifest.sha1())?;
            let bytes = download(url).await?;
            // Write back
            write(get_cache_root().join(dir).join(name), &bytes).await?;
            Ok(bytes)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test1() {
        let cache = Cache::new();
        let bytes = cache
            .get("Qwen--Qwen3-0.6B", "tokenizer.json")
            .await
            .unwrap();
        println!("Downloaded {}", bytes.len());
    }
}
