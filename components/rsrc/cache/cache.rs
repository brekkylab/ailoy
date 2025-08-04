use std::{collections::HashMap, env::var, sync::Arc};

use bytes::{Bytes, BytesMut};
use futures::stream::StreamExt;
use url::Url;

use crate::cache::{
    compat::RwLock,
    manifest::{Manifest, ManifestDirectory},
};

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

fn build_url(url: Url, dir: &str, name: &str) -> Result<Url, String> {
    let path = format!("{}/{}", dir, name);
    url.join(&path)
        .map_err(|_| format!("Invalid URL: {} path: {}", url, path))
}

fn download(url: Url) -> std::pin::Pin<Box<dyn futures::Stream<Item = Result<Bytes, String>>>> {
    let req = reqwest::get(url);

    Box::pin(async_stream::try_stream! {
        let resp = req.await.map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            Err(format!("HTTP error: {}", resp.status()))?;
        }
        let mut strm = resp.bytes_stream();
        while let Some(chunk) = strm.next().await {
            yield chunk.map_err(|e| e.to_string())?;
        }
    })
}

pub struct Cache {
    url: Url,
    fs_cache: super::fs::FSCache,
    manifests: Arc<RwLock<HashMap<String, ManifestDirectory>>>,
}

impl Cache {
    pub fn new() -> Self {
        Self {
            url: get_remote_cache_url(),
            fs_cache: super::fs::FSCache::new(),
            manifests: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn get_manifest(
        &self,
        dir: &str,
        name: &str,
    ) -> std::pin::Pin<Box<dyn futures::Future<Output = Result<Manifest, String>>>> {
        let fs_cache = self.fs_cache.clone();
        let dir = dir.to_owned();
        let name = name.to_owned();
        let manifests = self.manifests.clone();
        let url = build_url(self.url.clone(), &dir, "_manifest.json");
        Box::pin(async move {
            if !manifests.read().await.contains_key(&dir) {
                let mut buf = bytes::BytesMut::new();
                let mut strm_download = download(url?);
                while let Some(chunk) = strm_download.next().await {
                    let chunk = chunk.map_err(|e| format!("Failed to download manifest: {}", e))?;
                    buf.extend_from_slice(&chunk);
                }
                let str_ = String::from_utf8_lossy(buf.as_ref());
                let elem: ManifestDirectory = serde_json::from_str(&str_)
                    .map_err(|e| format!("Unable to parse manifest file: {}", e.to_string()))?;
                manifests.write().await.insert(dir.to_owned(), elem);
                fs_cache.put_sync(&dir, "_manifest.json", buf.freeze())?;
            };
            let manifests_lock = manifests.read().await;
            let files = &manifests_lock.get(&dir).unwrap().files;
            if files.contains_key(&name) {
                Ok(files.get(&name).unwrap().to_owned())
            } else {
                Err(format!("The file {} not exists in manifest", name))
            }
        })
    }

    pub fn get(
        &self,
        dir: &str,
        name: &str,
    ) -> std::pin::Pin<Box<dyn futures::Stream<Item = Result<Bytes, String>>>> {
        let fs_cache = self.fs_cache.clone();
        let local = match self.fs_cache.get_sync(dir, name) {
            Ok(v) => Some(v),
            Err(_) => None,
        };
        let local_manifest = local.clone().map(|v| Manifest::from_bytes(v));
        let fut_manifest = self.get_manifest(dir, name);
        let base_url = self.url.clone();
        let dir = dir.to_owned();
        let name = name.to_owned();

        Box::pin(async_stream::try_stream! {
            let manifest = fut_manifest.await?;
            let should_download = if let Some(local_manifest) = local_manifest {
                if manifest.sha1() != local_manifest.sha1() {
                    true
                } else {
                    false
                }
            } else {
                true
            };
            if should_download {
                let url = build_url(base_url, &dir, manifest.sha1())?;
                let mut strm_download = download(url);
                let mut buf = BytesMut::new();
                while let Some(chunk) = strm_download.next().await {
                    let chunk2 = chunk.clone();
                    buf.extend_from_slice(&chunk2?);
                    yield chunk?;
                }
                fs_cache.put_sync(&dir, &name, buf.freeze())?;
            } else {
                yield local.unwrap();
            }
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[cfg(test)]
mod tests {
    use super::*;
    use bytes::BytesMut;
    use futures::StreamExt;
    use indicatif::{ProgressBar, ProgressStyle};

    #[tokio::test]
    async fn test1() {
        let cache = Cache::new();
        let mut buf = BytesMut::new();
        let mut strm = cache.get("Qwen--Qwen3-0.6B", "tokenizer.json");

        let total_size = 11422654;
        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::with_template("{bar:40.cyan/blue} {bytes}/{total_bytes} ({eta})")
                .unwrap()
                .progress_chars("##-"),
        );

        while let Some(chunk) = strm.next().await {
            match chunk {
                Ok(chunk) => {
                    pb.inc(chunk.len() as u64);
                    buf.extend_from_slice(&chunk);
                }
                Err(e) => {
                    pb.abandon_with_message("Download failed");
                    panic!("{:?}", e);
                }
            }
        }

        pb.finish_with_message("Download complete");
    }
}
