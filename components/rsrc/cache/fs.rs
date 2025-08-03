use std::{env::var, path::PathBuf};

use bytes::Bytes;
use sha1::Digest;

use crate::cache::manifest::Manifest;

fn get_default_cache_root() -> PathBuf {
    match var("AILOY_CACHE_ROOT") {
        Ok(env_path) => PathBuf::from(env_path),
        Err(_) => {
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            {
                PathBuf::from(var("HOME").unwrap())
                    .join(".cache")
                    .join("ailoy")
            }
            #[cfg(target_os = "windows")]
            {
                PathBuf::from(var("LOCALAPPDATA").unwrap()).join("ailoy")
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct FSCache {
    root: std::path::PathBuf,
}

impl FSCache {
    pub fn new() -> FSCache {
        FSCache {
            root: get_default_cache_root(),
        }
    }

    pub fn from_root<T: Into<std::path::PathBuf>>(root: T) -> FSCache {
        FSCache { root: root.into() }
    }

    pub fn exists(&self, dir: &str, name: &str) -> bool {
        self.root.join(dir).join(name).exists()
    }

    pub fn get(
        &self,
        dir: &str,
        name: &str,
    ) -> std::pin::Pin<Box<dyn futures::Stream<Item = Result<bytes::Bytes, String>> + Send>> {
        let result = self.get_sync(dir, name);
        Box::pin(futures::stream::once(async move { result }))
    }

    pub fn get_sync(&self, dir: &str, name: &str) -> Result<bytes::Bytes, String> {
        let mut fhandle = match std::fs::File::open(&self.root.join(format!("{}/{}", dir, name))) {
            Ok(v) => v,
            Err(err) => match err.kind() {
                std::io::ErrorKind::NotFound => return Err("Not found".to_owned()),
                std::io::ErrorKind::PermissionDenied => {
                    return Err("Permisson denied".to_owned());
                }
                _ => {
                    return Err("Internal error".to_owned());
                }
            },
        };
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut fhandle, &mut buf).unwrap();
        Ok(bytes::Bytes::from(buf))
    }

    pub fn put_sync(&self, dir: &str, name: &str, buf: Bytes) -> Result<(), String> {
        let path = self.root.join(dir).join(name);

        // 상위 디렉토리가 없으면 생성
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "Failed to create directory {}: {}",
                    parent.display(),
                    e.to_string()
                )
            })?;
        }

        // 파일 쓰기
        match std::fs::File::create(&path) {
            Ok(mut f) => {
                std::io::Write::write_all(&mut f, &buf).map_err(|e| {
                    format!("Failed to write file {}: {}", path.display(), e.to_string())
                })?;
                Ok(())
            }
            Err(e) => {
                return Err(format!("Failed to create file {}: {}", path.display(), e));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use sha1::Digest;

    use super::*;

    #[tokio::test]
    async fn test1() {
        let mut cache =
            FSCache::from_root("/Users/ijaehwan/Workspace/ailoy/components/data".to_owned());
        let mut buf = bytes::BytesMut::new();
        let mut strm = cache.get("Qwen--Qwen3-0.6B", "tokenizer.json");

        while let Some(chunk) = strm.next().await {
            match chunk {
                Ok(chunk) => {
                    buf.extend_from_slice(&chunk);
                }
                Err(e) => {
                    panic!("{:?}", e);
                }
            }
        }

        let bytes = buf.freeze();
        let mut hasher = sha1::Sha1::new();
        hasher.update(bytes.as_ref());
        let v = hasher.finalize();
        let hex_str: String = v.iter().map(|b| format!("{:02x}", b)).collect();
        println!("{}", hex_str);
    }
}
