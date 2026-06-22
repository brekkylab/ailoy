use std::sync::Arc;

use object_store::aws::{AmazonS3, AmazonS3Builder};

use crate::vfs::accessor::S3Config;
use crate::vfs::path::VPath;

/// Holds the S3 client (one credential set) and the optional key prefix.
/// Mirrors mirage `accessor/s3.py`.
pub struct S3Accessor {
    pub store: Arc<AmazonS3>,
    key_prefix: String,
}

impl S3Accessor {
    pub fn new(config: &S3Config) -> anyhow::Result<Self> {
        let mut builder = AmazonS3Builder::new()
            .with_bucket_name(config.bucket.as_str())
            .with_region(config.region.as_str())
            .with_access_key_id(config.access_key_id.as_str())
            .with_secret_access_key(config.secret_access_key.as_str());
        if let Some(endpoint) = &config.endpoint {
            builder = builder.with_endpoint(endpoint.as_str()).with_allow_http(true);
        }
        let store = builder.build()?;
        let key_prefix = config
            .key_prefix
            .clone()
            .unwrap_or_default()
            .trim_matches('/')
            .to_string();
        Ok(Self {
            store: Arc::new(store),
            key_prefix,
        })
    }

    /// Map a mount-relative path to a full S3 object key (applying key_prefix).
    pub fn key(&self, path: &VPath) -> String {
        let rel = path.as_str().trim_start_matches('/');
        match (self.key_prefix.is_empty(), rel.is_empty()) {
            (true, _) => rel.to_string(),
            (false, true) => self.key_prefix.clone(),
            (false, false) => format!("{}/{}", self.key_prefix, rel),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn acc(prefix: Option<&str>) -> S3Accessor {
        S3Accessor::new(&S3Config {
            bucket: "b".into(),
            region: "us-east-1".into(),
            access_key_id: "k".into(),
            secret_access_key: "s".into(),
            endpoint: None,
            key_prefix: prefix.map(|p| p.into()),
        })
        .unwrap()
    }

    #[test]
    fn key_no_prefix() {
        let a = acc(None);
        assert_eq!(a.key(&VPath::root()), "");
        assert_eq!(a.key(&VPath::new("/data/x.csv")), "data/x.csv");
    }

    #[test]
    fn key_with_prefix() {
        let a = acc(Some("sub/inner"));
        assert_eq!(a.key(&VPath::root()), "sub/inner");
        assert_eq!(a.key(&VPath::new("/x.csv")), "sub/inner/x.csv");
    }
}
