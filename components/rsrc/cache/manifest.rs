use std::collections::HashMap;

use bytes::Bytes;
use serde::{Deserialize, de};
use sha1::{Digest, Sha1};

#[derive(Clone, Debug)]
pub struct Manifest {
    size: u64,
    sha1: String,
}

impl Manifest {
    pub fn new(size: u64, sha1: String) -> Self {
        Manifest { size, sha1 }
    }

    pub fn from_bytes(bytes: Bytes) -> Self {
        let mut hasher = Sha1::new();
        hasher.update(&bytes);
        let hash = hasher.finalize();
        let hash_hex: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
        Manifest::new(bytes.len() as u64, hash_hex)
    }

    pub fn sha1(&self) -> &str {
        &self.sha1
    }
}

struct ManifestVisitor;

impl<'de> de::Visitor<'de> for ManifestVisitor {
    type Value = Manifest;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a manifest JSON object with key \"*\" and object value")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Manifest, M::Error>
    where
        M: de::MapAccess<'de>,
    {
        let mut size: Option<u64> = None;
        let mut sha1: Option<String> = None;
        while let Some(key) = map.next_key::<String>()? {
            if key == "*" {
                // @jhlee
                // This field intended to version control of files. However, currently keys other than "*" are not supported.
                // We should implement:
                // 1. Definition of the key scheme (version specifier)
                // 2. Implementing filtering logic to compare it against the current library version (ailoy) for compatibility.
                return Ok(map.next_value()?);
            } else if key == "size" {
                size = Some(map.next_value()?);
            } else if key == "sha1" {
                sha1 = Some(map.next_value()?);
            } else {
                return Err(de::Error::unknown_field(&key, &["sha1"]));
            }
        }
        let size = size.ok_or_else(|| de::Error::missing_field("sha1"))?;
        let sha1 = sha1.ok_or_else(|| de::Error::missing_field("sha1"))?;
        Ok(Manifest { size, sha1 })
    }
}

impl<'de> de::Deserialize<'de> for Manifest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        deserializer.deserialize_map(ManifestVisitor)
    }
}

#[derive(Clone, Debug, Deserialize)]
pub(super) struct ManifestDirectory {
    pub version: String,
    pub files: HashMap<String, Manifest>,
}
