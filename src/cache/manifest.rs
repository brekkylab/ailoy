use std::collections::{BTreeMap, HashMap};

use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{MapAccess, Visitor},
    ser,
    ser::SerializeMap as _,
};
use sha1::{Digest, Sha1};
use version_compare::Version;

#[derive(Clone, Debug)]
pub struct Manifest {
    size: u64,
    sha1: String,
    min_version: Option<String>,
}

impl Manifest {
    pub fn new(size: u64, sha1: String) -> Self {
        Manifest {
            size,
            sha1,
            min_version: None,
        }
    }

    pub fn from_u8(bytes: impl AsRef<[u8]>) -> Self {
        let bytes = bytes.as_ref();
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

#[derive(Clone, Debug)]
pub struct ManifestFiles(pub BTreeMap<String, Vec<Manifest>>);

impl ManifestFiles {}

impl Serialize for ManifestFiles {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        struct FileItem<'a> {
            size: u64,
            sha1: &'a str,
        }

        impl<'a> Serialize for FileItem<'a> {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("size", &self.size)?;
                map.serialize_entry("sha1", &self.sha1)?;
                map.end()
            }
        }

        let mut outer = serializer.serialize_map(Some(1))?;
        for (filename, manifests) in self.0.iter() {
            let mut inner = HashMap::<String, FileItem>::new();
            for manifest in manifests.iter() {
                let min_version = match &manifest.min_version {
                    Some(version) => version.to_string(),
                    None => "*".into(),
                };
                if inner.contains_key(&min_version) {
                    return Err(ser::Error::custom(format!(
                        "duplicated min_version: {}",
                        min_version
                    )));
                }
                inner.insert(
                    min_version,
                    FileItem {
                        size: manifest.size,
                        sha1: &manifest.sha1,
                    },
                );
            }
            outer.serialize_entry(filename, &inner)?;
        }
        outer.end()
    }
}

impl<'de> Deserialize<'de> for ManifestFiles {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct FileItemVisitor;

        impl<'de> Visitor<'de> for FileItemVisitor {
            type Value = ManifestFiles;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a map of filenames to version maps")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut result = BTreeMap::new();

                while let Some((filename, version_map)) =
                    map.next_entry::<String, HashMap<String, serde_json::Value>>()?
                {
                    let versions_map = version_map.clone();
                    let mut manifests = Vec::new();

                    for (version, value) in versions_map.iter() {
                        let size = value.get("size").and_then(|v| v.as_u64()).ok_or_else(|| {
                            serde::de::Error::custom(format!(
                                "missing or invalid 'size' field for {}/{}",
                                filename, version
                            ))
                        })?;

                        let sha1 = value
                            .get("sha1")
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| {
                                serde::de::Error::custom(format!(
                                    "missing or invalid 'sha1' field for {}/{}",
                                    filename, version
                                ))
                            })?
                            .to_string();

                        let min_version = if version == "*" {
                            None
                        } else {
                            Some(version.clone())
                        };

                        manifests.push(Manifest {
                            size,
                            sha1,
                            min_version,
                        });
                    }

                    result.insert(filename, manifests);
                }

                Ok(ManifestFiles(result))
            }
        }

        deserializer.deserialize_map(FileItemVisitor)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ManifestDirectory {
    pub version: String,
    pub files: ManifestFiles,
}

#[allow(dead_code)]
impl ManifestDirectory {
    pub fn new() -> Self {
        ManifestDirectory {
            version: "1".to_owned(),
            files: ManifestFiles(BTreeMap::new()),
        }
    }

    pub fn insert_file(&mut self, filename: String, manifest: Manifest) {
        if self.files.0.contains_key(&filename) {
            self.files.0.get_mut(&filename).unwrap().push(manifest);
        } else {
            self.files.0.insert(filename, vec![manifest]);
        }
    }

    pub fn get_file_manifest(&self, filename: &str, target_version: &str) -> Option<Manifest> {
        if !self.files.0.contains_key(filename) {
            return None;
        }
        let target_version = Version::from(&target_version);
        if target_version.is_none() {
            return None;
        }
        let target_version = target_version.unwrap();

        // Sort the manifests by descending order of min_version
        let mut manifests = self.files.0.get(filename).cloned().unwrap();
        manifests.sort_by(|m1, m2| match (&m1.min_version, &m2.min_version) {
            (Some(v1), Some(v2)) => {
                let v1 = Version::from(v1).unwrap();
                let v2 = Version::from(v2).unwrap();
                v2.compare(v1).ord().unwrap()
            }
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, None) => std::cmp::Ordering::Equal,
        });

        let mut ret: Option<Manifest> = None;
        for manifest in manifests.iter() {
            if let Some(min_version) = &manifest.min_version {
                // If the largest min version less than target version is found, break immediately
                let min_version = Version::from(min_version).unwrap();
                if target_version >= min_version {
                    ret = Some(manifest.clone());
                    break;
                }
            } else {
                // This is the case of min_version = "*"
                ret = Some(manifest.clone());
            }
        }

        ret
    }

    pub fn get_file_manifests(&self, target_version: &str) -> Vec<(String, Manifest)> {
        let mut manifests = Vec::<(String, Manifest)>::new();
        for filename in self.files.0.keys() {
            match self.get_file_manifest(filename, target_version) {
                Some(manifest) => {
                    manifests.push((filename.to_string(), manifest));
                }
                None => {}
            }
        }
        manifests
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[tokio::test]
    async fn test_manifest_files() {
        let manifest_json = json!({
            "version": "1",
            "files": {
                "rt.dylib": {
                    "0.3.0": {
                        "size": 1282914,
                        "sha1": "040ba4836746c33166b60341cade6724fde73dcb"
                    },
                    "0.2.0": {
                        "size": 1169576,
                        "sha1": "1c842d06300e4d3d4880b9a36279597a7d541e97"
                    },
                    "*": {
                        "size": 867856,
                        "sha1": "a41c36b11632bb70eac7bd839839a479d8ea3ad1"
                    }
                },
                "new_file_from_0.3.0.json": {
                    "0.3.0": {
                        "size": 1234,
                        "sha1": "dd0beab4a8f2fbee9223aa6f54d37dda396e4828"
                    }
                }
            }
        });
        let manifest_dir = serde_json::from_value::<ManifestDirectory>(manifest_json).unwrap();

        let manifest_0_4_0 = manifest_dir.get_file_manifest("rt.dylib", "0.4.0");
        assert!(manifest_0_4_0.is_some());
        assert_eq!(manifest_0_4_0.unwrap().min_version.unwrap(), "0.3.0");

        let manifest_0_3_0_rc0 = manifest_dir.get_file_manifest("rt.dylib", "0.3.0-rc.0");
        assert!(manifest_0_3_0_rc0.is_some());
        assert_eq!(manifest_0_3_0_rc0.unwrap().min_version.unwrap(), "0.2.0");

        let manifest_0_1_0 = manifest_dir.get_file_manifest("rt.dylib", "0.1.0");
        assert!(manifest_0_1_0.is_some());
        assert_eq!(manifest_0_1_0.unwrap().min_version, None);

        let manifests_0_3_0 = manifest_dir.get_file_manifests("0.3.0");
        assert_eq!(manifests_0_3_0.len(), 2);

        let manifests_0_2_0 = manifest_dir.get_file_manifests("0.2.0");
        assert_eq!(manifests_0_2_0.len(), 1);
    }
}
