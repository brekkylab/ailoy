//! Provider credentials/endpoint configs. Mirrors mirage `accessor/`.
//!
//! Secret-bearing structs intentionally do not derive `Debug` to avoid
//! leaking credentials into logs. They stay host-only.

mod gdrive;
mod notion;
mod s3;

pub use gdrive::GDriveAccessor;
pub use notion::NotionAccessor;
pub use s3::S3Accessor;

use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct S3Config {
    pub bucket: String,
    #[serde(default = "default_region")]
    pub region: String,
    pub access_key_id: String,
    pub secret_access_key: String,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub key_prefix: Option<String>,
}

fn default_region() -> String {
    "us-east-1".to_string()
}

#[derive(Clone, Serialize, Deserialize)]
pub struct NotionConfig {
    pub api_key: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct GDriveConfig {
    pub client_id: String,
    pub client_secret: String,
    pub refresh_token: String,
}
