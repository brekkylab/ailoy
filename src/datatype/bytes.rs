use std::borrow::Cow;

use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Bytes(ByteBuf);

impl JsonSchema for Bytes {
    fn schema_name() -> Cow<'static, str> {
        "Bytes".into()
    }

    fn json_schema(_gen: &mut SchemaGenerator) -> Schema {
        schemars::json_schema!({
            "type": "string",
            "format": "base64"
        })
    }
}

impl Bytes {
    pub fn base64(&self) -> String {
        use base64::{Engine, engine::general_purpose::STANDARD};
        STANDARD.encode(&self.0)
    }

    /// Decode standard (padded) base64, the alphabet every model API uses for
    /// inline binary payloads. Inverse of [`base64`](Self::base64).
    pub fn from_base64(encoded: impl AsRef<[u8]>) -> anyhow::Result<Self> {
        use base64::{Engine, engine::general_purpose::STANDARD};
        let decoded = STANDARD
            .decode(encoded)
            .map_err(|e| anyhow::anyhow!("invalid base64: {e}"))?;
        Ok(Self::from(decoded))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl From<Vec<u8>> for Bytes {
    fn from(v: Vec<u8>) -> Self {
        Self(ByteBuf::from(v))
    }
}

impl AsRef<[u8]> for Bytes {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl std::fmt::Debug for Bytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Bytes(({} bytes total))", self.0.len())
    }
}

#[cfg(test)]
mod tests {
    use schemars::JsonSchema;

    use super::Bytes;

    #[test]
    fn base64_round_trip() {
        let original = Bytes::from(vec![0u8, 1, 2, 250, 251, 252]);
        let decoded = Bytes::from_base64(original.base64()).unwrap();
        assert_eq!(decoded, original);
        assert_eq!(decoded.len(), 6);
        assert!(!decoded.is_empty());
    }

    #[test]
    fn from_base64_rejects_garbage() {
        assert!(Bytes::from_base64("not base64!").is_err());
    }

    #[test]
    fn json_schema_preserves_base64_string_contract() {
        let schema = Bytes::json_schema(&mut schemars::SchemaGenerator::default());

        assert_eq!(
            serde_json::to_value(schema).unwrap(),
            serde_json::json!({
                "type": "string",
                "format": "base64"
            })
        );
    }
}
