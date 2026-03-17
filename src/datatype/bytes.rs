use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Bytes(ByteBuf);

impl Bytes {
    pub fn base64(&self) -> String {
        use base64::{Engine, engine::general_purpose::STANDARD};
        STANDARD.encode(&self.0)
    }
}

impl From<Vec<u8>> for Bytes {
    fn from(v: Vec<u8>) -> Self {
        Self(ByteBuf::from(v))
    }
}

impl std::fmt::Debug for Bytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Bytes(({} bytes total))", self.0.len())
    }
}
