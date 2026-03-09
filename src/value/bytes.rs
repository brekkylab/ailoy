use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;

use crate::utils::Ellipsis;

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Bytes(pub ByteBuf);

impl std::fmt::Debug for Bytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const MAX_LEN: usize = 64;
        let bytes_str = format!("{:?}", self.0);
        write!(
            f,
            "Bytes({} ({} bytes total))",
            bytes_str.truncate_ellipsis(MAX_LEN),
            self.0.len()
        )
    }
}
