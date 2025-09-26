use ailoy_macros::multi_platform_async_trait;
use anyhow::Result;

use crate::utils::{MaybeSend, MaybeSync};

pub type Embedding = Vec<f32>;

#[multi_platform_async_trait]
pub trait EmbeddingModel: MaybeSend + MaybeSync + 'static {
    async fn run(self: &mut Self, text: String) -> Result<Embedding>;
}
