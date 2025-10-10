mod api;
mod local;

use ailoy_macros::{maybe_send_sync, multi_platform_async_trait};
pub use api::*;
pub use local::*;
use serde_json::{Map, Value as Json};

pub type Embedding = Vec<f32>;
pub type Metadata = Map<String, Json>;

#[derive(Debug)]
pub struct VectorStoreAddInput {
    pub embedding: Embedding,
    pub document: String,
    pub metadata: Option<Metadata>,
}

#[derive(Debug)]
pub struct VectorStoreGetResult {
    pub id: String,
    pub document: String,
    pub metadata: Option<Metadata>,
    pub embedding: Embedding,
}

#[derive(Debug)]
pub struct VectorStoreRetrieveResult {
    pub id: String,
    pub document: String,
    pub metadata: Option<Metadata>,
    pub distance: f32,
}

#[maybe_send_sync]
#[multi_platform_async_trait]
pub trait VectorStore {
    async fn add_vector(&mut self, input: VectorStoreAddInput) -> anyhow::Result<String>;
    async fn add_vectors(
        &mut self,
        inputs: Vec<VectorStoreAddInput>,
    ) -> anyhow::Result<Vec<String>>;
    async fn get_by_id(&self, id: &str) -> anyhow::Result<Option<VectorStoreGetResult>>;
    async fn get_by_ids(&self, ids: &[&str]) -> anyhow::Result<Vec<VectorStoreGetResult>>;
    async fn retrieve(
        &self,
        query_embedding: Embedding,
        top_k: usize,
    ) -> anyhow::Result<Vec<VectorStoreRetrieveResult>>;
    async fn batch_retrieve(
        &self,
        query_embeddings: Vec<Embedding>,
        top_k: usize,
    ) -> anyhow::Result<Vec<Vec<VectorStoreRetrieveResult>>>;
    async fn remove_vector(&mut self, id: &str) -> anyhow::Result<()>;
    async fn remove_vectors(&mut self, ids: &[&str]) -> anyhow::Result<()>;
    async fn clear(&mut self) -> anyhow::Result<()>;

    async fn count(&self) -> anyhow::Result<usize>;
}
