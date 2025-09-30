mod api;
mod local;

use ailoy_macros::multi_platform_async_trait;
use anyhow::Result;
pub use api::*;
pub use local::*;
use serde_json::{Map, Value as Json};

use crate::utils::{MaybeSend, MaybeSync};

pub type Embedding = Vec<f32>;
pub type Metadata = Map<String, Json>;

pub struct VectorStoreKey {
    id: String,
}

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

#[multi_platform_async_trait]
pub trait VectorStore: MaybeSend + MaybeSync {
    async fn add_vector(&mut self, input: VectorStoreAddInput) -> Result<String>;
    async fn add_vectors(&mut self, inputs: Vec<VectorStoreAddInput>) -> Result<Vec<String>>;
    async fn get_by_id(&self, id: &str) -> Result<Option<VectorStoreGetResult>>;
    async fn get_by_ids(&self, ids: &[&str]) -> Result<Vec<VectorStoreGetResult>>;
    async fn retrieve(
        &self,
        query_embedding: Embedding,
        top_k: usize,
    ) -> Result<Vec<VectorStoreRetrieveResult>>;
    async fn batch_retrieve(
        &self,
        query_embeddings: Vec<Embedding>,
        top_k: usize,
    ) -> Result<Vec<Vec<VectorStoreRetrieveResult>>>;
    async fn remove_vector(&mut self, id: &str) -> Result<()>;
    async fn remove_vectors(&mut self, ids: &[&str]) -> Result<()>;
    async fn clear(&mut self) -> Result<()>;

    async fn count(&self) -> Result<usize>;
}
