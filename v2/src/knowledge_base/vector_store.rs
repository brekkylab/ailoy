use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value as Json;

pub type Embedding = Vec<f32>;
pub type Metadata = Json;

#[derive(Debug)]
pub struct AddInput {
    pub embedding: Embedding,
    pub document: String,
    pub metadata: Option<Metadata>,
}

#[derive(Debug)]
pub struct GetResult {
    pub id: String,
    pub document: String,
    pub metadata: Option<Metadata>,
    pub embedding: Embedding,
}

#[derive(Debug)]
pub struct RetrieveResult {
    pub id: String,
    pub document: String,
    pub metadata: Option<Metadata>,
    pub distance: f32,
}

#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn add_vector(&self, input: AddInput) -> Result<String>;
    async fn add_vectors(&self, inputs: Vec<AddInput>) -> Result<Vec<String>>;
    async fn get_by_id(&self, id: &str) -> Result<Option<GetResult>>;
    async fn get_by_ids(&self, ids: &[&str]) -> Result<Vec<GetResult>>;
    async fn retrieve(&self, query_embedding: Embedding, top_k: u64)
    -> Result<Vec<RetrieveResult>>;
    async fn remove_vector(&self, id: &str) -> Result<()>;
    async fn remove_vectors(&self, ids: &[&str]) -> Result<()>;
    async fn clear(&self) -> Result<()>;
}
