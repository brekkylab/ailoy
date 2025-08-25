use anyhow::Result;
use crate::async_trait;
use serde_json::{Map, Value as Json};

use crate::utils::{MaybeSend, MaybeSync};

pub type Embedding = Vec<f32>;
pub type Metadata = Map<String, Json>;

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

async_trait! {
    pub trait VectorStore: MaybeSend + MaybeSync {
        async fn add_vector(&mut self, input: AddInput) -> Result<String>;
        async fn add_vectors(&mut self, inputs: Vec<AddInput>) -> Result<Vec<String>>;
        async fn get_by_id(&self, id: &str) -> Result<Option<GetResult>>;
        async fn get_by_ids(&self, ids: &[&str]) -> Result<Vec<GetResult>>;
        async fn retrieve(
            &self,
            query_embedding: Embedding,
            top_k: usize,
        ) -> Result<Vec<RetrieveResult>>;
        async fn remove_vector(&mut self, id: &str) -> Result<()>;
        async fn remove_vectors(&mut self, ids: &[&str]) -> Result<()>;
        async fn clear(&mut self) -> Result<()>;

        async fn count(&self) -> Result<usize>;
    }
}
