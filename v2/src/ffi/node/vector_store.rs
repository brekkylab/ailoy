use std::sync::Arc;

use futures::lock::Mutex;
use napi::Status;
use napi_derive::napi;
use serde_json::{Map, Value};

use crate::{
    ffi::node::embedding_model::Embedding,
    knowledge_base::{AddInput, ChromaStore, FaissStore, GetResult, RetrieveResult, VectorStore},
};

#[napi]
pub type Metadata = Option<Map<String, Value>>;

#[napi(object, js_name = "VectorStoreAddInput")]
pub struct JsAddInput {
    pub embedding: Embedding,
    pub document: String,
    pub metadata: Metadata,
}

impl Into<AddInput> for JsAddInput {
    fn into(self) -> AddInput {
        AddInput {
            embedding: self.embedding.into_iter().map(|f| f as f32).collect(),
            document: self.document,
            metadata: self.metadata,
        }
    }
}

#[napi(object, js_name = "VectorStoreGetResult")]
pub struct JsGetResult {
    pub id: String,
    pub document: String,
    pub metadata: Metadata,
    pub embedding: Embedding,
}

impl From<GetResult> for JsGetResult {
    fn from(res: GetResult) -> Self {
        Self {
            id: res.id,
            document: res.document,
            metadata: res.metadata,
            embedding: res.embedding.into_iter().map(|f| f as f64).collect(),
        }
    }
}

#[napi(object, js_name = "VectorStoreRetrieveResult")]
pub struct JsRetrieveResult {
    pub id: String,
    pub document: String,
    pub metadata: Metadata,
    pub distance: f64,
}

impl From<RetrieveResult> for JsRetrieveResult {
    fn from(res: RetrieveResult) -> Self {
        Self {
            id: res.id,
            document: res.document,
            metadata: res.metadata,
            distance: res.distance as f64,
        }
    }
}

pub trait VectorStoreMethods<T: VectorStore + 'static> {
    fn inner(&self) -> Arc<Mutex<T>>;

    fn handle_error<TRes>(res: std::result::Result<TRes, anyhow::Error>) -> napi::Result<TRes> {
        res.map_err(|e| napi::Error::new(Status::GenericFailure, e.to_string()))
    }

    async fn _add_vector(&self, input: JsAddInput) -> napi::Result<String> {
        Self::handle_error(self.inner().lock().await.add_vector(input.into()).await)
    }

    async fn _add_vectors(&self, inputs: Vec<JsAddInput>) -> napi::Result<Vec<String>> {
        Self::handle_error(
            self.inner()
                .lock()
                .await
                .add_vectors(inputs.into_iter().map(|input| input.into()).collect())
                .await,
        )
    }

    async fn _get_by_id(&self, id: String) -> napi::Result<Option<JsGetResult>> {
        let result = Self::handle_error(self.inner().lock().await.get_by_id(id.as_str()).await)?;
        match result {
            Some(result) => Ok(Some(result.into())),
            None => Ok(None),
        }
    }

    async fn _get_by_ids(&self, ids: Vec<String>) -> napi::Result<Vec<JsGetResult>> {
        Ok(Self::handle_error(
            self.inner()
                .lock()
                .await
                .get_by_ids(&ids.iter().map(|id| id.as_str()).collect::<Vec<_>>())
                .await,
        )?
        .into_iter()
        .map(|res| res.into())
        .collect::<Vec<_>>())
    }

    async fn _retrieve(
        &self,
        query_embedding: Embedding,
        top_k: u32,
    ) -> napi::Result<Vec<JsRetrieveResult>> {
        Ok(Self::handle_error(
            self.inner()
                .lock()
                .await
                .retrieve(
                    query_embedding.into_iter().map(|f| f as f32).collect(),
                    top_k as usize,
                )
                .await,
        )?
        .into_iter()
        .map(|res| res.into())
        .collect::<Vec<_>>())
    }

    async fn _batch_retrieve(
        &self,
        query_embeddings: Vec<Embedding>,
        top_k: u32,
    ) -> napi::Result<Vec<Vec<JsRetrieveResult>>> {
        let query_embeddings = query_embeddings
            .into_iter()
            .map(|query_embedding| query_embedding.into_iter().map(|f| f as f32).collect())
            .collect();
        Ok(Self::handle_error(
            self.inner()
                .lock()
                .await
                .batch_retrieve(query_embeddings, top_k as usize)
                .await,
        )?
        .into_iter()
        .map(|batch| batch.into_iter().map(|item| item.into()).collect())
        .collect())
    }

    async fn _remove_vector(&self, id: String) -> napi::Result<()> {
        Self::handle_error(self.inner().lock().await.remove_vector(id.as_str()).await)
    }

    async fn _remove_vectors(&self, ids: Vec<String>) -> napi::Result<()> {
        Self::handle_error(
            self.inner()
                .lock()
                .await
                .remove_vectors(&ids.iter().map(|id| id.as_str()).collect::<Vec<_>>())
                .await,
        )
    }

    async fn _clear(&self) -> napi::Result<()> {
        Self::handle_error(self.inner().lock().await.clear().await)
    }

    async fn _count(&self) -> napi::Result<u32> {
        Self::handle_error(self.inner().lock().await.count().await.map(|c| c as u32))
    }
}

#[napi(js_name = "FaissVectorStore")]
pub struct JsFaissVectorStore {
    inner: Arc<Mutex<FaissStore>>,
}

impl VectorStoreMethods<FaissStore> for JsFaissVectorStore {
    fn inner(&self) -> Arc<Mutex<FaissStore>> {
        self.inner.clone()
    }
}

#[napi]
impl JsFaissVectorStore {
    #[napi(factory)]
    pub async fn create(dim: i32) -> napi::Result<Self> {
        let inner = FaissStore::new(dim)
            .await
            .map_err(|e| napi::Error::new(Status::GenericFailure, e.to_string()))?;
        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    #[napi]
    pub async fn add_vector(&self, input: JsAddInput) -> napi::Result<String> {
        self._add_vector(input).await
    }

    #[napi]
    pub async fn add_vectors(&self, inputs: Vec<JsAddInput>) -> napi::Result<Vec<String>> {
        self._add_vectors(inputs).await
    }

    #[napi]
    pub async fn get_by_id(&self, id: String) -> napi::Result<Option<JsGetResult>> {
        self._get_by_id(id).await
    }

    #[napi]
    pub async fn get_by_ids(&self, ids: Vec<String>) -> napi::Result<Vec<JsGetResult>> {
        self._get_by_ids(ids).await
    }

    #[napi]
    pub async fn retrieve(
        &self,
        query_embedding: Embedding,
        top_k: u32,
    ) -> napi::Result<Vec<JsRetrieveResult>> {
        self._retrieve(query_embedding, top_k).await
    }

    #[napi]
    pub async fn batch_retrieve(
        &self,
        query_embeddings: Vec<Embedding>,
        top_k: u32,
    ) -> napi::Result<Vec<Vec<JsRetrieveResult>>> {
        self._batch_retrieve(query_embeddings, top_k).await
    }

    #[napi]
    pub async fn remove_vector(&self, id: String) -> napi::Result<()> {
        self._remove_vector(id).await
    }

    #[napi]
    pub async fn remove_vectors(&self, ids: Vec<String>) -> napi::Result<()> {
        self._remove_vectors(ids).await
    }

    #[napi]
    pub async fn clear(&self) -> napi::Result<()> {
        self._clear().await
    }

    #[napi]
    pub async fn count(&self) -> napi::Result<u32> {
        self._count().await
    }
}

#[napi(js_name = "ChromaVectorStore")]
pub struct JsChromaVectorStore {
    inner: Arc<Mutex<ChromaStore>>,
}

impl VectorStoreMethods<ChromaStore> for JsChromaVectorStore {
    fn inner(&self) -> Arc<Mutex<ChromaStore>> {
        self.inner.clone()
    }
}

#[napi]
impl JsChromaVectorStore {
    #[napi(factory)]
    pub async fn create(chroma_url: String, collection_name: String) -> napi::Result<Self> {
        let inner = ChromaStore::new(chroma_url.as_str(), collection_name.as_str())
            .await
            .map_err(|e| napi::Error::new(Status::GenericFailure, e.to_string()))?;
        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    #[napi]
    pub async fn add_vector(&self, input: JsAddInput) -> napi::Result<String> {
        self._add_vector(input).await
    }

    #[napi]
    pub async fn add_vectors(&self, inputs: Vec<JsAddInput>) -> napi::Result<Vec<String>> {
        self._add_vectors(inputs).await
    }

    #[napi]
    pub async fn get_by_id(&self, id: String) -> napi::Result<Option<JsGetResult>> {
        self._get_by_id(id).await
    }

    #[napi]
    pub async fn get_by_ids(&self, ids: Vec<String>) -> napi::Result<Vec<JsGetResult>> {
        self._get_by_ids(ids).await
    }

    #[napi]
    pub async fn retrieve(
        &self,
        query_embedding: Embedding,
        top_k: u32,
    ) -> napi::Result<Vec<JsRetrieveResult>> {
        self._retrieve(query_embedding, top_k).await
    }

    #[napi]
    pub async fn batch_retrieve(
        &self,
        query_embeddings: Vec<Embedding>,
        top_k: u32,
    ) -> napi::Result<Vec<Vec<JsRetrieveResult>>> {
        self._batch_retrieve(query_embeddings, top_k).await
    }

    #[napi]
    pub async fn remove_vector(&self, id: String) -> napi::Result<()> {
        self._remove_vector(id).await
    }

    #[napi]
    pub async fn remove_vectors(&self, ids: Vec<String>) -> napi::Result<()> {
        self._remove_vectors(ids).await
    }

    #[napi]
    pub async fn clear(&self) -> napi::Result<()> {
        self._clear().await
    }

    #[napi]
    pub async fn count(&self) -> napi::Result<u32> {
        self._count().await
    }
}
