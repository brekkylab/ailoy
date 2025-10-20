mod api;
mod local;

use std::sync::Arc;

use ailoy_macros::{maybe_send_sync, multi_platform_async_trait};
pub use api::*;
use futures::lock::Mutex;
pub use local::*;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub type Embedding = Vec<f32>;

pub type Metadata = Map<String, Value>;

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(from_wasm_abi, into_wasm_abi))]
pub struct VectorStoreAddInput {
    pub embedding: Embedding,
    pub document: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(from_wasm_abi, into_wasm_abi))]
pub struct VectorStoreGetResult {
    pub id: String,
    pub document: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
    pub embedding: Embedding,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(from_wasm_abi, into_wasm_abi))]
pub struct VectorStoreRetrieveResult {
    pub id: String,
    pub document: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
    pub distance: f32,
}

#[maybe_send_sync]
#[multi_platform_async_trait]
pub trait VectorStoreBehavior {
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

#[derive(Debug, Clone)]
pub enum VectorStoreInner {
    Faiss(Arc<Mutex<FaissStore>>),
    Chroma(Arc<Mutex<ChromaStore>>),
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
pub struct VectorStore {
    inner: VectorStoreInner,
}

impl VectorStore {
    pub fn new_faiss(store: FaissStore) -> Self {
        Self {
            inner: VectorStoreInner::Faiss(Arc::new(Mutex::new(store))),
        }
    }

    pub fn new_chroma(store: ChromaStore) -> Self {
        Self {
            inner: VectorStoreInner::Chroma(Arc::new(Mutex::new(store))),
        }
    }

    pub async fn add_vector(&mut self, input: VectorStoreAddInput) -> anyhow::Result<String> {
        match &self.inner {
            VectorStoreInner::Faiss(inner) => inner.lock().await.add_vector(input).await,
            VectorStoreInner::Chroma(inner) => inner.lock().await.add_vector(input).await,
        }
    }

    pub async fn add_vectors(
        &mut self,
        inputs: Vec<VectorStoreAddInput>,
    ) -> anyhow::Result<Vec<String>> {
        match &self.inner {
            VectorStoreInner::Faiss(inner) => inner.lock().await.add_vectors(inputs).await,
            VectorStoreInner::Chroma(inner) => inner.lock().await.add_vectors(inputs).await,
        }
    }

    pub async fn get_by_id(&self, id: &str) -> anyhow::Result<Option<VectorStoreGetResult>> {
        match &self.inner {
            VectorStoreInner::Faiss(inner) => inner.lock().await.get_by_id(id).await,
            VectorStoreInner::Chroma(inner) => inner.lock().await.get_by_id(id).await,
        }
    }

    pub async fn get_by_ids(&self, ids: &[&str]) -> anyhow::Result<Vec<VectorStoreGetResult>> {
        match &self.inner {
            VectorStoreInner::Faiss(inner) => inner.lock().await.get_by_ids(ids).await,
            VectorStoreInner::Chroma(inner) => inner.lock().await.get_by_ids(ids).await,
        }
    }

    pub async fn retrieve(
        &self,
        query_embedding: Embedding,
        top_k: usize,
    ) -> anyhow::Result<Vec<VectorStoreRetrieveResult>> {
        match self.inner.clone() {
            VectorStoreInner::Faiss(inner) => {
                inner.lock().await.retrieve(query_embedding, top_k).await
            }
            VectorStoreInner::Chroma(inner) => {
                inner.lock().await.retrieve(query_embedding, top_k).await
            }
        }
    }

    pub async fn batch_retrieve(
        &self,
        query_embeddings: Vec<Embedding>,
        top_k: usize,
    ) -> anyhow::Result<Vec<Vec<VectorStoreRetrieveResult>>> {
        match &self.inner {
            VectorStoreInner::Faiss(inner) => {
                inner
                    .lock()
                    .await
                    .batch_retrieve(query_embeddings, top_k)
                    .await
            }
            VectorStoreInner::Chroma(inner) => {
                inner
                    .lock()
                    .await
                    .batch_retrieve(query_embeddings, top_k)
                    .await
            }
        }
    }

    pub async fn remove_vector(&mut self, id: &str) -> anyhow::Result<()> {
        match &self.inner {
            VectorStoreInner::Faiss(inner) => inner.lock().await.remove_vector(id).await,
            VectorStoreInner::Chroma(inner) => inner.lock().await.remove_vector(id).await,
        }
    }

    pub async fn remove_vectors(&mut self, ids: &[&str]) -> anyhow::Result<()> {
        match &self.inner {
            VectorStoreInner::Faiss(inner) => inner.lock().await.remove_vectors(ids).await,
            VectorStoreInner::Chroma(inner) => inner.lock().await.remove_vectors(ids).await,
        }
    }

    pub async fn clear(&mut self) -> anyhow::Result<()> {
        match &self.inner {
            VectorStoreInner::Faiss(inner) => inner.lock().await.clear().await,
            VectorStoreInner::Chroma(inner) => inner.lock().await.clear().await,
        }
    }

    pub async fn count(&self) -> anyhow::Result<usize> {
        match &self.inner {
            VectorStoreInner::Faiss(inner) => inner.lock().await.count().await,
            VectorStoreInner::Chroma(inner) => inner.lock().await.count().await,
        }
    }
}

#[cfg(feature = "wasm")]
mod wasm {
    use wasm_bindgen::prelude::*;

    use super::*;

    #[wasm_bindgen(typescript_custom_section)]
    const TS_APPEND_CONTENT: &'static str = dedent::dedent!(
        r#"
        type Embedding = Float32Array;
        type Metadata = Record<string, any>;
        "#
    );

    #[wasm_bindgen]
    impl VectorStore {
        #[wasm_bindgen(js_name = "newFaiss")]
        pub async fn new_faiss_js(dim: i32) -> Result<Self, js_sys::Error> {
            let store = FaissStore::new(dim)
                .await
                .map_err(|e| js_sys::Error::new(&e.to_string()))?;
            Ok(Self::new_faiss(store))
        }

        #[wasm_bindgen(js_name = "newChroma")]
        pub async fn new_chroma_js(
            url: String,
            #[wasm_bindgen(js_name = "collectionName")] collection_name: Option<String>,
        ) -> Result<Self, js_sys::Error> {
            let store = ChromaStore::new(&url, collection_name.as_deref())
                .await
                .map_err(|e| js_sys::Error::new(&e.to_string()))?;
            Ok(Self::new_chroma(store))
        }

        #[wasm_bindgen(js_name = "addVector")]
        pub async fn add_vector_js(
            &mut self,
            input: VectorStoreAddInput,
        ) -> Result<String, js_sys::Error> {
            self.add_vector(input)
                .await
                .map_err(|e| js_sys::Error::new(&e.to_string()))
        }

        #[wasm_bindgen(js_name = "addVectors")]
        pub async fn add_vectors_js(
            &mut self,
            inputs: Vec<VectorStoreAddInput>,
        ) -> Result<Vec<String>, js_sys::Error> {
            self.add_vectors(inputs)
                .await
                .map_err(|e| js_sys::Error::new(&e.to_string()))
        }

        #[wasm_bindgen(js_name = "getById")]
        pub async fn get_by_id_js(
            &self,
            id: String,
        ) -> Result<Option<VectorStoreGetResult>, js_sys::Error> {
            self.get_by_id(&id)
                .await
                .map_err(|e| js_sys::Error::new(&e.to_string()))
        }

        #[wasm_bindgen(js_name = "getByIds")]
        pub async fn get_by_ids_js(
            &self,
            ids: Vec<String>,
        ) -> Result<Vec<VectorStoreGetResult>, js_sys::Error> {
            self.get_by_ids(
                ids.iter()
                    .map(|id| id.as_str())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
            .await
            .map_err(|e| js_sys::Error::new(&e.to_string()))
        }

        #[wasm_bindgen(js_name = "retrieve")]
        pub async fn retrieve_js(
            &self,
            query_embedding: Embedding,
            top_k: usize,
        ) -> Result<Vec<VectorStoreRetrieveResult>, js_sys::Error> {
            self.retrieve(query_embedding, top_k)
                .await
                .map_err(|e| js_sys::Error::new(&e.to_string()))
        }

        #[wasm_bindgen(js_name = "removeVector")]
        pub async fn remove_vector_js(&mut self, id: String) -> Result<(), js_sys::Error> {
            self.remove_vector(&id)
                .await
                .map_err(|e| js_sys::Error::new(&e.to_string()))
        }

        #[wasm_bindgen(js_name = "removeVectors")]
        pub async fn remove_vectors_js(&mut self, ids: Vec<String>) -> Result<(), js_sys::Error> {
            self.remove_vectors(
                ids.iter()
                    .map(|id| id.as_str())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
            .await
            .map_err(|e| js_sys::Error::new(&e.to_string()))
        }

        #[wasm_bindgen(js_name = "clear")]
        pub async fn clear_js(&mut self) -> Result<(), js_sys::Error> {
            self.clear()
                .await
                .map_err(|e| js_sys::Error::new(&e.to_string()))
        }

        #[wasm_bindgen(js_name = "count")]
        pub async fn count_js(&self) -> Result<usize, js_sys::Error> {
            self.count()
                .await
                .map_err(|e| js_sys::Error::new(&e.to_string()))
        }
    }
}
