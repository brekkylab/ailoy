use std::{collections::HashMap, sync::Arc};

use ailoy_macros::{maybe_send_sync, multi_platform_async_trait};
use futures::lock::Mutex;
use serde::{Deserialize, Serialize};

use super::{api::ChromaStore, local::FaissStore};
use crate::value::{Embedding, Value};

pub type VectorStoreMetadata = HashMap<String, Value>;

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3_stub_gen_derive::gen_stub_pyclass)]
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(module = "ailoy._core", get_all, set_all)
)]
#[cfg_attr(feature = "nodejs", napi_derive::napi(object))]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(
    feature = "wasm",
    tsify(from_wasm_abi, into_wasm_abi, hashmap_as_object)
)]
pub struct VectorStoreAddInput {
    pub embedding: Embedding,
    pub document: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<VectorStoreMetadata>,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3_stub_gen_derive::gen_stub_pyclass)]
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(module = "ailoy._core", get_all, set_all)
)]
#[cfg_attr(feature = "nodejs", napi_derive::napi(object))]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(
    feature = "wasm",
    tsify(from_wasm_abi, into_wasm_abi, hashmap_as_object)
)]
pub struct VectorStoreGetResult {
    pub id: String,
    pub document: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<VectorStoreMetadata>,
    pub embedding: Embedding,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3_stub_gen_derive::gen_stub_pyclass)]
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(module = "ailoy._core", get_all, set_all)
)]
#[cfg_attr(feature = "nodejs", napi_derive::napi(object))]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(
    feature = "wasm",
    tsify(from_wasm_abi, into_wasm_abi, hashmap_as_object)
)]
pub struct VectorStoreRetrieveResult {
    pub id: String,
    pub document: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<VectorStoreMetadata>,
    pub distance: f64,
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
#[cfg_attr(feature = "python", pyo3_stub_gen_derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "ailoy._core"))]
#[cfg_attr(feature = "nodejs", napi_derive::napi)]
#[cfg_attr(feature = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
pub struct VectorStore {
    inner: VectorStoreInner,
}

impl VectorStore {
    pub async fn new_faiss(dim: u32) -> anyhow::Result<Self> {
        let store = FaissStore::new(dim).await?;
        Ok(Self {
            inner: VectorStoreInner::Faiss(Arc::new(Mutex::new(store))),
        })
    }

    pub async fn new_chroma(url: String, collection_name: Option<String>) -> anyhow::Result<Self> {
        let store = ChromaStore::new(url, collection_name).await?;
        Ok(Self {
            inner: VectorStoreInner::Chroma(Arc::new(Mutex::new(store))),
        })
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

#[cfg(feature = "python")]
mod py {
    use pyo3::{prelude::*, types::PyType};
    use pyo3_stub_gen_derive::*;

    use super::*;
    use crate::ffi::py::base::await_future;

    impl Into<VectorStoreAddInput> for Py<VectorStoreAddInput> {
        fn into(self) -> VectorStoreAddInput {
            Python::attach(|py| {
                let input = self.borrow(py);
                VectorStoreAddInput {
                    embedding: input.embedding.clone(),
                    document: input.document.clone(),
                    metadata: input.metadata.clone(),
                }
            })
        }
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl VectorStoreAddInput {
        #[new]
        #[pyo3(signature = (embedding, document, metadata = None))]
        fn __new__(
            embedding: Embedding,
            document: String,
            metadata: Option<VectorStoreMetadata>,
        ) -> Self {
            Self {
                embedding,
                document,
                metadata,
            }
        }
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl VectorStore {
        #[classmethod]
        #[pyo3(name = "new_faiss")]
        fn new_faiss_py<'a>(_cls: &Bound<'a, PyType>, py: Python<'a>, dim: u32) -> PyResult<Self> {
            await_future(py, VectorStore::new_faiss(dim))
        }

        #[classmethod]
        #[pyo3(name = "new_chroma")]
        fn new_chroma_py<'a>(
            _cls: &Bound<'a, PyType>,
            py: Python<'a>,
            url: String,
            collection_name: Option<String>,
        ) -> PyResult<Self> {
            await_future(py, VectorStore::new_chroma(url, collection_name))
        }

        #[pyo3(name = "add_vector")]
        fn add_vector_py(
            &mut self,
            py: Python<'_>,
            input: Py<VectorStoreAddInput>,
        ) -> PyResult<String> {
            await_future(py, self.add_vector(input.into()))
        }

        #[pyo3(name = "add_vectors")]
        fn add_vectors_py(
            &mut self,
            py: Python<'_>,
            inputs: Vec<Py<VectorStoreAddInput>>,
        ) -> PyResult<Vec<String>> {
            await_future(
                py,
                self.add_vectors(inputs.into_iter().map(|input| input.into()).collect()),
            )
        }

        #[pyo3(name = "get_by_id")]
        fn get_by_id_py(
            &self,
            py: Python<'_>,
            id: String,
        ) -> PyResult<Option<VectorStoreGetResult>> {
            let result = await_future(py, self.get_by_id(&id))?;
            match result {
                Some(result) => Ok(Some(result.into())),
                None => Ok(None),
            }
        }

        #[pyo3(name = "get_by_ids")]
        fn get_by_ids_py(
            &self,
            py: Python<'_>,
            ids: Vec<String>,
        ) -> PyResult<Vec<VectorStoreGetResult>> {
            Ok(await_future(
                py,
                self.get_by_ids(&ids.iter().map(|id| id.as_str()).collect::<Vec<_>>()),
            )?
            .into_iter()
            .map(|result| result.into())
            .collect::<Vec<_>>())
        }

        #[pyo3(name = "retrieve")]
        fn retrieve_py(
            &self,
            py: Python<'_>,
            query_embedding: Embedding,
            top_k: usize,
        ) -> PyResult<Vec<VectorStoreRetrieveResult>> {
            Ok(await_future(py, self.retrieve(query_embedding, top_k))?
                .into_iter()
                .map(|result| result.into())
                .collect::<Vec<_>>())
        }

        #[pyo3(name = "batch_retrieve")]
        fn batch_retrieve_py(
            &self,
            py: Python<'_>,
            query_embeddings: Vec<Embedding>,
            top_k: usize,
        ) -> PyResult<Vec<Vec<VectorStoreRetrieveResult>>> {
            Ok(
                await_future(py, self.batch_retrieve(query_embeddings, top_k))?
                    .into_iter()
                    .map(|batch| batch.into_iter().map(|item| item.into()).collect())
                    .collect(),
            )
        }

        #[pyo3(name = "remove_vector")]
        fn remove_vector_py(&mut self, py: Python<'_>, id: String) -> PyResult<()> {
            await_future(py, self.remove_vector(&id))
        }

        #[pyo3(name = "remove_vectors")]
        fn remove_vectors_py(&mut self, py: Python<'_>, ids: Vec<String>) -> PyResult<()> {
            await_future(
                py,
                self.remove_vectors(&ids.iter().map(|id| id.as_str()).collect::<Vec<_>>()),
            )
        }

        #[pyo3(name = "clear")]
        fn clear_py(&mut self, py: Python<'_>) -> PyResult<()> {
            await_future(py, self.clear())
        }

        #[pyo3(name = "count")]
        fn count_py(&self, py: Python<'_>) -> PyResult<usize> {
            await_future(py, self.count())
        }
    }
}

#[cfg(feature = "nodejs")]
mod node {
    use napi::Status;
    use napi_derive::napi;

    use super::*;

    #[allow(unused)]
    #[napi(js_name = "VectorStoreMetadata")]
    pub type JsVectorStoreMetadata = HashMap<String, Value>; // dummy type to generate type alias in d.ts

    #[napi]
    impl VectorStore {
        #[napi(js_name = "newFaiss")]
        pub async fn new_faiss_js(dim: u32) -> napi::Result<Self> {
            VectorStore::new_faiss(dim)
                .await
                .map_err(|e| napi::Error::new(Status::GenericFailure, e.to_string()))
        }

        #[napi(js_name = "newChroma")]
        pub async fn new_chroma_js(
            url: String,
            collection_name: Option<String>,
        ) -> napi::Result<Self> {
            VectorStore::new_chroma(url, collection_name)
                .await
                .map_err(|e| napi::Error::new(Status::GenericFailure, e.to_string()))
        }

        #[napi(js_name = "addVector")]
        pub async unsafe fn add_vector_js(
            &mut self,
            input: VectorStoreAddInput,
        ) -> napi::Result<String> {
            self.add_vector(input)
                .await
                .map_err(|e| napi::Error::new(Status::GenericFailure, e.to_string()))
        }

        #[napi(js_name = "addVectors")]
        pub async unsafe fn add_vectors_js(
            &mut self,
            inputs: Vec<VectorStoreAddInput>,
        ) -> napi::Result<Vec<String>> {
            self.add_vectors(inputs)
                .await
                .map_err(|e| napi::Error::new(Status::GenericFailure, e.to_string()))
        }

        #[napi(js_name = "getById")]
        pub async fn get_by_id_js(&self, id: String) -> napi::Result<Option<VectorStoreGetResult>> {
            self.get_by_id(&id)
                .await
                .map_err(|e| napi::Error::new(Status::GenericFailure, e.to_string()))
        }

        #[napi(js_name = "getByIds")]
        pub async fn get_by_ids_js(
            &self,
            ids: Vec<String>,
        ) -> napi::Result<Vec<VectorStoreGetResult>> {
            self.get_by_ids(
                ids.iter()
                    .map(|id| id.as_str())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
            .await
            .map_err(|e| napi::Error::new(Status::GenericFailure, e.to_string()))
        }

        #[napi(js_name = "retrieve")]
        pub async fn retrieve_js(
            &self,
            query_embedding: Embedding,
            top_k: u32,
        ) -> napi::Result<Vec<VectorStoreRetrieveResult>> {
            self.retrieve(query_embedding, top_k as usize)
                .await
                .map_err(|e| napi::Error::new(Status::GenericFailure, e.to_string()))
        }

        #[napi(js_name = "batchRetrieve")]
        pub async fn batch_retrieve_js(
            &self,
            query_embeddings: Vec<Embedding>,
            top_k: u32,
        ) -> napi::Result<Vec<Vec<VectorStoreRetrieveResult>>> {
            self.batch_retrieve(query_embeddings, top_k as usize)
                .await
                .map_err(|e| napi::Error::new(Status::GenericFailure, e.to_string()))
        }

        #[napi(js_name = "removeVector")]
        pub async unsafe fn remove_vector_js(&mut self, id: String) -> napi::Result<()> {
            self.remove_vector(&id)
                .await
                .map_err(|e| napi::Error::new(Status::GenericFailure, e.to_string()))
        }

        #[napi(js_name = "removeVectors")]
        pub async unsafe fn remove_vectors_js(&mut self, ids: Vec<String>) -> napi::Result<()> {
            self.remove_vectors(
                ids.iter()
                    .map(|id| id.as_str())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
            .await
            .map_err(|e| napi::Error::new(Status::GenericFailure, e.to_string()))
        }

        #[napi(js_name = "clear")]
        pub async unsafe fn clear_js(&mut self) -> napi::Result<()> {
            self.clear()
                .await
                .map_err(|e| napi::Error::new(Status::GenericFailure, e.to_string()))
        }

        #[napi(js_name = "count")]
        pub async fn count_js(&self) -> napi::Result<u32> {
            self.count()
                .await
                .map(|count| count as u32)
                .map_err(|e| napi::Error::new(Status::GenericFailure, e.to_string()))
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
        type VectorStoreMetadata = Record<string, any>;
        "#
    );

    #[wasm_bindgen]
    impl VectorStore {
        #[wasm_bindgen(js_name = "newFaiss")]
        pub async fn new_faiss_js(dim: u32) -> Result<Self, js_sys::Error> {
            VectorStore::new_faiss(dim)
                .await
                .map_err(|e| js_sys::Error::new(&e.to_string()))
        }

        #[wasm_bindgen(js_name = "newChroma")]
        pub async fn new_chroma_js(
            url: String,
            #[wasm_bindgen(js_name = "collectionName")] collection_name: Option<String>,
        ) -> Result<Self, js_sys::Error> {
            VectorStore::new_chroma(url, collection_name)
                .await
                .map_err(|e| js_sys::Error::new(&e.to_string()))
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
