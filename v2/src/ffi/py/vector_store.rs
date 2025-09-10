use std::sync::Arc;

use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyDict};
use pyo3_stub_gen::derive::*;

use crate::{
    ffi::py::base::{json_to_pydict, pydict_to_json},
    knowledge_base::{AddInput, ChromaStore, FaissStore, GetResult, RetrieveResult, VectorStore},
};

pub type Embedding = Vec<f32>;

#[gen_stub_pyclass]
#[pyclass(get_all, set_all)]
pub struct VectorStoreAddInput {
    pub embedding: Embedding,
    pub document: String,
    pub metadata: Option<Py<PyDict>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl VectorStoreAddInput {
    #[new]
    #[pyo3(signature = (embedding, document, metadata = None))]
    fn __new__(embedding: Embedding, document: String, metadata: Option<Py<PyDict>>) -> Self {
        Self {
            embedding,
            document,
            metadata,
        }
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        let _ = dict.set_item("embedding", &self.embedding);
        let _ = dict.set_item("document", &self.document);
        if let Some(metadata) = &self.metadata {
            let _ = dict.set_item("metadata", metadata);
        }
        Ok(dict.unbind())
    }
}

impl Into<AddInput> for Py<VectorStoreAddInput> {
    fn into(self) -> AddInput {
        Python::attach(|py| {
            let input = self.borrow(py);
            let metadata = match &input.metadata {
                Some(metadata) => {
                    let m = metadata.bind(py);
                    Some(pydict_to_json(py, m).unwrap())
                }
                None => None,
            };
            AddInput {
                embedding: input.embedding.clone(),
                document: input.document.clone(),
                metadata: metadata,
            }
        })
    }
}

#[gen_stub_pyclass]
#[pyclass(dict, get_all)]
pub struct VectorStoreGetResult {
    pub id: String,
    pub document: String,
    pub metadata: Option<Py<PyDict>>,
    pub embedding: Embedding,
}

impl From<GetResult> for VectorStoreGetResult {
    fn from(value: GetResult) -> Self {
        Python::attach(|py| {
            let metadata = match value.metadata {
                Some(metadata) => {
                    match json_to_pydict(py, &serde_json::Value::Object(metadata)).unwrap() {
                        Some(metadata) => Some(metadata.unbind()),
                        None => None,
                    }
                }
                None => None,
            };
            Self {
                id: value.id,
                document: value.document,
                metadata: metadata,
                embedding: value.embedding,
            }
        })
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl VectorStoreGetResult {
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        let _ = dict.set_item("id", &self.id);
        let _ = dict.set_item("embedding", &self.embedding);
        let _ = dict.set_item("document", &self.document);
        if let Some(metadata) = &self.metadata {
            let _ = dict.set_item("metadata", metadata);
        }
        Ok(dict.unbind())
    }

    fn __repr__(&self) -> String {
        format!("VectorStoreGetResult(id=\"{}\")", self.id)
    }
}

#[gen_stub_pyclass]
#[pyclass(dict, get_all)]
pub struct VectorStoreRetrieveResult {
    pub id: String,
    pub document: String,
    pub metadata: Option<Py<PyDict>>,
    pub distance: f32,
}

impl From<RetrieveResult> for VectorStoreRetrieveResult {
    fn from(value: RetrieveResult) -> Self {
        Python::attach(|py| {
            let metadata = match value.metadata {
                Some(metadata) => {
                    match json_to_pydict(py, &serde_json::Value::Object(metadata)).unwrap() {
                        Some(metadata) => Some(metadata.unbind()),
                        None => None,
                    }
                }
                None => None,
            };
            Self {
                id: value.id,
                document: value.document,
                metadata: metadata,
                distance: value.distance,
            }
        })
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl VectorStoreRetrieveResult {
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        let _ = dict.set_item("id", &self.id);
        let _ = dict.set_item("distance", &self.distance);
        let _ = dict.set_item("document", &self.document);
        if let Some(metadata) = &self.metadata {
            let _ = dict.set_item("metadata", metadata);
        }
        Ok(dict.unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "VectorStoreRetrieveResult(id=\"{}\", distance={})",
            self.id, self.distance
        )
    }
}

#[gen_stub_pyclass]
#[pyclass(subclass)]
pub struct BaseVectorStore {}

pub trait VectorStoreMethods<T: VectorStore + 'static> {
    fn inner(&self) -> &T;

    fn inner_mut(&mut self) -> &mut T;

    async fn _add_vector(&mut self, input: Py<VectorStoreAddInput>) -> PyResult<String> {
        let id = self
            .inner_mut()
            .add_vector(input.into())
            .await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(id)
    }

    async fn _add_vectors(
        &mut self,
        inputs: Vec<Py<VectorStoreAddInput>>,
    ) -> PyResult<Vec<String>> {
        let ids = self
            .inner_mut()
            .add_vectors(inputs.into_iter().map(|input| input.into()).collect())
            .await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(ids)
    }

    async fn _get_by_id(&self, id: String) -> PyResult<Option<VectorStoreGetResult>> {
        let result = self
            .inner()
            .get_by_id(id.as_str())
            .await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        match result {
            Some(result) => Ok(Some(VectorStoreGetResult::from(result))),
            None => Ok(None),
        }
    }

    async fn _get_by_ids(&self, ids: Vec<String>) -> PyResult<Vec<VectorStoreGetResult>> {
        Ok(self
            .inner()
            .get_by_ids(&ids.iter().map(|id| id.as_str()).collect::<Vec<_>>())
            .await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .into_iter()
            .map(|result| VectorStoreGetResult::from(result))
            .collect())
    }

    async fn _retrieve(
        &self,
        query_embedding: Embedding,
        top_k: usize,
    ) -> PyResult<Vec<VectorStoreRetrieveResult>> {
        Ok(self
            .inner()
            .retrieve(query_embedding, top_k)
            .await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .into_iter()
            .map(|result| VectorStoreRetrieveResult::from(result))
            .collect())
    }

    async fn _batch_retrieve(
        &self,
        query_embeddings: Vec<Embedding>,
        top_k: usize,
    ) -> PyResult<Vec<Vec<VectorStoreRetrieveResult>>> {
        Ok(self
            .inner()
            .batch_retrieve(query_embeddings, top_k)
            .await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .into_iter()
            .map(|batch| {
                batch
                    .into_iter()
                    .map(|item| VectorStoreRetrieveResult::from(item))
                    .collect()
            })
            .collect())
    }

    async fn _remove_vector(&mut self, id: String) -> PyResult<()> {
        self.inner_mut()
            .remove_vector(id.as_str())
            .await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    async fn _remove_vectors(&mut self, ids: Vec<String>) -> PyResult<()> {
        self.inner_mut()
            .remove_vectors(&ids.iter().map(|id| id.as_str()).collect::<Vec<_>>())
            .await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    async fn _clear(&mut self) -> PyResult<()> {
        self.inner_mut()
            .clear()
            .await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    async fn _count(&self) -> PyResult<usize> {
        self.inner()
            .count()
            .await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "FaissVectorStore", extends = BaseVectorStore)]
pub struct FaissVectorStore {
    inner: FaissStore,
}

impl VectorStoreMethods<FaissStore> for FaissVectorStore {
    fn inner(&self) -> &FaissStore {
        &self.inner
    }

    fn inner_mut(&mut self) -> &mut FaissStore {
        &mut self.inner
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl FaissVectorStore {
    #[new]
    fn __new__(py: Python<'_>, dim: i32) -> PyResult<Py<Self>> {
        let runtime = tokio::runtime::Runtime::new().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create runtime: {}", e.to_string()))
        })?;
        let inner = runtime.block_on(async {
            FaissStore::new(dim)
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        Py::new(py, (FaissVectorStore { inner }, BaseVectorStore {}))
    }

    async fn add_vector(&mut self, input: Py<VectorStoreAddInput>) -> PyResult<String> {
        self._add_vector(input).await
    }

    async fn add_vectors(&mut self, inputs: Vec<Py<VectorStoreAddInput>>) -> PyResult<Vec<String>> {
        self._add_vectors(inputs).await
    }

    async fn get_by_id(&self, id: String) -> PyResult<Option<VectorStoreGetResult>> {
        self._get_by_id(id).await
    }

    async fn get_by_ids(&self, ids: Vec<String>) -> PyResult<Vec<VectorStoreGetResult>> {
        self._get_by_ids(ids).await
    }

    async fn retrieve(
        &self,
        query_embedding: Embedding,
        top_k: usize,
    ) -> PyResult<Vec<VectorStoreRetrieveResult>> {
        self._retrieve(query_embedding, top_k).await
    }

    async fn batch_retrieve(
        &self,
        query_embeddings: Vec<Embedding>,
        top_k: usize,
    ) -> PyResult<Vec<Vec<VectorStoreRetrieveResult>>> {
        self._batch_retrieve(query_embeddings, top_k).await
    }

    async fn remove_vector(&mut self, id: String) -> PyResult<()> {
        self._remove_vector(id).await
    }

    async fn remove_vectors(&mut self, ids: Vec<String>) -> PyResult<()> {
        self._remove_vectors(ids).await
    }

    async fn clear(&mut self) -> PyResult<()> {
        self._clear().await
    }

    async fn count(&self) -> PyResult<usize> {
        self._count().await
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "ChromaVectorStore", extends = BaseVectorStore)]
pub struct ChromaVectorStore {
    inner: ChromaStore,
    rt: Arc<tokio::runtime::Runtime>,
}

impl VectorStoreMethods<ChromaStore> for ChromaVectorStore {
    fn inner(&self) -> &ChromaStore {
        &self.inner
    }

    fn inner_mut(&mut self) -> &mut ChromaStore {
        &mut self.inner
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl ChromaVectorStore {
    #[new]
    fn __new__(py: Python<'_>, chroma_url: String, collection_name: String) -> PyResult<Py<Self>> {
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create runtime: {}", e.to_string()))
        })?;
        let inner = rt.block_on(async {
            ChromaStore::new(chroma_url.as_str(), collection_name.as_str())
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        Py::new(
            py,
            (
                ChromaVectorStore {
                    inner,
                    rt: Arc::new(rt),
                },
                BaseVectorStore {},
            ),
        )
    }

    async fn add_vector(&mut self, input: Py<VectorStoreAddInput>) -> PyResult<String> {
        self.rt.clone().block_on(self._add_vector(input))
    }

    async fn add_vectors(&mut self, inputs: Vec<Py<VectorStoreAddInput>>) -> PyResult<Vec<String>> {
        self.rt.clone().block_on(self._add_vectors(inputs))
    }

    async fn get_by_id(&self, id: String) -> PyResult<Option<VectorStoreGetResult>> {
        self.rt.block_on(self._get_by_id(id))
    }

    async fn get_by_ids(&self, ids: Vec<String>) -> PyResult<Vec<VectorStoreGetResult>> {
        self.rt.block_on(self._get_by_ids(ids))
    }

    async fn retrieve(
        &self,
        query_embedding: Embedding,
        top_k: usize,
    ) -> PyResult<Vec<VectorStoreRetrieveResult>> {
        self.rt.block_on(self._retrieve(query_embedding, top_k))
    }

    async fn batch_retrieve(
        &self,
        query_embeddings: Vec<Embedding>,
        top_k: usize,
    ) -> PyResult<Vec<Vec<VectorStoreRetrieveResult>>> {
        self.rt
            .block_on(self._batch_retrieve(query_embeddings, top_k))
    }

    async fn remove_vector(&mut self, id: String) -> PyResult<()> {
        self.rt.clone().block_on(self._remove_vector(id))
    }

    async fn remove_vectors(&mut self, ids: Vec<String>) -> PyResult<()> {
        self.rt.clone().block_on(self._remove_vectors(ids))
    }

    async fn clear(&mut self) -> PyResult<()> {
        self.rt.clone().block_on(self._clear())
    }

    async fn count(&self) -> PyResult<usize> {
        self.rt.block_on(self._count())
    }
}
