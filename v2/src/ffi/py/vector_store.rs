use pyo3::{exceptions::PyNotImplementedError, prelude::*, types::PyDict};
use pyo3_stub_gen::derive::*;

use crate::{
    ffi::py::base::{await_future, json_to_pydict, pydict_to_json},
    vector_store::{
        ChromaStore, FaissStore, VectorStore, VectorStoreAddInput, VectorStoreGetResult,
        VectorStoreRetrieveResult,
    },
};

pub type Embedding = Vec<f32>;

#[gen_stub_pyclass]
#[pyclass(name = "VectorStoreAddInput", get_all, set_all)]
pub struct PyVectorStoreAddInput {
    pub embedding: Embedding,
    pub document: String,
    pub metadata: Option<Py<PyDict>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyVectorStoreAddInput {
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

impl Into<VectorStoreAddInput> for Py<PyVectorStoreAddInput> {
    fn into(self) -> VectorStoreAddInput {
        Python::attach(|py| {
            let input = self.borrow(py);
            let metadata = match &input.metadata {
                Some(metadata) => {
                    let m = metadata.bind(py);
                    Some(pydict_to_json(py, m).unwrap())
                }
                None => None,
            };
            VectorStoreAddInput {
                embedding: input.embedding.clone(),
                document: input.document.clone(),
                metadata: metadata,
            }
        })
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "VectorStoreGetResult", dict, get_all)]
pub struct PyVectorStoreGetResult {
    pub id: String,
    pub document: String,
    pub metadata: Option<Py<PyDict>>,
    pub embedding: Embedding,
}

impl From<VectorStoreGetResult> for PyVectorStoreGetResult {
    fn from(value: VectorStoreGetResult) -> Self {
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
impl PyVectorStoreGetResult {
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
#[pyclass(name = "VectorStoreRetrieveResult", dict, get_all)]
pub struct PyVectorStoreRetrieveResult {
    pub id: String,
    pub document: String,
    pub metadata: Option<Py<PyDict>>,
    pub distance: f32,
}

impl From<VectorStoreRetrieveResult> for PyVectorStoreRetrieveResult {
    fn from(value: VectorStoreRetrieveResult) -> Self {
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
impl PyVectorStoreRetrieveResult {
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

    fn _add_vector(&mut self, input: Py<PyVectorStoreAddInput>) -> PyResult<String> {
        await_future(self.inner_mut().add_vector(input.into()))
    }

    fn _add_vectors(&mut self, inputs: Vec<Py<PyVectorStoreAddInput>>) -> PyResult<Vec<String>> {
        await_future(
            self.inner_mut()
                .add_vectors(inputs.into_iter().map(|input| input.into()).collect()),
        )
    }

    fn _get_by_id(&self, id: String) -> PyResult<Option<PyVectorStoreGetResult>> {
        let result = await_future(self.inner().get_by_id(id.as_str()))?;
        match result {
            Some(result) => Ok(Some(result.into())),
            None => Ok(None),
        }
    }

    fn _get_by_ids(&self, ids: Vec<String>) -> PyResult<Vec<PyVectorStoreGetResult>> {
        Ok(await_future(
            self.inner()
                .get_by_ids(&ids.iter().map(|id| id.as_str()).collect::<Vec<_>>()),
        )?
        .into_iter()
        .map(|result| result.into())
        .collect::<Vec<_>>())
    }

    fn _retrieve(
        &self,
        query_embedding: Embedding,
        top_k: usize,
    ) -> PyResult<Vec<PyVectorStoreRetrieveResult>> {
        Ok(await_future(self.inner().retrieve(query_embedding, top_k))?
            .into_iter()
            .map(|result| result.into())
            .collect::<Vec<_>>())
    }

    fn _batch_retrieve(
        &self,
        query_embeddings: Vec<Embedding>,
        top_k: usize,
    ) -> PyResult<Vec<Vec<PyVectorStoreRetrieveResult>>> {
        Ok(
            await_future(self.inner().batch_retrieve(query_embeddings, top_k))?
                .into_iter()
                .map(|batch| batch.into_iter().map(|item| item.into()).collect())
                .collect(),
        )
    }

    fn _remove_vector(&mut self, id: String) -> PyResult<()> {
        await_future(self.inner_mut().remove_vector(id.as_str()))
    }

    fn _remove_vectors(&mut self, ids: Vec<String>) -> PyResult<()> {
        await_future(
            self.inner_mut()
                .remove_vectors(&ids.iter().map(|id| id.as_str()).collect::<Vec<_>>()),
        )
    }

    fn _clear(&mut self) -> PyResult<()> {
        await_future(self.inner_mut().clear())
    }

    fn _count(&self) -> PyResult<usize> {
        await_future(self.inner().count())
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl BaseVectorStore {
    #[allow(unused_variables)]
    fn add_vector(&mut self, input: Py<PyVectorStoreAddInput>) -> PyResult<String> {
        Err(PyNotImplementedError::new_err(
            "Subclass of BaseVectorStore must implement 'add_vector'",
        ))
    }

    #[allow(unused_variables)]
    fn add_vectors(&mut self, inputs: Vec<Py<PyVectorStoreAddInput>>) -> PyResult<Vec<String>> {
        Err(PyNotImplementedError::new_err(
            "Subclass of BaseVectorStore must implement 'add_vectors'",
        ))
    }

    #[allow(unused_variables)]
    fn get_by_id(&self, id: String) -> PyResult<Option<PyVectorStoreGetResult>> {
        Err(PyNotImplementedError::new_err(
            "Subclass of BaseVectorStore must implement 'get_by_id'",
        ))
    }

    #[allow(unused_variables)]
    fn get_by_ids(&self, ids: Vec<String>) -> PyResult<Vec<PyVectorStoreGetResult>> {
        Err(PyNotImplementedError::new_err(
            "Subclass of BaseVectorStore must implement 'get_by_ids'",
        ))
    }

    #[allow(unused_variables)]
    fn retrieve(
        &self,
        query_embedding: Embedding,
        top_k: usize,
    ) -> PyResult<Vec<PyVectorStoreRetrieveResult>> {
        Err(PyNotImplementedError::new_err(
            "Subclass of BaseVectorStore must implement 'retrieve'",
        ))
    }

    #[allow(unused_variables)]
    fn batch_retrieve(
        &self,
        query_embeddings: Vec<Embedding>,
        top_k: usize,
    ) -> PyResult<Vec<Vec<PyVectorStoreRetrieveResult>>> {
        Err(PyNotImplementedError::new_err(
            "Subclass of BaseVectorStore must implement 'batch_retrieve'",
        ))
    }

    #[allow(unused_variables)]
    fn remove_vector(&mut self, id: String) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "Subclass of BaseVectorStore must implement 'remove_vector'",
        ))
    }

    #[allow(unused_variables)]
    fn remove_vectors(&mut self, ids: Vec<String>) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "Subclass of BaseVectorStore must implement 'remove_vectors'",
        ))
    }

    #[allow(unused_variables)]
    fn clear(&mut self) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "Subclass of BaseVectorStore must implement 'clear'",
        ))
    }

    #[allow(unused_variables)]
    fn count(&self) -> PyResult<usize> {
        Err(PyNotImplementedError::new_err(
            "Subclass of BaseVectorStore must implement 'count'",
        ))
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
        let inner = await_future(FaissStore::new(dim))?;
        Py::new(py, (FaissVectorStore { inner }, BaseVectorStore {}))
    }

    fn add_vector(&mut self, input: Py<PyVectorStoreAddInput>) -> PyResult<String> {
        self._add_vector(input)
    }

    fn add_vectors(&mut self, inputs: Vec<Py<PyVectorStoreAddInput>>) -> PyResult<Vec<String>> {
        self._add_vectors(inputs)
    }

    fn get_by_id(&self, id: String) -> PyResult<Option<PyVectorStoreGetResult>> {
        self._get_by_id(id)
    }

    fn get_by_ids(&self, ids: Vec<String>) -> PyResult<Vec<PyVectorStoreGetResult>> {
        self._get_by_ids(ids)
    }

    fn retrieve(
        &self,
        query_embedding: Embedding,
        top_k: usize,
    ) -> PyResult<Vec<PyVectorStoreRetrieveResult>> {
        self._retrieve(query_embedding, top_k)
    }

    fn batch_retrieve(
        &self,
        query_embeddings: Vec<Embedding>,
        top_k: usize,
    ) -> PyResult<Vec<Vec<PyVectorStoreRetrieveResult>>> {
        self._batch_retrieve(query_embeddings, top_k)
    }

    fn remove_vector(&mut self, id: String) -> PyResult<()> {
        self._remove_vector(id)
    }

    fn remove_vectors(&mut self, ids: Vec<String>) -> PyResult<()> {
        self._remove_vectors(ids)
    }

    fn clear(&mut self) -> PyResult<()> {
        self._clear()
    }

    fn count(&self) -> PyResult<usize> {
        self._count()
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "ChromaVectorStore", extends = BaseVectorStore)]
pub struct ChromaVectorStore {
    inner: ChromaStore,
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
        let inner = await_future(ChromaStore::new(
            chroma_url.as_str(),
            collection_name.as_str(),
        ))?;
        Py::new(py, (ChromaVectorStore { inner }, BaseVectorStore {}))
    }

    fn collection_exists(&self, collection_name: String) -> PyResult<bool> {
        let col = await_future(self.inner.get_collection(collection_name.as_str()));
        Ok(col.is_ok())
    }

    #[pyo3(signature = (collection_name, metadata = None))]
    fn create_collection(
        &self,
        py: Python<'_>,
        collection_name: String,
        metadata: Option<Py<PyDict>>,
    ) -> PyResult<Py<PyDict>> {
        let metadata = metadata.map(|metadata| pydict_to_json(py, metadata.bind(py)).unwrap());
        let col = await_future(
            self.inner
                .create_collection(collection_name.as_str(), metadata),
        )?;
        let dict = PyDict::new(py);
        dict.set_item("id", col.id())?;
        dict.set_item("name", col.name())?;
        if let Some(m) = col.metadata() {
            dict.set_item(
                "metadata",
                json_to_pydict(py, &serde_json::Value::Object(m.clone()))?,
            )?;
        }
        Ok(dict.unbind())
    }

    fn delete_collection(&self, collection_name: String) -> PyResult<()> {
        await_future(self.inner.delete_collection(collection_name.as_str()))
    }

    fn add_vector(&mut self, input: Py<PyVectorStoreAddInput>) -> PyResult<String> {
        self._add_vector(input)
    }

    fn add_vectors(&mut self, inputs: Vec<Py<PyVectorStoreAddInput>>) -> PyResult<Vec<String>> {
        self._add_vectors(inputs)
    }

    fn get_by_id(&self, id: String) -> PyResult<Option<PyVectorStoreGetResult>> {
        self._get_by_id(id)
    }

    fn get_by_ids(&self, ids: Vec<String>) -> PyResult<Vec<PyVectorStoreGetResult>> {
        self._get_by_ids(ids)
    }

    fn retrieve(
        &self,
        query_embedding: Embedding,
        top_k: usize,
    ) -> PyResult<Vec<PyVectorStoreRetrieveResult>> {
        self._retrieve(query_embedding, top_k)
    }

    fn batch_retrieve(
        &self,
        query_embeddings: Vec<Embedding>,
        top_k: usize,
    ) -> PyResult<Vec<Vec<PyVectorStoreRetrieveResult>>> {
        self._batch_retrieve(query_embeddings, top_k)
    }

    fn remove_vector(&mut self, id: String) -> PyResult<()> {
        self._remove_vector(id)
    }

    fn remove_vectors(&mut self, ids: Vec<String>) -> PyResult<()> {
        self._remove_vectors(ids)
    }

    fn clear(&mut self) -> PyResult<()> {
        self._clear()
    }

    fn count(&self) -> PyResult<usize> {
        self._count()
    }
}
