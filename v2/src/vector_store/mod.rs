mod api;
mod local;

use std::{collections::HashMap, sync::Arc};

use ailoy_macros::{maybe_send_sync, multi_platform_async_trait};
pub use api::*;
use futures::lock::Mutex;
pub use local::*;
use serde::{Deserialize, Serialize};

use crate::{utils::Normalize, value::Value};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Embedding(pub Vec<f32>);

impl From<Vec<f32>> for Embedding {
    fn from(value: Vec<f32>) -> Self {
        Self(value)
    }
}

impl Into<Vec<f32>> for Embedding {
    fn into(self) -> Vec<f32> {
        self.0
    }
}

impl Normalize for Embedding {
    fn normalized(&self) -> Self {
        Self(self.0.normalized())
    }
}

impl Embedding {
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

pub type Metadata = HashMap<String, Value>;

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3_stub_gen_derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, set_all))]
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
    pub metadata: Option<Metadata>,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3_stub_gen_derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, set_all))]
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
    pub metadata: Option<Metadata>,
    pub embedding: Embedding,
}

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3_stub_gen_derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, set_all))]
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
    pub metadata: Option<Metadata>,
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
#[cfg_attr(feature = "python", pyo3::pyclass)]
#[cfg_attr(feature = "nodejs", napi_derive::napi)]
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

#[cfg(feature = "python")]
mod py {
    use pyo3::{
        prelude::*,
        types::{PyFloat, PyList, PyType},
    };
    use pyo3_stub_gen::{PyStubType, TypeInfo};
    use pyo3_stub_gen_derive::*;

    use super::*;
    use crate::ffi::py::base::await_future;

    impl<'py> IntoPyObject<'py> for Embedding {
        type Target = PyList;
        type Output = Bound<'py, PyList>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            PyList::new(py, self.0).into()
        }
    }

    impl<'py> FromPyObject<'py> for Embedding {
        fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
            let vec: Vec<f32> = ob.extract()?;
            Ok(Embedding(vec))
        }
    }

    impl PyStubType for Embedding {
        fn type_output() -> pyo3_stub_gen::TypeInfo {
            TypeInfo::list_of::<PyFloat>()
        }
    }

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
        fn __new__(embedding: Embedding, document: String, metadata: Option<Metadata>) -> Self {
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
        fn new_faiss_py<'a>(
            _cls: &Bound<'a, PyType>,
            py: Python<'a>,
            dim: i32,
        ) -> PyResult<Py<Self>> {
            let store = await_future(FaissStore::new(dim))?;
            Py::new(py, Self::new_faiss(store))
        }

        #[classmethod]
        #[pyo3(name = "new_chroma")]
        fn new_chroma_py<'a>(
            _cls: &Bound<'a, PyType>,
            py: Python<'a>,
            url: String,
            collection_name: Option<String>,
        ) -> PyResult<Py<Self>> {
            let store = await_future(ChromaStore::new(&url, collection_name.as_deref()))?;
            Py::new(py, Self::new_chroma(store))
        }

        #[pyo3(name = "add_vector")]
        fn add_vector_py(&mut self, input: Py<VectorStoreAddInput>) -> PyResult<String> {
            await_future(self.add_vector(input.into()))
        }

        #[pyo3(name = "add_vectors")]
        fn add_vectors_py(
            &mut self,
            inputs: Vec<Py<VectorStoreAddInput>>,
        ) -> PyResult<Vec<String>> {
            await_future(self.add_vectors(inputs.into_iter().map(|input| input.into()).collect()))
        }

        #[pyo3(name = "get_by_id")]
        fn get_by_id_py(&self, id: String) -> PyResult<Option<VectorStoreGetResult>> {
            let result = await_future(self.get_by_id(&id))?;
            match result {
                Some(result) => Ok(Some(result.into())),
                None => Ok(None),
            }
        }

        #[pyo3(name = "get_by_ids")]
        fn get_by_ids_py(&self, ids: Vec<String>) -> PyResult<Vec<VectorStoreGetResult>> {
            Ok(await_future(
                self.get_by_ids(&ids.iter().map(|id| id.as_str()).collect::<Vec<_>>()),
            )?
            .into_iter()
            .map(|result| result.into())
            .collect::<Vec<_>>())
        }

        #[pyo3(name = "retrieve")]
        fn retrieve_py(
            &self,
            query_embedding: Embedding,
            top_k: usize,
        ) -> PyResult<Vec<VectorStoreRetrieveResult>> {
            Ok(await_future(self.retrieve(query_embedding, top_k))?
                .into_iter()
                .map(|result| result.into())
                .collect::<Vec<_>>())
        }

        #[pyo3(name = "batch_retrieve")]
        fn batch_retrieve_py(
            &self,
            query_embeddings: Vec<Embedding>,
            top_k: usize,
        ) -> PyResult<Vec<Vec<VectorStoreRetrieveResult>>> {
            Ok(await_future(self.batch_retrieve(query_embeddings, top_k))?
                .into_iter()
                .map(|batch| batch.into_iter().map(|item| item.into()).collect())
                .collect())
        }

        #[pyo3(name = "remove_vector")]
        fn remove_vector_py(&mut self, id: String) -> PyResult<()> {
            await_future(self.remove_vector(&id))
        }

        #[pyo3(name = "remove_vectors")]
        fn remove_vectors_py(&mut self, ids: Vec<String>) -> PyResult<()> {
            await_future(self.remove_vectors(&ids.iter().map(|id| id.as_str()).collect::<Vec<_>>()))
        }

        #[pyo3(name = "clear")]
        fn clear_py(&mut self) -> PyResult<()> {
            await_future(self.clear())
        }

        #[pyo3(name = "count")]
        fn count_py(&self) -> PyResult<usize> {
            await_future(self.count())
        }
    }
}

#[cfg(feature = "nodejs")]
mod node {
    use napi::{Status, bindgen_prelude::*};
    use napi_derive::*;

    use super::*;

    #[allow(unused)]
    #[napi(js_name = "Embedding")]
    pub type JsEmbedding = Float32Array; // dummy type to generate type alias in d.ts

    #[allow(unused)]
    #[napi(js_name = "Metadata")]
    pub type JsMetadata = HashMap<String, Value>; // dummy type to generate type alias in d.ts

    impl FromNapiValue for Embedding {
        unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> Result<Self> {
            let array = unsafe { Float32Array::from_napi_value(env, napi_val) }?;
            Ok(Self(array.to_vec()))
        }
    }

    impl ToNapiValue for Embedding {
        unsafe fn to_napi_value(env: sys::napi_env, val: Self) -> Result<sys::napi_value> {
            let env = Env::from_raw(env);
            let array = Float32Array::new(val.0.clone());
            let unknown = array.into_unknown(&env)?;
            Ok(unknown.raw())
        }
    }

    impl TypeName for Embedding {
        fn type_name() -> &'static str {
            "Float32Array"
        }

        fn value_type() -> ValueType {
            ValueType::Object
        }
    }

    impl ValidateNapiValue for Embedding {}

    #[napi]
    impl VectorStore {
        #[napi(js_name = "newFaiss")]
        pub async fn new_faiss_js(dim: i32) -> napi::Result<Self> {
            let store = FaissStore::new(dim)
                .await
                .map_err(|e| napi::Error::new(Status::GenericFailure, e.to_string()))?;
            Ok(Self::new_faiss(store))
        }

        #[napi(js_name = "newChroma")]
        pub async fn new_chroma_js(
            url: String,
            collection_name: Option<String>,
        ) -> napi::Result<Self> {
            let store = ChromaStore::new(&url, collection_name.as_deref())
                .await
                .map_err(|e| napi::Error::new(Status::GenericFailure, e.to_string()))?;
            Ok(Self::new_chroma(store))
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
    use js_sys::Float32Array;
    use wasm_bindgen::{
        convert::{FromWasmAbi, IntoWasmAbi},
        describe::WasmDescribe,
        prelude::*,
    };

    use super::*;

    #[wasm_bindgen(typescript_custom_section)]
    const TS_APPEND_CONTENT: &'static str = dedent::dedent!(
        r#"
        type Embedding = Float32Array;
        type Metadata = Record<string, any>;
        "#
    );

    impl WasmDescribe for Embedding {
        fn describe() {
            Float32Array::describe()
        }
    }

    impl IntoWasmAbi for Embedding {
        type Abi = <Float32Array as IntoWasmAbi>::Abi;

        fn into_abi(self) -> Self::Abi {
            let array = Float32Array::new_with_length(self.0.len() as u32);
            array.copy_from(&self.0);
            array.into_abi()
        }
    }

    impl FromWasmAbi for Embedding {
        type Abi = <Float32Array as IntoWasmAbi>::Abi;

        unsafe fn from_abi(js: Self::Abi) -> Self {
            let array = unsafe { Float32Array::from_abi(js) };
            let vec = array.to_vec();
            Embedding(vec)
        }
    }

    impl From<JsValue> for Embedding {
        fn from(value: JsValue) -> Self {
            let array = Float32Array::from(value);
            let vec = array.to_vec();
            Self(vec)
        }
    }

    impl Into<JsValue> for Embedding {
        fn into(self) -> JsValue {
            let array = Float32Array::new_with_length(self.0.len() as u32);
            array.copy_from(&self.0);
            array.into()
        }
    }

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
