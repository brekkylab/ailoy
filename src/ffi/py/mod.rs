pub(crate) mod base;
pub(crate) mod cache_progress;
#[cfg(feature = "ailoy-model-cli")]
pub(crate) mod cli;
pub(crate) mod string_enum;

use pyo3::prelude::*;
use pyo3_stub_gen::{Result, generate::StubInfo};

use crate::{
    agent::{Agent, AgentConfig},
    ffi::py::cache_progress::PyCacheProgress as CacheProgress,
    knowledge::{Knowledge, KnowledgeConfig},
    model::{
        DocumentPolyfill, EmbeddingModel, Grammar, KVCacheConfig, LangModel, LangModelInferConfig,
    },
    tool::{MCPClient, Tool},
    value::{
        Document, FinishReason, Message, MessageDelta, MessageDeltaOutput, MessageOutput, Part,
        PartDelta, PartDeltaFunction, PartFunction, PartImage, ToolDesc,
        message::py::{
            MessageDeltaOutputIterator, MessageDeltaOutputSyncIterator, MessageOutputIterator,
            MessageOutputSyncIterator,
        },
    },
    vector_store::{
        VectorStore, VectorStoreAddInput, VectorStoreGetResult, VectorStoreRetrieveResult,
    },
};

#[pymodule(name = "_core")]
fn ailoy_py(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    // Add classes in alphabetical order
    m.add_class::<Agent>()?;
    m.add_class::<AgentConfig>()?;
    m.add_class::<CacheProgress>()?;
    m.add_class::<Document>()?;
    m.add_class::<DocumentPolyfill>()?;
    m.add_class::<EmbeddingModel>()?;
    m.add_class::<FinishReason>()?;
    m.add_class::<Grammar>()?;
    m.add_class::<Knowledge>()?;
    m.add_class::<KnowledgeConfig>()?;
    m.add_class::<KVCacheConfig>()?;
    m.add_class::<LangModel>()?;
    m.add_class::<LangModelInferConfig>()?;
    m.add_class::<MCPClient>()?;
    m.add_class::<Message>()?;
    m.add_class::<MessageDelta>()?;
    m.add_class::<MessageDeltaOutput>()?;
    m.add_class::<MessageDeltaOutputIterator>()?;
    m.add_class::<MessageDeltaOutputSyncIterator>()?;
    m.add_class::<MessageOutput>()?;
    m.add_class::<MessageOutputIterator>()?;
    m.add_class::<MessageOutputSyncIterator>()?;
    m.add_class::<Part>()?;
    m.add_class::<PartDelta>()?;
    m.add_class::<PartDeltaFunction>()?;
    m.add_class::<PartFunction>()?;
    m.add_class::<PartImage>()?;
    m.add_class::<Tool>()?;
    m.add_class::<ToolDesc>()?;
    m.add_class::<VectorStore>()?;
    m.add_class::<VectorStoreAddInput>()?;
    m.add_class::<VectorStoreGetResult>()?;
    m.add_class::<VectorStoreRetrieveResult>()?;

    #[cfg(feature = "ailoy-model-cli")]
    m.add_function(wrap_pyfunction!(cli::ailoy_model_cli, m)?)?;

    Ok(())
}

#[doc(hidden)]
pub fn py_stub_info() -> Result<StubInfo> {
    let manifest_dir: &::std::path::Path = env!("CARGO_MANIFEST_DIR").as_ref();
    StubInfo::from_pyproject_toml(manifest_dir.join("bindings/python/pyproject.toml"))
}
