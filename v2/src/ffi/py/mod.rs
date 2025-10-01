// mod agent;
pub(crate) mod base;
pub(crate) mod cache_progress;
pub(crate) mod vector_store;

// use agent::{
//     PyAgent as Agent, PyAgentRunIterator as AgentRunIterator,
//     PyAgentRunSyncIterator as AgentRunSyncIterator,
// };
use pyo3::prelude::*;
use pyo3_stub_gen::{Result, generate::StubInfo};

use crate::ffi::py::vector_store::{
    BaseVectorStore, ChromaVectorStore, FaissVectorStore,
    PyVectorStoreAddInput as VectorStoreAddInput, PyVectorStoreGetResult as VectorStoreGetResult,
    PyVectorStoreRetrieveResult as VectorStoreRetrieveResult,
};

#[pymodule(name = "_core")]
fn ailoy_py(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    // Add classes in alphabetical order
    // m.add_class::<Agent>()?;
    // m.add_class::<AgentRunIterator>()?;
    // m.add_class::<AgentRunSyncIterator>()?;
    // m.add_class::<AnthropicLanguageModel>()?;
    m.add_class::<BaseVectorStore>()?;
    // m.add_class::<BaseTool>()?;
    // m.add_class::<BuiltinTool>()?;
    m.add_class::<ChromaVectorStore>()?;
    m.add_class::<FaissVectorStore>()?;
    // m.add_class::<MCPTool>()?;
    // m.add_class::<MCPTransport>()?;
    // m.add_class::<PythonAsyncFunctionTool>()?;
    // m.add_class::<PythonFunctionTool>()?;
    // m.add_class::<ToolDesc>()?;
    m.add_class::<VectorStoreAddInput>()?;
    m.add_class::<VectorStoreGetResult>()?;
    m.add_class::<VectorStoreRetrieveResult>()?;
    m.add_class::<crate::model::EmbeddingModel>()?;
    m.add_class::<crate::model::LangModel>()?;
    m.add_class::<crate::value::FinishReason>()?;
    m.add_class::<crate::value::Message>()?;
    m.add_class::<crate::value::MessageDelta>()?;
    m.add_class::<crate::value::MessageOutput>()?;
    m.add_class::<crate::value::Part>()?;
    m.add_class::<crate::value::PartDelta>()?;
    m.add_class::<crate::value::PartDeltaFunction>()?;
    m.add_class::<crate::value::PartImage>()?;
    m.add_class::<crate::value::Role>()?;
    m.add_class::<crate::value::ToolDesc>()?;
    Ok(())
}

pub fn stub_info() -> Result<StubInfo> {
    let manifest_dir: &::std::path::Path = env!("CARGO_MANIFEST_DIR").as_ref();
    StubInfo::from_pyproject_toml(manifest_dir.join("bindings/python/pyproject.toml"))
}
