mod agent;
mod base;
mod cache_progress;
mod embedding_model;
mod language_model;
mod tool;
mod value;
mod vector_store;

use agent::{
    PyAgent as Agent, PyAgentRunIterator as AgentRunIterator,
    PyAgentRunSyncIterator as AgentRunSyncIterator,
};
use cache_progress::{
    PyCacheProgress as CacheProgress, PyCacheProgressIterator as CacheProgressIterator,
    PyCacheProgressSyncIterator as CacheProgressSyncIterator,
};
use embedding_model::{
    PyEmbeddingModel as EmbeddingModel, PyLocalEmbeddingModel as LocalEmbeddingModel,
};
use language_model::{
    PyAnthropicLanguageModel as AnthropicLanguageModel,
    PyGeminiLanguageModel as GeminiLanguageModel, PyLanguageModel as LanguageModel,
    PyLanguageModelRunIterator as LanguageModelRunIterator,
    PyLanguageModelRunSyncIterator as LanguageModelRunSyncIterator,
    PyLocalLanguageModel as LocalLanguageModel, PyOpenAILanguageModel as OpenAILanguageModel,
    PyXAILanguageModel as XAILanguageModel,
};
use pyo3::prelude::*;
use pyo3_stub_gen::{Result, generate::StubInfo};
use tool::{
    PyBuiltinTool as BuiltinTool, PyMCPTool as MCPTool, PyTool as Tool, PythonAsyncFunctionTool,
    PythonFunctionTool,
};
use vector_store::{BaseVectorStore, ChromaVectorStore, FaissVectorStore};

use crate::{
    ffi::py::vector_store::{VectorStoreAddInput, VectorStoreGetResult, VectorStoreRetrieveResult},
    tool::mcp::MCPTransport,
    value::{FinishReason, Message, MessageAggregator, MessageOutput, Part, Role, ToolDesc},
};

#[pymodule(name = "_core")]
fn ailoy_py(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    // Add classes in alphabetical order
    m.add_class::<Agent>()?;
    m.add_class::<AgentRunIterator>()?;
    m.add_class::<AgentRunSyncIterator>()?;
    m.add_class::<AnthropicLanguageModel>()?;
    m.add_class::<BaseVectorStore>()?;
    m.add_class::<BuiltinTool>()?;
    m.add_class::<CacheProgress>()?;
    m.add_class::<CacheProgressIterator>()?;
    m.add_class::<CacheProgressSyncIterator>()?;
    m.add_class::<ChromaVectorStore>()?;
    m.add_class::<EmbeddingModel>()?;
    m.add_class::<FaissVectorStore>()?;
    m.add_class::<FinishReason>()?;
    m.add_class::<GeminiLanguageModel>()?;
    m.add_class::<LanguageModel>()?;
    m.add_class::<LanguageModelRunIterator>()?;
    m.add_class::<LanguageModelRunSyncIterator>()?;
    m.add_class::<LocalEmbeddingModel>()?;
    m.add_class::<LocalLanguageModel>()?;
    m.add_class::<Message>()?;
    m.add_class::<MessageAggregator>()?;
    m.add_class::<MessageOutput>()?;
    m.add_class::<MCPTool>()?;
    m.add_class::<MCPTransport>()?;
    m.add_class::<OpenAILanguageModel>()?;
    m.add_class::<Part>()?;
    m.add_class::<PythonAsyncFunctionTool>()?;
    m.add_class::<PythonFunctionTool>()?;
    m.add_class::<Role>()?;
    m.add_class::<Tool>()?;
    m.add_class::<ToolDesc>()?;
    m.add_class::<VectorStoreAddInput>()?;
    m.add_class::<VectorStoreGetResult>()?;
    m.add_class::<VectorStoreRetrieveResult>()?;
    m.add_class::<XAILanguageModel>()?;
    Ok(())
}

pub fn stub_info() -> Result<StubInfo> {
    let manifest_dir: &::std::path::Path = env!("CARGO_MANIFEST_DIR").as_ref();
    StubInfo::from_pyproject_toml(manifest_dir.join("bindings/python/pyproject.toml"))
}
