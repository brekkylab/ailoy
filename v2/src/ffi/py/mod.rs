// mod agent;
pub(crate) mod base;
pub(crate) mod cache_progress;
#[cfg(feature = "ailoy-model-cli")]
pub(crate) mod cli;

// use agent::{
//     PyAgent as Agent, PyAgentRunIterator as AgentRunIterator,
//     PyAgentRunSyncIterator as AgentRunSyncIterator,
// };
use pyo3::prelude::*;

#[pymodule(name = "_core")]
fn ailoy_py(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    // Add classes in alphabetical order
    // m.add_class::<Agent>()?;
    // m.add_class::<AgentRunIterator>()?;
    // m.add_class::<AgentRunSyncIterator>()?;
    // m.add_class::<AnthropicLanguageModel>()?;
    // m.add_class::<BaseTool>()?;
    // m.add_class::<BuiltinTool>()?;
    // m.add_class::<MCPTool>()?;
    // m.add_class::<MCPTransport>()?;
    // m.add_class::<PythonAsyncFunctionTool>()?;
    // m.add_class::<PythonFunctionTool>()?;
    // m.add_class::<ToolDesc>()?;
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
    m.add_class::<crate::vector_store::VectorStore>()?;
    m.add_class::<crate::vector_store::VectorStoreAddInput>()?;
    m.add_class::<crate::vector_store::VectorStoreGetResult>()?;
    m.add_class::<crate::vector_store::VectorStoreRetrieveResult>()?;

    #[cfg(feature = "ailoy-model-cli")]
    m.add_function(wrap_pyfunction!(cli::ailoy_model_cli, m)?)?;

    Ok(())
}
