// mod agent;
pub(crate) mod base;
pub(crate) mod cache_progress;
#[cfg(feature = "ailoy-model-cli")]
pub(crate) mod cli;
pub(crate) mod tool;
pub(crate) mod value;
pub(crate) mod vector_store;

use pyo3::prelude::*;

#[pymodule(name = "_core")]
fn ailoy_py(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    // Add classes in alphabetical order
    m.add_class::<crate::agent::Agent>()?;
    m.add_class::<crate::agent::AgentResponse>()?;
    m.add_class::<crate::tool::Tool>()?;
    m.add_class::<crate::tool::MCPClient>()?;
    m.add_class::<vector_store::ChromaVectorStore>()?;
    m.add_class::<vector_store::FaissVectorStore>()?;
    m.add_class::<crate::value::ToolDesc>()?;
    m.add_class::<vector_store::PyVectorStoreAddInput>()?;
    m.add_class::<vector_store::PyVectorStoreGetResult>()?;
    m.add_class::<vector_store::PyVectorStoreRetrieveResult>()?;
    m.add_class::<crate::model::EmbeddingModel>()?;
    m.add_class::<crate::model::ThinkEffort>()?;
    m.add_class::<crate::model::LangModel>()?;
    m.add_class::<crate::model::InferenceConfig>()?;
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

    #[cfg(feature = "ailoy-model-cli")]
    m.add_function(wrap_pyfunction!(cli::ailoy_model_cli, m)?)?;

    Ok(())
}
