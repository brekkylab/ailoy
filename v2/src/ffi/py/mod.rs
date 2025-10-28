pub(crate) mod base;
pub(crate) mod cache_progress;
#[cfg(feature = "ailoy-model-cli")]
pub(crate) mod cli;
pub(crate) mod string_enum;
pub(crate) mod value;

use pyo3::prelude::*;

use crate::{
    agent::{Agent, AgentConfig},
    model::{EmbeddingModel, InferenceConfig, LangModel},
    tool::{MCPClient, Tool},
    value::{
        FinishReason, Message, MessageDelta, MessageDeltaOutput, MessageOutput, Part, PartDelta,
        PartDeltaFunction, PartFunction, PartImage, ToolDesc,
        py::{
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
    m.add_class::<EmbeddingModel>()?;
    m.add_class::<FinishReason>()?;
    m.add_class::<InferenceConfig>()?;
    m.add_class::<LangModel>()?;
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
