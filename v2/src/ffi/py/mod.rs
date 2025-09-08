mod agent;
mod base;
mod cache_progress;
mod model;
mod tool;
mod value;

use agent::{
    PyAgent as Agent, PyAgentRunIterator as AgentRunIterator,
    PyAgentRunSyncIterator as AgentRunSyncIterator,
};
use cache_progress::{
    PyCacheProgress as CacheProgress, PyCacheProgressIterator as CacheProgressIterator,
    PyCacheProgressSyncIterator as CacheProgressSyncIterator,
};
use model::{
    PyAnthropicLanguageModel as AnthropicLanguageModel,
    PyGeminiLanguageModel as GeminiLanguageModel, PyLanguageModel as LanguageModel,
    PyLanguageModelRunIterator as LanguageModelRunIterator,
    PyLanguageModelRunSyncIterator as LanguageModelRunSyncIterator,
    PyLocalLanguageModel as LocalLanguageModel, PyOpenAILanguageModel as OpenAILanguageModel,
    PyXAILanguageModel as XAILanguageModel,
};
use pyo3::prelude::*;
use pyo3_stub_gen::{Result, generate::StubInfo};
use tool::PyBuiltinTool as BuiltinTool;

use crate::value::{FinishReason, Message, MessageAggregator, MessageOutput, Part, Role, ToolDesc};

#[pymodule(name = "_core")]
fn ailoy_py(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    // Add classes in alphabetical order
    m.add_class::<Agent>()?;
    m.add_class::<AgentRunIterator>()?;
    m.add_class::<AgentRunSyncIterator>()?;
    m.add_class::<AnthropicLanguageModel>()?;
    m.add_class::<BuiltinTool>()?;
    m.add_class::<CacheProgress>()?;
    m.add_class::<CacheProgressIterator>()?;
    m.add_class::<CacheProgressSyncIterator>()?;
    m.add_class::<FinishReason>()?;
    m.add_class::<GeminiLanguageModel>()?;
    m.add_class::<LanguageModel>()?;
    m.add_class::<LanguageModelRunIterator>()?;
    m.add_class::<LanguageModelRunSyncIterator>()?;
    m.add_class::<LocalLanguageModel>()?;
    m.add_class::<Message>()?;
    m.add_class::<MessageAggregator>()?;
    m.add_class::<MessageOutput>()?;
    m.add_class::<OpenAILanguageModel>()?;
    m.add_class::<Part>()?;
    m.add_class::<Role>()?;
    m.add_class::<ToolDesc>()?;
    m.add_class::<XAILanguageModel>()?;
    Ok(())
}

pub fn stub_info() -> Result<StubInfo> {
    let manifest_dir: &::std::path::Path = env!("CARGO_MANIFEST_DIR").as_ref();
    StubInfo::from_pyproject_toml(manifest_dir.join("bindings/python/pyproject.toml"))
}
