use std::sync::Arc;

use ailoy_macros::maybe_send_sync;

use crate::{
    model::{InferenceConfig, LangModelInference},
    utils::BoxStream,
    value::{Document, Message, MessageOutput, ToolDesc},
};

#[maybe_send_sync]
pub(super) type CustomLangModelInferFunc =
    dyn Fn(
        Vec<Message>,
        Vec<ToolDesc>,
        Vec<Document>,
        InferenceConfig,
    ) -> BoxStream<'static, anyhow::Result<MessageOutput>>;

#[derive(Clone)]
pub(super) struct CustomLangModel {
    pub infer_func: Arc<CustomLangModelInferFunc>,
}

impl LangModelInference for CustomLangModel {
    fn infer<'a>(
        &'a mut self,
        msg: Vec<Message>,
        tools: Vec<ToolDesc>,
        docs: Vec<Document>,
        config: InferenceConfig,
    ) -> BoxStream<'a, anyhow::Result<MessageOutput>> {
        (self.infer_func)(msg, tools, docs, config)
    }
}
