use std::sync::Arc;

use ailoy_macros::maybe_send_sync;

use super::language_model::{LangModelInferConfig, LangModelInference};
use crate::{
    utils::BoxStream,
    value::{Document, Message, MessageDeltaOutput, ToolDesc},
};

#[maybe_send_sync]
pub(super) type CustomLangModelInferFunc =
    dyn Fn(
        Vec<Message>,
        Vec<ToolDesc>,
        Vec<Document>,
        LangModelInferConfig,
    ) -> BoxStream<'static, anyhow::Result<MessageDeltaOutput>>;

#[derive(Clone)]
pub(super) struct CustomLangModel {
    pub infer_func: Arc<CustomLangModelInferFunc>,
}

impl LangModelInference for CustomLangModel {
    fn infer_delta<'a>(
        &'a mut self,
        msg: Vec<Message>,
        tools: Vec<ToolDesc>,
        docs: Vec<Document>,
        config: LangModelInferConfig,
    ) -> BoxStream<'a, anyhow::Result<MessageDeltaOutput>> {
        (self.infer_func)(msg, tools, docs, config)
    }
}
