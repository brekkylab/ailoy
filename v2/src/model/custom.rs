use std::sync::Arc;

use ailoy_macros::maybe_send_sync;

use crate::{
    model::{InferenceConfig, LangModelInference},
    utils::BoxStream,
    value::{Message, MessageOutput, ToolDesc},
};

#[maybe_send_sync]
pub(super) type CustomLangModelInferFunc =
    dyn Fn(
        Vec<Message>,
        Vec<ToolDesc>,
        InferenceConfig,
    ) -> BoxStream<'static, Result<MessageOutput, String>>;

#[derive(Clone)]
pub(super) struct CustomLangModel {
    pub infer_func: Arc<CustomLangModelInferFunc>,
}

impl LangModelInference for CustomLangModel {
    fn infer<'a>(
        &'a mut self,
        msg: Vec<Message>,
        tools: Vec<ToolDesc>,
        config: InferenceConfig,
    ) -> BoxStream<'a, Result<MessageOutput, String>> {
        (self.infer_func)(msg, tools, config)
    }
}
