use std::sync::Arc;

use crate::{
    model::{InferenceConfig, LangModelInference},
    utils::{BoxStream, MaybeSend, MaybeSync},
    value::{Message, MessageOutput, ToolDesc},
};

#[derive(Clone)]
pub(super) struct CustomLangModel {
    pub run: Arc<
        dyn Fn(
                Vec<Message>,
                Vec<ToolDesc>,
                InferenceConfig,
            ) -> BoxStream<'static, Result<MessageOutput, String>>
            + MaybeSend
            + MaybeSync,
    >,
}

impl LangModelInference for CustomLangModel {
    fn infer<'a>(
        &'a mut self,
        msg: Vec<Message>,
        tools: Vec<ToolDesc>,
        config: InferenceConfig,
    ) -> BoxStream<'a, Result<MessageOutput, String>> {
        (self.run)(msg, tools, config)
    }
}
