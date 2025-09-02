use crate::{
    utils::{BoxStream, MaybeSend, MaybeSync},
    value::{Message, MessageOutput, ToolDesc},
};

pub trait LanguageModel: MaybeSend + MaybeSync {
    /// Runs the language model with the given tools and messages, returning a stream of `MessageOutput`s.
    fn run<'a>(
        self: &'a mut Self,
        msg: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> BoxStream<'a, Result<MessageOutput, String>>;
}
