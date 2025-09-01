use crate::{
    utils::{BoxStream, MaybeSend, MaybeSync},
    value::{Message, MessageOutput, ToolDesc},
};

/// Runs the language model with the given tools and messages, returning a stream of `MessageOutput`s.
pub trait LanguageModel: MaybeSend + MaybeSync {
    // Runs the language model with the given tools and messages, returning a stream of `MessageOutput`s.
    /// See [`LanguageModel`] trait document for the details.
    fn run<'a>(
        self: &'a mut Self,
        msg: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> BoxStream<'a, Result<MessageOutput, String>>;
}
