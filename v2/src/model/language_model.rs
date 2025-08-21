use std::sync::Arc;

#[cfg(not(target_family = "wasm"))]
use futures::stream::BoxStream;
use futures::stream::LocalBoxStream;

use crate::value::{Message, MessageOutput, ToolDesc};

/// Runs the language model with the given tools and messages, returning a stream of `MessageOutput`s.
///
/// Note that a user of this trait should store it using `Arc<dyn LanguageModel>`, like:
///
/// ```rust
/// struct LanguageModelUser {
///     lm: Arc<dyn LanguageModel>,
/// }
/// ```
pub trait LanguageModel: 'static {
    // Runs the language model with the given tools and messages, returning a stream of `MessageOutput`s.
    /// See [`LanguageModel`] trait document for the details.
    fn run_nonsend(
        self: Arc<Self>,
        msg: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> LocalBoxStream<'static, Result<MessageOutput, String>>;

    #[cfg(not(target_family = "wasm"))]
    fn run(
        self: Arc<Self>,
        msg: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> BoxStream<'static, Result<MessageOutput, String>>;
}
