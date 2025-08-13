use std::sync::Arc;

use futures::stream::BoxStream;

use crate::value::{Message, MessageDelta, ToolDescription};

/// Runs the language model with the given tools and messages, returning a stream of `MessageDelta`s.
///
/// Note that a user of this trait should store it using `Arc<dyn LanguageModel>`, like:
///
/// ```rust
/// struct LanguageModelUser {
///     lm: Arc<dyn LanguageModel>,
/// }
/// ```
///
/// # Why `self: Arc<Self>`
///
/// ## 1. Object safety
///
/// Using `Arc<Self>` ensures the method is object-safe, allowing calls via `Arc<dyn LanguageModel>`.
/// Taking `lm` by value without a smart pointer would require knowing `dyn LanguageModel`’s concrete size,
/// which is not possible for trait objects.  
/// `Arc<Self>` (like `Box<Self>` or `Rc<Self>`) has a fixed size and supports
/// unsizing to `Arc<dyn LanguageModel>`, enabling dynamic dispatch.
///
/// ## 2. Clonability
///
/// The method may need to capture `self` multiple times inside the returned stream.  
/// If `run` were intended to be a one-time call that consumes the instance, `Box<Self>` could be used instead.
/// However, because `run` is expected to be called repeatedly, the receiver must be clonable.  
/// `Arc` provides cheap, thread-safe cloning of the underlying object,
/// making it well-suited for asynchronous and concurrent use.
pub trait LanguageModel: Send + Sync + 'static {
    // Runs the language model with the given tools and messages, returning a stream of `MessageDelta`s.
    /// See [`LanguageModel`] trait document for the details.
    fn run(
        self: Arc<Self>,
        tools: Vec<ToolDescription>,
        msg: Vec<Message>,
    ) -> BoxStream<'static, Result<MessageDelta, String>>;
}
