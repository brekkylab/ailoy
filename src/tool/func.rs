use std::future::Future;
#[cfg(feature = "sandbox")]
use std::sync::Arc;

use futures::{
    Stream, StreamExt,
    stream::{self, BoxStream},
};

#[cfg(feature = "sandbox")]
use crate::sandbox::Sandbox;
use crate::{
    datatype::Value,
    message::{FinishReason, Message, MessageOutput, Part, Role},
};

/// Runtime context forwarded to every tool call.
///
/// Tools that don't need the context simply ignore it.  Constructed by the
/// caller (typically the agent) and passed through [`ToolFunc::call`].
pub struct ToolContext {
    pub id: String,

    #[cfg(feature = "sandbox")]
    pub sandbox: Option<Arc<Sandbox>>,
}

impl ToolContext {
    pub(crate) fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            #[cfg(feature = "sandbox")]
            sandbox: None,
        }
    }

    #[cfg(feature = "sandbox")]
    pub(crate) fn sandbox(mut self, sandbox: Arc<Sandbox>) -> Self {
        self.sandbox = Some(sandbox);
        self
    }
}

pub struct ToolFunc {
    inner: Box<dyn Fn(Value, ToolContext) -> BoxStream<'static, MessageOutput> + Send + Sync>,
}

impl ToolFunc {
    /// Construct a [`ToolFunc`] from any value that implements [`IntoToolFunc`].
    ///
    /// The `Marker` type parameter is inferred from the closure's return type —
    /// callers never need to name it.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ailoy::tool::{ToolFunc, ToolContext};
    /// # use ailoy::datatype::Value;
    /// // sync
    /// let f = ToolFunc::new(|_args: Value, _ctx: ToolContext| Value::string("ok"));
    ///
    /// // async
    /// let f = ToolFunc::new(|_args: Value, _ctx: ToolContext| async move { Value::string("ok") });
    /// ```
    pub fn new<M>(f: impl IntoToolFunc<M>) -> Self {
        f.into_tool_func()
    }

    pub fn call(&self, args: Value, ctx: ToolContext) -> BoxStream<'static, MessageOutput> {
        (self.inner)(args, ctx)
    }
}

/// Marker types used to disambiguate `IntoToolFunc` impls by function signature.
/// These are never constructed — they exist only as type-level tags.
mod marker {
    /// `Fn(Value) -> Value`
    pub struct ArgValueRetValueFn;

    /// `Fn(Value, ToolContext) -> Value`
    pub struct ArgValueContextRetValueFn;

    /// `Fn(Value) -> Message`
    pub struct ArgValueRetMessageFn;

    /// `Fn(Value, ToolContext) -> Message`
    pub struct ArgValueContextRetMessageFn;

    /// `Fn(Value) -> Future<Output = Value>`
    pub struct ArgValueRetValueAsyncFn;

    /// `Fn(Value, ToolContext) -> Future<Output = Value>`
    pub struct ArgValueContextRetValueAsyncFn;

    /// `Fn(Value) -> Future<Output = Message>`
    pub struct ArgValueRetMessageAsyncFn;

    /// `Fn(Value, ToolContext) -> Future<Output = Message>`
    pub struct ArgValueContextRetMessageAsyncFn;

    /// `Fn(Value) -> Stream<Item = Value>`
    pub struct ArgValueRetValueStreamFn;

    /// `Fn(Value, ToolContext) -> Stream<Item = Value>`
    pub struct ArgValueContextRetValueStreamFn;

    /// `Fn(Value) -> Stream<Item = Message>`
    pub struct ArgValueRetMessageStreamFn;

    /// `Fn(Value, ToolContext) -> Stream<Item = Message>`
    pub struct ArgValueContextRetMessageStreamFn;
}

/// Converts a function into a [`ToolFunc`].
///
/// The `Marker` type parameter is used only to avoid coherence conflicts —
/// callers never need to name or specify it.
pub trait IntoToolFunc<Marker> {
    fn into_tool_func(self) -> ToolFunc;
}

// Fn(Value) -> Value
impl<F> IntoToolFunc<marker::ArgValueRetValueFn> for F
where
    F: Fn(Value) -> Value + Send + Sync + 'static,
{
    fn into_tool_func(self) -> ToolFunc {
        ToolFunc {
            inner: Box::new(move |args, ctx| {
                let id = ctx.id.clone();
                let value = self(args);
                stream::once(std::future::ready(MessageOutput {
                    depth: None,
                    message: Message::new(Role::Tool)
                        .with_contents([Part::value(value)])
                        .with_id(id),
                    finish_reason: FinishReason::Stop {},
                }))
                .boxed()
            }),
        }
    }
}

// Fn(Value, ToolContext) -> Value
impl<F> IntoToolFunc<marker::ArgValueContextRetValueFn> for F
where
    F: Fn(Value, ToolContext) -> Value + Send + Sync + 'static,
{
    fn into_tool_func(self) -> ToolFunc {
        ToolFunc {
            inner: Box::new(move |args, ctx| {
                let id = ctx.id.clone();
                let value = self(args, ctx);
                stream::once(std::future::ready(MessageOutput {
                    depth: None,
                    message: Message::new(Role::Tool)
                        .with_contents([Part::value(value)])
                        .with_id(id),
                    finish_reason: FinishReason::Stop {},
                }))
                .boxed()
            }),
        }
    }
}

// Fn(Value) -> Message
impl<F> IntoToolFunc<marker::ArgValueRetMessageFn> for F
where
    F: Fn(Value) -> Message + Send + Sync + 'static,
{
    fn into_tool_func(self) -> ToolFunc {
        ToolFunc {
            inner: Box::new(move |args, _ctx| {
                let message = self(args);
                stream::once(std::future::ready(MessageOutput {
                    depth: None,
                    message,
                    finish_reason: FinishReason::Stop {},
                }))
                .boxed()
            }),
        }
    }
}

// Fn(Value, ToolContext) -> Message
impl<F> IntoToolFunc<marker::ArgValueContextRetMessageFn> for F
where
    F: Fn(Value, ToolContext) -> Message + Send + Sync + 'static,
{
    fn into_tool_func(self) -> ToolFunc {
        ToolFunc {
            inner: Box::new(move |args, ctx| {
                let message = self(args, ctx);
                stream::once(std::future::ready(MessageOutput {
                    depth: None,
                    message,
                    finish_reason: FinishReason::Stop {},
                }))
                .boxed()
            }),
        }
    }
}

// Fn(Value) -> Future<Output = Value>
impl<F, Fut> IntoToolFunc<marker::ArgValueRetValueAsyncFn> for F
where
    F: Fn(Value) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Value> + Send + 'static,
{
    fn into_tool_func(self) -> ToolFunc {
        ToolFunc {
            inner: Box::new(move |args, ctx| {
                let id = ctx.id.clone();
                let fut = self(args);
                stream::once(Box::pin(async move {
                    MessageOutput {
                        depth: None,
                        message: Message::new(Role::Tool)
                            .with_contents([Part::value(fut.await)])
                            .with_id(id),
                        finish_reason: FinishReason::Stop {},
                    }
                }))
                .boxed()
            }),
        }
    }
}

// Fn(Value, ToolContext) -> Future<Output = Value>
impl<F, Fut> IntoToolFunc<marker::ArgValueContextRetValueAsyncFn> for F
where
    F: Fn(Value, ToolContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Value> + Send + 'static,
{
    fn into_tool_func(self) -> ToolFunc {
        ToolFunc {
            inner: Box::new(move |args, ctx| {
                let id = ctx.id.clone();
                let fut = self(args, ctx);
                stream::once(Box::pin(async move {
                    MessageOutput {
                        depth: None,
                        message: Message::new(Role::Tool)
                            .with_contents([Part::value(fut.await)])
                            .with_id(id),
                        finish_reason: FinishReason::Stop {},
                    }
                }))
                .boxed()
            }),
        }
    }
}

// Fn(Value) -> Future<Output = Message>
impl<F, Fut> IntoToolFunc<marker::ArgValueRetMessageAsyncFn> for F
where
    F: Fn(Value) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Message> + Send + 'static,
{
    fn into_tool_func(self) -> ToolFunc {
        ToolFunc {
            inner: Box::new(move |args, _ctx| {
                let fut = self(args);
                stream::once(Box::pin(async move {
                    MessageOutput {
                        depth: None,
                        message: fut.await,
                        finish_reason: FinishReason::Stop {},
                    }
                }))
                .boxed()
            }),
        }
    }
}

// Fn(Value, ToolContext) -> Future<Output = Message>
impl<F, Fut> IntoToolFunc<marker::ArgValueContextRetMessageAsyncFn> for F
where
    F: Fn(Value, ToolContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Message> + Send + 'static,
{
    fn into_tool_func(self) -> ToolFunc {
        ToolFunc {
            inner: Box::new(move |args, ctx| {
                let fut = self(args, ctx);
                stream::once(Box::pin(async move {
                    MessageOutput {
                        depth: None,
                        message: fut.await,
                        finish_reason: FinishReason::Stop {},
                    }
                }))
                .boxed()
            }),
        }
    }
}

// Fn(Value) -> Stream<Item = Value>
impl<F, S> IntoToolFunc<marker::ArgValueRetValueStreamFn> for F
where
    F: Fn(Value) -> S + Send + Sync + 'static,
    S: Stream<Item = Value> + Send + 'static,
{
    fn into_tool_func(self) -> ToolFunc {
        ToolFunc {
            inner: Box::new(move |args, ctx| {
                let id = ctx.id.clone();
                self(args)
                    .map(move |value| MessageOutput {
                        depth: None,
                        message: Message::new(Role::Tool)
                            .with_contents([Part::value(value)])
                            .with_id(id.clone()),
                        finish_reason: FinishReason::Stop {},
                    })
                    .boxed()
            }),
        }
    }
}

// Fn(Value, ToolContext) -> Stream<Item = Value>
impl<F, S> IntoToolFunc<marker::ArgValueContextRetValueStreamFn> for F
where
    F: Fn(Value, ToolContext) -> S + Send + Sync + 'static,
    S: Stream<Item = Value> + Send + 'static,
{
    fn into_tool_func(self) -> ToolFunc {
        ToolFunc {
            inner: Box::new(move |args, ctx| {
                let id = ctx.id.clone();
                self(args, ctx)
                    .map(move |value| MessageOutput {
                        depth: None,
                        message: Message::new(Role::Tool)
                            .with_contents([Part::value(value)])
                            .with_id(id.clone()),
                        finish_reason: FinishReason::Stop {},
                    })
                    .boxed()
            }),
        }
    }
}

// Fn(Value) -> Stream<Item = Message>
impl<F, S> IntoToolFunc<marker::ArgValueRetMessageStreamFn> for F
where
    F: Fn(Value) -> S + Send + Sync + 'static,
    S: Stream<Item = Message> + Send + 'static,
{
    fn into_tool_func(self) -> ToolFunc {
        ToolFunc {
            inner: Box::new(move |args, _ctx| {
                self(args)
                    .map(|message| MessageOutput {
                        depth: None,
                        message,
                        finish_reason: FinishReason::Stop {},
                    })
                    .boxed()
            }),
        }
    }
}

// Fn(Value, ToolContext) -> Stream<Item = Message>
impl<F, S> IntoToolFunc<marker::ArgValueContextRetMessageStreamFn> for F
where
    F: Fn(Value, ToolContext) -> S + Send + Sync + 'static,
    S: Stream<Item = Message> + Send + 'static,
{
    fn into_tool_func(self) -> ToolFunc {
        ToolFunc {
            inner: Box::new(move |args, ctx| {
                self(args, ctx)
                    .map(|message| MessageOutput {
                        depth: None,
                        message,
                        finish_reason: FinishReason::Stop {},
                    })
                    .boxed()
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ArgValueRetValueFn
    #[tokio::test]
    async fn test_sync_no_ctx() {
        let f = ToolFunc::new(|_args: Value| Value::string("ok"));
        let out = f
            .call(Value::object_empty(), ToolContext::new("call-1"))
            .next()
            .await
            .unwrap();
        assert_eq!(
            out.message.contents[0].as_value(),
            Some(&Value::string("ok"))
        );
        assert_eq!(out.message.id.as_deref(), Some("call-1"));
    }

    // ArgValueContextRetValueFn
    #[tokio::test]
    async fn test_sync_with_ctx() {
        let f = ToolFunc::new(|_args: Value, _ctx: ToolContext| Value::string("ok"));
        let out = f
            .call(Value::object_empty(), ToolContext::new("call-1"))
            .next()
            .await
            .unwrap();
        assert_eq!(
            out.message.contents[0].as_value(),
            Some(&Value::string("ok"))
        );
        assert_eq!(out.message.id.as_deref(), Some("call-1"));
    }

    // ArgValueContextRetValueFn — echoes args
    #[tokio::test]
    async fn test_sync_echoes_args() {
        let f = ToolFunc::new(|args: Value, _ctx: ToolContext| args);
        let input = Value::integer(99);
        let out = f
            .call(input.clone(), ToolContext::new("call-1"))
            .next()
            .await
            .unwrap();
        assert_eq!(out.message.contents[0].as_value(), Some(&input));
    }

    // ArgValueRetValueAsyncFn
    #[tokio::test]
    async fn test_async_no_ctx() {
        let f = ToolFunc::new(|_args: Value| async move { Value::bool(true) });
        let out = f
            .call(Value::object_empty(), ToolContext::new("call-1"))
            .next()
            .await
            .unwrap();
        assert_eq!(out.message.contents[0].as_value(), Some(&Value::bool(true)));
    }

    // ArgValueContextRetValueAsyncFn
    #[tokio::test]
    async fn test_async_with_ctx() {
        let f = ToolFunc::new(|_args: Value, _ctx: ToolContext| async move { Value::bool(true) });
        let out = f
            .call(Value::object_empty(), ToolContext::new("call-1"))
            .next()
            .await
            .unwrap();
        assert_eq!(out.message.contents[0].as_value(), Some(&Value::bool(true)));
    }

    // ArgValueRetValueStreamFn
    #[tokio::test]
    async fn test_stream_value_no_ctx() {
        let f = ToolFunc::new(|_args: Value| {
            stream::iter(vec![
                Value::integer(1),
                Value::integer(2),
                Value::integer(3),
            ])
        });
        let outputs: Vec<_> = f
            .call(Value::object_empty(), ToolContext::new("call-1"))
            .collect()
            .await;
        assert_eq!(outputs.len(), 3);
        assert_eq!(
            outputs[2].message.contents[0].as_value(),
            Some(&Value::integer(3))
        );
        assert_eq!(outputs[0].message.id.as_deref(), Some("call-1"));
    }

    // ArgValueContextRetValueStreamFn
    #[tokio::test]
    async fn test_stream_value_with_ctx() {
        let f = ToolFunc::new(|_args: Value, _ctx: ToolContext| {
            stream::iter(vec![Value::bool(false), Value::bool(true)])
        });
        let outputs: Vec<_> = f
            .call(Value::object_empty(), ToolContext::new("call-1"))
            .collect()
            .await;
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].message.id.as_deref(), Some("call-1"));
    }
}
