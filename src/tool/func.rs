use std::{future::Future, sync::Arc};

use futures::{
    Stream, StreamExt,
    future::BoxFuture,
    stream::{self, BoxStream},
};

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
    pub sandbox: Option<Arc<crate::sandbox::Sandbox>>,
}

impl Default for ToolContext {
    fn default() -> Self {
        Self {
            id: String::new(),
            sandbox: None,
        }
    }
}

pub enum ToolFunc {
    Simple(Box<dyn Fn(Value, ToolContext) -> MessageOutput + Send + Sync>),
    Future(Box<dyn Fn(Value, ToolContext) -> BoxFuture<'static, MessageOutput> + Send + Sync>),
    Stream(Box<dyn Fn(Value, ToolContext) -> BoxStream<'static, MessageOutput> + Send + Sync>),
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
        match self {
            ToolFunc::Simple(f) => stream::once(std::future::ready(f(args, ctx))).boxed(),
            ToolFunc::Future(f) => stream::once(f(args, ctx)).boxed(),
            ToolFunc::Stream(f) => f(args, ctx),
        }
    }
}

/// Marker types used to disambiguate `IntoToolFunc` impls by function signature.
/// These are never constructed — they exist only as type-level tags.
mod marker {
    /// `Fn(String, Value, ToolContext) -> Value`
    pub struct SyncValue;

    /// `Fn(String, Value, ToolContext) -> Future<Output = Value>`
    pub struct AsyncValue;

    /// `Fn(String, Value, ToolContext) -> MessageOutput`
    pub struct SyncMessage;

    /// `Fn(String, Value, ToolContext) -> Future<Output = MessageOutput>`
    pub struct AsyncMessage;

    /// `Fn(String, Value, ToolContext) -> Stream<Item = MessageOutput>`
    pub struct AsyncStream;
}

/// Converts a function into a [`ToolFunc`].
///
/// The `Marker` type parameter is used only to avoid coherence conflicts —
/// callers never need to name or specify it.
pub trait IntoToolFunc<Marker> {
    fn into_tool_func(self) -> ToolFunc;
}

impl<F> IntoToolFunc<marker::SyncValue> for F
where
    F: Fn(Value, ToolContext) -> Value + Send + Sync + 'static,
{
    fn into_tool_func(self) -> ToolFunc {
        ToolFunc::Simple(Box::new(move |args, ctx| {
            let id = ctx.id.clone();
            MessageOutput {
                depth: None,
                message: Message::new(Role::Tool)
                    .with_contents([Part::value(self(args, ctx))])
                    .with_id(id),
                finish_reason: FinishReason::Stop {},
            }
        }))
    }
}

impl<F> IntoToolFunc<marker::SyncMessage> for F
where
    F: Fn(Value, ToolContext) -> MessageOutput + Send + Sync + 'static,
{
    fn into_tool_func(self) -> ToolFunc {
        ToolFunc::Simple(Box::new(self))
    }
}

impl<F, Fut> IntoToolFunc<marker::AsyncValue> for F
where
    F: Fn(Value, ToolContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Value> + Send + 'static,
{
    fn into_tool_func(self) -> ToolFunc {
        ToolFunc::Future(Box::new(move |args, ctx| {
            let id = ctx.id.clone();
            let fut = self(args, ctx);
            Box::pin(async move {
                MessageOutput {
                    depth: None,
                    message: Message::new(Role::Tool)
                        .with_contents([Part::value(fut.await)])
                        .with_id(id),
                    finish_reason: FinishReason::Stop {},
                }
            })
        }))
    }
}

impl<F, Fut> IntoToolFunc<marker::AsyncMessage> for F
where
    F: Fn(Value, ToolContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = MessageOutput> + Send + 'static,
{
    fn into_tool_func(self) -> ToolFunc {
        ToolFunc::Future(Box::new(move |args, ctx| Box::pin(self(args, ctx))))
    }
}

impl<F, S> IntoToolFunc<marker::AsyncStream> for F
where
    F: Fn(Value, ToolContext) -> S + Send + Sync + 'static,
    S: Stream<Item = MessageOutput> + Send + 'static,
{
    fn into_tool_func(self) -> ToolFunc {
        ToolFunc::Stream(Box::new(move |args, ctx| Box::pin(self(args, ctx))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(id: &str) -> ToolContext {
        ToolContext {
            id: id.to_owned(),
            sandbox: None,
        }
    }

    #[tokio::test]
    async fn test_sync() {
        let f = ToolFunc::new(|_args: Value, _ctx: ToolContext| Value::string("ok"));
        let out = f
            .call(Value::object_empty(), ctx("call-1"))
            .next()
            .await
            .unwrap();
        assert_eq!(
            out.message.contents[0].as_value(),
            Some(&Value::string("ok"))
        );
        assert_eq!(out.message.id.as_deref(), Some("call-1"));
    }

    #[tokio::test]
    async fn test_sync_echoes_args() {
        let f = ToolFunc::new(|args: Value, _ctx: ToolContext| args);
        let input = Value::integer(99);
        let out = f.call(input.clone(), ctx("call-1")).next().await.unwrap();
        assert_eq!(out.message.contents[0].as_value(), Some(&input));
    }

    #[tokio::test]
    async fn test_async() {
        let f = ToolFunc::new(|_args: Value, _ctx: ToolContext| async move { Value::bool(true) });
        let out = f
            .call(Value::object_empty(), ctx("call-1"))
            .next()
            .await
            .unwrap();
        assert_eq!(out.message.contents[0].as_value(), Some(&Value::bool(true)));
    }
}
