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
    pub sandbox: Option<Arc<crate::sandbox::Sandbox>>,
}

impl Default for ToolContext {
    fn default() -> Self {
        Self { sandbox: None }
    }
}

pub enum ToolFunc {
    Simple(Box<dyn Fn(String, Value) -> MessageOutput + Send + Sync>),
    Future(Box<dyn Fn(String, Value) -> BoxFuture<'static, MessageOutput> + Send + Sync>),
    Stream(Box<dyn Fn(String, Value) -> BoxStream<'static, MessageOutput> + Send + Sync>),
    SandboxFuture(
        Box<
            dyn Fn(
                    String,
                    Value,
                    Option<Arc<crate::sandbox::Sandbox>>,
                ) -> BoxFuture<'static, MessageOutput>
                + Send
                + Sync,
        >,
    ),
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
    /// # use ailoy::tool::ToolFunc;
    /// # use ailoy::datatype::Value;
    /// // sync
    /// let f = ToolFunc::new(|_args: Value| Value::string("ok"));
    ///
    /// // async
    /// let f = ToolFunc::new(|_args: Value| async move { Value::string("ok") });
    /// ```
    pub fn new<M>(f: impl IntoToolFunc<M>) -> Self {
        f.into_tool_func()
    }

    pub fn call(
        &self,
        tool_call: Part,
        ctx: ToolContext,
    ) -> anyhow::Result<BoxStream<'static, MessageOutput>> {
        let (id, _, args) = tool_call
            .as_function()
            .ok_or(anyhow::anyhow!("Part is not function"))?;
        let id = id.to_owned();
        let args = args.to_owned();
        Ok(match self {
            ToolFunc::Simple(f) => stream::once(std::future::ready(f(id, args))).boxed(),
            ToolFunc::Future(f) => stream::once(f(id, args)).boxed(),
            ToolFunc::Stream(f) => f(id, args),
            ToolFunc::SandboxFuture(f) => stream::once(f(id, args, ctx.sandbox)).boxed(),
        })
    }
}

/// Marker types used to disambiguate `IntoToolFunc` impls by function signature.
/// These are never constructed — they exist only as type-level tags.
mod marker {
    /// `Fn(Value) -> Value`
    pub struct SyncValueOutput;

    /// `Fn(Value) -> Future<Output = Value>`
    pub struct AsyncValueOutput;

    /// `Fn(String, Value) -> Future<Output = MessageOutput>`
    pub struct AsyncMessageOutput;

    /// `Fn(String, Value) -> Stream<Item = MessageOutput>`
    pub struct AsyncMessageStreamOutput;

    /// `Fn(Value, Option<Arc<Sandbox>>) -> Future<Output = Value>`
    pub struct AsyncSandboxValueOutput;
}

/// Converts a function into a [`ToolFunc`].
///
/// The `Marker` type parameter is used only to avoid coherence conflicts —
/// callers never need to name or specify it.
pub trait IntoToolFunc<Marker> {
    fn into_tool_func(self) -> ToolFunc;
}

impl<F> IntoToolFunc<marker::SyncValueOutput> for F
where
    F: Fn(Value) -> Value + Send + Sync + 'static,
{
    fn into_tool_func(self) -> ToolFunc {
        ToolFunc::Simple(Box::new(move |id, args| MessageOutput {
            depth: None,
            message: Message::new(Role::Tool)
                .with_contents([Part::value(self(args))])
                .with_id(id),
            finish_reason: FinishReason::Stop {},
        }))
    }
}

impl<F, Fut> IntoToolFunc<marker::AsyncValueOutput> for F
where
    F: Fn(Value) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Value> + Send + 'static,
{
    fn into_tool_func(self) -> ToolFunc {
        ToolFunc::Future(Box::new(move |id, args| {
            let fut = self(args);
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

impl<F, Fut> IntoToolFunc<marker::AsyncMessageOutput> for F
where
    F: Fn(String, Value) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = MessageOutput> + Send + 'static,
{
    fn into_tool_func(self) -> ToolFunc {
        ToolFunc::Future(Box::new(move |id, args| Box::pin(self(id, args))))
    }
}

impl<F, S> IntoToolFunc<marker::AsyncMessageStreamOutput> for F
where
    F: Fn(String, Value) -> S + Send + Sync + 'static,
    S: Stream<Item = MessageOutput> + Send + 'static,
{
    fn into_tool_func(self) -> ToolFunc {
        ToolFunc::Stream(Box::new(move |id, args| Box::pin(self(id, args))))
    }
}

impl<F, Fut> IntoToolFunc<marker::AsyncSandboxValueOutput> for F
where
    F: Fn(Value, Option<Arc<crate::sandbox::Sandbox>>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Value> + Send + 'static,
{
    fn into_tool_func(self) -> ToolFunc {
        ToolFunc::SandboxFuture(Box::new(move |id, args, sandbox| {
            let fut = self(args, sandbox);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_call(args: Value) -> Part {
        Part::function("call-1", "my_tool", args)
    }

    #[tokio::test]
    async fn test_sync() {
        let f = ToolFunc::new(|_args: Value| Value::string("ok"));
        let out = f
            .call(tool_call(Value::object_empty()), ToolContext::default())
            .unwrap()
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
        let f = ToolFunc::new(|args: Value| args);
        let input = Value::integer(99);
        let out = f
            .call(tool_call(input.clone()), ToolContext::default())
            .unwrap()
            .next()
            .await
            .unwrap();
        assert_eq!(out.message.contents[0].as_value(), Some(&input));
    }

    #[tokio::test]
    async fn test_async() {
        let f = ToolFunc::new(|_args: Value| async move { Value::bool(true) });
        let out = f
            .call(tool_call(Value::object_empty()), ToolContext::default())
            .unwrap()
            .next()
            .await
            .unwrap();
        assert_eq!(out.message.contents[0].as_value(), Some(&Value::bool(true)));
    }
}
