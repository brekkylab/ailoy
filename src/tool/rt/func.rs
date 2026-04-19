//! Tool function representation and ergonomic construction.
//!
//! # Design intent
//!
//! A tool exposed to a language model is ultimately a function that receives a
//! [`Value`] (the model's arguments) and produces a [`MessageOutput`] (the tool
//! result sent back to the model).  In practice, tool implementations come in
//! several shapes — synchronous or async — and forcing callers to manually box
//! closures or convert between these forms creates unnecessary boilerplate.
//!
//! This module addresses that by splitting responsibilities across two types:
//!
//! - **[`ToolFunc`]** is the *stored* representation.  It is an enum whose
//!   variants hold type-erased, heap-allocated function objects.  The internal
//!   signature always includes the tool-call `id` (`String`) and `args`
//!   ([`Value`]) so that the runtime can build a well-formed [`MessageOutput`]
//!   without knowing anything about the original closure type.
//!
//! - **[`IntoToolFunc`]** is the *construction* interface.  It is a trait
//!   implemented for common function shapes, letting callers pass plain closures
//!   directly to [`ToolRuntime::new`] without any boxing or naming.
//!
//! # Phantom marker pattern
//!
//! Rust's coherence rules prevent multiple blanket `impl`s of the same trait for
//! overlapping types.  Because the `Fn` shapes are distinguished only by their
//! return type — not by a separate wrapper struct — a naïve approach would produce
//! conflicting impls.
//!
//! The solution is the [`marker`] module: each shape is paired with a unique,
//! zero-sized marker type.  The `IntoToolFunc<Marker>` trait is generic over
//! `Marker`, so each impl targets a distinct trait instantiation and there is no
//! overlap.  Rust's type inference resolves the correct `Marker` from the
//! closure's return type automatically — callers never need to name or specify it.

use std::future::Future;

use futures::{
    Stream, StreamExt,
    future::BoxFuture,
    stream::{self, BoxStream},
};

use crate::{
    datatype::Value,
    message::{FinishReason, Message, MessageOutput, Part, Role},
};

pub enum ToolFunc {
    Simple(Box<dyn Fn(String, Value) -> MessageOutput + Send + Sync>),
    Future(Box<dyn Fn(String, Value) -> BoxFuture<'static, MessageOutput> + Send + Sync>),
    Stream(Box<dyn Fn(String, Value) -> BoxStream<'static, MessageOutput> + Send + Sync>),
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
    /// // sync
    /// let f = ToolFunc::new(|args| Value::string("ok"));
    ///
    /// // async
    /// let f = ToolFunc::new(|args| async move { Value::string("ok") });
    /// ```
    pub fn new<M>(f: impl IntoToolFunc<M>) -> Self {
        f.into_tool_func()
    }

    pub fn call(&self, tool_call: Part) -> anyhow::Result<BoxStream<'static, MessageOutput>> {
        let (id, _, args) = tool_call
            .as_function()
            .ok_or(anyhow::anyhow!("Part is not function"))?;
        let id = id.to_owned();
        let args = args.to_owned();
        Ok(match self {
            ToolFunc::Simple(f) => stream::once(std::future::ready(f(id, args))).boxed(),
            ToolFunc::Future(f) => stream::once(f(id, args)).boxed(),
            ToolFunc::Stream(f) => f(id, args),
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
            .call(tool_call(Value::object_empty()))
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
            .call(tool_call(input.clone()))
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
            .call(tool_call(Value::object_empty()))
            .unwrap()
            .next()
            .await
            .unwrap();
        assert_eq!(out.message.contents[0].as_value(), Some(&Value::bool(true)));
    }
}
