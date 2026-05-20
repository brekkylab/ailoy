use std::sync::Arc;

use futures::{
    StreamExt,
    future::BoxFuture,
    stream::{self, BoxStream},
};

use crate::{
    datatype::Value,
    message::{FinishReason, Message, MessageOutput, Part, Role},
    runenv::RunEnvHandle,
};

#[derive(Clone)]
pub struct ToolFunc {
    // The runenv is passed in as `Arc<RunEnvHandle>`, owned, so the resulting
    // stream is always `'static`. Tools that need to hold the env across await
    // points clone the Arc into their stream/future captures.
    inner: Arc<
        dyn Fn(Value, String, Arc<RunEnvHandle>) -> BoxStream<'static, MessageOutput> + Send + Sync,
    >,
}

impl ToolFunc {
    /// Construct a [`ToolFunc`] from a closure of the canonical inner shape.
    ///
    /// We recommend using the [`crate::tool_func!`] macro instead. Use `new`
    /// only for advanced usage that the macro does not cover.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(Value, String, Arc<RunEnvHandle>) -> BoxStream<'static, MessageOutput>
            + Send
            + Sync
            + 'static,
    {
        Self { inner: Arc::new(f) }
    }

    pub fn call(
        &self,
        args: Value,
        id: impl Into<String>,
        runenv: Arc<RunEnvHandle>,
    ) -> BoxStream<'static, MessageOutput> {
        (self.inner)(args, id.into(), runenv)
    }
}

/// Wrapping helpers used by the [`crate::tool_func!`] macro to turn user output
/// (a [`Value`]/[`Message`], an async future of one, or a stream of them) into
/// the canonical `BoxStream<'static, MessageOutput>` shape.
///
/// Not part of the public API.
#[doc(hidden)]
pub mod __private {
    use super::*;

    pub fn value_to_stream(id: String, value: Value) -> BoxStream<'static, MessageOutput> {
        stream::once(std::future::ready(MessageOutput {
            message: Message::new(Role::Tool)
                .with_contents([Part::value(value)])
                .with_id(id),
            finish_reason: FinishReason::Stop {},
            usage: None,
            depth: None,
            source_agent: None,
        }))
        .boxed()
    }

    pub fn message_to_stream(message: Message) -> BoxStream<'static, MessageOutput> {
        stream::once(std::future::ready(MessageOutput {
            message,
            finish_reason: FinishReason::Stop {},
            usage: None,
            depth: None,
            source_agent: None,
        }))
        .boxed()
    }

    pub fn value_fut_to_stream(
        id: String,
        fut: BoxFuture<'static, Value>,
    ) -> BoxStream<'static, MessageOutput> {
        stream::once(async move {
            MessageOutput {
                message: Message::new(Role::Tool)
                    .with_contents([Part::value(fut.await)])
                    .with_id(id),
                finish_reason: FinishReason::Stop {},
                usage: None,
                depth: None,
                source_agent: None,
            }
        })
        .boxed()
    }

    pub fn message_fut_to_stream(
        fut: BoxFuture<'static, Message>,
    ) -> BoxStream<'static, MessageOutput> {
        stream::once(async move {
            MessageOutput {
                message: fut.await,
                finish_reason: FinishReason::Stop {},
                usage: None,
                depth: None,
                source_agent: None,
            }
        })
        .boxed()
    }

    pub fn value_stream_to_msg_stream(
        id: String,
        s: BoxStream<'static, Value>,
    ) -> BoxStream<'static, MessageOutput> {
        s.map(move |value| MessageOutput {
            message: Message::new(Role::Tool)
                .with_contents([Part::value(value)])
                .with_id(id.clone()),
            finish_reason: FinishReason::Stop {},
            usage: None,
            depth: None,
            source_agent: None,
        })
        .boxed()
    }

    pub fn message_stream_to_msg_stream(
        s: BoxStream<'static, Message>,
    ) -> BoxStream<'static, MessageOutput> {
        s.map(|message| MessageOutput {
            message,
            finish_reason: FinishReason::Stop {},
            usage: None,
            depth: None,
            source_agent: None,
        })
        .boxed()
    }
}

/// Build a [`ToolFunc`] from a closure without writing the canonical
/// `Fn(Value, String, Arc<RunEnvHandle>) -> BoxStream<'static, MessageOutput>`
/// shape or its `BoxFuture`/`BoxStream`/`Box::pin` boilerplate by hand.
///
/// # Forms
///
/// Sync — closure body returns a value/message directly:
///
/// ```ignore
/// tool_func!(|args: Value| -> Value { … })
/// tool_func!(|args: Value, runenv: Arc<RunEnvHandle>| -> Value { … })
/// tool_func!(|args: Value, id: String| -> Message { … })
/// tool_func!(|args: Value, id: String, runenv: Arc<RunEnvHandle>| -> Message { … })
/// ```
///
/// Async — closure body is `async`-awaitable:
///
/// ```ignore
/// tool_func!(async |args: Value| -> Value { … })
/// tool_func!(async |args: Value, runenv: Arc<RunEnvHandle>| -> Value { … })
/// tool_func!(async |args: Value, id: String| -> Message { … })
/// tool_func!(async |args: Value, id: String, runenv: Arc<RunEnvHandle>| -> Message { … })
/// ```
///
/// Stream — closure body produces a `Stream`/`BoxStream`:
///
/// ```ignore
/// tool_func!(stream |args: Value| -> Value { … })
/// tool_func!(stream |args: Value, runenv: Arc<RunEnvHandle>| -> Value { … })
/// tool_func!(stream |args: Value, id: String| -> Message { … })
/// tool_func!(stream |args: Value, id: String, runenv: Arc<RunEnvHandle>| -> Message { … })
/// ```
#[macro_export]
macro_rules! tool_func {
    // ─── sync ──────────────────────────────────────────────────────────────
    (|$args:ident : Value| -> Value $body:block) => {
        $crate::tool::ToolFunc::new(
            move |$args: $crate::datatype::Value,
                  id: ::std::string::String,
                  _runenv: ::std::sync::Arc<$crate::runenv::RunEnvHandle>| {
                let value: $crate::datatype::Value = $body;
                $crate::tool::__private::value_to_stream(id, value)
            },
        )
    };

    (|$args:ident : Value, $runenv:ident : Arc<RunEnvHandle>| -> Value $body:block) => {
        $crate::tool::ToolFunc::new(
            move |$args: $crate::datatype::Value,
                  id: ::std::string::String,
                  $runenv: ::std::sync::Arc<$crate::runenv::RunEnvHandle>| {
                let value: $crate::datatype::Value = $body;
                $crate::tool::__private::value_to_stream(id, value)
            },
        )
    };

    (|$args:ident : Value, $id:ident : String| -> Message $body:block) => {
        $crate::tool::ToolFunc::new(
            move |$args: $crate::datatype::Value,
                  $id: ::std::string::String,
                  _runenv: ::std::sync::Arc<$crate::runenv::RunEnvHandle>| {
                let message: $crate::message::Message = $body;
                $crate::tool::__private::message_to_stream(message)
            },
        )
    };

    (
        |$args:ident : Value, $id:ident : String, $runenv:ident : Arc<RunEnvHandle>| -> Message
            $body:block
    ) => {
        $crate::tool::ToolFunc::new(
            move |$args: $crate::datatype::Value,
                  $id: ::std::string::String,
                  $runenv: ::std::sync::Arc<$crate::runenv::RunEnvHandle>| {
                let message: $crate::message::Message = $body;
                $crate::tool::__private::message_to_stream(message)
            },
        )
    };

    // ─── async ─────────────────────────────────────────────────────────────
    //
    // Note: async arms wrap the body in `async move { ... }`, which captures
    // outer variables by move. If the body needs to clone an `Arc`/`String`/etc.
    // captured from the surrounding scope per call, that clone must happen
    // *outside* the `async move` (otherwise the surrounding `Fn` closure tries
    // to move the same outer variable on every call). Use the optional
    // `with [name = expr, ...]` clause to lift those clones up:
    //
    // ```ignore
    // tool_func!(async |args: Value, runenv: Arc<RunEnvHandle>| -> Value
    //     with [runner = runner.clone()]
    //     {
    //         // body sees local `runner` (the per-call clone)
    //     }
    // )
    // ```
    (async |$args:ident : Value| -> Value $body:block) => {
        $crate::tool_func!(async |$args: Value| -> Value with [] $body)
    };

    (async |$args:ident : Value| -> Value with [$($cap:ident = $expr:expr),* $(,)?] $body:block) => {
        $crate::tool::ToolFunc::new(
            move |$args: $crate::datatype::Value,
                  id: ::std::string::String,
                  _runenv: ::std::sync::Arc<$crate::runenv::RunEnvHandle>| {
                $(let $cap = $expr;)*
                let fut: ::futures::future::BoxFuture<'static, $crate::datatype::Value> =
                    ::std::boxed::Box::pin(async move $body);
                $crate::tool::__private::value_fut_to_stream(id, fut)
            },
        )
    };

    (async |$args:ident : Value, $runenv:ident : Arc<RunEnvHandle>| -> Value $body:block) => {
        $crate::tool_func!(
            async |$args: Value, $runenv: Arc<RunEnvHandle>| -> Value with [] $body
        )
    };

    (
        async |$args:ident : Value, $runenv:ident : Arc<RunEnvHandle>| -> Value
            with [$($cap:ident = $expr:expr),* $(,)?] $body:block
    ) => {
        $crate::tool::ToolFunc::new(
            move |$args: $crate::datatype::Value,
                  id: ::std::string::String,
                  $runenv: ::std::sync::Arc<$crate::runenv::RunEnvHandle>| {
                $(let $cap = $expr;)*
                let fut: ::futures::future::BoxFuture<'static, $crate::datatype::Value> =
                    ::std::boxed::Box::pin(async move $body);
                $crate::tool::__private::value_fut_to_stream(id, fut)
            },
        )
    };

    (async |$args:ident : Value, $id:ident : String| -> Message $body:block) => {
        $crate::tool_func!(
            async |$args: Value, $id: String| -> Message with [] $body
        )
    };

    (
        async |$args:ident : Value, $id:ident : String| -> Message
            with [$($cap:ident = $expr:expr),* $(,)?] $body:block
    ) => {
        $crate::tool::ToolFunc::new(
            move |$args: $crate::datatype::Value,
                  $id: ::std::string::String,
                  _runenv: ::std::sync::Arc<$crate::runenv::RunEnvHandle>| {
                $(let $cap = $expr;)*
                let fut: ::futures::future::BoxFuture<'static, $crate::message::Message> =
                    ::std::boxed::Box::pin(async move $body);
                $crate::tool::__private::message_fut_to_stream(fut)
            },
        )
    };

    (
        async |$args:ident : Value, $id:ident : String, $runenv:ident : Arc<RunEnvHandle>|
            -> Message $body:block
    ) => {
        $crate::tool_func!(
            async |$args: Value, $id: String, $runenv: Arc<RunEnvHandle>| -> Message with [] $body
        )
    };

    (
        async |$args:ident : Value, $id:ident : String, $runenv:ident : Arc<RunEnvHandle>|
            -> Message with [$($cap:ident = $expr:expr),* $(,)?] $body:block
    ) => {
        $crate::tool::ToolFunc::new(
            move |$args: $crate::datatype::Value,
                  $id: ::std::string::String,
                  $runenv: ::std::sync::Arc<$crate::runenv::RunEnvHandle>| {
                $(let $cap = $expr;)*
                let fut: ::futures::future::BoxFuture<'static, $crate::message::Message> =
                    ::std::boxed::Box::pin(async move $body);
                $crate::tool::__private::message_fut_to_stream(fut)
            },
        )
    };

    // ─── stream ────────────────────────────────────────────────────────────
    (stream |$args:ident : Value| -> Value $body:block) => {
        $crate::tool::ToolFunc::new(
            move |$args: $crate::datatype::Value,
                  id: ::std::string::String,
                  _runenv: ::std::sync::Arc<$crate::runenv::RunEnvHandle>| {
                let s: ::futures::stream::BoxStream<'static, $crate::datatype::Value> =
                    ::futures::StreamExt::boxed($body);
                $crate::tool::__private::value_stream_to_msg_stream(id, s)
            },
        )
    };

    (stream |$args:ident : Value, $runenv:ident : Arc<RunEnvHandle>| -> Value $body:block) => {
        $crate::tool::ToolFunc::new(
            move |$args: $crate::datatype::Value,
                  id: ::std::string::String,
                  $runenv: ::std::sync::Arc<$crate::runenv::RunEnvHandle>| {
                let s: ::futures::stream::BoxStream<'static, $crate::datatype::Value> =
                    ::futures::StreamExt::boxed($body);
                $crate::tool::__private::value_stream_to_msg_stream(id, s)
            },
        )
    };

    (stream |$args:ident : Value, $id:ident : String| -> Message $body:block) => {
        $crate::tool::ToolFunc::new(
            move |$args: $crate::datatype::Value,
                  $id: ::std::string::String,
                  _runenv: ::std::sync::Arc<$crate::runenv::RunEnvHandle>| {
                let s: ::futures::stream::BoxStream<'static, $crate::message::Message> =
                    ::futures::StreamExt::boxed($body);
                $crate::tool::__private::message_stream_to_msg_stream(s)
            },
        )
    };

    (
        stream |$args:ident : Value, $id:ident : String, $runenv:ident : Arc<RunEnvHandle>|
            -> Message $body:block
    ) => {
        $crate::tool::ToolFunc::new(
            move |$args: $crate::datatype::Value,
                  $id: ::std::string::String,
                  $runenv: ::std::sync::Arc<$crate::runenv::RunEnvHandle>| {
                let s: ::futures::stream::BoxStream<'static, $crate::message::Message> =
                    ::futures::StreamExt::boxed($body);
                $crate::tool::__private::message_stream_to_msg_stream(s)
            },
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runenv::RunEnv;

    async fn local_handle() -> Arc<RunEnvHandle> {
        RunEnv::local().get().await.unwrap()
    }

    #[tokio::test]
    async fn test_sync_value() {
        let runenv = local_handle().await;
        let f = tool_func!(|_args: Value| -> Value { Value::string("ok") });
        let out = f
            .call(Value::object_empty(), "call-1", runenv)
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
    async fn test_sync_value_with_runenv() {
        let runenv = local_handle().await;
        let f =
            tool_func!(|_args: Value, _runenv: Arc<RunEnvHandle>| -> Value { Value::string("ok") });
        let out = f
            .call(Value::object_empty(), "call-1", runenv)
            .next()
            .await
            .unwrap();
        assert_eq!(
            out.message.contents[0].as_value(),
            Some(&Value::string("ok"))
        );
    }

    #[tokio::test]
    async fn test_async_value() {
        let runenv = local_handle().await;
        let f = tool_func!(async |_args: Value| -> Value { Value::bool(true) });
        let out = f
            .call(Value::object_empty(), "call-1", runenv)
            .next()
            .await
            .unwrap();
        assert_eq!(out.message.contents[0].as_value(), Some(&Value::bool(true)));
    }

    #[tokio::test]
    async fn test_async_value_with_runenv() {
        let runenv = local_handle().await;
        let f = tool_func!(async |_args: Value, _runenv: Arc<RunEnvHandle>| -> Value {
            Value::bool(true)
        });
        let out = f
            .call(Value::object_empty(), "call-1", runenv)
            .next()
            .await
            .unwrap();
        assert_eq!(out.message.contents[0].as_value(), Some(&Value::bool(true)));
    }

    #[tokio::test]
    async fn test_stream_value() {
        let runenv = local_handle().await;
        let f = tool_func!(stream |_args: Value| -> Value {
            stream::iter(vec![Value::integer(1), Value::integer(2), Value::integer(3)])
        });
        let outputs: Vec<_> = f
            .call(Value::object_empty(), "call-1", runenv)
            .collect()
            .await;
        assert_eq!(outputs.len(), 3);
        assert_eq!(
            outputs[2].message.contents[0].as_value(),
            Some(&Value::integer(3))
        );
    }
}
