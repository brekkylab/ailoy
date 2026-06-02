use std::sync::Arc;

use futures::{
    StreamExt,
    future::BoxFuture,
    stream::{self, BoxStream},
};

use crate::{
    datatype::Value,
    message::{FinishReason, Message, MessageOutput, Part, Role},
    runenv::Console,
};

/// Inner closure of a `ToolFunc`. Pure tools don't see the console at all;
/// console tools receive `&dyn Console` and may borrow from it across awaits.
type PureFn = dyn Fn(Value, String) -> BoxStream<'static, MessageOutput> + Send + Sync;
type ConsoleFn =
    dyn for<'a> Fn(Value, String, &'a dyn Console) -> BoxStream<'a, MessageOutput> + Send + Sync;

#[derive(Clone)]
enum ToolFuncInner {
    Pure(Arc<PureFn>),
    WithConsole(Arc<ConsoleFn>),
}

#[derive(Clone)]
pub struct ToolFunc {
    inner: ToolFuncInner,
}

impl ToolFunc {
    /// Construct a [`ToolFunc`] from a closure that does not need a console.
    ///
    /// Prefer the [`crate::tool_func!`] macro; use `new` only for advanced
    /// cases the macro does not cover.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(Value, String) -> BoxStream<'static, MessageOutput> + Send + Sync + 'static,
    {
        Self {
            inner: ToolFuncInner::Pure(Arc::new(f)),
        }
    }

    /// Construct a [`ToolFunc`] from a closure that borrows a `&dyn Console`.
    /// The returned stream's lifetime is tied to the borrow.
    ///
    /// Prefer the [`crate::tool_func!`] macro; use `new_with_console` only
    /// for advanced cases the macro does not cover.
    pub fn new_with_console<F>(f: F) -> Self
    where
        F: for<'a> Fn(Value, String, &'a dyn Console) -> BoxStream<'a, MessageOutput>
            + Send
            + Sync
            + 'static,
    {
        Self {
            inner: ToolFuncInner::WithConsole(Arc::new(f)),
        }
    }

    /// Whether this tool's inner closure actually consumes the `&dyn Console`.
    /// Callers may use this to skip starting a machine for pure tools.
    pub fn needs_console(&self) -> bool {
        matches!(self.inner, ToolFuncInner::WithConsole(_))
    }

    /// Invoke the tool. The `console` argument is always required for API
    /// uniformity; pure variants simply ignore it.
    pub fn call<'a>(
        &self,
        args: Value,
        id: impl Into<String>,
        console: &'a dyn Console,
    ) -> BoxStream<'a, MessageOutput> {
        match &self.inner {
            ToolFuncInner::Pure(f) => f(args, id.into()),
            ToolFuncInner::WithConsole(f) => f(args, id.into(), console),
        }
    }
}

/// Wrapping helpers used by the [`crate::tool_func!`] macro to turn user output
/// (a [`Value`]/[`Message`], an async future of one, or a stream of them) into
/// the canonical `BoxStream<_, MessageOutput>` shape.
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

    pub fn value_fut_to_stream<'a>(
        id: String,
        fut: BoxFuture<'a, Value>,
    ) -> BoxStream<'a, MessageOutput> {
        Box::pin(stream::once(async move {
            MessageOutput {
                message: Message::new(Role::Tool)
                    .with_contents([Part::value(fut.await)])
                    .with_id(id),
                finish_reason: FinishReason::Stop {},
                usage: None,
                depth: None,
                source_agent: None,
            }
        }))
    }

    pub fn message_fut_to_stream<'a>(fut: BoxFuture<'a, Message>) -> BoxStream<'a, MessageOutput> {
        Box::pin(stream::once(async move {
            MessageOutput {
                message: fut.await,
                finish_reason: FinishReason::Stop {},
                usage: None,
                depth: None,
                source_agent: None,
            }
        }))
    }

    pub fn value_stream_to_msg_stream<'a>(
        id: String,
        s: BoxStream<'a, Value>,
    ) -> BoxStream<'a, MessageOutput> {
        Box::pin(s.map(move |value| {
            MessageOutput {
                message: Message::new(Role::Tool)
                    .with_contents([Part::value(value)])
                    .with_id(id.clone()),
                finish_reason: FinishReason::Stop {},
                usage: None,
                depth: None,
                source_agent: None,
            }
        }))
    }

    pub fn message_stream_to_msg_stream<'a>(
        s: BoxStream<'a, Message>,
    ) -> BoxStream<'a, MessageOutput> {
        Box::pin(s.map(|message| MessageOutput {
            message,
            finish_reason: FinishReason::Stop {},
            usage: None,
            depth: None,
            source_agent: None,
        }))
    }
}

/// Build a [`ToolFunc`] from a closure without writing the canonical
/// closure shape or its `BoxFuture`/`BoxStream`/`Box::pin` boilerplate by hand.
///
/// # Forms
///
/// Sync — closure body returns a value/message directly:
///
/// ```ignore
/// tool_func!(|args: Value| -> Value { … })
/// tool_func!(|args: Value, console: &dyn Console| -> Value { … })
/// tool_func!(|args: Value, id: String| -> Message { … })
/// tool_func!(|args: Value, id: String, console: &dyn Console| -> Message { … })
/// ```
///
/// Async — closure body is `async`-awaitable:
///
/// ```ignore
/// tool_func!(async |args: Value| -> Value { … })
/// tool_func!(async |args: Value, console: &dyn Console| -> Value { … })
/// tool_func!(async |args: Value, id: String| -> Message { … })
/// tool_func!(async |args: Value, id: String, console: &dyn Console| -> Message { … })
/// ```
///
/// Stream — closure body produces a `Stream`/`BoxStream`:
///
/// ```ignore
/// tool_func!(stream |args: Value| -> Value { … })
/// tool_func!(stream |args: Value, console: &dyn Console| -> Value { … })
/// tool_func!(stream |args: Value, id: String| -> Message { … })
/// tool_func!(stream |args: Value, id: String, console: &dyn Console| -> Message { … })
/// ```
///
/// Async/stream variants also accept a `with [name = expr, ...]` clause to lift
/// per-call clones out of the `async move` capture (otherwise the surrounding
/// `Fn` closure tries to move the same outer variable on every call):
///
/// ```ignore
/// tool_func!(async |args: Value, console: &dyn Console| -> Value
///     with [runner = runner.clone()]
///     {
///         // body sees local `runner` (the per-call clone)
///     }
/// )
/// ```
#[macro_export]
macro_rules! tool_func {
    // ─── sync, pure ───────────────────────────────────────────────────────
    (|$args:ident : Value| -> Value $body:block) => {
        $crate::tool::ToolFunc::new(
            move |$args: $crate::datatype::Value, id: ::std::string::String| {
                let value: $crate::datatype::Value = $body;
                $crate::tool::__private::value_to_stream(id, value)
            },
        )
    };

    (|$args:ident : Value, $id:ident : String| -> Message $body:block) => {
        $crate::tool::ToolFunc::new(
            move |$args: $crate::datatype::Value, $id: ::std::string::String| {
                let message: $crate::message::Message = $body;
                $crate::tool::__private::message_to_stream(message)
            },
        )
    };

    // ─── sync, with console ───────────────────────────────────────────────
    (|$args:ident : Value, $console:ident : &dyn Console| -> Value $body:block) => {
        $crate::tool::ToolFunc::new_with_console(
            move |$args: $crate::datatype::Value,
                  id: ::std::string::String,
                  $console: &dyn $crate::runenv::Console|
                  -> ::futures::stream::BoxStream<'_, $crate::message::MessageOutput> {
                let value: $crate::datatype::Value = $body;
                $crate::tool::__private::value_to_stream(id, value)
            },
        )
    };

    (
        |$args:ident : Value, $id:ident : String, $console:ident : &dyn Console| -> Message
            $body:block
    ) => {
        $crate::tool::ToolFunc::new_with_console(
            move |$args: $crate::datatype::Value,
                  $id: ::std::string::String,
                  $console: &dyn $crate::runenv::Console|
                  -> ::futures::stream::BoxStream<'_, $crate::message::MessageOutput> {
                let message: $crate::message::Message = $body;
                $crate::tool::__private::message_to_stream(message)
            },
        )
    };

    // ─── async, pure ──────────────────────────────────────────────────────
    (async |$args:ident : Value| -> Value $body:block) => {
        $crate::tool_func!(async |$args: Value| -> Value with [] $body)
    };

    (
        async |$args:ident : Value| -> Value
            with [$($cap:ident = $expr:expr),* $(,)?] $body:block
    ) => {
        $crate::tool::ToolFunc::new(
            move |$args: $crate::datatype::Value, id: ::std::string::String| {
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
            move |$args: $crate::datatype::Value, $id: ::std::string::String| {
                $(let $cap = $expr;)*
                let fut: ::futures::future::BoxFuture<'static, $crate::message::Message> =
                    ::std::boxed::Box::pin(async move $body);
                $crate::tool::__private::message_fut_to_stream(fut)
            },
        )
    };

    // ─── async, with console ──────────────────────────────────────────────
    (async |$args:ident : Value, $console:ident : &dyn Console| -> Value $body:block) => {
        $crate::tool_func!(
            async |$args: Value, $console: &dyn Console| -> Value with [] $body
        )
    };

    (
        async |$args:ident : Value, $console:ident : &dyn Console| -> Value
            with [$($cap:ident = $expr:expr),* $(,)?] $body:block
    ) => {
        $crate::tool::ToolFunc::new_with_console(
            move |$args: $crate::datatype::Value,
                  id: ::std::string::String,
                  $console: &dyn $crate::runenv::Console|
                  -> ::futures::stream::BoxStream<'_, $crate::message::MessageOutput> {
                $(let $cap = $expr;)*
                let fut: ::futures::future::BoxFuture<'_, $crate::datatype::Value> =
                    ::std::boxed::Box::pin(async move $body);
                $crate::tool::__private::value_fut_to_stream(id, fut)
            },
        )
    };

    (
        async |$args:ident : Value, $id:ident : String, $console:ident : &dyn Console|
            -> Message $body:block
    ) => {
        $crate::tool_func!(
            async |$args: Value, $id: String, $console: &dyn Console| -> Message with [] $body
        )
    };

    (
        async |$args:ident : Value, $id:ident : String, $console:ident : &dyn Console|
            -> Message with [$($cap:ident = $expr:expr),* $(,)?] $body:block
    ) => {
        $crate::tool::ToolFunc::new_with_console(
            move |$args: $crate::datatype::Value,
                  $id: ::std::string::String,
                  $console: &dyn $crate::runenv::Console|
                  -> ::futures::stream::BoxStream<'_, $crate::message::MessageOutput> {
                $(let $cap = $expr;)*
                let fut: ::futures::future::BoxFuture<'_, $crate::message::Message> =
                    ::std::boxed::Box::pin(async move $body);
                $crate::tool::__private::message_fut_to_stream(fut)
            },
        )
    };

    // ─── stream, pure ─────────────────────────────────────────────────────
    (stream |$args:ident : Value| -> Value $body:block) => {
        $crate::tool::ToolFunc::new(
            move |$args: $crate::datatype::Value, id: ::std::string::String| {
                let s: ::futures::stream::BoxStream<'static, $crate::datatype::Value> =
                    ::futures::StreamExt::boxed($body);
                $crate::tool::__private::value_stream_to_msg_stream(id, s)
            },
        )
    };

    (stream |$args:ident : Value, $id:ident : String| -> Message $body:block) => {
        $crate::tool::ToolFunc::new(
            move |$args: $crate::datatype::Value, $id: ::std::string::String| {
                let s: ::futures::stream::BoxStream<'static, $crate::message::Message> =
                    ::futures::StreamExt::boxed($body);
                $crate::tool::__private::message_stream_to_msg_stream(s)
            },
        )
    };

    // ─── stream, with console ─────────────────────────────────────────────
    (stream |$args:ident : Value, $console:ident : &dyn Console| -> Value $body:block) => {
        $crate::tool::ToolFunc::new_with_console(
            move |$args: $crate::datatype::Value,
                  id: ::std::string::String,
                  $console: &dyn $crate::runenv::Console|
                  -> ::futures::stream::BoxStream<'_, $crate::message::MessageOutput> {
                let s: ::futures::stream::BoxStream<'_, $crate::datatype::Value> =
                    ::std::boxed::Box::pin($body);
                $crate::tool::__private::value_stream_to_msg_stream(id, s)
            },
        )
    };

    (
        stream |$args:ident : Value, $id:ident : String, $console:ident : &dyn Console|
            -> Message $body:block
    ) => {
        $crate::tool::ToolFunc::new_with_console(
            move |$args: $crate::datatype::Value,
                  $id: ::std::string::String,
                  $console: &dyn $crate::runenv::Console|
                  -> ::futures::stream::BoxStream<'_, $crate::message::MessageOutput> {
                let s: ::futures::stream::BoxStream<'_, $crate::message::Message> =
                    ::std::boxed::Box::pin($body);
                $crate::tool::__private::message_stream_to_msg_stream(s)
            },
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runenv::{Local, Machine};

    #[tokio::test]
    async fn test_sync_value() {
        let mut local = Local::new();
        let console = local.start().await.unwrap();
        let f = tool_func!(|_args: Value| -> Value { Value::string("ok") });
        let out = f
            .call(Value::object_empty(), "call-1", console)
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
    async fn test_sync_value_with_console() {
        let mut local = Local::new();
        let console = local.start().await.unwrap();
        let f = tool_func!(|_args: Value, _console: &dyn Console| -> Value { Value::string("ok") });
        let out = f
            .call(Value::object_empty(), "call-1", console)
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
        let mut local = Local::new();
        let console = local.start().await.unwrap();
        let f = tool_func!(async |_args: Value| -> Value { Value::bool(true) });
        let out = f
            .call(Value::object_empty(), "call-1", console)
            .next()
            .await
            .unwrap();
        assert_eq!(out.message.contents[0].as_value(), Some(&Value::bool(true)));
    }

    #[tokio::test]
    async fn test_async_value_with_console() {
        let mut local = Local::new();
        let console = local.start().await.unwrap();
        let f = tool_func!(async |_args: Value, console: &dyn Console| -> Value {
            let r = console
                .exec_shell("echo hi".to_string(), None)
                .await
                .unwrap();
            Value::string(r.stdout.trim().to_string())
        });
        let out = f
            .call(Value::object_empty(), "call-1", console)
            .next()
            .await
            .unwrap();
        assert_eq!(
            out.message.contents[0].as_value(),
            Some(&Value::string("hi"))
        );
    }

    #[tokio::test]
    async fn test_stream_value() {
        let mut local = Local::new();
        let console = local.start().await.unwrap();
        let f = tool_func!(stream |_args: Value| -> Value {
            stream::iter(vec![Value::integer(1), Value::integer(2), Value::integer(3)])
        });
        let outputs: Vec<_> = f
            .call(Value::object_empty(), "call-1", console)
            .collect()
            .await;
        assert_eq!(outputs.len(), 3);
        assert_eq!(
            outputs[2].message.contents[0].as_value(),
            Some(&Value::integer(3))
        );
    }

    #[tokio::test]
    async fn test_needs_console() {
        let pure = tool_func!(|_args: Value| -> Value { Value::bool(false) });
        let with_console =
            tool_func!(|_args: Value, _console: &dyn Console| -> Value { Value::bool(true) });
        assert!(!pure.needs_console());
        assert!(with_console.needs_console());
    }
}
