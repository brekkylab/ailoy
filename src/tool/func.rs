use std::sync::Arc;

use futures::{
    StreamExt,
    future::BoxFuture,
    stream::{self, BoxStream},
};

use crate::{
    console::Sandbox,
    datatype::Value,
    message::{FinishReason, Message, MessageOutput, Part, Role},
};

/// A tool body that borrows the sandbox for as long as its stream runs.
#[rustfmt::skip]
type SandboxFn =
    dyn for<'a> Fn(Value, String, &'a Sandbox) -> BoxStream<'a, MessageOutput> + Send + Sync;

#[derive(Clone)]
enum ToolFuncInner {
    Pure(Arc<dyn Fn(Value, String) -> BoxStream<'static, MessageOutput> + Send + Sync>),
    WithSandbox(Arc<SandboxFn>),
}

#[derive(Clone)]
pub struct ToolFunc {
    inner: ToolFuncInner,
}

impl ToolFunc {
    /// Construct a [`ToolFunc`] from a closure that does not need a sandbox.
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

    /// Construct a [`ToolFunc`] from a closure that borrows the agent's `&Sandbox`.
    /// The returned stream's lifetime is tied to the borrow.
    ///
    /// Prefer the [`crate::tool_func!`] macro; use `new_with_sandbox` only
    /// for advanced cases the macro does not cover.
    pub fn new_with_sandbox<F>(f: F) -> Self
    where
        F: for<'a> Fn(Value, String, &'a Sandbox) -> BoxStream<'a, MessageOutput>
            + Send
            + Sync
            + 'static,
    {
        Self {
            inner: ToolFuncInner::WithSandbox(Arc::new(f)),
        }
    }

    /// Whether this tool's inner closure takes the `&Sandbox`.
    pub fn needs_sandbox(&self) -> bool {
        matches!(self.inner, ToolFuncInner::WithSandbox(_))
    }

    /// Invoke the tool without a sandbox, or `None` if it needs one.
    ///
    /// `None` for a sandbox tool, since a sandbox can't be fabricated.
    pub fn call_pure(
        &self,
        args: Value,
        id: impl Into<String>,
    ) -> Option<BoxStream<'static, MessageOutput>> {
        match &self.inner {
            ToolFuncInner::Pure(f) => Some(f(args, id.into())),
            ToolFuncInner::WithSandbox(_) => None,
        }
    }

    /// Invoke the tool. `sandbox` is always required; pure variants ignore it.
    /// A caller with no sandbox to lend uses [`call_pure`](Self::call_pure).
    pub fn call<'a>(
        &self,
        args: Value,
        id: impl Into<String>,
        sandbox: &'a Sandbox,
    ) -> BoxStream<'a, MessageOutput> {
        match &self.inner {
            ToolFuncInner::Pure(f) => f(args, id.into()),
            ToolFuncInner::WithSandbox(f) => f(args, id.into(), sandbox),
        }
    }
}

/// Wrapping helpers used by the [`crate::tool_func!`] macro to turn user output
/// (a [`Value`]/[`Message`], an async future of one, or a stream of them) into
/// the canonical `BoxStream<_, MessageOutput>` shape.
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
        }))
        .boxed()
    }

    pub fn message_to_stream(message: Message) -> BoxStream<'static, MessageOutput> {
        stream::once(std::future::ready(MessageOutput {
            message,
            finish_reason: FinishReason::Stop {},
            usage: None,
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
            }
        }))
    }

    pub fn message_fut_to_stream<'a>(fut: BoxFuture<'a, Message>) -> BoxStream<'a, MessageOutput> {
        Box::pin(stream::once(async move {
            MessageOutput {
                message: fut.await,
                finish_reason: FinishReason::Stop {},
                usage: None,
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
/// tool_func!(|args: Value, sandbox: &Sandbox| -> Value { … })
/// tool_func!(|args: Value, id: String| -> Message { … })
/// tool_func!(|args: Value, id: String, sandbox: &Sandbox| -> Message { … })
/// ```
///
/// Async — closure body is `async`-awaitable:
///
/// ```ignore
/// tool_func!(async |args: Value| -> Value { … })
/// tool_func!(async |args: Value, sandbox: &Sandbox| -> Value { … })
/// tool_func!(async |args: Value, id: String| -> Message { … })
/// tool_func!(async |args: Value, id: String, sandbox: &Sandbox| -> Message { … })
/// ```
///
/// Stream — closure body produces a `Stream`/`BoxStream`:
///
/// ```ignore
/// tool_func!(stream |args: Value| -> Value { … })
/// tool_func!(stream |args: Value, sandbox: &Sandbox| -> Value { … })
/// tool_func!(stream |args: Value, id: String| -> Message { … })
/// tool_func!(stream |args: Value, id: String, sandbox: &Sandbox| -> Message { … })
/// ```
///
/// Async/stream variants also accept a `with [name = expr, ...]` clause to lift
/// per-call clones out of the `async move` capture (otherwise the surrounding
/// `Fn` closure tries to move the same outer variable on every call):
///
/// ```ignore
/// tool_func!(async |args: Value, sandbox: &Sandbox| -> Value
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

    // ─── sync, with sandbox ───────────────────────────────────────────────
    (|$args:ident : Value, $sandbox:ident : &Sandbox| -> Value $body:block) => {
        $crate::tool::ToolFunc::new_with_sandbox(
            move |$args: $crate::datatype::Value,
                  id: ::std::string::String,
                  $sandbox: &$crate::console::Sandbox|
                  -> ::futures::stream::BoxStream<'_, $crate::message::MessageOutput> {
                let value: $crate::datatype::Value = $body;
                $crate::tool::__private::value_to_stream(id, value)
            },
        )
    };

    (
        |$args:ident : Value, $id:ident : String, $sandbox:ident : &Sandbox| -> Message
            $body:block
    ) => {
        $crate::tool::ToolFunc::new_with_sandbox(
            move |$args: $crate::datatype::Value,
                  $id: ::std::string::String,
                  $sandbox: &$crate::console::Sandbox|
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

    // ─── async, with sandbox ──────────────────────────────────────────────
    (async |$args:ident : Value, $sandbox:ident : &Sandbox| -> Value $body:block) => {
        $crate::tool_func!(
            async |$args: Value, $sandbox: &Sandbox| -> Value with [] $body
        )
    };

    (
        async |$args:ident : Value, $sandbox:ident : &Sandbox| -> Value
            with [$($cap:ident = $expr:expr),* $(,)?] $body:block
    ) => {
        $crate::tool::ToolFunc::new_with_sandbox(
            move |$args: $crate::datatype::Value,
                  id: ::std::string::String,
                  $sandbox: &$crate::console::Sandbox|
                  -> ::futures::stream::BoxStream<'_, $crate::message::MessageOutput> {
                $(let $cap = $expr;)*
                let fut: ::futures::future::BoxFuture<'_, $crate::datatype::Value> =
                    ::std::boxed::Box::pin(async move $body);
                $crate::tool::__private::value_fut_to_stream(id, fut)
            },
        )
    };

    (
        async |$args:ident : Value, $id:ident : String, $sandbox:ident : &Sandbox|
            -> Message $body:block
    ) => {
        $crate::tool_func!(
            async |$args: Value, $id: String, $sandbox: &Sandbox| -> Message with [] $body
        )
    };

    (
        async |$args:ident : Value, $id:ident : String, $sandbox:ident : &Sandbox|
            -> Message with [$($cap:ident = $expr:expr),* $(,)?] $body:block
    ) => {
        $crate::tool::ToolFunc::new_with_sandbox(
            move |$args: $crate::datatype::Value,
                  $id: ::std::string::String,
                  $sandbox: &$crate::console::Sandbox|
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

    // ─── stream, with sandbox ─────────────────────────────────────────────
    (stream |$args:ident : Value, $sandbox:ident : &Sandbox| -> Value $body:block) => {
        $crate::tool::ToolFunc::new_with_sandbox(
            move |$args: $crate::datatype::Value,
                  id: ::std::string::String,
                  $sandbox: &$crate::console::Sandbox|
                  -> ::futures::stream::BoxStream<'_, $crate::message::MessageOutput> {
                let s: ::futures::stream::BoxStream<'_, $crate::datatype::Value> =
                    ::std::boxed::Box::pin($body);
                $crate::tool::__private::value_stream_to_msg_stream(id, s)
            },
        )
    };

    (
        stream |$args:ident : Value, $id:ident : String, $sandbox:ident : &Sandbox|
            -> Message $body:block
    ) => {
        $crate::tool::ToolFunc::new_with_sandbox(
            move |$args: $crate::datatype::Value,
                  $id: ::std::string::String,
                  $sandbox: &$crate::console::Sandbox|
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
    use crate::test_sandbox;

    #[tokio::test]
    async fn test_sync_value() {
        let sandbox = test_sandbox().await;
        let f = tool_func!(|_args: Value| -> Value { Value::string("ok") });
        let out = f
            .call(Value::object_empty(), "call-1", &sandbox)
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
    async fn test_sync_value_with_sandbox() {
        let sandbox = test_sandbox().await;
        let f = tool_func!(|_args: Value, _sandbox: &Sandbox| -> Value { Value::string("ok") });
        let out = f
            .call(Value::object_empty(), "call-1", &sandbox)
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
        let sandbox = test_sandbox().await;
        let f = tool_func!(async |_args: Value| -> Value { Value::bool(true) });
        let out = f
            .call(Value::object_empty(), "call-1", &sandbox)
            .next()
            .await
            .unwrap();
        assert_eq!(out.message.contents[0].as_value(), Some(&Value::bool(true)));
    }

    #[tokio::test]
    async fn test_async_value_with_sandbox() {
        let sandbox = test_sandbox().await;
        let f = tool_func!(async |_args: Value, sandbox: &Sandbox| -> Value {
            // `exec` takes an argv: a shell must be invoked explicitly.
            let r = crate::console::exec(sandbox, ["sh", "-c", "echo hi"], None)
                .await
                .unwrap();
            Value::string(r.stdout.trim().to_string())
        });
        let out = f
            .call(Value::object_empty(), "call-1", &sandbox)
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
        let sandbox = test_sandbox().await;
        let f = tool_func!(stream |_args: Value| -> Value {
            stream::iter(vec![Value::integer(1), Value::integer(2), Value::integer(3)])
        });
        let outputs: Vec<_> = f
            .call(Value::object_empty(), "call-1", &sandbox)
            .collect()
            .await;
        assert_eq!(outputs.len(), 3);
        assert_eq!(
            outputs[2].message.contents[0].as_value(),
            Some(&Value::integer(3))
        );
    }

    #[tokio::test]
    async fn test_needs_sandbox() {
        let pure = tool_func!(|_args: Value| -> Value { Value::bool(false) });
        let with_sandbox =
            tool_func!(|_args: Value, _sandbox: &Sandbox| -> Value { Value::bool(true) });
        assert!(!pure.needs_sandbox());
        assert!(with_sandbox.needs_sandbox());
    }
}
