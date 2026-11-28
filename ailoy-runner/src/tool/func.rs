//! `ToolFunc` is `ailoy-core`'s; these tests run it in a sandbox.

#[cfg(test)]
mod tests {
    use futures::{StreamExt, stream};

    use crate::test_sandbox;
    use crate::{datatype::Value, tool_func};

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
