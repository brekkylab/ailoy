use std::sync::Arc;

use ailoy_macros::{maybe_send_sync, multi_platform_async_trait};
use serde_json::Value as JsonValue;

use crate::{tool::Tool, value::Value};

/// A document returned by a knowledge source.
#[derive(Debug, Clone)]
pub struct Document {
    pub content: String,
    pub metadata: JsonValue,
}

impl Document {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            metadata: JsonValue::Null,
        }
    }

    pub fn with_metadata(mut self, metadata: JsonValue) -> Self {
        self.metadata = metadata;
        self
    }
}

/// A knowledge source that can retrieve relevant documents for a query.
#[maybe_send_sync]
#[multi_platform_async_trait]
pub trait Knowledge {
    /// Retrieve relevant documents for the given query.
    async fn retrieve(&self, query: &str) -> anyhow::Result<Vec<Document>>;

    /// Format retrieved documents into a context string to prepend to the user message.
    /// Override to customize the format.
    fn format_context(&self, _query: &str, docs: &[Document]) -> String {
        let mut out = String::from("[Retrieved Context]\n");
        for (i, doc) in docs.iter().enumerate() {
            out.push_str(&format!("{}. {}\n", i + 1, doc.content));
        }
        out
    }
}

/// Platform-appropriate `Arc`-compatible type alias for a `Knowledge` trait object.
/// Expands to `dyn Knowledge + Send + Sync` on non-wasm32, `dyn Knowledge` on wasm32.
#[maybe_send_sync]
pub(crate) type KnowledgeDyn = dyn Knowledge;

/// Extension trait providing `into_tool` for wrapping a [`Knowledge`] source as a [`Tool`].
///
/// Separate from [`Knowledge`] because `into_tool` requires `Self: Sized`, which is
/// incompatible with object safety.
pub trait KnowledgeExt: Knowledge + Sized + 'static {
    /// Wrap this knowledge source as a named, callable tool the LLM can invoke (Pattern A).
    fn into_tool(self, name: impl Into<String>, description: impl Into<String>) -> Tool;
}

impl<K: Knowledge + Sized + 'static> KnowledgeExt for K {
    fn into_tool(self, name: impl Into<String>, description: impl Into<String>) -> Tool {
        let arc = Arc::new(self);
        Tool::builder()
            .name(name)
            .description(description)
            .parameter::<String>("query")
            .handler(move |args: Value| {
                let arc = arc.clone();
                async move {
                    let query: String = serde_json::from_value(
                        serde_json::to_value(
                            args.as_object()
                                .ok_or_else(|| anyhow::anyhow!("Tool args must be a JSON object"))?
                                .get("query")
                                .ok_or_else(|| {
                                    anyhow::anyhow!("Missing required parameter: query")
                                })?,
                        )
                        .map_err(anyhow::Error::from)?,
                    )?;
                    let docs = arc.retrieve(&query).await?;
                    let text = arc.format_context(&query, &docs);
                    Ok(Value::String(text))
                }
            })
            .build()
            .expect("KnowledgeExt::into_tool: should not fail")
    }
}

#[cfg(test)]
mod tests {
    use ailoy_macros::{multi_platform_async_trait, multi_platform_test};

    use super::*;
    use crate::{to_value, tool::ToolBehavior as _};

    struct MockKnowledge {
        docs: Vec<Document>,
    }

    #[multi_platform_async_trait]
    impl Knowledge for MockKnowledge {
        async fn retrieve(&self, _query: &str) -> anyhow::Result<Vec<Document>> {
            Ok(self.docs.clone())
        }
    }

    #[test]
    fn format_context_default_output() {
        let k = MockKnowledge { docs: vec![] };
        let docs = vec![Document::new("foo"), Document::new("bar")];
        let ctx = k.format_context("q", &docs);
        assert!(ctx.starts_with("[Retrieved Context]"), "ctx: {ctx}");
        assert!(ctx.contains("1. foo"), "ctx: {ctx}");
        assert!(ctx.contains("2. bar"), "ctx: {ctx}");
    }

    struct CustomKnowledge;

    #[multi_platform_async_trait]
    impl Knowledge for CustomKnowledge {
        async fn retrieve(&self, _: &str) -> anyhow::Result<Vec<Document>> {
            Ok(vec![])
        }

        fn format_context(&self, query: &str, _docs: &[Document]) -> String {
            format!("Custom: {query}")
        }
    }

    #[test]
    fn format_context_custom_override() {
        let k = CustomKnowledge;
        let ctx = k.format_context("myquery", &[]);
        assert_eq!(ctx, "Custom: myquery");
    }

    #[multi_platform_test]
    async fn into_tool_name_description_and_schema() {
        let store = MockKnowledge {
            docs: vec![Document::new("hello world")],
        };
        let tool = store.into_tool("search_docs", "Search the docs");
        let desc = tool.get_description();
        assert_eq!(desc.name, "search_docs");
        assert_eq!(desc.description.as_deref(), Some("Search the docs"));
        let params_json = serde_json::to_value(&desc.parameters).unwrap();
        assert!(
            params_json["properties"]["query"].is_object(),
            "expected query parameter in schema"
        );
        let required = params_json["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::Value::String("query".into())));
    }

    #[multi_platform_test]
    async fn into_tool_handler_returns_formatted_context() {
        let store = MockKnowledge {
            docs: vec![Document::new("hello world")],
        };
        let tool = store.into_tool("search", "desc");
        let result = tool.run(to_value!({"query": "test"})).await.unwrap();
        let s = result.as_str().expect("expected String value");
        assert!(s.contains("hello world"), "result: {s}");
    }
}
