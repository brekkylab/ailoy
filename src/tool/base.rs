use std::{fmt::Debug, future::Future, pin::Pin, sync::Arc};

use ailoy_macros::{maybe_send_sync, multi_platform_async_trait};

use super::{
    builtin::{
        BuiltinToolKind, create_terminal_tool, create_web_fetch_tool,
        create_web_search_duckduckgo_tool,
    },
    function::{FunctionTool, ToolFunc},
    mcp::MCPTool,
};
use crate::message::{ToolDesc, Value};

#[maybe_send_sync]
#[multi_platform_async_trait]
pub trait ToolBehavior: Debug + Clone {
    fn get_description(&self) -> ToolDesc;
    async fn run(&self, args: Value) -> anyhow::Result<Value>;
}

#[derive(Debug, Clone)]
pub enum ToolInner {
    Function(FunctionTool),
    MCP(MCPTool),
}

#[derive(Debug, Clone)]
pub struct Tool {
    inner: ToolInner,
}

/// Fluent builder for [`Tool`]. Obtain via [`Tool::builder()`].
pub struct ToolBuilder {
    name: Option<String>,
    description: Option<String>,
    parameters: Option<Value>,
    returns: Option<Value>,
    handler: Option<Arc<ToolFunc>>,
    typed_params: Vec<(String, serde_json::Value)>,
    optional_params: Vec<(String, serde_json::Value)>,
}

impl ToolBuilder {
    fn new() -> Self {
        Self {
            name: None,
            description: None,
            parameters: None,
            returns: None,
            handler: None,
            typed_params: Vec::new(),
            optional_params: Vec::new(),
        }
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    pub fn parameters(mut self, params: impl Into<Value>) -> Self {
        self.parameters = Some(params.into());
        self
    }

    pub fn returns(mut self, ret: impl Into<Value>) -> Self {
        self.returns = Some(ret.into());
        self
    }

    /// Add a typed parameter whose JSON Schema is derived automatically via [`schemars::JsonSchema`].
    pub fn parameter<T: schemars::JsonSchema>(mut self, name: impl Into<String>) -> Self {
        let mut schema_gen = schemars::r#gen::SchemaGenerator::default();
        let schema = schema_gen.subschema_for::<T>();
        let schema_json = serde_json::to_value(schema).unwrap();
        self.typed_params.push((name.into(), schema_json));
        self
    }

    /// Add a typed optional parameter (not included in "required" array).
    pub fn optional_parameter<T: schemars::JsonSchema>(mut self, name: impl Into<String>) -> Self {
        let mut schema_gen = schemars::r#gen::SchemaGenerator::default();
        let schema = schema_gen.subschema_for::<T>();
        let schema_json = serde_json::to_value(schema).unwrap();
        self.optional_params.push((name.into(), schema_json));
        self
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn handler<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = anyhow::Result<Value>> + Send + 'static,
    {
        self.handler = Some(Arc::new(move |args| {
            Box::pin(f(args)) as Pin<Box<super::function::ToolFuncResult>>
        }));
        self
    }

    #[cfg(target_arch = "wasm32")]
    pub fn handler<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(Value) -> Fut + 'static,
        Fut: Future<Output = anyhow::Result<Value>> + 'static,
    {
        self.handler = Some(Arc::new(move |args| {
            Box::pin(f(args)) as Pin<Box<super::function::ToolFuncResult>>
        }));
        self
    }

    pub fn build(self) -> anyhow::Result<Tool> {
        let name = self
            .name
            .ok_or_else(|| anyhow::anyhow!("Tool name is required"))?;
        let handler = self
            .handler
            .ok_or_else(|| anyhow::anyhow!("Tool handler is required"))?;
        let parameters = if let Some(p) = self.parameters {
            p
        } else if !self.typed_params.is_empty() || !self.optional_params.is_empty() {
            let mut properties: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
            for (name, schema) in &self.typed_params {
                properties.insert(name.clone(), schema.clone());
            }
            for (name, schema) in &self.optional_params {
                properties.insert(name.clone(), schema.clone());
            }
            let required: Vec<serde_json::Value> = self
                .typed_params
                .iter()
                .map(|(name, _)| serde_json::Value::String(name.clone()))
                .collect();
            let mut json = serde_json::json!({
                "type": "object",
                "properties": properties,
            });
            if !required.is_empty() {
                json["required"] = serde_json::Value::Array(required);
            }
            serde_json::from_value(json).unwrap_or(Value::Null)
        } else {
            Value::Null
        };
        let desc = ToolDesc {
            name,
            description: self.description,
            parameters,
            returns: self.returns,
        };
        Ok(Tool {
            inner: ToolInner::Function(FunctionTool::new(desc, handler)),
        })
    }
}

impl Tool {
    pub fn builder() -> ToolBuilder {
        ToolBuilder::new()
    }

    pub fn new_builtin(kind: BuiltinToolKind, config: Value) -> anyhow::Result<Self> {
        let tool = match kind {
            BuiltinToolKind::Terminal => create_terminal_tool(config),
            BuiltinToolKind::WebSearchDuckduckgo => create_web_search_duckduckgo_tool(config),
            BuiltinToolKind::WebFetch => create_web_fetch_tool(config),
        }?;
        Ok(Self {
            inner: ToolInner::Function(tool),
        })
    }

    pub fn new_function(desc: ToolDesc, f: Arc<ToolFunc>) -> Self {
        Self {
            inner: ToolInner::Function(FunctionTool::new(desc, f)),
        }
    }

    pub fn new_mcp(tool: MCPTool) -> Self {
        Self {
            inner: ToolInner::MCP(tool),
        }
    }
}

#[multi_platform_async_trait]
impl ToolBehavior for Tool {
    fn get_description(&self) -> ToolDesc {
        match &self.inner {
            ToolInner::Function(tool) => tool.get_description(),
            ToolInner::MCP(tool) => tool.get_description(),
        }
    }

    async fn run(&self, args: Value) -> anyhow::Result<Value> {
        match &self.inner {
            ToolInner::Function(tool) => tool.run(args).await,
            ToolInner::MCP(tool) => tool.run(args).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use ailoy_macros::{multi_platform_test, tool};

    use super::{ToolBehavior, *};
    use crate::{message::Value, to_value};

    #[test]
    fn tool_builder_with_typed_params() {
        let tool = Tool::builder()
            .name("search")
            .description("Search the web")
            .parameter::<String>("query")
            .parameter::<i32>("limit")
            .handler(|_args| async move { Ok(Value::Null) })
            .build()
            .unwrap();
        let desc = tool.get_description();
        assert_eq!(desc.name, "search");
        assert_eq!(desc.description.as_deref(), Some("Search the web"));
        // Verify the parameters schema has the expected structure
        let params_json = serde_json::to_value(&desc.parameters).unwrap();
        assert_eq!(params_json["type"], "object");
        assert!(params_json["properties"]["query"].is_object());
        assert!(params_json["properties"]["limit"].is_object());
        let required = params_json["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::Value::String("query".into())));
        assert!(required.contains(&serde_json::Value::String("limit".into())));
    }

    #[test]
    fn tool_builder_explicit_parameters_takes_precedence() {
        let explicit_params = serde_json::json!({"type": "object", "properties": {}});
        let explicit_value: Value = serde_json::from_value(explicit_params).unwrap();
        let tool = Tool::builder()
            .name("test")
            .parameters(explicit_value.clone())
            .parameter::<String>("ignored")
            .handler(|_args| async move { Ok(Value::Null) })
            .build()
            .unwrap();
        let desc = tool.get_description();
        // explicit parameters should win over typed_params
        assert_eq!(desc.parameters, explicit_value);
    }

    #[tool(description = "Add two integers and return the sum")]
    async fn add(a: i64, b: i64) -> anyhow::Result<Value> {
        Ok(to_value!(a + b))
    }

    #[test]
    fn tool_macro_name_and_description() {
        let tool = add_tool();
        let desc = tool.get_description();
        assert_eq!(desc.name, "add");
        assert_eq!(
            desc.description.as_deref(),
            Some("Add two integers and return the sum")
        );
    }

    #[test]
    fn tool_macro_parameter_schema() {
        let desc = add_tool().get_description();
        let params = serde_json::to_value(&desc.parameters).unwrap();
        assert_eq!(params["type"], "object");
        assert!(params["properties"]["a"].is_object());
        assert!(params["properties"]["b"].is_object());
        let req = params["required"].as_array().unwrap();
        assert!(req.contains(&serde_json::Value::String("a".into())));
        assert!(req.contains(&serde_json::Value::String("b".into())));
    }

    #[multi_platform_test]
    async fn tool_macro_handler_runs() {
        let tool = add_tool();
        let result = tool.run(to_value!({"a": 3, "b": 4})).await.unwrap();
        assert_eq!(result, to_value!(7));
    }

    #[multi_platform_test]
    async fn tool_macro_handler_missing_param_error() {
        let tool = add_tool();
        let err = tool.run(to_value!({"a": 1})).await.unwrap_err();
        assert!(
            err.to_string().contains("Missing required parameter"),
            "unexpected error: {err}"
        );
    }
}
