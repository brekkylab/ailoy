use std::sync::Arc;

use futures::stream::BoxStream;

use crate::{
    agent::AgentSpec,
    datatype::Value,
    message::{MessageOutput, ToolDesc},
    tool::{ToolContext, ToolFunc},
};

/// Deferred constructor that produces a [`Tool`] tailored to a given [`AgentSpec`].
///
/// Each entry in a [`ToolProvider`](super::ToolProvider) resolves to a `ToolFactory`,
/// which is then called via [`ToolFactory::make`] with the agent's spec to select the
/// right [`ToolFunc`] implementation (e.g. a sandbox-aware variant) and return a
/// ready-to-call [`Tool`].
#[derive(Clone)]
pub struct ToolFactory {
    name: String,
    f: Arc<dyn Fn(&AgentSpec) -> (ToolDesc, Arc<ToolFunc>) + Send + Sync>,
}

impl std::fmt::Debug for ToolFactory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolFactory")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl ToolFactory {
    pub fn new(
        name: impl Into<String>,
        f: Arc<dyn Fn(&AgentSpec) -> (ToolDesc, Arc<ToolFunc>) + Send + Sync>,
    ) -> Self {
        Self {
            name: name.into(),
            f,
        }
    }
    // Always returning same F
    pub fn simple(desc: ToolDesc, f: ToolFunc) -> Self {
        let f = Arc::new(f);
        Self::new(
            desc.name.clone(),
            Arc::new(move |_| (desc.clone(), f.clone())),
        )
    }

    // Create with initializer
    pub fn with_initializer(
        desc: ToolDesc,
        f: impl Fn(&AgentSpec) -> ToolFunc + Send + Sync + 'static,
    ) -> Self {
        Self::new(
            desc.name.clone(),
            Arc::new(move |spec| {
                let f = f(spec);
                (desc.clone(), Arc::new(f))
            }),
        )
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }

    pub fn make(&self, spec: &AgentSpec) -> Tool {
        let (desc, f) = (self.f)(spec);
        Tool::new(desc, f)
    }
}

/// Runtime tool bound to a specific [`AgentSpec`].
///
/// Produced by [`ToolFactory::make`], it pairs a [`ToolDesc`] (name, description,
/// JSON schema exposed to the LLM) with the concrete [`ToolFunc`] chosen for
/// that agent's configuration.  Call [`Tool::call`] to execute it.
#[derive(Clone)]
pub struct Tool {
    desc: ToolDesc,
    f: Arc<ToolFunc>,
}

impl Tool {
    pub(crate) fn new(desc: ToolDesc, f: Arc<ToolFunc>) -> Self {
        Self { desc, f }
    }

    pub fn get_desc(&self) -> &ToolDesc {
        &self.desc
    }

    pub fn get_func(&self) -> Arc<ToolFunc> {
        self.f.clone()
    }

    /// Execute this tool and return a stream of [`MessageOutput`] items.
    ///
    /// The agent iterates the full stream to forward intermediate sub-agent
    /// outputs and collect the final tool result.  For simple tools that emit
    /// exactly one item, callers can just call `.next().await`.
    pub fn call(&self, args: Value, ctx: ToolContext) -> BoxStream<'static, MessageOutput> {
        self.f.call(args, ctx)
    }
}

#[cfg(test)]
pub(crate) mod test_helpers {
    use futures::StreamExt as _;

    use crate::{
        datatype::Value,
        message::Message,
        tool::{Tool, ToolContext},
    };

    impl Tool {
        pub async fn call_next(&self, args: Value, ctx: ToolContext) -> Message {
            self.call(args, ctx).next().await.unwrap().message
        }
    }
}
