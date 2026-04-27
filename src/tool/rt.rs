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
/// Held in a [`super::ToolSet`] until an agent is instantiated.  At that point
/// [`ToolFactory::make`] is called with the agent's spec to select the right
/// [`ToolFunc`] implementation (e.g. sandbox vs. no-sandbox) and return a
/// ready-to-call [`Tool`].
#[derive(Clone)]
pub struct ToolFactory {
    name: String,
    f: Arc<dyn Fn(&AgentSpec) -> (ToolDesc, Arc<ToolFunc>)>,
}

impl ToolFactory {
    pub fn new(
        name: impl Into<String>,
        f: Arc<dyn Fn(&AgentSpec) -> (ToolDesc, Arc<ToolFunc>)>,
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
    ) -> Self
    {
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
