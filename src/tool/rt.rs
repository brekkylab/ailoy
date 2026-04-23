use std::sync::Arc;

use futures::stream::BoxStream;

use crate::{
    agent::AgentSpec,
    datatype::Value,
    message::{MessageOutput, ToolDesc},
    tool::{ToolContext, ToolFunc},
};

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

    // Sandbox-aware, if agent have sandbox: f1, or f2
    pub fn sandbox_aware(desc: ToolDesc, f1: ToolFunc, f2: ToolFunc) -> Self {
        let f1 = Arc::new(f1);
        let f2 = Arc::new(f2);
        Self::new(
            desc.name.clone(),
            Arc::new(move |spec| {
                if let Some(_) = spec.sandbox {
                    (desc.clone(), f1.clone())
                } else {
                    (desc.clone(), f2.clone())
                }
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
