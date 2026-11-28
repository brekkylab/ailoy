use crate::tool::{ToolDesc, ToolFunc};

/// Describes and makes the tools it offers, where the agent runs.
pub trait ToolProvider: Send + Sync {
    /// The description of every tool this provider makes.
    fn descs(&self) -> Vec<ToolDesc>;

    /// Make the tool `name`, one of [`descs`](Self::descs). An error fails the run that
    /// asked for it.
    fn make(&self, name: &str) -> anyhow::Result<ToolFunc>;
}

/// A function of this process as a tool: `desc` describes it, and every agent gets `func`.
impl ToolProvider for (ToolDesc, ToolFunc) {
    fn descs(&self) -> Vec<ToolDesc> {
        vec![self.0.clone()]
    }

    fn make(&self, name: &str) -> anyhow::Result<ToolFunc> {
        anyhow::ensure!(
            name == self.0.name,
            "this provides '{}', not '{name}'",
            self.0.name
        );
        Ok(self.1.clone())
    }
}
