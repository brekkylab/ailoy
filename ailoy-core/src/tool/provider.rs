use crate::tool::{ToolDesc, ToolFunc};

/// Describes and makes the tools it offers, where the agent runs.
pub trait ToolProvider: Send + Sync {
    /// The description of every tool this provider makes.
    fn descs(&self) -> Vec<ToolDesc>;

    /// Make the tool `name`, one of [`descs`](Self::descs). An error fails the run that
    /// asked for it.
    fn make(&self, name: &str) -> anyhow::Result<ToolFunc>;
}
