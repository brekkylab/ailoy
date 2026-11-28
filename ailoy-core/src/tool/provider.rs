use std::sync::Arc;

use indexmap::IndexMap;

use crate::{
    agent::Refusal,
    tool::{ToolDesc, ToolFunc},
};

/// Describes and makes the tools it offers, where the agent runs.
pub trait ToolSet: Send + Sync {
    /// The description of every tool this set makes.
    fn descs(&self) -> Vec<ToolDesc>;

    /// Make the tool `name`, one of [`descs`](Self::descs). An error fails the run that
    /// asked for it.
    fn make(&self, name: &str) -> anyhow::Result<ToolFunc>;
}

/// A function of this process as a tool: `desc` describes it, and every agent gets `func`.
impl ToolSet for (ToolDesc, ToolFunc) {
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

/// The tools a spec may name, each with its description and the set that makes it.
#[derive(Default)]
pub struct ToolProvider {
    /// Each tool's description and the set that makes it, by the tool's name.
    tools: IndexMap<String, (ToolDesc, Arc<dyn ToolSet>)>,
}

impl ToolProvider {
    pub fn new() -> Self {
        Self::default()
    }

    /// Provide each tool `set` describes, in place of one of its name already provided.
    /// Answers their names, which a spec lists to give an agent the tools.
    pub fn set(&mut self, set: impl ToolSet + 'static) -> Vec<String> {
        let set: Arc<dyn ToolSet> = Arc::new(set);
        let descs = set.descs();
        let mut names = Vec::with_capacity(descs.len());
        for desc in descs {
            names.push(desc.name.clone());
            self.tools.insert(desc.name.clone(), (desc, set.clone()));
        }
        names
    }

    /// Stop providing the tool `name`. Answers whether it was provided.
    pub fn remove(&mut self, name: &str) -> bool {
        self.tools.shift_remove(name).is_some()
    }

    /// Stop providing every tool whose name starts with `prefix`. Answers how many there were.
    pub fn remove_by_prefix(&mut self, prefix: &str) -> usize {
        let before = self.tools.len();
        self.tools.retain(|name, _| !name.starts_with(prefix));
        before - self.tools.len()
    }

    /// Whether the tool `name` is provided.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// The description of every tool provided, in the order each name was first provided.
    pub fn descs(&self) -> Vec<ToolDesc> {
        self.tools.values().map(|(desc, _)| desc.clone()).collect()
    }

    /// Make the tool `name` through its set, with the description it is shown to a model by.
    pub fn create(&self, name: &str) -> anyhow::Result<(ToolDesc, ToolFunc)> {
        let (desc, set) = self
            .tools
            .get(name)
            .ok_or_else(|| Refusal::NoTool(name.to_owned()))?;
        Ok((desc.clone(), set.make(name)?))
    }
}
