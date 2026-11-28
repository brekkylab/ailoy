//! The tools that reach the web from this process.

mod fetch;
mod search;

pub use fetch::*;
pub use search::*;

use crate::tool::{ToolDesc, ToolFunc, ToolSet};

/// `web_search` and `web_fetch`.
pub struct Web {
    /// Makes `web_fetch`: every one it makes shares its pace per host.
    fetch: Box<dyn Fn() -> anyhow::Result<ToolFunc> + Send + Sync>,
}

impl Default for Web {
    fn default() -> Self {
        Self {
            fetch: Box::new(get_web_fetch_tool_factory()),
        }
    }
}

impl ToolSet for Web {
    fn descs(&self) -> Vec<ToolDesc> {
        vec![get_web_search_tool_desc(), get_web_fetch_tool_desc()]
    }

    fn make(&self, name: &str) -> anyhow::Result<ToolFunc> {
        match name {
            // Made for each agent: it searches with the provider set at the time.
            "web_search" => get_web_search_tool_factory()(),
            "web_fetch" => (self.fetch)(),
            _ => anyhow::bail!("'{name}' is no web tool"),
        }
    }
}
