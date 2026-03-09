mod terminal;
mod web_search;

use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq, Display, EnumString)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum BuiltinToolKind {
    Terminal,
    WebSearchDuckduckgo,
    WebFetch,
}

pub use terminal::create_terminal_tool;
pub use web_search::{create_web_fetch_tool, create_web_search_duckduckgo_tool};
