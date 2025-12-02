mod terminal;
mod web_search;

use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq, Display, EnumString)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
#[cfg_attr(feature = "python", derive(ailoy_macros::PyStringEnum))]
#[cfg_attr(feature = "nodejs", napi_derive::napi(string_enum = "snake_case"))]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(from_wasm_abi, into_wasm_abi))]
pub enum BuiltinToolKind {
    Terminal,
    WebSearchDuckduckgo,
    WebFetch,
}

pub use terminal::*;
pub use web_search::*;
