mod bing;
mod brave;
mod duckduckgo;
mod google;
mod librex;
mod mojeek;
mod startpage;
mod yahoo;
mod yandex;

pub use bing::Bing;
pub use brave::Brave;
pub use duckduckgo::DuckDuckGo;
pub use google::Google;
pub use librex::LibreX;
pub use mojeek::Mojeek;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
pub use startpage::Startpage;
pub use yahoo::Yahoo;
pub use yandex::Yandex;

use super::engine::SearchEngine;

/// Identifies a specific web search engine available to [`MetaSearcher`](super::aggregator::MetaSearcher).
///
/// Pass a subset to [`MetaSearcher::new`](super::aggregator::MetaSearcher::new) to restrict
/// which engines are used. An empty slice falls back to all engines.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub enum WebSearchEngineKind {
    Bing,
    Brave,
    DuckDuckGo,
    Google,
    LibreX,
    Mojeek,
    Startpage,
    Yahoo,
    Yandex,
}

impl WebSearchEngineKind {
    pub const ALL: &'static [WebSearchEngineKind] = &[
        WebSearchEngineKind::Bing,
        WebSearchEngineKind::Brave,
        WebSearchEngineKind::DuckDuckGo,
        WebSearchEngineKind::Google,
        WebSearchEngineKind::LibreX,
        WebSearchEngineKind::Mojeek,
        WebSearchEngineKind::Startpage,
        WebSearchEngineKind::Yahoo,
        WebSearchEngineKind::Yandex,
    ];

    /// Returns the engine's canonical name string, matching `SearchEngine::name()`.
    pub fn name(&self) -> &'static str {
        match self {
            WebSearchEngineKind::Bing => "Bing",
            WebSearchEngineKind::Brave => "Brave",
            WebSearchEngineKind::DuckDuckGo => "DuckDuckGo",
            WebSearchEngineKind::Google => "Google",
            WebSearchEngineKind::LibreX => "LibreX",
            WebSearchEngineKind::Mojeek => "Mojeek",
            WebSearchEngineKind::Startpage => "Startpage",
            WebSearchEngineKind::Yahoo => "Yahoo",
            WebSearchEngineKind::Yandex => "Yandex",
        }
    }

    /// Constructs a boxed [`SearchEngine`] for this kind.
    pub fn instantiate(&self) -> Box<dyn SearchEngine> {
        match self {
            WebSearchEngineKind::Bing => Box::new(Bing::new().expect("Bing init failed")),
            WebSearchEngineKind::Brave => Box::new(Brave::new().expect("Brave init failed")),
            WebSearchEngineKind::DuckDuckGo => {
                Box::new(DuckDuckGo::new().expect("DuckDuckGo init failed"))
            }
            WebSearchEngineKind::Google => Box::new(Google::new().expect("Google init failed")),
            WebSearchEngineKind::LibreX => Box::new(LibreX::new().expect("LibreX init failed")),
            WebSearchEngineKind::Mojeek => Box::new(Mojeek::new().expect("Mojeek init failed")),
            WebSearchEngineKind::Startpage => {
                Box::new(Startpage::new().expect("Startpage init failed"))
            }
            WebSearchEngineKind::Yahoo => Box::new(Yahoo::new().expect("Yahoo init failed")),
            WebSearchEngineKind::Yandex => Box::new(Yandex::new().expect("Yandex init failed")),
        }
    }
}
