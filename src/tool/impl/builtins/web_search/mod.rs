//! The `web_search` tool, run by the [`WebSearchProvider`] set with
//! [`set_web_search_provider`]: the web's own result pages by default, or a search API.

mod brave;
mod engine;
mod exa;
mod naver;
mod tavily;
mod web;

use std::sync::{Arc, RwLock};

use engine::SearchEngine;

use crate::{
    datatype::Value,
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
    tool_func,
};

/// How the `web_search` tool searches, spelled `"web"`, `"brave"`, `"tavily"`, `"exa"` or
/// `"naver"` in its serde form. For a search of your own, set a provider for the tool named
/// `web_search` with [`set_tool_provider`](crate::tool::set_tool_provider).
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize, strum::Display,
)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum WebSearchProvider {
    /// The result pages of Google, Bing, DuckDuckGo, Startpage, Yahoo, Mojeek and Yandex, one
    /// engine a call in turn, the next taking over when one blocks or finds nothing. Needs no
    /// key.
    #[default]
    Web,
    /// The Brave Search API, with the key in `BRAVE_API_KEY`.
    Brave,
    /// The Tavily API, with the key in `TAVILY_API_KEY`.
    Tavily,
    /// The Exa API, with the key in `EXA_API_KEY`.
    Exa,
    /// Naver's web document search API, with `NAVER_CLIENT_ID` and `NAVER_CLIENT_SECRET`.
    Naver,
}

static PROVIDER: RwLock<WebSearchProvider> = RwLock::new(WebSearchProvider::Web);

/// Search with `provider` in every `web_search` tool built from now on.
pub fn set_web_search_provider(provider: WebSearchProvider) {
    *PROVIDER.write().unwrap() = provider;
}

/// The provider `web_search` tools are built with.
pub fn get_web_search_provider() -> WebSearchProvider {
    *PROVIDER.read().unwrap()
}

pub fn get_web_search_tool_desc() -> ToolDesc {
    ToolDescBuilder::new("web_search")
        .description(
            "Search the web. Returns each result's title, URL, a short description and the \
             engine or API it came from. Use this tool to find current information, facts, \
             documentation, news, or any web-accessible content.",
        )
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 10. Max: 20.",
                    "default": 10
                }
            },
            "required": ["query"]
        }))
        .build()
}

/// A factory for the `web_search` tool, searching with the provider set when it is built. A
/// search API without its key fails the build.
pub fn get_web_search_tool_factory() -> impl Fn(&ToolDesc) -> anyhow::Result<ToolFunc> {
    |_| {
        let searcher: Arc<dyn SearchEngine> = match get_web_search_provider() {
            WebSearchProvider::Web => Arc::new(web::Web::new()?),
            WebSearchProvider::Brave => Arc::new(brave::Brave::from_env()?),
            WebSearchProvider::Tavily => Arc::new(tavily::Tavily::from_env()?),
            WebSearchProvider::Exa => Arc::new(exa::Exa::from_env()?),
            WebSearchProvider::Naver => Arc::new(naver::Naver::from_env()?),
        };
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .user_agent(engine::gen_useragent())
            .build()?;
        Ok(tool_func!(async |args: Value| -> Value
            with [searcher = searcher.clone(), client = client.clone()]
            {
            let query = args.pointer("/query").and_then(|v| v.as_str()).unwrap_or("");
            if query.is_empty() {
                return crate::to_value!({ "results": [], "total": 0 });
            }
            let max_results = args
                .pointer("/max_results")
                .and_then(|v| v.as_unsigned())
                .unwrap_or(10)
                .clamp(1, 20) as usize;

            match searcher.search(&client, query, max_results).await {
                Ok(results) => {
                    let total = results.len() as u64;
                    let results = results.into_iter().map(|r| {
                        crate::to_value!({
                            "title": r.title.as_str(),
                            "url": r.url.as_str(),
                            "description": r.description.as_str(),
                            "source": r.engine
                        })
                    });
                    crate::to_value!({ "results": Value::array(results), "total": total })
                }
                Err(e) => crate::to_value!({ "error": e.to_string() }),
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_serde_names() {
        for (provider, name) in [
            (WebSearchProvider::Web, "web"),
            (WebSearchProvider::Brave, "brave"),
            (WebSearchProvider::Tavily, "tavily"),
            (WebSearchProvider::Exa, "exa"),
            (WebSearchProvider::Naver, "naver"),
        ] {
            assert_eq!(serde_json::to_value(provider).unwrap(), name);
            assert_eq!(provider.to_string(), name);
        }
    }

    /// Each search API is built from the environment, so a missing key fails the build
    /// rather than the first search.
    #[test]
    fn test_an_api_without_its_key_fails_the_build() {
        for (var, build) in [
            ("BRAVE_API_KEY", brave::Brave::from_env().err()),
            ("TAVILY_API_KEY", tavily::Tavily::from_env().err()),
            ("EXA_API_KEY", exa::Exa::from_env().err()),
            ("NAVER_CLIENT_ID", naver::Naver::from_env().err()),
        ] {
            if std::env::var(var).is_err() {
                assert!(build.unwrap().to_string().contains(var));
            }
        }
    }

    /// One query through whichever API has its key in the environment.
    #[tokio::test]
    #[ignore = "requires network and keys"]
    async fn test_each_api_finds_results() {
        let client = reqwest::Client::new();
        let searchers: Vec<Arc<dyn SearchEngine>> = [
            brave::Brave::from_env().ok().map(|s| Arc::new(s) as _),
            tavily::Tavily::from_env().ok().map(|s| Arc::new(s) as _),
            exa::Exa::from_env().ok().map(|s| Arc::new(s) as _),
            naver::Naver::from_env().ok().map(|s| Arc::new(s) as _),
        ]
        .into_iter()
        .flatten()
        .collect();
        for searcher in searchers {
            let results = searcher.search(&client, "rust language", 5).await.unwrap();
            assert!(!results.is_empty(), "{}", searcher.name());
            assert!(results.len() <= 5, "{}", searcher.name());
            assert!(results.iter().all(|r| r.url.starts_with("http")));
        }
    }
}
