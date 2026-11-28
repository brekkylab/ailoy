//! [`WebSearchProvider::Tavily`](super::WebSearchProvider::Tavily): the Tavily API.

use anyhow::Context as _;
use async_trait::async_trait;
use reqwest::Client;
use serde_json::{Value, json};

use super::engine::{SearchEngine, SearchError, SearchResult};

pub struct Tavily {
    key: String,
}

impl Tavily {
    pub fn from_env() -> anyhow::Result<Self> {
        let key = std::env::var("TAVILY_API_KEY")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .context("TAVILY_API_KEY is not set")?;
        Ok(Self { key })
    }
}

#[async_trait]
impl SearchEngine for Tavily {
    fn name(&self) -> &'static str {
        "tavily"
    }

    async fn search(
        &self,
        client: &Client,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let response = client
            .post("https://api.tavily.com/search")
            .bearer_auth(&self.key)
            .json(&json!({ "query": query, "max_results": max_results.min(20) }))
            .send()
            .await?;
        let status = response.status();
        let body: Value = response.json().await?;
        if !status.is_success() {
            return Err(SearchError::Api(format!(
                "Tavily answered {status}: {body}"
            )));
        }
        Ok(body["results"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|r| {
                Some(SearchResult {
                    title: r.get("title")?.as_str()?.to_owned(),
                    url: r.get("url")?.as_str()?.to_owned(),
                    description: r["content"].as_str().unwrap_or_default().to_owned(),
                    engine: "tavily",
                })
            })
            .take(max_results)
            .collect())
    }
}
