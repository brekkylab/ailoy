//! [`WebSearchProvider::Brave`](super::WebSearchProvider::Brave): the Brave Search API.

use anyhow::Context as _;
use async_trait::async_trait;
use reqwest::Client;
use serde_json::Value;

use super::engine::{SearchEngine, SearchError, SearchResult};

pub struct Brave {
    key: String,
}

impl Brave {
    pub fn from_env() -> anyhow::Result<Self> {
        let key = std::env::var("BRAVE_API_KEY")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .context("BRAVE_API_KEY is not set")?;
        Ok(Self { key })
    }
}

#[async_trait]
impl SearchEngine for Brave {
    fn name(&self) -> &'static str {
        "brave"
    }

    async fn search(
        &self,
        client: &Client,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let response = client
            .get(format!(
                "https://api.search.brave.com/res/v1/web/search?q={}&count={}",
                urlencoding::encode(query),
                max_results.min(20)
            ))
            .header("X-Subscription-Token", &self.key)
            .header("Accept", "application/json")
            .send()
            .await?;
        let status = response.status();
        let body: Value = response.json().await?;
        if !status.is_success() {
            return Err(SearchError::Api(format!("Brave answered {status}: {body}")));
        }
        Ok(body
            .pointer("/web/results")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|r| {
                Some(SearchResult {
                    title: r.get("title")?.as_str()?.to_owned(),
                    url: r.get("url")?.as_str()?.to_owned(),
                    description: r["description"].as_str().unwrap_or_default().to_owned(),
                    engine: "brave",
                })
            })
            .take(max_results)
            .collect())
    }
}
