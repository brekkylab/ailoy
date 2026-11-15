//! [`WebSearchProvider::Exa`](super::WebSearchProvider::Exa): the Exa API.

use anyhow::Context as _;
use async_trait::async_trait;
use reqwest::Client;
use serde_json::{Value, json};

use super::engine::{SearchEngine, SearchError, SearchResult};

pub struct Exa {
    key: String,
}

impl Exa {
    pub fn from_env() -> anyhow::Result<Self> {
        let key = std::env::var("EXA_API_KEY")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .context("EXA_API_KEY is not set")?;
        Ok(Self { key })
    }
}

#[async_trait]
impl SearchEngine for Exa {
    fn name(&self) -> &'static str {
        "exa"
    }

    /// Asks for each result's highlights, up to 300 characters, as its description.
    async fn search(
        &self,
        client: &Client,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let response = client
            .post("https://api.exa.ai/search")
            .header("x-api-key", &self.key)
            .json(&json!({
                "query": query,
                "numResults": max_results,
                "contents": { "highlights": { "maxCharacters": 300 } }
            }))
            .send()
            .await?;
        let status = response.status();
        let body: Value = response.json().await?;
        if !status.is_success() {
            return Err(SearchError::Api(format!("Exa answered {status}: {body}")));
        }
        Ok(body["results"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|r| {
                let highlights = r["highlights"].as_array().into_iter().flatten();
                Some(SearchResult {
                    title: r["title"].as_str().unwrap_or_default().to_owned(),
                    url: r.get("url")?.as_str()?.to_owned(),
                    description: highlights
                        .filter_map(Value::as_str)
                        .collect::<Vec<_>>()
                        .join(" … "),
                    engine: "exa",
                })
            })
            .take(max_results)
            .collect())
    }
}
