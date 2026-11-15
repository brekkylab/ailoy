//! [`WebSearchProvider::Naver`](super::WebSearchProvider::Naver): Naver's web document search
//! API, with an application's id and secret from Naver Developers.

use anyhow::Context as _;
use async_trait::async_trait;
use reqwest::Client;
use serde_json::Value;

use super::engine::{SearchEngine, SearchError, SearchResult};

pub struct Naver {
    id: String,
    secret: String,
}

impl Naver {
    pub fn from_env() -> anyhow::Result<Self> {
        let var = |name: &str| {
            std::env::var(name)
                .ok()
                .filter(|v| !v.trim().is_empty())
                .with_context(|| format!("{name} is not set"))
        };
        Ok(Self {
            id: var("NAVER_CLIENT_ID")?,
            secret: var("NAVER_CLIENT_SECRET")?,
        })
    }
}

#[async_trait]
impl SearchEngine for Naver {
    fn name(&self) -> &'static str {
        "naver"
    }

    /// The API marks the query's words with `<b>` in titles and descriptions; they are taken
    /// out.
    async fn search(
        &self,
        client: &Client,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let response = client
            .get(format!(
                "https://openapi.naver.com/v1/search/webkr.json?query={}&display={max_results}",
                urlencoding::encode(query)
            ))
            .header("X-Naver-Client-Id", &self.id)
            .header("X-Naver-Client-Secret", &self.secret)
            .send()
            .await?;
        let status = response.status();
        let body: Value = response.json().await?;
        if !status.is_success() {
            return Err(SearchError::Api(format!("Naver answered {status}: {body}")));
        }
        let plain = |text: &str| text.replace("<b>", "").replace("</b>", "");
        Ok(body["items"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|r| {
                Some(SearchResult {
                    title: plain(r.get("title")?.as_str()?),
                    url: r.get("link")?.as_str()?.to_owned(),
                    description: plain(r["description"].as_str().unwrap_or_default()),
                    engine: "naver",
                })
            })
            .take(max_results)
            .collect())
    }
}
