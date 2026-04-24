use async_trait::async_trait;
use reqwest::Client;
use scraper::Html;

use crate::tool_impl::builtins::web_search::engine::{
    SearchEngine, SearchError, SearchResult, SearchResultParser, USER_AGENT,
};

pub struct Yahoo {
    parser: SearchResultParser,
}

impl Yahoo {
    pub fn new() -> Result<Self, SearchError> {
        // Yahoo wraps <h3 class="title"> inside <a>, so "h3 a" finds nothing.
        // The anchor (.compTitle a) is the parent of <h3>, not a child.
        let parser = SearchResultParser::new(
            ".no-results",
            ".algo",
            "h3.title",
            ".compTitle a",
            ".compText p",
        )?;
        Ok(Self { parser })
    }

    /// Extract the actual destination URL from Yahoo's redirect wrapper.
    /// Yahoo wraps URLs as: /RU=<encoded_url>/RK=...
    pub fn extract_redirect_url(href: &str) -> String {
        if let Some(ru_start) = href.find("/RU=") {
            let after_ru = &href[ru_start + 4..];
            let end = after_ru.find('/').unwrap_or(after_ru.len());
            let encoded_url = &after_ru[..end];
            return urlencoding::decode(encoded_url)
                .map(|s| s.into_owned())
                .unwrap_or_else(|_| href.to_string());
        }
        href.to_string()
    }
}

#[async_trait]
impl SearchEngine for Yahoo {
    fn name(&self) -> &'static str {
        "Yahoo"
    }

    async fn search(
        &self,
        client: &Client,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let url = format!(
            "https://search.yahoo.com/search?p={}&ei=UTF-8",
            urlencoding::encode(query)
        );

        let response = client
            .get(&url)
            .header("User-Agent", USER_AGENT)
            .header("Accept", "text/html,application/xhtml+xml")
            .header("Accept-Language", "en-US,en;q=0.9")
            .send()
            .await?;

        if response.status() == 429 {
            return Err(SearchError::Blocked);
        }

        let html_text = response.error_for_status()?.text().await?;
        let document = Html::parse_document(&html_text);

        if self.parser.has_no_results(&document) {
            return Ok(vec![]);
        }

        let mut results = self.parser.extract(&document, self.name(), |el| {
            el.select(&self.parser.result_url)
                .next()
                .and_then(|a| a.value().attr("href"))
                .map(|href| Self::extract_redirect_url(href))
                .filter(|url| !url.is_empty())
        });

        results.truncate(max_results);
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use scraper::Html;

    use super::*;

    #[test]
    fn test_yahoo_extract_redirect_url() {
        assert_eq!(
            Yahoo::extract_redirect_url("/RU=https%3A%2F%2Fexample.com/RK=2"),
            "https://example.com"
        );
    }

    #[test]
    fn test_yahoo_extract_direct_url() {
        assert_eq!(
            Yahoo::extract_redirect_url("https://example.com"),
            "https://example.com"
        );
    }

    #[test]
    fn test_yahoo_parser_extracts_results() {
        let engine = Yahoo::new().expect("Failed to create Yahoo engine");
        // Structure matches current Yahoo HTML: <a> wraps <h3>, not the other way around
        let html = r#"
            <html><body>
              <div class="algo">
                <div class="compTitle">
                  <a href="/RU=https%3A%2F%2Fexample.com/RK=2">
                    <h3 class="title">Yahoo Result</h3>
                  </a>
                </div>
                <div class="compText"><p>Yahoo desc</p></div>
              </div>
            </body></html>
        "#;
        let document = Html::parse_document(html);
        let results = engine.parser.extract(&document, engine.name(), |el| {
            el.select(&engine.parser.result_url)
                .next()
                .and_then(|a| a.value().attr("href"))
                .map(|href| Yahoo::extract_redirect_url(href))
                .filter(|url| !url.is_empty())
        });

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Yahoo Result");
        assert_eq!(results[0].url, "https://example.com");
        assert_eq!(results[0].description, "Yahoo desc");
        assert_eq!(results[0].engine, "Yahoo");
    }

    #[test]
    fn test_yahoo_parser_handles_no_results() {
        let engine = Yahoo::new().expect("Failed to create Yahoo engine");
        let html = r#"<html><body><div class="no-results">No results found</div></body></html>"#;
        let document = Html::parse_document(html);
        assert!(engine.parser.has_no_results(&document));
    }

    #[tokio::test]
    #[ignore = "requires network"]
    async fn test_search_returns_results() {
        let engine = Yahoo::new().expect("Failed to create Yahoo engine");
        let client = Client::new();
        match engine.search(&client, "ailoy", 5).await {
            Ok(results) => {
                assert!(!results.is_empty(), "Expected at least one result");
                for r in &results {
                    println!("{:?}", r);
                    assert!(!r.title.is_empty(), "title must not be empty");
                    assert!(
                        r.url.starts_with("http"),
                        "url must start with http: {}",
                        r.url
                    );
                    assert_eq!(r.engine, "Yahoo");
                }
            }
            Err(SearchError::Blocked) => {
                eprintln!("Yahoo is blocking requests — skipping assertions")
            }
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }
}
