use async_trait::async_trait;
use reqwest::Client;
use scraper::{Html, Selector};

/// A single search result extracted from a search engine.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub description: String,
    pub engine: &'static str,
}

/// Errors that can occur during a search engine request.
#[derive(Debug, thiserror::Error)]
#[allow(dead_code)]
pub enum SearchError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Failed to parse HTML response: {0}")]
    Parse(String),
    #[error("Engine returned no results")]
    NoResults,
    #[error("Engine is blocking requests (rate limit or CAPTCHA)")]
    Blocked,
}

/// Holds pre-compiled CSS selectors for extracting results from a search engine's HTML.
pub struct SearchResultParser {
    pub no_result: Selector,
    pub results: Selector,
    pub result_title: Selector,
    pub result_url: Selector,
    pub result_desc: Selector,
}

impl SearchResultParser {
    pub fn new(
        no_result: &str,
        results: &str,
        result_title: &str,
        result_url: &str,
        result_desc: &str,
    ) -> Result<Self, SearchError> {
        let parse = |s: &str| {
            Selector::parse(s)
                .map_err(|e| SearchError::Parse(format!("Invalid CSS selector '{}': {:?}", s, e)))
        };
        Ok(Self {
            no_result: parse(no_result)?,
            results: parse(results)?,
            result_title: parse(result_title)?,
            result_url: parse(result_url)?,
            result_desc: parse(result_desc)?,
        })
    }

    pub fn has_no_results(&self, document: &Html) -> bool {
        document.select(&self.no_result).next().is_some()
    }

    pub fn extract<F>(
        &self,
        document: &Html,
        engine_name: &'static str,
        url_extractor: F,
    ) -> Vec<SearchResult>
    where
        F: Fn(&scraper::ElementRef) -> Option<String>,
    {
        document
            .select(&self.results)
            .filter_map(|result_el| {
                let title = result_el
                    .select(&self.result_title)
                    .next()
                    .map(|el| el.text().collect::<String>().trim().to_string())?;

                let url = url_extractor(&result_el)?;
                if url.is_empty() {
                    return None;
                }

                let description = result_el
                    .select(&self.result_desc)
                    .next()
                    .map(|el| el.text().collect::<String>().trim().to_string())
                    .unwrap_or_default();

                Some(SearchResult {
                    title,
                    url,
                    description,
                    engine: engine_name,
                })
            })
            .collect()
    }
}

/// The interface all search engines must implement.
#[async_trait]
pub trait SearchEngine: Send + Sync {
    fn name(&self) -> &'static str;
    async fn search(
        &self,
        client: &Client,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>, SearchError>;
}

/// Shared browser-like User-Agent used by all engines.
pub const USER_AGENT: &str =
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0";

#[cfg(test)]
mod tests {
    use super::*;

    fn make_parser() -> SearchResultParser {
        SearchResultParser::new(
            ".no-results",
            ".result",
            ".result-title",
            ".result-url",
            ".result-desc",
        )
        .unwrap()
    }

    #[test]
    fn test_extract_basic_results() {
        let parser = make_parser();
        let html = r#"
            <html><body>
              <div class="result">
                <a class="result-title">First Result</a>
                <a class="result-url" href="https://example.com">https://example.com</a>
                <p class="result-desc">A description</p>
              </div>
              <div class="result">
                <a class="result-title">Second Result</a>
                <a class="result-url" href="https://other.com">https://other.com</a>
                <p class="result-desc">Another description</p>
              </div>
            </body></html>
        "#;
        let document = Html::parse_document(html);
        let results = parser.extract(&document, "test", |el| {
            el.select(&parser.result_url)
                .next()
                .and_then(|a| a.value().attr("href"))
                .map(String::from)
        });

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "First Result");
        assert_eq!(results[0].url, "https://example.com");
        assert_eq!(results[0].description, "A description");
        assert_eq!(results[0].engine, "test");
        assert_eq!(results[1].title, "Second Result");
    }

    #[test]
    fn test_has_no_results() {
        let parser = make_parser();
        let html_with_no_results = r#"<html><body><div class="no-results">No results found</div></body></html>"#;
        let html_with_results = r#"<html><body><div class="result"><a class="result-title">X</a></div></body></html>"#;

        assert!(parser.has_no_results(&Html::parse_document(html_with_no_results)));
        assert!(!parser.has_no_results(&Html::parse_document(html_with_results)));
    }

    #[test]
    fn test_invalid_selector_returns_error() {
        let result = SearchResultParser::new(">>>invalid", ".results", ".title", ".url", ".desc");
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_skips_missing_title() {
        let parser = make_parser();
        let html = r#"
            <html><body>
              <div class="result">
                <a class="result-url" href="https://example.com"></a>
                <p class="result-desc">desc</p>
              </div>
            </body></html>
        "#;
        let document = Html::parse_document(html);
        let results = parser.extract(&document, "test", |el| {
            el.select(&parser.result_url)
                .next()
                .and_then(|a| a.value().attr("href"))
                .map(String::from)
        });
        assert_eq!(results.len(), 0, "Results missing title should be skipped");
    }
}
