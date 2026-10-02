use async_trait::async_trait;
use reqwest::Client;
use scraper::Html;

use crate::tool::r#impl::builtins::web_search::engine::{
    SearchEngine, SearchError, SearchResult, SearchResultParser, gen_useragent,
};

pub struct Mojeek {
    parser: SearchResultParser,
}

impl Mojeek {
    pub fn new() -> Result<Self, SearchError> {
        let parser = SearchResultParser::new(
            ".no-results",
            "ul.results-standard li",
            "h2 a",
            "h2 a",
            "p.s",
        )?;
        Ok(Self { parser })
    }
}

#[async_trait]
impl SearchEngine for Mojeek {
    fn name(&self) -> &'static str {
        "Mojeek"
    }

    async fn search(
        &self,
        client: &Client,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let url = format!(
            "https://www.mojeek.com/search?q={}",
            urlencoding::encode(query)
        );

        let response = client
            .get(&url)
            .header("User-Agent", gen_useragent())
            .header("Accept", "text/html,application/xhtml+xml")
            .header("Accept-Language", "en-US,en;q=0.9")
            .send()
            .await?;

        if response.status() == 429 || response.status() == 403 {
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
                .map(String::from)
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
    fn test_mojeek_parser_extracts_results() {
        let engine = Mojeek::new().expect("Failed to create Mojeek engine");
        let html = r#"
            <html><body>
              <ul class="results-standard">
                <li>
                  <h2><a href="https://example.com">Mojeek Result</a></h2>
                  <p class="s">Mojeek desc</p>
                </li>
              </ul>
            </body></html>
        "#;
        let document = Html::parse_document(html);
        let results = engine.parser.extract(&document, engine.name(), |el| {
            el.select(&engine.parser.result_url)
                .next()
                .and_then(|a| a.value().attr("href"))
                .map(String::from)
                .filter(|url| !url.is_empty())
        });

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Mojeek Result");
        assert_eq!(results[0].url, "https://example.com");
        assert_eq!(results[0].description, "Mojeek desc");
        assert_eq!(results[0].engine, "Mojeek");
    }

    #[test]
    fn test_mojeek_parser_handles_no_results() {
        let engine = Mojeek::new().expect("Failed to create Mojeek engine");
        let html = r#"<html><body><div class="no-results">No results found</div></body></html>"#;
        let document = Html::parse_document(html);
        assert!(engine.parser.has_no_results(&document));
    }

    #[tokio::test]
    #[ignore = "requires network"]
    async fn test_search_returns_results() {
        let engine = Mojeek::new().expect("Failed to create Mojeek engine");
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
                    assert_eq!(r.engine, "Mojeek");
                }
            }
            Err(SearchError::Blocked) => {
                eprintln!("Mojeek is blocking requests — skipping assertions")
            }
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }
}
