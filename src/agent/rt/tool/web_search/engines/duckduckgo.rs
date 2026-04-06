use async_trait::async_trait;
use reqwest::Client;
use scraper::{Html, Selector};

use crate::agent::rt::tool::web_search::engine::{
    SearchEngine, SearchError, SearchResult, SearchResultParser, USER_AGENT,
};

pub struct DuckDuckGo {
    parser: SearchResultParser,
    /// Selector for the result title+URL anchor inside each web-result block.
    /// DDG HTML uses `<h2><a class="result__a" href="...">title</a></h2>`.
    title_url_selector: Selector,
    /// Selector for the snippet anchor inside each web-result block.
    snippet_selector: Selector,
    /// Selector to detect DDG's bot-challenge / CAPTCHA page.
    captcha_selector: Selector,
}

impl DuckDuckGo {
    pub fn new() -> Result<Self, SearchError> {
        let parse = |s: &str| {
            Selector::parse(s)
                .map_err(|e| SearchError::Parse(format!("Invalid CSS selector '{}': {:?}", s, e)))
        };

        // DDG HTML response structure (verified April 2026):
        //
        //   <div id="links">
        //     <div class="result results_links results_links_deep web-result">
        //       <div class="links_main links_deep result__body">
        //         <h2 class="result__title">
        //           <a rel="nofollow" class="result__a" href="https://...">Title text</a>
        //         </h2>
        //         <a class="result__snippet" href="https://...">Snippet text</a>
        //       </div>
        //     </div>
        //     ...
        //   </div>
        //
        // Ads have class "result--ad" and are NOT inside #links, so the
        // `#links .web-result` selector naturally excludes them.

        let parser = SearchResultParser::new(
            ".no-results",
            // Select only web-result blocks inside the #links container.
            // Ads appear outside #links (or carry "result--ad") so this selector
            // naturally excludes them.
            "#links .web-result",
            // Placeholder – title extraction is done via title_url_selector below.
            "h2.result__title",
            "h2.result__title",
            "a.result__snippet",
        )?;

        Ok(Self {
            parser,
            title_url_selector: parse("h2.result__title > a.result__a")?,
            snippet_selector: parse("a.result__snippet")?,
            captcha_selector: parse("form#challenge-form")?,
        })
    }

    /// Extract the actual destination URL from DuckDuckGo's redirect URLs.
    /// DDG sometimes wraps URLs as: //duckduckgo.com/l/?uddg=<encoded_url>&...
    fn extract_url(href: &str) -> String {
        if let Some(uddg_start) = href.find("uddg=") {
            let encoded = &href[uddg_start + 5..];
            let end = encoded.find('&').unwrap_or(encoded.len());
            let encoded_url = &encoded[..end];
            return urlencoding::decode(encoded_url)
                .map(|s| s.into_owned())
                .unwrap_or_else(|_| href.to_string());
        }
        href.to_string()
    }

    /// Returns `true` when DDG served its bot-challenge / CAPTCHA page.
    fn is_captcha_page(&self, document: &Html) -> bool {
        document.select(&self.captcha_selector).next().is_some()
    }

    /// Parse a pre-fetched HTML document and return search results.
    /// Exposed for unit tests; in production, `search()` fetches and calls this.
    fn search_document(&self, document: &Html) -> Result<Vec<SearchResult>, SearchError> {
        if self.is_captcha_page(document) {
            return Err(SearchError::Blocked);
        }
        if self.parser.has_no_results(document) {
            return Ok(vec![]);
        }

        let title_url_sel = &self.title_url_selector;
        let snippet_sel = &self.snippet_selector;

        let results: Vec<SearchResult> = document
            .select(&self.parser.results)
            .filter_map(|result_el| {
                let anchor = result_el.select(title_url_sel).next()?;
                let title = anchor.text().collect::<String>().trim().to_string();
                if title.is_empty() {
                    return None;
                }
                let href = anchor.value().attr("href")?;
                let url = Self::extract_url(href);
                if url.is_empty() || !url.starts_with("http") {
                    return None;
                }

                let description = result_el
                    .select(snippet_sel)
                    .next()
                    .map(|el| el.text().collect::<String>().trim().to_string())
                    .unwrap_or_default();

                Some(SearchResult {
                    title,
                    url,
                    description,
                    engine: self.name(),
                })
            })
            .collect();

        Ok(results)
    }
}

#[async_trait]
impl SearchEngine for DuckDuckGo {
    fn name(&self) -> &'static str {
        "DuckDuckGo"
    }

    async fn search(
        &self,
        client: &Client,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>, SearchError> {
        // POST to html.duckduckgo.com/html/ – the no-JS endpoint.
        //
        // Key headers required to avoid bot-detection (202 CAPTCHA challenge):
        //   • Sec-Fetch-{Dest,Mode,Site,User} – mimic a real browser form submit
        //   • Referer set to the DDG HTML page (Referrer-Policy: origin)
        //
        // Sending `kl=wt-wt` (all regions) gives broader results than an empty
        // string.  `b=""` is the first-page marker; omit `df` for no date filter.
        let response = client
            .post("https://html.duckduckgo.com/html/")
            .form(&[("q", query), ("b", ""), ("kl", "wt-wt")])
            .header("User-Agent", USER_AGENT)
            .header(
                "Accept",
                "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            )
            .header("Accept-Language", "en-US,en;q=0.5")
            .header("Content-Type", "application/x-www-form-urlencoded")
            .header("Referer", "https://html.duckduckgo.com/")
            .header("Sec-Fetch-Dest", "document")
            .header("Sec-Fetch-Mode", "navigate")
            .header("Sec-Fetch-Site", "same-origin")
            .header("Sec-Fetch-User", "?1")
            .send()
            .await?;

        let status = response.status();
        // HTTP 429 = explicit rate-limit; HTTP 202 = bot-challenge page
        if status == 429 || status == 202 {
            return Err(SearchError::Blocked);
        }

        let html_text = response.error_for_status()?.text().await?;
        let document = Html::parse_document(&html_text);

        let mut results = self.search_document(&document)?;
        results.truncate(max_results);
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use scraper::Html;

    use super::*;

    #[test]
    fn test_extract_url_with_uddg_redirect() {
        let redirect_url =
            "//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.rust-lang.org%2F&rut=123&h=456";
        let extracted = DuckDuckGo::extract_url(redirect_url);
        assert_eq!(extracted, "https://www.rust-lang.org/");
    }

    #[test]
    fn test_extract_url_with_direct_url() {
        let direct_url = "https://example.com/page";
        let extracted = DuckDuckGo::extract_url(direct_url);
        assert_eq!(extracted, "https://example.com/page");
    }

    /// Build a minimal DDG-style HTML page with the real structure:
    ///
    ///   <div id="links">
    ///     <div class="result results_links results_links_deep web-result">
    ///       <h2 class="result__title">
    ///         <a class="result__a" href="URL">Title</a>
    ///       </h2>
    ///       <a class="result__snippet" href="URL">Snippet</a>
    ///     </div>
    ///     ...
    ///   </div>
    fn make_ddg_html(results: &[(&str, &str, &str)]) -> String {
        let mut items = String::new();
        for (title, url, snippet) in results {
            items.push_str(&format!(
                r#"<div class="result results_links results_links_deep web-result">
                     <h2 class="result__title">
                       <a class="result__a" href="{url}">{title}</a>
                     </h2>
                     <a class="result__snippet" href="{url}">{snippet}</a>
                   </div>"#
            ));
        }
        format!("<html><body><div id=\"links\">{items}</div></body></html>")
    }

    #[test]
    fn test_parser_excludes_ads() {
        // Ads appear OUTSIDE #links (or with result--ad class) so the
        // `#links .web-result` selector naturally skips them.
        let engine = DuckDuckGo::new().expect("Failed to create DuckDuckGo engine");
        // Ad lives outside #links
        let html = format!(
            r#"<html><body>
              <div class="result result--ad">
                <h2 class="result__title">
                  <a class="result__a" href="https://ads.example.com">Sponsored Result</a>
                </h2>
                <a class="result__snippet" href="https://ads.example.com">Ad description</a>
              </div>
              <div id="links">
                <div class="result results_links web-result">
                  <h2 class="result__title">
                    <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Flegit.com%2F">Legitimate Result</a>
                  </h2>
                  <a class="result__snippet" href="https://legit.com/">Real description</a>
                </div>
              </div>
            </body></html>"#
        );
        let document = Html::parse_document(&html);

        let results = engine
            .search_document(&document)
            .expect("parsing should succeed");

        assert_eq!(
            results.len(),
            1,
            "Should have exactly 1 result (ad excluded)"
        );
        assert_eq!(results[0].title, "Legitimate Result");
        assert_eq!(results[0].url, "https://legit.com/");
        assert_eq!(results[0].description, "Real description");
        assert_eq!(results[0].engine, "DuckDuckGo");
    }

    #[test]
    fn test_parser_extracts_all_fields() {
        let engine = DuckDuckGo::new().expect("Failed to create DuckDuckGo engine");
        let html = make_ddg_html(&[(
            "Learning Rust",
            "https://example.com/rust",
            "The Rust programming language homepage with documentation and resources",
        )]);
        let document = Html::parse_document(&html);
        let results = engine
            .search_document(&document)
            .expect("parsing should succeed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Learning Rust");
        assert_eq!(results[0].url, "https://example.com/rust");
        assert_eq!(
            results[0].description,
            "The Rust programming language homepage with documentation and resources"
        );
        assert_eq!(results[0].engine, "DuckDuckGo");
    }

    #[test]
    fn test_parser_handles_no_results() {
        let engine = DuckDuckGo::new().expect("Failed to create DuckDuckGo engine");
        let html = r#"
            <html><body>
              <div class="no-results">
                <p>No results found</p>
              </div>
            </body></html>
        "#;
        let document = Html::parse_document(html);
        assert!(engine.parser.has_no_results(&document));
    }

    #[test]
    fn test_parser_with_multiple_results() {
        let engine = DuckDuckGo::new().expect("Failed to create DuckDuckGo engine");
        let html = make_ddg_html(&[
            ("Result One", "https://example.com/1", "Description one"),
            ("Result Two", "https://example.com/2", "Description two"),
            ("Result Three", "https://example.com/3", "Description three"),
        ]);
        let document = Html::parse_document(&html);
        let results = engine
            .search_document(&document)
            .expect("parsing should succeed");

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].title, "Result One");
        assert_eq!(results[1].title, "Result Two");
        assert_eq!(results[2].title, "Result Three");
    }

    #[test]
    fn test_captcha_detection() {
        let engine = DuckDuckGo::new().expect("Failed to create DuckDuckGo engine");
        let html = r#"<html><body><form id="challenge-form" action="//duckduckgo.com/anomaly.js"></form></body></html>"#;
        let document = Html::parse_document(html);
        assert!(engine.is_captcha_page(&document));
    }

    #[tokio::test]
    #[ignore = "requires network"]
    async fn test_search_returns_results() {
        let engine = DuckDuckGo::new().expect("Failed to create DuckDuckGo engine");
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
                    assert_eq!(r.engine, "DuckDuckGo");
                }
            }
            Err(SearchError::Blocked) => {
                eprintln!("DuckDuckGo is blocking requests — skipping assertions")
            }
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }
}
