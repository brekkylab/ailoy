use async_trait::async_trait;
use reqwest::{Client, header::SET_COOKIE};
use scraper::{Html, Selector};
use url::Url;

use crate::agent::rt::tool::web_search::engine::{
    SearchEngine, SearchError, SearchResult, USER_AGENT,
};

pub struct Brave {
    /// Selector for a "no results" indicator element.
    no_result: Selector,
    /// Selector for individual result snippet containers.
    results: Selector,
    /// Selector for the title element inside a snippet.
    result_title: Selector,
    /// Selector for any anchor tag with an `href` inside a snippet.
    result_link: Selector,
    /// Selector for the description/content element inside a snippet.
    result_desc: Selector,
}

impl Brave {
    pub fn new() -> Result<Self, SearchError> {
        let parse = |s: &str| {
            Selector::parse(s)
                .map_err(|e| SearchError::Parse(format!("Invalid CSS selector '{}': {:?}", s, e)))
        };
        Ok(Self {
            no_result: parse(".no-results")?,
            // Brave renders each web result in a <div class="snippet ..."> container.
            // The leading-space variant (`div[class*=' snippet']`) avoids false-positives
            // on sibling classes like "snippet-url" or "snippet-title".
            results: parse("div[class^='snippet'], div[class*=' snippet']")?,
            result_title: parse("div[class*='title']")?,
            // The actual destination URL lives in the first <a href> of the snippet.
            result_link: parse("a[href]")?,
            // Brave wraps descriptions in a div whose class contains both
            // "content" and "t-primary".  We match on "t-primary" to avoid
            // accidentally selecting the breadcrumb (site-name-content) or
            // other "content"-bearing divs.
            result_desc: parse("div[class*='t-primary']")?,
        })
    }
}

#[async_trait]
impl SearchEngine for Brave {
    fn name(&self) -> &'static str {
        "Brave"
    }

    async fn search(
        &self,
        client: &Client,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>, SearchError> {
        // Pre-fetch the search homepage to acquire session cookies (__cf_bm, etc.)
        // and make the subsequent search request look like an organic browser
        // navigation (same-origin Sec-Fetch headers).
        let init_resp = client
            .get("https://search.brave.com/")
            .header("User-Agent", USER_AGENT)
            .header(
                "Accept",
                "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            )
            .header("Accept-Language", "en-US,en;q=0.9")
            .send()
            .await?;

        let cookies: Vec<String> = init_resp
            .headers()
            .get_all(SET_COOKIE)
            .iter()
            .filter_map(|v| v.to_str().ok())
            .map(|s| s.split(';').next().unwrap_or("").trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let cookie_header = cookies.join("; ");

        // Disable spell-check so Brave doesn't silently rewrite the query
        // (e.g. "ailoy" → "alloy").
        let url = format!(
            "https://search.brave.com/search?q={}&source=web&spellcheck=0",
            urlencoding::encode(query)
        );

        let response = client
            .get(&url)
            .header("User-Agent", USER_AGENT)
            .header(
                "Accept",
                "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            )
            .header("Accept-Language", "en-US,en;q=0.9")
            .header("Referer", "https://search.brave.com/")
            .header("Sec-Fetch-Dest", "document")
            .header("Sec-Fetch-Mode", "navigate")
            .header("Sec-Fetch-Site", "same-origin")
            .header("Sec-Fetch-User", "?1")
            .header("Cookie", &cookie_header)
            .send()
            .await?;

        if response.status() == 429 {
            return Err(SearchError::Blocked);
        }

        let html_text = response.error_for_status()?.text().await?;
        let document = Html::parse_document(&html_text);

        if document.select(&self.no_result).next().is_some() {
            return Ok(vec![]);
        }

        let mut results: Vec<SearchResult> = document
            .select(&self.results)
            .filter_map(|snippet| {
                // Get URL from the first anchor's `href` attribute (not text content).
                let href = snippet
                    .select(&self.result_link)
                    .find_map(|a| a.value().attr("href").map(str::to_owned))?;

                // Only keep results with a proper absolute URL (filters out ads
                // that have partial/relative paths).
                let parsed = Url::parse(&href).ok()?;
                if parsed.host().is_none() {
                    return None;
                }

                // Title: text of the first element whose class contains "title".
                let title = snippet
                    .select(&self.result_title)
                    .next()
                    .map(|el| el.text().collect::<String>().trim().to_string())
                    .filter(|s| !s.is_empty())?;

                // Description: try a few common content selectors.
                let description = snippet
                    .select(&self.result_desc)
                    .next()
                    .map(|el| el.text().collect::<String>().trim().to_string())
                    .unwrap_or_default();

                Some(SearchResult {
                    title,
                    url: href,
                    description,
                    engine: self.name(),
                })
            })
            .collect();

        results.truncate(max_results);
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use scraper::Html;

    use super::*;

    #[test]
    fn test_brave_parser_extracts_results() {
        let engine = Brave::new().expect("Failed to create Brave engine");
        // Simulate the actual Brave HTML structure (Svelte-based, 2025+).
        // Key observations from live page inspection:
        //   - Snippet container:  class="snippet  svelte-jmfu5f"
        //   - Title:              class="title search-snippet-title line-clamp-1 svelte-14r20fy"
        //   - Description/content: class="content desktop-default-regular t-primary line-clamp-dynamic svelte-1cwdgg3"
        //   - URL: from <a href="..."> attribute (not text of <cite class="snippet-url">)
        let html = r#"
            <html><body>
              <div id="results">
                <div class="snippet  svelte-jmfu5f">
                  <a href="https://example.com/page" class="svelte-abc l1">
                    <div class="title search-snippet-title line-clamp-1 svelte-14r20fy">Brave Result</div>
                  </a>
                  <cite class="snippet-url desktop-small-regular t-tertiary svelte-on1hvy">example.com <span>  › page</span></cite>
                  <div class="generic-snippet svelte-1cwdgg3">
                    <div class="content desktop-default-regular t-primary line-clamp-dynamic svelte-1cwdgg3">A description of the result.</div>
                  </div>
                </div>
              </div>
            </body></html>
        "#;
        let document = Html::parse_document(html);

        let results: Vec<SearchResult> = document
            .select(&engine.results)
            .filter_map(|snippet| {
                let href = snippet
                    .select(&engine.result_link)
                    .find_map(|a| a.value().attr("href").map(str::to_owned))?;
                let parsed = url::Url::parse(&href).ok()?;
                if parsed.host().is_none() {
                    return None;
                }
                let title = snippet
                    .select(&engine.result_title)
                    .next()
                    .map(|el| el.text().collect::<String>().trim().to_string())
                    .filter(|s| !s.is_empty())?;
                let description = snippet
                    .select(&engine.result_desc)
                    .next()
                    .map(|el| el.text().collect::<String>().trim().to_string())
                    .unwrap_or_default();
                Some(SearchResult {
                    title,
                    url: href,
                    description,
                    engine: engine.name(),
                })
            })
            .collect();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Brave Result");
        assert_eq!(results[0].url, "https://example.com/page");
        assert_eq!(results[0].description, "A description of the result.");
        assert_eq!(results[0].engine, "Brave");
    }

    #[test]
    fn test_brave_parser_handles_no_results() {
        let engine = Brave::new().expect("Failed to create Brave engine");
        let html = r#"<html><body><div class="no-results">No results found</div></body></html>"#;
        let document = Html::parse_document(html);
        assert!(document.select(&engine.no_result).next().is_some());
    }

    #[tokio::test]
    #[ignore = "requires network"]
    async fn test_search_returns_results() {
        let engine = Brave::new().expect("Failed to create Brave engine");
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
                    assert_eq!(r.engine, "Brave");
                }
            }
            Err(SearchError::Blocked) => {
                eprintln!("Brave is blocking requests — skipping assertions")
            }
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }
}
