use async_trait::async_trait;
use reqwest::{Client, header::SET_COOKIE};
use scraper::{Html, Selector};
use url::Url;

use crate::tool::r#impl::builtins::web_search::engine::{
    SearchEngine, SearchError, SearchResult, gen_useragent,
};

fn brave_default_cookies() -> &'static str {
    "safesearch=off; useLocation=0; summarizer=0; country=us; ui_lang=en-US"
}

pub struct Brave {
    no_result: Selector,
    results: Selector,
    result_title: Selector,
    result_link: Selector,
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
            // Leading-space variant excludes sibling classes (snippet-url, snippet-title).
            results: parse("div[class^='snippet'], div[class*=' snippet']")?,
            result_title: parse("div[class*='title']")?,
            result_link: parse("a[href]")?,
            // t-primary disambiguates from breadcrumb's site-name-content.
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
        // Pre-fetch seeds session cookies (__cf_bm etc.) for the same-origin
        // navigation that follows.
        let init_resp = client
            .get("https://search.brave.com/")
            .header("User-Agent", gen_useragent())
            .header(
                "Accept",
                "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            )
            .header("Accept-Language", "en-US,en;q=0.9")
            .send()
            .await?;

        let server_cookies: Vec<String> = init_resp
            .headers()
            .get_all(SET_COOKIE)
            .iter()
            .filter_map(|v| v.to_str().ok())
            .map(|s| s.split(';').next().unwrap_or("").trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let cookie_header = if server_cookies.is_empty() {
            brave_default_cookies().to_string()
        } else {
            format!("{}; {}", server_cookies.join("; "), brave_default_cookies())
        };

        // spellcheck=0 prevents silent query rewrites (e.g. "ailoy" → "alloy").
        let url = format!(
            "https://search.brave.com/search?q={}&source=web&spellcheck=0",
            urlencoding::encode(query)
        );

        let response = client
            .get(&url)
            .header("User-Agent", gen_useragent())
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
                let href = snippet
                    .select(&self.result_link)
                    .find_map(|a| a.value().attr("href").map(str::to_owned))?;

                // Require absolute URL — filters out ad anchors with relative paths.
                let parsed = Url::parse(&href).ok()?;
                parsed.host()?;

                let title = snippet
                    .select(&self.result_title)
                    .next()
                    .map(|el| el.text().collect::<String>().trim().to_string())
                    .filter(|s| !s.is_empty())?;

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
                parsed.host()?;
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
    fn test_brave_default_cookies_contain_all_required_keys() {
        let c = brave_default_cookies();
        for required in [
            "safesearch=off",
            "useLocation=0",
            "summarizer=0",
            "country=us",
            "ui_lang=en-US",
        ] {
            assert!(
                c.contains(required),
                "default cookie header missing `{required}`: got `{c}`"
            );
        }
    }

    #[test]
    fn test_brave_default_cookies_use_semicolon_separator() {
        let c = brave_default_cookies();
        let pairs: Vec<&str> = c.split("; ").collect();
        assert_eq!(pairs.len(), 5, "expected 5 cookie pairs, got: {c:?}");
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
