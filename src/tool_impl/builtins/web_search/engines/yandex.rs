use async_trait::async_trait;
use reqwest::Client;
use scraper::{Html, Selector};

use crate::tool_impl::builtins::web_search::engine::{
    SearchEngine, SearchError, SearchResult, USER_AGENT,
};

/// Yandex web search via the Yandex Site Search widget endpoint.
///
/// The widget endpoint (`/search/site/`) bypasses the main Yandex search
/// anti-bot protections and returns a lightweight HTML page with a stable
/// BEM-class structure that is straightforward to parse.
///
/// HTML structure (verified April 2026):
///
///   <ul class="b-serp-list">
///     <li class="b-serp-item">
///       <div class="b-serp-item__content">
///         <h3 class="b-serp-item__title">
///           <a class="b-serp-item__title-link" href="https://...">
///             <span>Title text with <b>highlight</b></span>
///           </a>
///         </h3>
///         <div class="b-serp-item__text">Description…</div>
///       </div>
///     </li>
///   </ul>
pub struct Yandex {
    /// Selector for individual search result items.
    results: Selector,
    /// Selector for the title anchor inside a result item.
    result_link: Selector,
    /// Selector for the description text inside a result item.
    result_desc: Selector,
}

impl Yandex {
    pub fn new() -> Result<Self, SearchError> {
        let parse = |s: &str| {
            Selector::parse(s)
                .map_err(|e| SearchError::Parse(format!("Invalid CSS selector '{}': {:?}", s, e)))
        };
        Ok(Self {
            results: parse("li.b-serp-item")?,
            result_link: parse("a.b-serp-item__title-link")?,
            result_desc: parse("div.b-serp-item__text")?,
        })
    }
}

#[async_trait]
impl SearchEngine for Yandex {
    fn name(&self) -> &'static str {
        "Yandex"
    }

    async fn search(
        &self,
        client: &Client,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>, SearchError> {
        // Yandex.com redirects to yandex.ru for search; use the .ru host directly.
        // The `searchid` identifies a public Site Search widget instance.
        let url = format!(
            "https://yandex.ru/search/site/?tmpl_version=releases&text={}&web=1&frame=1&searchid=3131712",
            urlencoding::encode(query)
        );

        let response = client
            .get(&url)
            .header("User-Agent", USER_AGENT)
            .header(
                "Accept",
                "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            )
            .header("Accept-Language", "en-US,en;q=0.5")
            .send()
            .await?;

        // Yandex signals CAPTCHA challenges via a response header.
        if response.headers().get("x-yandex-captcha").is_some() {
            return Err(SearchError::Blocked);
        }

        if response.status() == 429 {
            return Err(SearchError::Blocked);
        }

        let html_text = response.error_for_status()?.text().await?;
        let document = Html::parse_document(&html_text);

        let mut results: Vec<SearchResult> = document
            .select(&self.results)
            .filter_map(|item| {
                let link = item.select(&self.result_link).next()?;

                let url = link.value().attr("href")?.to_string();
                if !url.starts_with("http") {
                    return None;
                }

                let title = link.text().collect::<String>().trim().to_string();
                if title.is_empty() {
                    return None;
                }

                let description = item
                    .select(&self.result_desc)
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

        results.truncate(max_results);
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use scraper::Html;

    use super::*;

    fn make_yandex_html(items: &[(&str, &str, &str)]) -> String {
        let mut list = String::new();
        for (title, url, desc) in items {
            list.push_str(&format!(
                r#"<li class="b-serp-item">
                     <div class="b-serp-item__content">
                       <h3 class="b-serp-item__title">
                         <a class="b-serp-item__title-link" href="{url}">
                           <span>{title}</span>
                         </a>
                       </h3>
                       <div class="b-serp-item__text">{desc}</div>
                     </div>
                   </li>"#
            ));
        }
        format!("<html><body><ul class=\"b-serp-list\">{list}</ul></body></html>")
    }

    #[test]
    fn test_yandex_parser_extracts_results() {
        let engine = Yandex::new().expect("Failed to create Yandex engine");
        let html = make_yandex_html(&[(
            "Introducing Ailoy",
            "https://medium.com/ailoy/introducing-ailoy",
            "Ailoy is a lightweight library for building AI applications.",
        )]);
        let document = Html::parse_document(&html);

        let results: Vec<SearchResult> = document
            .select(&engine.results)
            .filter_map(|item| {
                let link = item.select(&engine.result_link).next()?;
                let url = link.value().attr("href")?.to_string();
                if !url.starts_with("http") {
                    return None;
                }
                let title = link.text().collect::<String>().trim().to_string();
                if title.is_empty() {
                    return None;
                }
                let description = item
                    .select(&engine.result_desc)
                    .next()
                    .map(|el| el.text().collect::<String>().trim().to_string())
                    .unwrap_or_default();
                Some(SearchResult {
                    title,
                    url,
                    description,
                    engine: engine.name(),
                })
            })
            .collect();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Introducing Ailoy");
        assert_eq!(results[0].url, "https://medium.com/ailoy/introducing-ailoy");
        assert_eq!(
            results[0].description,
            "Ailoy is a lightweight library for building AI applications."
        );
        assert_eq!(results[0].engine, "Yandex");
    }

    #[test]
    fn test_yandex_parser_filters_non_http_urls() {
        let engine = Yandex::new().expect("Failed to create Yandex engine");
        let html = r#"
            <html><body><ul class="b-serp-list">
              <li class="b-serp-item">
                <div class="b-serp-item__content">
                  <h3 class="b-serp-item__title">
                    <a class="b-serp-item__title-link" href="/relative/path">
                      <span>Relative Link</span>
                    </a>
                  </h3>
                </div>
              </li>
              <li class="b-serp-item">
                <div class="b-serp-item__content">
                  <h3 class="b-serp-item__title">
                    <a class="b-serp-item__title-link" href="https://example.com">
                      <span>Absolute Link</span>
                    </a>
                  </h3>
                </div>
              </li>
            </ul></body></html>
        "#;
        let document = Html::parse_document(html);
        let results: Vec<SearchResult> = document
            .select(&engine.results)
            .filter_map(|item| {
                let link = item.select(&engine.result_link).next()?;
                let url = link.value().attr("href")?.to_string();
                if !url.starts_with("http") {
                    return None;
                }
                let title = link.text().collect::<String>().trim().to_string();
                if title.is_empty() {
                    return None;
                }
                Some(SearchResult {
                    title,
                    url,
                    description: String::new(),
                    engine: engine.name(),
                })
            })
            .collect();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].url, "https://example.com");
    }

    #[test]
    fn test_yandex_parser_multiple_results() {
        let engine = Yandex::new().expect("Failed to create Yandex engine");
        let html = make_yandex_html(&[
            ("Result One", "https://example.com/1", "Desc one"),
            ("Result Two", "https://example.com/2", "Desc two"),
            ("Result Three", "https://example.com/3", "Desc three"),
        ]);
        let document = Html::parse_document(&html);
        let results: Vec<SearchResult> = document
            .select(&engine.results)
            .filter_map(|item| {
                let link = item.select(&engine.result_link).next()?;
                let url = link.value().attr("href")?.to_string();
                if !url.starts_with("http") {
                    return None;
                }
                let title = link.text().collect::<String>().trim().to_string();
                if title.is_empty() {
                    return None;
                }
                let description = item
                    .select(&engine.result_desc)
                    .next()
                    .map(|el| el.text().collect::<String>().trim().to_string())
                    .unwrap_or_default();
                Some(SearchResult {
                    title,
                    url,
                    description,
                    engine: engine.name(),
                })
            })
            .collect();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].title, "Result One");
        assert_eq!(results[1].title, "Result Two");
        assert_eq!(results[2].title, "Result Three");
    }

    #[tokio::test]
    #[ignore = "requires network"]
    async fn test_search_returns_results() {
        let engine = Yandex::new().expect("Failed to create Yandex engine");
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
                    assert_eq!(r.engine, "Yandex");
                }
            }
            Err(SearchError::Blocked) => {
                eprintln!("Yandex is blocking requests — skipping assertions")
            }
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }
}
