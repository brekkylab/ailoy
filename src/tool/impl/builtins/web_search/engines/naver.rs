use async_trait::async_trait;
use reqwest::Client;
use scraper::{Html, Selector};

use crate::tool::r#impl::builtins::web_search::engine::{
    SearchEngine, SearchError, SearchResult, USER_AGENT,
};

pub struct Naver {
    sel_results: Selector,
    sel_link: Selector,
    sel_title: Selector,
    sel_desc: Selector,
}

impl Naver {
    pub fn new() -> Result<Self, SearchError> {
        let parse = |s: &str| {
            Selector::parse(s)
                .map_err(|e| SearchError::Parse(format!("Invalid CSS selector '{}': {:?}", s, e)))
        };
        Ok(Self {
            sel_results: parse("div.fds-web-doc-root")?,
            sel_link: parse(r#"a[nocr="1"]"#)?,
            sel_title: parse(".sds-comps-text-type-headline1")?,
            sel_desc: parse(".sds-comps-text-type-body1")?,
        })
    }

    fn pick_result_url<'a>(&self, block: &scraper::ElementRef<'a>) -> Option<String> {
        for a in block.select(&self.sel_link) {
            let href = a.value().attr("href")?;
            if !href.starts_with("http") {
                continue;
            }
            if href.contains("help.naver.com") || href.contains("policy.naver.com") {
                continue;
            }
            return Some(href.to_string());
        }
        None
    }
}

#[async_trait]
impl SearchEngine for Naver {
    fn name(&self) -> &'static str {
        "Naver"
    }

    async fn search(
        &self,
        client: &Client,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let url = format!(
            "https://search.naver.com/search.naver?where=webkr&query={}",
            urlencoding::encode(query)
        );

        let response = client
            .get(&url)
            .header("User-Agent", USER_AGENT)
            .header(
                "Accept",
                "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            )
            .header("Accept-Language", "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7")
            .send()
            .await?;

        if response.status() == 429 || response.status() == 403 {
            return Err(SearchError::Blocked);
        }

        let html_text = response.error_for_status()?.text().await?;
        let document = Html::parse_document(&html_text);

        let mut results = Vec::new();
        for block in document.select(&self.sel_results) {
            let url = match self.pick_result_url(&block) {
                Some(u) => u,
                None => continue,
            };

            let title = block
                .select(&self.sel_title)
                .next()
                .map(|el| el.text().collect::<String>().trim().to_string())
                .filter(|s| !s.is_empty())
                .or_else(|| {
                    block.select(&self.sel_link).next().map(|a| {
                        a.text()
                            .collect::<String>()
                            .split_whitespace()
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
                })
                .unwrap_or_default();
            if title.is_empty() {
                continue;
            }

            let description = block
                .select(&self.sel_desc)
                .next()
                .map(|el| el.text().collect::<String>().trim().to_string())
                .unwrap_or_default();

            results.push(SearchResult {
                title,
                url,
                description,
                engine: self.name(),
            });

            if results.len() >= max_results {
                break;
            }
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use scraper::Html;

    use super::*;

    fn make_naver_html(items: &[(&str, &str, &str)]) -> String {
        let mut blocks = String::new();
        for (title, url, desc) in items {
            blocks.push_str(&format!(
                r##"<div class="sds-comps-vertical-layout sds-comps-full-layout fds-web-doc-root auVQnhF7QaL2HDJPkxei fds-web-normal-doc-root">
                       <a nocr="1" href="{url}" class="fender-ui_228e3bd1 mJuTmoYILQDUdgQsX1eg" target="_blank">
                         <span class="sds-comps-text sds-comps-text-type-headline1">{title}</span>
                       </a>
                       <span class="sds-comps-text sds-comps-text-type-body1">{desc}</span>
                     </div>"##
            ));
        }
        format!("<html><body>{blocks}</body></html>")
    }

    #[test]
    fn test_naver_parser_extracts_results() {
        let engine = Naver::new().expect("Failed to create Naver engine");
        let html = make_naver_html(&[(
            "GitHub - brekkylab/ailoy",
            "https://github.com/brekkylab/ailoy",
            "A comprehensive library for building intelligent AI agents.",
        )]);
        let document = Html::parse_document(&html);

        let mut results = Vec::new();
        for block in document.select(&engine.sel_results) {
            let url = engine.pick_result_url(&block).unwrap();
            let title = block
                .select(&engine.sel_title)
                .next()
                .map(|el| el.text().collect::<String>().trim().to_string())
                .unwrap();
            let desc = block
                .select(&engine.sel_desc)
                .next()
                .map(|el| el.text().collect::<String>().trim().to_string())
                .unwrap_or_default();
            results.push((title, url, desc));
        }

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "GitHub - brekkylab/ailoy");
        assert_eq!(results[0].1, "https://github.com/brekkylab/ailoy");
        assert_eq!(
            results[0].2,
            "A comprehensive library for building intelligent AI agents."
        );
    }

    #[test]
    fn test_naver_parser_skips_help_chrome_links() {
        let engine = Naver::new().expect("Failed to create Naver engine");
        let html = r##"
            <html><body>
              <div class="fds-web-doc-root">
                <a nocr="1" href="https://help.naver.com/service/5626/contents/24844">도움말</a>
                <a nocr="1" href="https://pypi.org/project/ailoy-py/">
                  <span class="sds-comps-text-type-headline1">ailoy-py - PyPI</span>
                </a>
                <span class="sds-comps-text-type-body1">Python package</span>
              </div>
            </body></html>
        "##;
        let document = Html::parse_document(html);
        let block = document.select(&engine.sel_results).next().unwrap();
        let url = engine.pick_result_url(&block).unwrap();
        assert_eq!(url, "https://pypi.org/project/ailoy-py/");
    }

    #[test]
    fn test_naver_parser_falls_back_to_anchor_text_when_headline_missing() {
        let engine = Naver::new().expect("Failed to create Naver engine");
        let html = r##"
            <html><body>
              <div class="fds-web-doc-root">
                <a nocr="1" href="https://example.com/x">  Anchor Title  </a>
                <span class="sds-comps-text-type-body1">desc</span>
              </div>
            </body></html>
        "##;
        let document = Html::parse_document(html);
        let block = document.select(&engine.sel_results).next().unwrap();
        let url = engine.pick_result_url(&block).unwrap();
        assert_eq!(url, "https://example.com/x");

        let title = block
            .select(&engine.sel_title)
            .next()
            .map(|el| el.text().collect::<String>().trim().to_string())
            .filter(|s| !s.is_empty())
            .or_else(|| {
                block.select(&engine.sel_link).next().map(|a| {
                    a.text()
                        .collect::<String>()
                        .split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" ")
                })
            })
            .unwrap_or_default();
        assert_eq!(title, "Anchor Title");
    }

    #[test]
    fn test_naver_parser_skips_relative_pagination_links() {
        let engine = Naver::new().expect("Failed to create Naver engine");
        let html = r##"
            <html><body>
              <div class="fds-web-doc-root">
                <a nocr="1" href="?page=2&amp;query=ailoy">2</a>
                <a nocr="1" href="https://github.com/brekkylab/ailoy">
                  <span class="sds-comps-text-type-headline1">brekkylab/ailoy</span>
                </a>
              </div>
            </body></html>
        "##;
        let document = Html::parse_document(html);
        let block = document.select(&engine.sel_results).next().unwrap();
        let url = engine.pick_result_url(&block).unwrap();
        assert_eq!(url, "https://github.com/brekkylab/ailoy");
    }

    #[test]
    fn test_naver_parser_returns_empty_when_no_doc_roots() {
        let engine = Naver::new().expect("Failed to create Naver engine");
        let html = r#"<html><body><div id="content">No web results</div></body></html>"#;
        let document = Html::parse_document(html);
        let count = document.select(&engine.sel_results).count();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    #[ignore = "requires network"]
    async fn test_search_returns_results() {
        let engine = Naver::new().expect("Failed to create Naver engine");
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
                    assert_eq!(r.engine, "Naver");
                }
            }
            Err(SearchError::Blocked) => {
                eprintln!("Naver is blocking requests — skipping assertions")
            }
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }
}
