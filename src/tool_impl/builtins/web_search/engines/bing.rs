use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use reqwest::Client;
use scraper::{Html, Selector};
use wreq::header as rh;
use wreq_util::Emulation;

use crate::tool_impl::builtins::web_search::engine::{
    SearchEngine, SearchError, SearchResult, SearchResultParser,
};

pub struct Bing {
    parser: SearchResultParser,
    sel_results: Selector,
    sel_title_link: Selector,
    sel_caption: Selector,
}

impl Bing {
    pub fn new() -> Result<Self, SearchError> {
        let parser = SearchResultParser::new(".b_no", ".b_algo", "h2 a", "h2 a", ".b_caption p")?;
        Ok(Self {
            parser,
            sel_results: Selector::parse("ol#b_results li.b_algo")
                .map_err(|e| SearchError::Parse(format!("{:?}", e)))?,
            sel_title_link: Selector::parse("h2 a")
                .map_err(|e| SearchError::Parse(format!("{:?}", e)))?,
            sel_caption: Selector::parse(".b_caption p")
                .map_err(|e| SearchError::Parse(format!("{:?}", e)))?,
        })
    }

    /// Bing sometimes wraps the real URL in a tracking redirect of the form
    ///   https://www.bing.com/ck/a?...&u=a1<base64url-no-pad>&...
    /// Decode the `u` query parameter: strip the "a1" prefix, then
    /// base64url-decode to recover the original URL.
    fn decode_bing_redirect(href: &str) -> String {
        if !href.starts_with("https://www.bing.com/ck/a?") {
            return href.to_string();
        }

        let qs = match href.find('?') {
            Some(pos) => &href[pos + 1..],
            None => return href.to_string(),
        };

        let u_val = qs.split('&').find_map(|pair| {
            let mut kv = pair.splitn(2, '=');
            if kv.next() == Some("u") {
                kv.next()
            } else {
                None
            }
        });

        let u_val = match u_val {
            Some(v) => v,
            None => return href.to_string(),
        };

        if !u_val.starts_with("a1") {
            return href.to_string();
        }
        let encoded = &u_val[2..];

        match URL_SAFE_NO_PAD.decode(encoded) {
            Ok(bytes) => String::from_utf8(bytes).unwrap_or_else(|_| href.to_string()),
            Err(_) => href.to_string(),
        }
    }
}

#[async_trait]
impl SearchEngine for Bing {
    fn name(&self) -> &'static str {
        "Bing"
    }

    async fn search(
        &self,
        _client: &Client,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>, SearchError> {
        // Use wreq with Firefox TLS fingerprint emulation.
        // Bing/Cloudflare uses JA3 TLS fingerprinting to detect bots.
        // Standard Rust TLS backends (native-tls / rustls) have fingerprints that
        // are blocked, while Firefox's TLS ClientHello passes.
        // wreq with Emulation::Firefox135 replicates the Firefox TLS handshake.
        let rq_client = wreq::Client::builder()
            .emulation(Emulation::Firefox135)
            .timeout(std::time::Duration::from_secs(15))
            .build()
            .map_err(|e| SearchError::Parse(e.to_string()))?;

        // A homepage pre-fetch seeds cookies (e.g. _EDGE_S=F=1&SID=...) that
        // cause Bing to switch to JS-rendered results with no SSR b_algo nodes.
        // Without prior cookies, Bing serves server-side-rendered results directly.
        //
        // nfpr=1 disables Bing's automatic query reformulation (spell correction),
        // which would otherwise silently rewrite short/uncommon terms like "ailoy"
        // into a different word and return completely irrelevant results.
        // Locale is left to Bing's geolocation so results match the user's region.
        let url = format!(
            "https://www.bing.com/search?q={}&adlt=off&nfpr=1",
            urlencoding::encode(query)
        );

        let response = rq_client
            .get(&url)
            .header(rh::ACCEPT, "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8")
            .header(rh::ACCEPT_LANGUAGE, "en-US,en;q=0.9")
            .header(rh::ACCEPT_ENCODING, "gzip, deflate")
            .header(rh::CACHE_CONTROL, "no-cache")
            .header("DNT", "1")
            .header(rh::CONNECTION, "keep-alive")
            .send()
            .await
            .map_err(|e| SearchError::Parse(e.to_string()))?;

        if response.status() == 429 {
            return Err(SearchError::Blocked);
        }

        if !response.status().is_success() {
            return Err(SearchError::Blocked);
        }

        let html_text = response
            .text()
            .await
            .map_err(|e| SearchError::Parse(e.to_string()))?;

        if !html_text.contains("b_algo") {
            return Err(SearchError::Blocked);
        }

        let document = Html::parse_document(&html_text);

        if self.parser.has_no_results(&document) {
            return Ok(vec![]);
        }

        let mut results = Vec::new();

        for item in document.select(&self.sel_results) {
            let link = match item.select(&self.sel_title_link).next() {
                Some(a) => a,
                None => continue,
            };
            let href = match link.value().attr("href") {
                Some(h) if !h.is_empty() => h,
                _ => continue,
            };
            let title: String = link.text().collect::<Vec<_>>().join("");
            if title.is_empty() {
                continue;
            }

            // Decode Bing tracking redirects to recover the real destination URL.
            let url = Self::decode_bing_redirect(href);

            // Description: text from .b_caption p, skipping decorative icon spans
            // (<span class="algoSlug_icon">) that Bing injects into snippet text.
            let description = item
                .select(&self.sel_caption)
                .flat_map(|p| {
                    p.children().filter_map(|child| {
                        if let Some(el) = child.value().as_element() {
                            if el.name() == "span" {
                                if el.attr("class").unwrap_or("").contains("algoSlug_icon") {
                                    return None;
                                }
                            }
                        }
                        Some(
                            scraper::ElementRef::wrap(child)
                                .map(|er| er.text().collect::<String>())
                                .unwrap_or_else(|| {
                                    child
                                        .value()
                                        .as_text()
                                        .map(|t| t.to_string())
                                        .unwrap_or_default()
                                }),
                        )
                    })
                })
                .collect::<String>()
                .trim()
                .to_string();

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

    #[test]
    fn test_decode_bing_redirect_passthrough() {
        let url = "https://example.com/page";
        assert_eq!(Bing::decode_bing_redirect(url), url);
    }

    #[test]
    fn test_decode_bing_redirect_decodes_u_param() {
        // "https://example.com" base64url-encoded without padding
        let encoded = "aHR0cHM6Ly9leGFtcGxlLmNvbQ";
        let href = format!("https://www.bing.com/ck/a?foo=bar&u=a1{encoded}&other=val");
        assert_eq!(Bing::decode_bing_redirect(&href), "https://example.com");
    }

    #[test]
    fn test_bing_parser_extracts_results() {
        let engine = Bing::new().expect("Failed to create Bing engine");
        let html = r#"
            <html><body>
              <ol id="b_results">
                <li class="b_algo">
                  <h2><a href="https://example.com">Bing Result</a></h2>
                  <div class="b_caption"><p>Bing desc</p></div>
                </li>
              </ol>
            </body></html>
        "#;
        let document = Html::parse_document(html);
        let results: Vec<_> = document.select(&engine.sel_results).collect();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_bing_parser_handles_no_results() {
        let engine = Bing::new().expect("Failed to create Bing engine");
        let html = r#"<html><body><div class="b_no">No results found</div></body></html>"#;
        let document = Html::parse_document(html);
        assert!(engine.parser.has_no_results(&document));
    }

    #[test]
    fn test_icon_span_excluded_from_description() {
        let engine = Bing::new().expect("Failed to create Bing engine");
        let html = r#"
            <html><body>
              <ol id="b_results">
                <li class="b_algo">
                  <h2><a href="https://example.com">Title</a></h2>
                  <div class="b_caption">
                    <p><span class="algoSlug_icon">ICON</span>Real description text</p>
                  </div>
                </li>
              </ol>
            </body></html>
        "#;
        let document = Html::parse_document(html);
        let item = document.select(&engine.sel_results).next().unwrap();
        let desc: String = item
            .select(&engine.sel_caption)
            .flat_map(|p| {
                p.children().filter_map(|child| {
                    if let Some(el) = child.value().as_element() {
                        if el.name() == "span" {
                            if el.attr("class").unwrap_or("").contains("algoSlug_icon") {
                                return None;
                            }
                        }
                    }
                    Some(
                        scraper::ElementRef::wrap(child)
                            .map(|er| er.text().collect::<String>())
                            .unwrap_or_else(|| {
                                child
                                    .value()
                                    .as_text()
                                    .map(|t| t.to_string())
                                    .unwrap_or_default()
                            }),
                    )
                })
            })
            .collect::<String>();
        assert!(!desc.contains("ICON"), "icon text must be excluded");
        assert!(desc.contains("Real description text"));
    }

    #[tokio::test]
    #[ignore = "requires network"]
    async fn test_search_returns_results() {
        let engine = Bing::new().expect("Failed to create Bing engine");
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
                    assert_eq!(r.engine, "Bing");
                }
            }
            Err(SearchError::Blocked) => {
                eprintln!("Bing is blocking requests — skipping assertions")
            }
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }
}
