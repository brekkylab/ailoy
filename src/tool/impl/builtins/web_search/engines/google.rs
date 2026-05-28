use async_trait::async_trait;
use reqwest::Client;
use scraper::{ElementRef, Html, Selector};

use crate::tool::r#impl::builtins::web_search::engine::{SearchEngine, SearchError, SearchResult};

/// Android Chrome Mobile UAs.
///
/// Google routes requests to a server-rendered HTML code path (rather than the
/// JS-only SPA) when the UA identifies as the "Google Go" native Android app
/// (`com.google.android.apps.searchlite`).  The `NSTNWV` suffix — appended by
/// `random_ua()` — is the token that triggers this path on Google's backend.
static MOBILE_UAS: &[&str] = &[
    "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.8459.1387 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.4665.1706 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.8434.1860 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.6976.1608 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.5155.1496 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3280.1347 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 8.0; Pixel 2 Build/OPD3.170816.012) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.3126.1891 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 8.0; Pixel 2 Build/OPD3.170816.012) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.3462.1505 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 8.0; Pixel 2 Build/OPD3.170816.012) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.3958.1590 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 8.0; Pixel 2 Build/OPD3.170816.012) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2978.1556 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 8.0; Pixel 2 Build/OPD3.170816.012) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.5114.1832 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 8.0; Pixel 2 Build/OPD3.170816.012) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.9357.1059 Mobile Safari/537.36",
];

/// Body-marker captcha detection (status 200 + challenge HTML).
fn is_captcha_page(html: &str) -> bool {
    const BLOCK_MARKERS: &[&str] = &[
        "Our systems have detected unusual traffic",
        "/sorry/index",
        "id=\"captcha\"",
        "CaptchaRedirect",
    ];
    BLOCK_MARKERS.iter().any(|m| html.contains(m))
}

/// URL-based block detection (302 to sorry.google.com or /sorry/* path).
fn is_sorry_url(host: Option<&str>, path: &str) -> bool {
    host == Some("sorry.google.com") || path.starts_with("/sorry/")
}

/// Pick a random base UA and append the `NSTNWV` suffix.
///
/// `NSTNWV` is the token that identifies the "Google Go" native Android app
/// (com.google.android.apps.searchlite) to Google's backend.  Without it,
/// Google serves a JS-only SPA shell; with it, Google serves server-rendered
/// HTML with parseable `data-ved` anchors and `div[style]` title elements.
///
/// Uses subsecond nanosecond timestamp as a lightweight entropy source —
/// sufficient for UA rotation; no cryptographic quality needed.
fn random_ua() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as usize;
    let base = MOBILE_UAS[nanos % MOBILE_UAS.len()];
    format!("{base} NSTNWV")
}

pub struct Google {
    /// `a[data-ved]` — further filtered in code to skip anchors that have a class attribute.
    sel_results: Selector,
    /// Title text lives in the first `div` with a `style` attribute inside the anchor.
    sel_title: Selector,
    /// Description lives 2 DOM levels above the anchor, in a div carrying these three classes.
    sel_desc: Selector,
}

impl Google {
    pub fn new() -> Result<Self, SearchError> {
        Ok(Self {
            sel_results: Selector::parse("a[data-ved]")
                .map_err(|e| SearchError::Parse(format!("{:?}", e)))?,
            sel_title: Selector::parse("div[style]")
                .map_err(|e| SearchError::Parse(format!("{:?}", e)))?,
            sel_desc: Selector::parse("div.ilUpNd.H66NU.aSRlid")
                .map_err(|e| SearchError::Parse(format!("{:?}", e)))?,
        })
    }

    /// Strip Google's `/url?q=<percent-encoded-url>&sa=U&...` redirect wrapper.
    ///
    /// Direct `http(s)://` hrefs are returned as-is.  Everything else (relative
    /// paths, fragment-only anchors, etc.) returns `None` and will be skipped.
    fn clean_url(href: &str) -> Option<String> {
        if let Some(rest) = href.strip_prefix("/url?q=") {
            // Take the part after `/url?q=` and before the first `&sa=U` token.
            let encoded = rest.split("&sa=U").next().unwrap_or("");
            let decoded = urlencoding::decode(encoded).ok()?.into_owned();
            if decoded.starts_with("http") {
                Some(decoded)
            } else {
                None
            }
        } else if href.starts_with("http") {
            Some(href.to_string())
        } else {
            None
        }
    }
}

#[async_trait]
impl SearchEngine for Google {
    fn name(&self) -> &'static str {
        "Google"
    }

    async fn search(
        &self,
        client: &Client,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>, SearchError> {
        // Google does not use TLS fingerprinting for bot detection; plain reqwest works.
        // The mobile Android Chrome UA is what triggers Google's SSR code path —
        // desktop or unknown UAs receive a JavaScript-only shell that cannot be parsed.

        // hl=en-US   : interface language
        // lr=lang_en : restrict results to English documents
        // ie/oe      : character encoding (utf8)
        // filter=0   : disable duplicate result filtering
        let url = format!(
            "https://www.google.com/search?q={}&hl=en-US&lr=lang_en&ie=utf8&oe=utf8&filter=0",
            urlencoding::encode(query)
        );

        let response = client
            .get(&url)
            .header("User-Agent", random_ua())
            .header("Accept", "*/*")
            .header("Accept-Language", "en-US,en;q=0.9")
            .header("Cache-Control", "no-cache")
            // CONSENT=YES+ bypasses Google's cookie-consent interstitial.
            .header("Cookie", "CONSENT=YES+")
            .send()
            .await?;

        // Explicit rate-limit signal.
        if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            log::warn!("Google blocked: HTTP 429");
            return Err(SearchError::Blocked);
        }

        let final_url = response.url().clone();
        if is_sorry_url(final_url.host_str(), final_url.path()) {
            log::warn!("Google blocked: redirected to {}", final_url);
            return Err(SearchError::Blocked);
        }

        // Other non-success statuses are not necessarily a block (transient 5xx, etc.).
        // Surface them as SearchError::Http so callers can distinguish from Blocked.
        let response = response.error_for_status()?;

        let html_text = response.text().await?;

        // Soft block: status 200 but body is a CAPTCHA challenge page.
        if is_captcha_page(&html_text) {
            log::warn!("Google blocked: body matched CAPTCHA markers");
            return Err(SearchError::Blocked);
        }

        let document = Html::parse_document(&html_text);
        let mut results = Vec::new();

        for a_el in document.select(&self.sel_results) {
            // XPath equivalent: //a[@data-ved and not(@class)]
            // Anchors that carry a class attribute are navigation links, image tiles, etc.
            if a_el.value().attr("class").is_some() {
                continue;
            }

            let href = match a_el.value().attr("href") {
                Some(h) if !h.is_empty() => h,
                _ => continue,
            };
            let url = match Self::clean_url(href) {
                Some(u) => u,
                None => continue,
            };

            // Title: first div[style] descendant of the anchor.
            let title: String = match a_el.select(&self.sel_title).next() {
                Some(el) => el.text().collect::<String>().trim().to_string(),
                None => continue,
            };
            if title.is_empty() {
                continue;
            }

            // Description: XPath `../..//div[contains(@class, "ilUpNd H66NU aSRlid")]`
            // Walk up 2 DOM levels from the anchor, then search descendants.
            let description = a_el
                .parent()
                .and_then(|p| p.parent())
                .and_then(ElementRef::wrap)
                .and_then(|gp| {
                    gp.select(&self.sel_desc)
                        .next()
                        .map(|el| el.text().collect::<String>().trim().to_string())
                })
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
    use reqwest::Client;
    use scraper::Html;

    use super::*;

    #[test]
    fn test_clean_url_strips_google_redirect() {
        let href = "/url?q=https%3A%2F%2Fwww.rust-lang.org%2F&sa=U&ved=xxx";
        assert_eq!(
            Google::clean_url(href),
            Some("https://www.rust-lang.org/".to_string())
        );
    }

    #[test]
    fn test_clean_url_passthrough_for_direct_http() {
        let href = "https://example.com/page";
        assert_eq!(
            Google::clean_url(href),
            Some("https://example.com/page".to_string())
        );
    }

    #[test]
    fn test_clean_url_rejects_relative_and_anchor_paths() {
        assert_eq!(Google::clean_url("/search?q=foo"), None);
        assert_eq!(Google::clean_url("#top"), None);
        assert_eq!(Google::clean_url(""), None);
    }

    #[test]
    fn test_google_parser_extracts_title_and_url() {
        let engine = Google::new().expect("Failed to create Google engine");
        // HTML mirrors Google's structure: grandparent wraps parent+anchor and the description.
        let html = r#"
            <html><body>
              <div>
                <div>
                  <a data-ved="yyy" href="/url?q=https%3A%2F%2Fwww.rust-lang.org&sa=U&ved=zzz">
                    <div style="font-size:20px">Rust Programming Language</div>
                  </a>
                </div>
                <div class="ilUpNd H66NU aSRlid">A language empowering everyone.</div>
              </div>
            </body></html>
        "#;
        let document = Html::parse_document(html);
        let a_el = document
            .select(&engine.sel_results)
            .find(|a| a.value().attr("class").is_none())
            .expect("no classless a[data-ved] found");

        let href = a_el.value().attr("href").unwrap();
        assert_eq!(
            Google::clean_url(href),
            Some("https://www.rust-lang.org".to_string())
        );

        let title: String = a_el
            .select(&engine.sel_title)
            .next()
            .unwrap()
            .text()
            .collect::<String>()
            .trim()
            .to_string();
        assert_eq!(title, "Rust Programming Language");
    }

    #[test]
    fn test_google_parser_skips_anchors_with_class() {
        let engine = Google::new().expect("Failed to create Google engine");
        let html = r#"
            <html><body>
              <a data-ved="1" class="nav-link" href="https://example.com">
                <div style="">Should be skipped</div>
              </a>
              <a data-ved="2" href="https://example.com">
                <div style="">Should be kept</div>
              </a>
            </body></html>
        "#;
        let document = Html::parse_document(html);
        let classless: Vec<_> = document
            .select(&engine.sel_results)
            .filter(|a| a.value().attr("class").is_none())
            .collect();
        assert_eq!(classless.len(), 1);
        let title: String = classless[0]
            .select(&engine.sel_title)
            .next()
            .unwrap()
            .text()
            .collect();
        assert_eq!(title.trim(), "Should be kept");
    }

    #[test]
    fn test_description_extracted_from_grandparent() {
        let engine = Google::new().expect("Failed to create Google engine");
        let html = r#"
            <html><body>
              <div>
                <div>
                  <a data-ved="1" href="https://example.com">
                    <div style="">Title</div>
                  </a>
                </div>
                <div class="ilUpNd H66NU aSRlid">Description text here</div>
              </div>
            </body></html>
        "#;
        let document = Html::parse_document(html);
        let a_el = document.select(&engine.sel_results).next().unwrap();
        let desc = a_el
            .parent()
            .and_then(|p| p.parent())
            .and_then(ElementRef::wrap)
            .and_then(|gp| {
                gp.select(&engine.sel_desc)
                    .next()
                    .map(|el| el.text().collect::<String>().trim().to_string())
            });
        assert_eq!(desc, Some("Description text here".to_string()));
    }

    #[test]
    fn test_is_sorry_url_matches_host() {
        assert!(is_sorry_url(Some("sorry.google.com"), "/index"));
    }

    #[test]
    fn test_is_sorry_url_matches_path() {
        assert!(is_sorry_url(Some("www.google.com"), "/sorry/index"));
    }

    #[test]
    fn test_is_sorry_url_rejects_normal_path() {
        assert!(!is_sorry_url(Some("www.google.com"), "/search"));
    }

    #[test]
    fn test_is_sorry_url_handles_missing_host() {
        assert!(!is_sorry_url(None, "/search"));
        assert!(is_sorry_url(None, "/sorry/index"));
    }

    #[test]
    fn test_is_captcha_page_detects_visible_english_copy() {
        let body = r#"<html><body>
            Our systems have detected unusual traffic from your computer network.
        </body></html>"#;
        assert!(is_captcha_page(body));
    }

    #[test]
    fn test_is_captcha_page_detects_sorry_index_path() {
        let body = r#"<form action="/sorry/index">...</form>"#;
        assert!(is_captcha_page(body));
    }

    #[test]
    fn test_is_captcha_page_detects_captcha_id_attr() {
        let body = r#"<input id="captcha" name="captcha" />"#;
        assert!(is_captcha_page(body));
    }

    #[test]
    fn test_is_captcha_page_detects_captcha_redirect_marker() {
        let body = "window.CaptchaRedirect = function() {};";
        assert!(is_captcha_page(body));
    }

    #[test]
    fn test_is_captcha_page_rejects_normal_results() {
        // Realistic-shaped HTML that happens to contain the word "sorry" but is
        // NOT a block page — make sure we do not false-positive on result text.
        let body = r#"<html><body>
            <a data-ved="xxx" href="/url?q=https%3A%2F%2Fexample.com">
              <div style="">Example — sorry, no results for legacy paths</div>
            </a>
            <div class="ilUpNd H66NU aSRlid">Product info</div>
        </body></html>"#;
        assert!(!is_captcha_page(body));
    }

    #[tokio::test]
    #[ignore = "requires network"]
    async fn test_search_returns_results() {
        let engine = Google::new().expect("Failed to create Google engine");
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
                    assert_eq!(r.engine, "Google");
                }
            }
            Err(SearchError::Blocked) => {
                eprintln!("Google is blocking requests — skipping assertions")
            }
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }
}
