use std::time::{Duration, Instant};

use async_trait::async_trait;
use parking_lot::Mutex;
use reqwest::Client;
use scraper::{Html, Selector};

use crate::tool::r#impl::builtins::web_search::engine::{
    SearchEngine, SearchError, SearchResult, gen_useragent,
};

const SC_TOKEN_TTL: Duration = Duration::from_secs(3600);

/// Search requires an `sc` token from the homepage form, then a POST to
/// `/sp/search`, then a follow-up POST through the interstitial JS gate.
pub struct Startpage {
    sc_cache: Mutex<Option<(String, Instant)>>,
}

impl Startpage {
    pub fn new() -> Result<Self, SearchError> {
        Ok(Self {
            sc_cache: Mutex::new(None),
        })
    }

    fn cached_sc(&self) -> Option<String> {
        let guard = self.sc_cache.lock();
        match &*guard {
            Some((tok, ts)) if ts.elapsed() < SC_TOKEN_TTL => Some(tok.clone()),
            _ => None,
        }
    }

    fn store_sc(&self, token: String) {
        *self.sc_cache.lock() = Some((token, Instant::now()));
    }

    /// Pull the hidden `<input name="sc" value="…">` from the homepage form.
    fn extract_sc_from_homepage(html: &str) -> Option<String> {
        let document = Html::parse_document(html);
        let sel = Selector::parse(r#"input[name="sc"]"#).ok()?;
        document
            .select(&sel)
            .next()?
            .value()
            .attr("value")
            .map(str::to_owned)
            .filter(|s| !s.is_empty())
    }

    async fn fetch_sc_token(client: &Client) -> Result<String, SearchError> {
        let response = client
            .get("https://www.startpage.com/")
            .header("User-Agent", gen_useragent())
            .header("Accept", "text/html,application/xhtml+xml")
            .header("Accept-Language", "en-US,en;q=0.9")
            .send()
            .await?;

        if response.status() == 429 {
            return Err(SearchError::Blocked);
        }

        let html = response.error_for_status()?.text().await?;
        Self::extract_sc_from_homepage(&html)
            .ok_or_else(|| SearchError::Parse("sc token not found on homepage".to_string()))
    }

    async fn ensure_sc_token(&self, client: &Client) -> Result<String, SearchError> {
        if let Some(tok) = self.cached_sc() {
            return Ok(tok);
        }
        let tok = Self::fetch_sc_token(client).await?;
        self.store_sc(tok.clone());
        Ok(tok)
    }

    /// Parse the JS literal `var data = {…}` from the interstitial page
    /// into form fields for the follow-up POST (carries `sgt`, etc.).
    fn extract_interstitial_form_data(html: &str) -> Option<Vec<(String, String)>> {
        let start_marker = "var data = {";
        let start = html.find(start_marker)?;
        let after = &html[start + start_marker.len() - 1..]; // include the `{`
        let bytes = after.as_bytes();
        let mut depth: usize = 0;
        let mut end = 0;
        for (i, &b) in bytes.iter().enumerate() {
            match b {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        if end == 0 {
            return None;
        }
        let obj = &after[..end];
        let v: serde_json::Value = serde_json::from_str(obj).ok()?;
        let map = v.as_object()?;
        let mut out = Vec::with_capacity(map.len());
        for (k, val) in map {
            if let Some(s) = val.as_str() {
                out.push((k.clone(), s.to_string()));
            }
        }
        if out.is_empty() { None } else { Some(out) }
    }

    fn is_interstitial(html: &str) -> bool {
        html.contains("js-interstitial-spinner") || html.contains("var data = {")
    }

    fn strip_html(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        let mut in_tag = false;
        for c in s.chars() {
            match c {
                '<' => in_tag = true,
                '>' => in_tag = false,
                _ if !in_tag => out.push(c),
                _ => {}
            }
        }
        out
    }

    /// Carve out the `{…}` after `React.createElement(UIStartpage.AppSerpWeb,`.
    fn extract_props_json(html: &str) -> Option<&str> {
        let marker = "React.createElement(UIStartpage.AppSerpWeb,";
        let marker_pos = html.find(marker)?;
        let after_marker = &html[marker_pos + marker.len()..];
        let brace_offset = after_marker.find('{')?;
        let json_start = marker_pos + marker.len() + brace_offset;

        let mut depth: usize = 0;
        let bytes = html.as_bytes();
        for i in json_start..html.len() {
            match bytes[i] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(&html[json_start..=i]);
                    }
                }
                _ => {}
            }
        }
        None
    }
}

#[async_trait]
impl SearchEngine for Startpage {
    fn name(&self) -> &'static str {
        "Startpage"
    }

    async fn search(
        &self,
        client: &Client,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let sc = self.ensure_sc_token(client).await?;

        let initial_form = [
            ("query", query),
            ("cat", "web"),
            ("t", "device"),
            ("sc", sc.as_str()),
            ("abd", "1"),
            ("abe", "1"),
            ("qsr", "all"),
        ];

        let response = client
            .post("https://www.startpage.com/sp/search")
            .form(&initial_form)
            .header("User-Agent", gen_useragent())
            .header("Accept", "text/html,application/xhtml+xml")
            .header("Accept-Language", "en-US,en;q=0.9")
            .header("Origin", "https://www.startpage.com")
            .header("Referer", "https://www.startpage.com/")
            .send()
            .await?;

        if response.status() == 429 {
            return Err(SearchError::Blocked);
        }

        // Stale sc → invalidate cache so the next call refetches.
        if response.status() == 403 {
            *self.sc_cache.lock() = None;
            return Err(SearchError::Blocked);
        }

        let mut html_text = response.error_for_status()?.text().await?;

        if Self::is_interstitial(&html_text) {
            let follow_fields =
                Self::extract_interstitial_form_data(&html_text).ok_or_else(|| {
                    SearchError::Parse("interstitial form data not found".to_string())
                })?;

            let follow_pairs: Vec<(&str, &str)> = follow_fields
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();

            let follow_response = client
                .post("https://www.startpage.com/sp/search")
                .form(&follow_pairs)
                .header("User-Agent", gen_useragent())
                .header("Accept", "text/html,application/xhtml+xml")
                .header("Accept-Language", "en-US,en;q=0.9")
                .header("Origin", "https://www.startpage.com")
                .header("Referer", "https://www.startpage.com/sp/search")
                .send()
                .await?;

            if follow_response.status() == 429 {
                return Err(SearchError::Blocked);
            }
            if follow_response.status() == 403 {
                *self.sc_cache.lock() = None;
                return Err(SearchError::Blocked);
            }

            html_text = follow_response.error_for_status()?.text().await?;
        }

        let props_json = Self::extract_props_json(&html_text)
            .ok_or_else(|| SearchError::Parse("React props JSON not found".to_string()))?;

        let props: serde_json::Value = serde_json::from_str(props_json)
            .map_err(|e| SearchError::Parse(format!("Failed to parse props JSON: {e}")))?;

        let mainline = props
            .pointer("/render/presenter/regions/mainline")
            .and_then(|v| v.as_array())
            .ok_or_else(|| SearchError::Parse("regions.mainline not found in props".to_string()))?;

        let web_section = mainline
            .iter()
            .find(|item| item.get("display_type").and_then(|v| v.as_str()) == Some("web-google"))
            .ok_or_else(|| SearchError::Parse("web-google section not found".to_string()))?;

        let raw_results = web_section
            .get("results")
            .and_then(|v| v.as_array())
            .ok_or_else(|| SearchError::Parse("web-google results array missing".to_string()))?;

        let results = raw_results
            .iter()
            .take(max_results)
            .filter_map(|item| {
                let url = item.get("clickUrl")?.as_str()?.to_string();
                if url.is_empty() {
                    return None;
                }
                let title = item
                    .get("title")
                    .and_then(|v| v.as_str())
                    .map(Self::strip_html)
                    .unwrap_or_default();
                if title.is_empty() {
                    return None;
                }
                let description = item
                    .get("description")
                    .and_then(|v| v.as_str())
                    .map(Self::strip_html)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_html_removes_bold_tags() {
        assert_eq!(
            Startpage::strip_html("<b>Rust Programming</b> Language"),
            "Rust Programming Language"
        );
        assert_eq!(Startpage::strip_html("No tags here"), "No tags here");
        assert_eq!(
            Startpage::strip_html("<b>Rust</b> is <b>fast</b>"),
            "Rust is fast"
        );
    }

    #[test]
    fn test_extract_sc_from_homepage_parses_form_input() {
        let html = r#"
            <html><body>
              <form id="search">
                <input name="query" value="" />
                <input name="sc" value="abc123token" />
              </form>
            </body></html>
        "#;
        assert_eq!(
            Startpage::extract_sc_from_homepage(html),
            Some("abc123token".to_string())
        );
    }

    #[test]
    fn test_extract_sc_from_homepage_returns_none_when_missing() {
        let html = r#"<html><body><form id="search"></form></body></html>"#;
        assert_eq!(Startpage::extract_sc_from_homepage(html), None);
    }

    #[test]
    fn test_extract_sc_from_homepage_rejects_empty_value() {
        let html = r#"<input name="sc" value="" />"#;
        assert_eq!(Startpage::extract_sc_from_homepage(html), None);
    }

    #[test]
    fn test_cached_sc_returns_none_when_empty() {
        let engine = Startpage::new().unwrap();
        assert!(engine.cached_sc().is_none());
    }

    #[test]
    fn test_store_and_read_sc_round_trip() {
        let engine = Startpage::new().unwrap();
        engine.store_sc("fresh-token".to_string());
        assert_eq!(engine.cached_sc(), Some("fresh-token".to_string()));
    }

    #[test]
    fn test_is_interstitial_detects_spinner_class() {
        let html = r#"<div class="js-interstitial-spinner"></div>"#;
        assert!(Startpage::is_interstitial(html));
    }

    #[test]
    fn test_is_interstitial_detects_var_data_block() {
        let html = r#"<script>var data = {"foo":"bar"};</script>"#;
        assert!(Startpage::is_interstitial(html));
    }

    #[test]
    fn test_is_interstitial_rejects_normal_serp_html() {
        let html = r#"<script>React.createElement(UIStartpage.AppSerpWeb,{})</script>"#;
        assert!(!Startpage::is_interstitial(html));
    }

    #[test]
    fn test_extract_interstitial_form_data_collects_all_string_fields() {
        let html = r##"
            <html><body>
              <form action="/sp/search" method="POST"></form>
              <script>
                window.addEventListener('DOMContentLoaded', function() {});
                (function () {
                  var data = {"abd":"1","abe":"1","cat":"web","language":"english","lui":"english","qsr":"all","query":"rust","sc":"sctok","segment":"organic","sgt":"sgtok","t":"device"};
                  // submit code here
                })();
              </script>
            </body></html>
        "##;
        let fields = Startpage::extract_interstitial_form_data(html).unwrap();
        let map: std::collections::HashMap<_, _> = fields.into_iter().collect();
        assert_eq!(map.get("sgt"), Some(&"sgtok".to_string()));
        assert_eq!(map.get("sc"), Some(&"sctok".to_string()));
        assert_eq!(map.get("query"), Some(&"rust".to_string()));
        assert_eq!(map.get("segment"), Some(&"organic".to_string()));
        assert_eq!(map.len(), 11);
    }

    #[test]
    fn test_extract_interstitial_form_data_returns_none_when_absent() {
        let html = r#"<html><body>no spinner here</body></html>"#;
        assert!(Startpage::extract_interstitial_form_data(html).is_none());
    }

    #[test]
    fn test_extract_interstitial_form_data_handles_nested_braces() {
        let html = r##"<script>var data = {"a":"1","nested":"{x}","b":"2"};</script>"##;
        let fields = Startpage::extract_interstitial_form_data(html).unwrap();
        let map: std::collections::HashMap<_, _> = fields.into_iter().collect();
        assert_eq!(map.get("a"), Some(&"1".to_string()));
        assert_eq!(map.get("b"), Some(&"2".to_string()));
        assert_eq!(map.get("nested"), Some(&"{x}".to_string()));
    }

    #[test]
    fn test_extract_props_json_not_found() {
        let html = "<html><body>no react here</body></html>";
        assert!(Startpage::extract_props_json(html).is_none());
    }

    #[test]
    fn test_extract_props_json_finds_balanced_braces() {
        let html = r#"
            React.createElement(UIStartpage.AppSerpWeb, {"key": "value", "nested": {"a": 1}})
            extra stuff
        "#;
        let json = Startpage::extract_props_json(html).unwrap();
        assert_eq!(json, r#"{"key": "value", "nested": {"a": 1}}"#);
    }

    #[test]
    fn test_parse_web_results_from_props() {
        let props_json = r#"{
            "render": {
                "presenter": {
                    "regions": {
                        "mainline": [
                            {"display_type": "ads-top", "results": []},
                            {
                                "display_type": "web-google",
                                "presented_count": 2,
                                "results": [
                                    {
                                        "clickUrl": "https://rust-lang.org/",
                                        "title": "<b>Rust Programming</b> Language",
                                        "description": "<b>Rust</b> is blazingly fast"
                                    },
                                    {
                                        "clickUrl": "https://doc.rust-lang.org/book/",
                                        "title": "The <b>Rust</b> Book",
                                        "description": "Learn <b>Rust</b> programming"
                                    }
                                ]
                            }
                        ]
                    }
                }
            }
        }"#;

        let props: serde_json::Value = serde_json::from_str(props_json).unwrap();
        let mainline = props
            .pointer("/render/presenter/regions/mainline")
            .and_then(|v| v.as_array())
            .unwrap();

        let web_section = mainline
            .iter()
            .find(|item| item.get("display_type").and_then(|v| v.as_str()) == Some("web-google"))
            .unwrap();

        let raw = web_section["results"].as_array().unwrap();
        assert_eq!(raw.len(), 2);

        let title = Startpage::strip_html(raw[0]["title"].as_str().unwrap());
        let url = raw[0]["clickUrl"].as_str().unwrap();
        let desc = Startpage::strip_html(raw[0]["description"].as_str().unwrap());

        assert_eq!(title, "Rust Programming Language");
        assert_eq!(url, "https://rust-lang.org/");
        assert_eq!(desc, "Rust is blazingly fast");
    }

    #[test]
    fn test_no_results_when_web_google_missing() {
        // If the web-google section is absent, we get a parse error (not a panic)
        let props: serde_json::Value = serde_json::from_str(
            r#"{
            "render": {"presenter": {"regions": {"mainline": [
                {"display_type": "ads-top", "results": []}
            ]}}}
        }"#,
        )
        .unwrap();

        let mainline = props
            .pointer("/render/presenter/regions/mainline")
            .and_then(|v| v.as_array())
            .unwrap();
        let web = mainline
            .iter()
            .find(|i| i.get("display_type").and_then(|v| v.as_str()) == Some("web-google"));
        assert!(web.is_none());
    }

    #[tokio::test]
    #[ignore = "requires network"]
    async fn test_search_returns_results() {
        let engine = Startpage::new().expect("Failed to create Startpage engine");
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
                    assert_eq!(r.engine, "Startpage");
                }
            }
            Err(SearchError::Blocked) => {
                eprintln!("Startpage is blocking requests — skipping assertions")
            }
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }
}
