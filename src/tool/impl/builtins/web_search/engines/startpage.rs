use async_trait::async_trait;
use reqwest::Client;

use crate::tool::r#impl::builtins::web_search::engine::{
    SearchEngine, SearchError, SearchResult, USER_AGENT,
};

/// Startpage search engine.
///
/// Startpage serves results via a React SPA. Results are embedded in the HTML as
/// JSON props passed to `React.createElement(UIStartpage.AppSerpWeb, {...})`.
/// The relevant path is: props["render"]["presenter"]["regions"]["mainline"]
/// → find the item with `display_type == "web-google"` → `results[]`.
///
/// Each result has:
/// - `clickUrl`: the actual destination URL
/// - `title`: HTML string (bold tags around matched terms)
/// - `description`: HTML string (bold tags around matched terms)
pub struct Startpage;

impl Startpage {
    pub fn new() -> Result<Self, SearchError> {
        Ok(Self)
    }

    /// Strip HTML tags from a string, returning plain text.
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

    /// Extract the React props JSON string from the raw HTML.
    ///
    /// Looks for `React.createElement(UIStartpage.AppSerpWeb, {` and finds the
    /// matching closing `}` by bracket-counting.
    fn extract_props_json(html: &str) -> Option<&str> {
        let marker = "React.createElement(UIStartpage.AppSerpWeb,";
        let marker_pos = html.find(marker)?;
        let after_marker = &html[marker_pos + marker.len()..];

        // Skip optional whitespace to reach the opening '{'
        let brace_offset = after_marker.find('{')?;
        let json_start = marker_pos + marker.len() + brace_offset;

        // Find the matching closing '}'
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
        let url = format!(
            "https://www.startpage.com/sp/search?q={}",
            urlencoding::encode(query)
        );

        let response = client
            .get(&url)
            .header("User-Agent", USER_AGENT)
            .header("Accept", "text/html,application/xhtml+xml")
            .header("Accept-Language", "en-US,en;q=0.9")
            .send()
            .await?;

        if response.status() == 429 {
            return Err(SearchError::Blocked);
        }

        let html_text = response.error_for_status()?.text().await?;

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
