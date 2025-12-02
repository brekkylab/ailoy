use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

use anyhow::anyhow;
use futures::lock::Mutex;
use reqwest::Client;
use scraper::{Html, Selector};
use serde::Deserialize;
#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};

use crate::{
    to_value,
    tool::{FunctionTool, ToolFunc},
    utils::sleep,
    value::{ToolDescBuilder, Value},
};

#[derive(Clone, Debug)]
struct SearchResult {
    pub title: String,
    pub link: String,
    pub snippet: String,
    pub position: usize,
}

struct RateLimiter {
    last_request: Arc<Mutex<Instant>>,
    min_interval: Duration,
}

impl RateLimiter {
    pub fn new(requests_per_minute: usize) -> Self {
        Self {
            last_request: Arc::new(Mutex::new(Instant::now())),
            min_interval: Duration::from_secs(60 / requests_per_minute as u64),
        }
    }

    pub async fn acquire(&self) {
        let mut last = self.last_request.lock().await;
        let elapsed = last.elapsed();

        if elapsed < self.min_interval {
            let wait_time = self.min_interval - elapsed;
            drop(last);
            sleep(wait_time.as_millis() as i32).await;
            let mut last = self.last_request.lock().await;
            *last = Instant::now();
        } else {
            *last = Instant::now();
        }
    }
}

const DUCKDUCKGO_BASE_URL: &str = "https://html.duckduckgo.com/html";

struct DuckDuckGoSearcher {
    base_url: String,
    rate_limiter: RateLimiter,
    client: Client,
}

impl DuckDuckGoSearcher {
    pub fn new(base_url: Option<String>, requests_per_minute: Option<usize>) -> Self {
        let client = Client::builder().user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36").build().expect("Failed to create HTTP client");
        let base_url = base_url.unwrap_or(DUCKDUCKGO_BASE_URL.to_string());
        let requests_per_minute = requests_per_minute.unwrap_or(60);
        Self {
            base_url,
            rate_limiter: RateLimiter::new(requests_per_minute),
            client,
        }
    }

    pub fn format_results(&self, results: &[SearchResult]) -> String {
        if results.is_empty() {
            return "No results were found for your search query. This could be due to DuckDuckGo's bot detection or the query returned no matches. Please try rephrasing your search or try again in a few minutes.".to_string();
        }

        let mut output = vec![format!("Found {} search results:\n", results.len())];

        for result in results {
            output.push(format!("{}. {}", result.position, result.title));
            output.push(format!("   URL: {}", result.link));
            output.push(format!("   Summary: {}", result.snippet));
            output.push(String::new());
        }

        output.join("\n")
    }

    pub async fn search(
        &self,
        query: &str,
        max_results: usize,
    ) -> anyhow::Result<Vec<SearchResult>> {
        self.rate_limiter.acquire().await;

        let params = [("q", query), ("b", ""), ("kl", "")];
        let response = self
            .client
            .post(&self.base_url)
            .form(&params)
            .timeout(Duration::from_secs(30))
            .send()
            .await?;

        let body = response.error_for_status()?.text().await?;
        let document = Html::parse_document(&body);

        let mut results = Vec::new();
        let result_selector =
            Selector::parse(".result").map_err(|_| anyhow!("Invalid CSS selector"))?;
        let title_selector =
            Selector::parse(".result__title").map_err(|_| anyhow!("Invalid CSS selector"))?;
        let snippet_selector =
            Selector::parse(".result__snippet").map_err(|_| anyhow!("Invalid CSS selector"))?;

        for element in document.select(&result_selector) {
            if let Some(title_elem) = element.select(&title_selector).next() {
                if let Some(link_elem) = title_elem.select(&Selector::parse("a").unwrap()).next() {
                    let title = link_elem.inner_html();
                    let mut link = link_elem.value().attr("href").unwrap_or("").to_string();

                    // Skip ad results
                    if link.contains("y.js") {
                        continue;
                    }

                    // Clean up DuckDuckGo redirect URLs
                    if link.starts_with("//duckduckgo.com/l/?uddg=") {
                        if let Some(encoded_url) = link.split("uddg=").nth(1) {
                            if let Some(clean_url) = encoded_url.split('&').next() {
                                link = urlencoding::decode(clean_url)
                                    .unwrap_or_default()
                                    .to_string();
                            }
                        }
                    }

                    let snippet = element
                        .select(&snippet_selector)
                        .next()
                        .map(|e| e.inner_html())
                        .unwrap_or_default();

                    results.push(SearchResult {
                        title,
                        link,
                        snippet,
                        position: results.len() + 1,
                    });

                    if results.len() >= max_results {
                        break;
                    }
                }
            }
        }

        Ok(results)
    }
}

struct WebContentFetcher {
    rate_limiter: RateLimiter,
    client: Client,
}

impl WebContentFetcher {
    pub fn new() -> Self {
        let client = Client::builder()
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            .build()
            .expect("Failed to create HTTP client");

        Self {
            rate_limiter: RateLimiter::new(20),
            client,
        }
    }

    pub async fn fetch_and_parse(&self, url: &str) -> anyhow::Result<String> {
        self.rate_limiter.acquire().await;

        let response = self
            .client
            .get(url)
            .timeout(Duration::from_secs(30))
            .send()
            .await?;

        let body = response.error_for_status()?.text().await?;
        let document = Html::parse_document(&body);

        // Extract and clean text
        let text = Self::extract_and_clean_text(&document);

        // Truncate if too long
        let final_text = if text.len() > 8000 {
            format!(
                "{}... [content truncated]",
                text.chars().take(8000).collect::<String>()
            )
        } else {
            text
        };

        Ok(final_text)
    }

    fn extract_and_clean_text(document: &Html) -> String {
        // Get all text nodes, excluding script, style, nav, header, footer
        let excluded_selectors = vec!["script", "style", "nav", "header", "footer"];

        let mut text = String::new();
        Self::extract_text_recursive(&document.root_element(), &excluded_selectors, &mut text);

        // Clean up text
        let lines: Vec<&str> = text.lines().map(|l| l.trim()).collect();
        let mut cleaned = Vec::new();

        for line in lines {
            if !line.is_empty() {
                let chunks: Vec<&str> = line
                    .split("  ")
                    .map(|c| c.trim())
                    .filter(|c| !c.is_empty())
                    .collect();
                cleaned.extend(chunks);
            }
        }

        let result = cleaned.join(" ");

        // Remove extra whitespace
        #[cfg(not(target_arch = "wasm32"))]
        {
            let re = onig::Regex::new(r"\s+").unwrap();
            re.replace_all(&result, " ").trim().to_string()
        }
        #[cfg(target_arch = "wasm32")]
        {
            let re = regex::Regex::new(r"\s+").unwrap();
            re.replace_all(&result, " ").trim().to_string()
        }
    }

    fn extract_text_recursive(
        node: &scraper::element_ref::ElementRef,
        excluded: &[&str],
        text: &mut String,
    ) {
        let tag_name = node.value().name();

        // Skip excluded elements
        if excluded.contains(&tag_name) {
            return;
        }

        // Extract text from this node
        for child in node.children() {
            if let Some(text_node) = child.value().as_text() {
                let content = text_node.text.trim();
                if !content.is_empty() {
                    text.push_str(content);
                    text.push(' ');
                }
            } else if let Some(_) = child.value().as_element() {
                if let Some(elem_ref) = scraper::element_ref::ElementRef::wrap(child) {
                    Self::extract_text_recursive(&elem_ref, excluded, text);
                }
            }
        }
    }
}

pub fn create_web_search_duckduckgo_tool(config: Value) -> anyhow::Result<FunctionTool> {
    #[derive(Clone, Default, Deserialize)]
    struct WebSearchDuckduckgoToolConfig {
        base_url: Option<String>,
        requests_per_minute: Option<usize>,
    }

    let config =
        serde_json::from_value::<WebSearchDuckduckgoToolConfig>(config.into()).unwrap_or_default();

    if cfg!(feature = "wasm") {
        if config.base_url.is_none()
            || config
                .base_url
                .clone()
                .is_some_and(|val| val == DUCKDUCKGO_BASE_URL)
        {
            return Err(anyhow::anyhow!(dedent::dedent!(
                r#"
                Builtin tool \"web_search_duckduckgo\" is not available on web browser environment due to the CORS policy.
                Try to setup proxy server and configure `base_url` to point there.
            "#
            )));
        }
    }

    let desc = ToolDescBuilder::new("web_search_duckduckgo").description("Search webpages using DuckDuckGo and return formatted results.").parameters(to_value!({
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query string"},
            "max_results": {"type": "number", "description": "Maximum number of results to return (default: 10)"},
        },
        "required": ["query"]
    })).returns(to_value!({
        "type": "object",
        "properties": {
            "results": {"type": "string", "description": "Formatted results in a single string"}
        },
        "required": ["results"]
    })).build();

    let f: Box<ToolFunc> = Box::new(move |args: Value| {
        let config = config.clone();
        let base_url = config.base_url;
        let requests_per_minute = config.requests_per_minute;
        Box::pin(async move {
            let args = match args.as_object() {
                Some(a) => a,
                None => {
                    return Ok(to_value!({
                        "error": "Invalid arguments: expected object"
                    }));
                }
            };

            let query = match args.get("query").and_then(|v| v.as_str()) {
                Some(s) => s,
                None => {
                    return Ok(to_value!({
                        "error": "Missing required 'query' string"
                    }));
                }
            };
            let max_results = args
                .get("max_results")
                .and_then(|v| v.as_unsigned())
                .unwrap_or(10) as usize;

            let searcher = DuckDuckGoSearcher::new(base_url, requests_per_minute);
            let results = match searcher.search(query, max_results).await {
                Ok(results) => results,
                Err(err) => {
                    return Ok(to_value!({
                        "error": format!("{}", err.context("Failed to search results"))
                    }));
                }
            };
            let formatted = searcher.format_results(&results);

            Ok(to_value!({"results": formatted}))
        })
    });

    Ok(FunctionTool::new(desc, std::sync::Arc::new(f)))
}

pub fn create_web_fetch_tool(_config: Value) -> anyhow::Result<FunctionTool> {
    let desc = ToolDescBuilder::new("web_fetch")
        .description("Fetch and parse content from a webpage URL.")
        .parameters(to_value!({
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The webpage URL to fetch content from"},
            },
            "required": ["url"]
        }))
        .returns(to_value!({
            "type": "object",
            "properties": {
                "results": {"type": "string", "description": "Parsed web contents"}
            },
            "required": ["results"]
        }))
        .build();

    let f: Box<ToolFunc> = Box::new(move |args: Value| {
        Box::pin(async move {
            let args = match args.as_object() {
                Some(a) => a,
                None => {
                    return Ok(to_value!({
                        "error": "Invalid arguments: expected object"
                    }));
                }
            };

            let url = match args.get("url").and_then(|v| v.as_str()) {
                Some(s) => s,
                None => {
                    return Ok(to_value!({
                        "error": "Missing required 'query' string"
                    }));
                }
            };

            let fetcher = WebContentFetcher::new();
            let results = match fetcher.fetch_and_parse(url).await {
                Ok(results) => results,
                Err(err) => {
                    return Ok(to_value!({
                        "error": format!("{}", err.context("Failed to fetch web contents"))
                    }));
                }
            };

            Ok(to_value!({"results": results}))
        })
    });

    Ok(FunctionTool::new(desc, std::sync::Arc::new(f)))
}
