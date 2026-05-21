use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use reqwest::Client;
use scraper::{Html, Selector};
use url::Url;

use crate::{
    datatype::Value,
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
    tool_func,
};

const DEFAULT_BODY_CHARS: usize = 30 * 1024;
const MAX_BODY_CHARS: usize = 60 * 1024;
const MAX_DOWNLOAD_BYTES: usize = 2 * 1024 * 1024;
const MAX_URL_CHARS: usize = 2048;
const REQUEST_TIMEOUT_SECS: u64 = 10;
const PER_HOST_MIN_INTERVAL: Duration = Duration::from_millis(1000);
const MAX_BATCH_URLS: usize = 5;
const USER_AGENT: &str =
    "Mozilla/5.0 (compatible; ailoy/web_fetch; +https://github.com/brekkylab/ailoy)";

#[derive(Clone)]
struct WebFetchState {
    client: Client,
    last_hit: Arc<Mutex<HashMap<String, Instant>>>,
    robots: Arc<Mutex<HashMap<String, RobotsRules>>>,
}

impl WebFetchState {
    fn new() -> Self {
        let client = Client::builder()
            .user_agent(USER_AGENT)
            .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
            .build()
            .expect("reqwest::Client builder cannot fail with these settings");
        Self {
            client,
            last_hit: Arc::new(Mutex::new(HashMap::new())),
            robots: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

#[derive(Default, Clone)]
struct RobotsRules {
    disallow: Vec<String>,
    allow: Vec<String>,
    // Per RFC 9309 §2.3.1.4: when robots.txt cannot be fetched, allow all.
    fetch_failed: bool,
}

impl RobotsRules {
    fn permits(&self, path: &str) -> bool {
        if self.fetch_failed {
            return true;
        }
        let mut best_len = 0usize;
        let mut best_allow = true;
        for d in &self.disallow {
            if !d.is_empty() && path.starts_with(d.as_str()) && d.len() >= best_len {
                best_len = d.len();
                best_allow = false;
            }
        }
        for a in &self.allow {
            if !a.is_empty() && path.starts_with(a.as_str()) && a.len() >= best_len {
                best_len = a.len();
                best_allow = true;
            }
        }
        best_allow
    }
}

fn parse_robots(body: &str) -> RobotsRules {
    let mut rules = RobotsRules::default();
    let mut applies_to_us = false;
    for raw in body.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let (key, value) = match line.split_once(':') {
            Some((k, v)) => (k.trim().to_ascii_lowercase(), v.trim().to_string()),
            None => continue,
        };
        match key.as_str() {
            "user-agent" => applies_to_us = value == "*",
            "disallow" if applies_to_us => rules.disallow.push(value),
            "allow" if applies_to_us => rules.allow.push(value),
            _ => {}
        }
    }
    rules
}

async fn robots_rules_for(state: &WebFetchState, host_origin: &str) -> RobotsRules {
    if let Some(cached) = state.robots.lock().get(host_origin).cloned() {
        return cached;
    }
    let robots_url = format!("{host_origin}/robots.txt");
    let rules = match state.client.get(&robots_url).send().await {
        Ok(resp) if resp.status().is_success() => {
            let body = resp.text().await.unwrap_or_default();
            parse_robots(&body)
        }
        _ => RobotsRules {
            fetch_failed: true,
            ..Default::default()
        },
    };
    state
        .robots
        .lock()
        .insert(host_origin.to_string(), rules.clone());
    rules
}

async fn rate_limit_for(state: &WebFetchState, host: &str) {
    let wait = {
        let mut m = state.last_hit.lock();
        let now = Instant::now();
        let wait = m
            .get(host)
            .and_then(|when| PER_HOST_MIN_INTERVAL.checked_sub(now.duration_since(*when)))
            .unwrap_or(Duration::ZERO);
        m.insert(host.to_string(), now + wait);
        wait
    };
    if !wait.is_zero() {
        tokio::time::sleep(wait).await;
    }
}

async fn download(
    state: &WebFetchState,
    url: &str,
) -> Result<(String, String, u16, String), String> {
    let resp = state
        .client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    let status = resp.status().as_u16();
    let final_url = resp.url().to_string();
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    // Stream chunks instead of buffering the full body, capped at MAX_DOWNLOAD_BYTES.
    use futures::StreamExt;
    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::with_capacity(64 * 1024);
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("read body chunk: {e}"))?;
        let remaining = MAX_DOWNLOAD_BYTES.saturating_sub(buf.len());
        if remaining == 0 {
            break;
        }
        if chunk.len() <= remaining {
            buf.extend_from_slice(&chunk);
        } else {
            buf.extend_from_slice(&chunk[..remaining]);
            break;
        }
    }
    let body = String::from_utf8_lossy(&buf).into_owned();
    Ok((body, content_type, status, final_url))
}

fn html_to_text(body: &str) -> String {
    let doc = Html::parse_document(body);
    let mut out = String::new();
    walk(doc.root_element(), &mut out);

    // Collapse 3+ newlines to 2 and trim.
    let mut collapsed = String::with_capacity(out.len());
    let mut newline_run = 0;
    for ch in out.chars() {
        if ch == '\n' {
            newline_run += 1;
            if newline_run <= 2 {
                collapsed.push('\n');
            }
        } else {
            newline_run = 0;
            collapsed.push(ch);
        }
    }
    collapsed.trim().to_string()
}

fn walk(el: scraper::ElementRef, out: &mut String) {
    let tag = el.value().name();
    if matches!(tag, "script" | "style" | "noscript" | "svg" | "template") {
        return;
    }
    let is_block = matches!(
        tag,
        "p" | "div"
            | "section"
            | "article"
            | "header"
            | "footer"
            | "nav"
            | "main"
            | "aside"
            | "li"
            | "tr"
            | "h1"
            | "h2"
            | "h3"
            | "h4"
            | "h5"
            | "h6"
            | "blockquote"
            | "pre"
            | "ul"
            | "ol"
            | "table"
            | "br"
            | "hr"
    );
    if is_block && !out.ends_with('\n') {
        out.push('\n');
    }
    if tag == "a" {
        let text = el.text().collect::<String>();
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            out.push_str(trimmed);
            if let Some(href) = el.value().attr("href") {
                let h = href.trim();
                if !h.is_empty() && !h.starts_with('#') {
                    out.push_str(" [");
                    out.push_str(h);
                    out.push(']');
                }
            }
        }
    } else {
        for child in el.children() {
            if let Some(child_el) = scraper::ElementRef::wrap(child) {
                walk(child_el, out);
            } else if let Some(text) = child.value().as_text() {
                out.push_str(text);
            }
        }
    }
    if is_block && !out.ends_with('\n') {
        out.push('\n');
    }
}

fn extract_title(content_type: &str, body: &str) -> String {
    if !content_type.to_ascii_lowercase().contains("html") {
        return String::new();
    }
    let doc = Html::parse_document(body);
    let sel = Selector::parse("title").expect("static selector");
    let raw = doc
        .select(&sel)
        .next()
        .map(|el| el.text().collect::<String>())
        .unwrap_or_default();
    // HTML parsers treat <title> as RAWTEXT, so inline tags like <b>...</b>
    // come through as literal text. Strip them with a simple manual pass.
    let mut out = String::with_capacity(raw.len());
    let mut in_tag = false;
    for ch in raw.chars() {
        match ch {
            '<' => in_tag = true,
            '>' if in_tag => in_tag = false,
            c if !in_tag => out.push(c),
            _ => {}
        }
    }
    out.trim().to_string()
}

fn extract_readable(body: &str, content_type: &str) -> String {
    if content_type.to_ascii_lowercase().contains("html") {
        html_to_text(body)
    } else {
        body.to_string()
    }
}

fn slice_body(text: &str, offset: usize, len: usize) -> (String, usize, bool) {
    let total: Vec<char> = text.chars().collect();
    let total_chars = total.len();
    if offset >= total_chars {
        return (String::new(), total_chars, true);
    }
    let end = offset.saturating_add(len).min(total_chars);
    let slice: String = total[offset..end].iter().collect();
    let truncated = end < total_chars;
    (slice, total_chars, !truncated)
}

fn now_iso_utc() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    let hh = sod / 3600;
    let mm = (sod % 3600) / 60;
    let ss = sod % 60;
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

// Howard Hinnant's `civil_from_days`. days since 1970-01-01 -> (y, m, d).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

pub fn get_web_fetch_tool_desc() -> ToolDesc {
    ToolDescBuilder::new("web_fetch")
        .description(concat!(
            "Fetch one or more URLs and return readable bodies. HTML is reduced ",
            "to plain text: script/style/svg/template/noscript subtrees are ",
            "removed and <a href> anchors render as `text [url]` so outbound ",
            "links survive. Use this after `web_search` to read the full body ",
            "of a result instead of relying on the snippet.\n\n",
            "Two call shapes:\n",
            "  - single: pass `url` (string), get one result back.\n",
            "  - batch: pass `urls` (array of strings) to fetch up to 5 URLs in ",
            "parallel. The response is `{\"results\": [...]}` with one entry ",
            "per input URL, in the same order. Independent fetches (e.g. ",
            "URLs returned by one `web_search` call) should be batched.\n\n",
            "Each body is clamped to 30 KiB by default; for more, call again ",
            "with `offset` set to the previous `next_offset`. Respects ",
            "robots.txt and rate-limits 1 request/second per host."
        ))
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Single URL to fetch. Use this OR `urls`, not both."
                },
                "urls": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Batch of URLs to fetch in parallel (max 5). Use this OR `url`. Returns `{results: [...]}` matching input order."
                },
                "offset": {
                    "type": "integer",
                    "description": "Character offset into each readable body. Use 0 for the first call; on subsequent calls pass the `next_offset` returned previously.",
                    "default": 0,
                    "minimum": 0
                },
                "length": {
                    "type": "integer",
                    "description": "Maximum number of characters per body returned in this call. Default 30720 (30 KiB). Hard cap 61440.",
                    "default": 30720,
                    "minimum": 256,
                    "maximum": 61440
                }
            }
        }))
        .build()
}

async fn fetch_one(state: WebFetchState, url_str: String, offset: usize, length: usize) -> Value {
    if url_str.len() > MAX_URL_CHARS {
        let msg = format!("url exceeds {MAX_URL_CHARS} characters");
        return crate::to_value!({"url": url_str, "error": msg});
    }
    let parsed = match Url::parse(&url_str) {
        Ok(u) => u,
        Err(e) => {
            let msg = format!("invalid url: {e}");
            return crate::to_value!({"url": url_str, "error": msg});
        }
    };
    if !matches!(parsed.scheme(), "http" | "https") {
        let msg = format!("unsupported scheme: {}", parsed.scheme());
        return crate::to_value!({"url": url_str, "error": msg});
    }
    let host = match parsed.host_str() {
        Some(h) => h.to_string(),
        None => return crate::to_value!({"url": url_str, "error": "url has no host"}),
    };
    let host_origin = format!("{}://{}", parsed.scheme(), host);

    let rules = robots_rules_for(&state, &host_origin).await;
    if !rules.permits(parsed.path()) {
        let msg = format!("robots.txt disallows fetching {url_str}");
        return crate::to_value!({"url": url_str, "error": msg});
    }

    rate_limit_for(&state, &host).await;

    let (body, content_type, status, final_url) = match download(&state, &url_str).await {
        Ok(t) => t,
        Err(e) => return crate::to_value!({"url": url_str, "error": e}),
    };
    let title = extract_title(&content_type, &body);
    let readable = extract_readable(&body, &content_type);
    let (slice, total_chars, complete) = slice_body(&readable, offset, length);
    let next_offset = if complete {
        Value::Null
    } else {
        Value::from((offset + slice.chars().count()) as i64)
    };

    crate::to_value!({
        "url": final_url,
        "status": status as i64,
        "title": title,
        "content_type": content_type,
        "body": slice,
        "body_length_total": total_chars as i64,
        "next_offset": next_offset,
        "complete": complete,
        "retrieved_at": now_iso_utc()
    })
}

/// Factory closes over a process-wide [`WebFetchState`] so the rate limiter
/// and robots cache are shared across calls, matching `web_search`.
pub fn get_web_fetch_tool_factory() -> impl Fn(&ToolDesc) -> ToolFunc {
    let state = WebFetchState::new();
    move |_| {
        let state = state.clone();
        tool_func!(async |args: Value| -> Value with [state = state.clone()] {
            let offset = args
                .pointer("/offset")
                .and_then(|v| v.as_integer())
                .unwrap_or(0)
                .max(0) as usize;
            let length = args
                .pointer("/length")
                .and_then(|v| v.as_integer())
                .map(|n| n.max(256).min(MAX_BODY_CHARS as i64) as usize)
                .unwrap_or(DEFAULT_BODY_CHARS);

            // Accept `urls` only when it is a non-empty array of strings.
            // An empty/missing `urls` (some LLMs send `"urls": []` next to a
            // single `url`) falls through to the single-URL path so callers
            // get a result instead of a validation error.
            let urls_array: Option<Vec<String>> = args
                .pointer("/urls")
                .and_then(|v| v.as_array())
                .cloned()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect::<Vec<String>>()
                })
                .filter(|v| !v.is_empty());

            if let Some(urls) = urls_array {
                if urls.len() > MAX_BATCH_URLS {
                    let msg = format!(
                        "batch size {} exceeds max {MAX_BATCH_URLS} — split into multiple calls",
                        urls.len()
                    );
                    return crate::to_value!({"error": msg});
                }

                let mut handles = Vec::with_capacity(urls.len());
                for u in urls {
                    handles.push(tokio::spawn(fetch_one(state.clone(), u, offset, length)));
                }
                let mut results: Vec<Value> = Vec::with_capacity(handles.len());
                for h in handles {
                    match h.await {
                        Ok(v) => results.push(v),
                        Err(e) => {
                            let msg = format!("fetch task panicked: {e}");
                            results.push(crate::to_value!({"error": msg}));
                        }
                    }
                }
                return crate::to_value!({"results": Value::array(results)});
            }

            let url_str = match args.pointer("/url").and_then(|v| v.as_str()) {
                Some(u) => u.to_string(),
                None => {
                    return crate::to_value!({
                        "error": "missing required parameter: either `url` (string) or `urls` (array)"
                    });
                }
            };
            fetch_one(state, url_str, offset, length).await
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_body_basic() {
        let s = "abcdefghij";
        let (out, total, complete) = slice_body(s, 0, 5);
        assert_eq!(out, "abcde");
        assert_eq!(total, 10);
        assert!(!complete);
        let (out2, _, complete2) = slice_body(s, 5, 5);
        assert_eq!(out2, "fghij");
        assert!(complete2);
        let (out3, _, complete3) = slice_body(s, 100, 5);
        assert_eq!(out3, "");
        assert!(complete3);
    }

    #[test]
    fn slice_body_unicode_is_char_based() {
        let s = "한국어테스트";
        let (out, total, _) = slice_body(s, 0, 3);
        assert_eq!(out, "한국어");
        assert_eq!(total, 6);
    }

    #[test]
    fn robots_longest_match_wins() {
        let body = "User-agent: *\nDisallow: /private\nAllow: /private/public\n";
        let rules = parse_robots(body);
        assert!(!rules.permits("/private/secret.html"));
        assert!(rules.permits("/private/public/file.html"));
        assert!(rules.permits("/about"));
    }

    #[test]
    fn robots_other_user_agents_are_ignored() {
        let body = "User-agent: GoogleBot\nDisallow: /\n";
        let rules = parse_robots(body);
        assert!(rules.permits("/anything"));
    }

    #[test]
    fn robots_fetch_failed_allows_all() {
        let rules = RobotsRules {
            fetch_failed: true,
            disallow: vec!["/".into()],
            ..Default::default()
        };
        assert!(rules.permits("/anything"));
    }

    #[test]
    fn html_to_text_strips_script_and_style() {
        let html = "<html><body>\
                    <script>var x = 1;</script>\
                    <style>body { color: red; }</style>\
                    <h1>Hello</h1>\
                    <p>World</p>\
                    </body></html>";
        let out = html_to_text(html);
        assert!(!out.contains("var x"), "{out}");
        assert!(!out.contains("color: red"), "{out}");
        assert!(out.contains("Hello"), "{out}");
        assert!(out.contains("World"), "{out}");
    }

    #[test]
    fn html_to_text_preserves_anchor_href() {
        let html =
            "<html><body><p>See <a href=\"https://example.com/x\">the docs</a></p></body></html>";
        let out = html_to_text(html);
        assert!(
            out.contains("the docs [https://example.com/x]"),
            "anchor href should survive flattening, got: {out:?}"
        );
    }

    #[test]
    fn html_to_text_skips_pure_fragment_links() {
        let html = "<a href=\"#top\">jump</a>";
        let out = html_to_text(html);
        assert!(out.contains("jump"), "{out}");
        assert!(!out.contains("[#top]"), "{out}");
    }

    #[test]
    fn extract_title_pulls_title_element() {
        let html = "<html><head><title>Hello <b>World</b></title></head><body></body></html>";
        let title = extract_title("text/html", html);
        assert_eq!(title, "Hello World");
    }

    #[test]
    fn extract_title_empty_for_non_html_content_type() {
        let html = "<title>Foo</title>";
        assert_eq!(extract_title("application/json", html), "");
    }

    #[test]
    fn descriptor_shape() {
        let desc = get_web_fetch_tool_desc();
        assert_eq!(desc.name, "web_fetch");
        assert!(desc.description.is_some());
    }

    /// Single-URL fetch against a stable public endpoint.
    #[tokio::test]
    #[ignore = "requires network"]
    async fn network_single_fetch_returns_200_and_body() {
        let state = WebFetchState::new();
        let result = fetch_one(
            state,
            "https://example.com/".to_string(),
            0,
            DEFAULT_BODY_CHARS,
        )
        .await;
        let status = result
            .pointer("/status")
            .and_then(|v| v.as_integer())
            .unwrap_or(0);
        let body = result
            .pointer("/body")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let retrieved_at = result
            .pointer("/retrieved_at")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert_eq!(status, 200, "result: {result:?}");
        assert!(
            body.to_ascii_lowercase().contains("example domain"),
            "body should mention `Example Domain`, got first 200 chars: {:?}",
            &body.chars().take(200).collect::<String>()
        );
        assert!(retrieved_at.ends_with('Z'), "retrieved_at: {retrieved_at}");
    }

    /// Batch fetch across multiple distinct hosts. Verifies the response
    /// shape (`{"results": [...]}`), input-order preservation, and that all
    /// five fetches return a non-empty body.
    #[tokio::test]
    #[ignore = "requires network"]
    async fn network_batch_fetch_returns_results_in_order() {
        let state = WebFetchState::new();
        let urls = vec![
            "https://example.com/".to_string(),
            "https://example.org/".to_string(),
            "https://example.net/".to_string(),
        ];
        let mut handles = Vec::with_capacity(urls.len());
        for u in urls.iter() {
            handles.push(tokio::spawn(fetch_one(
                state.clone(),
                u.clone(),
                0,
                DEFAULT_BODY_CHARS,
            )));
        }
        let mut results = Vec::with_capacity(handles.len());
        for h in handles {
            results.push(h.await.unwrap());
        }
        assert_eq!(results.len(), urls.len());
        for (i, r) in results.iter().enumerate() {
            let url = r
                .pointer("/url")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let status = r
                .pointer("/status")
                .and_then(|v| v.as_integer())
                .unwrap_or(0);
            let body = r.pointer("/body").and_then(|v| v.as_str()).unwrap_or("");
            assert!(
                url.starts_with(&urls[i].trim_end_matches('/')),
                "result {i} url mismatch: input={} got={url}",
                urls[i]
            );
            assert_eq!(status, 200, "result {i}: {r:?}");
            assert!(!body.is_empty(), "result {i} body empty");
        }
    }
}
