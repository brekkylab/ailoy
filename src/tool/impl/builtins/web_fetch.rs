use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use html_to_markdown_rs::{ConversionOptions, OutputFormat};
use parking_lot::Mutex;
use url::Url;
use wreq::Client;
use wreq_util::Emulation;

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

#[derive(Clone)]
struct WebFetchState {
    client: Client,
    last_hit: Arc<Mutex<HashMap<String, Instant>>>,
}

impl WebFetchState {
    fn new() -> Self {
        let client = Client::builder()
            .emulation(Emulation::Firefox135)
            .redirect(wreq::redirect::Policy::limited(10))
            .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
            .build()
            .expect("wreq::Client builder cannot fail with these settings");
        Self {
            client,
            last_hit: Arc::new(Mutex::new(HashMap::new())),
        }
    }
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
        // Self-trim: only keep hosts hit within the rate-limit window.
        // Future timestamps (the entry we just inserted with non-zero wait)
        // survive because `duration_since` saturates to zero for them.
        m.retain(|_, when| now.duration_since(*when) < PER_HOST_MIN_INTERVAL);
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
        .get(wreq::header::CONTENT_TYPE)
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

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum BodyFormat {
    Text,
    Markdown,
    Html,
}

impl BodyFormat {
    fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "text" | "plain" | "plaintext" => Some(Self::Text),
            "markdown" | "md" => Some(Self::Markdown),
            "html" | "raw" => Some(Self::Html),
            _ => None,
        }
    }
}

// Output of the HTML→{text,markdown} conversion path. We return both the
// readable body and the document title because `html_to_markdown_rs` already
// extracts both in one pass — no point parsing twice.
struct Converted {
    body: String,
    title: String,
}

fn convert_with_crate(html: &str, output_format: OutputFormat) -> Converted {
    let opts = ConversionOptions::builder()
        .output_format(output_format)
        // Inline base64 data-URI images blow up the byte budget for no
        // benefit to an LLM caller; skip them.
        .skip_images(true)
        .build();
    match html_to_markdown_rs::convert(html, Some(opts)) {
        Ok(result) => {
            // Prefer the `<title>` element; fall back to `og:title` so SPAs
            // that set only the OG tag still surface something useful.
            let title = result
                .metadata
                .document
                .title
                .clone()
                .or_else(|| result.metadata.document.open_graph.get("title").cloned())
                .unwrap_or_default()
                .trim()
                .to_string();
            Converted {
                body: result.content.unwrap_or_default(),
                title,
            }
        }
        // On conversion error (malformed input, etc.), return the raw input
        // and an empty title rather than surfacing an error — `web_fetch` is
        // best-effort.
        Err(_) => Converted {
            body: html.to_string(),
            title: String::new(),
        },
    }
}

// Pick the conversion path for the requested format and content-type.
//
// - `format=html`: raw passthrough, regardless of content type.
// - `format={text,markdown}`: route HTML through `html_to_markdown_rs`;
//   non-HTML content (JSON, plain text, etc.) passes through verbatim so a
//   caller asking for a JSON body gets a JSON body, not an empty conversion.
fn convert(body: &str, content_type: &str, format: BodyFormat) -> Converted {
    if matches!(format, BodyFormat::Html) {
        return Converted {
            body: body.to_string(),
            title: String::new(),
        };
    }
    let is_html = content_type.to_ascii_lowercase().contains("html");
    if !is_html {
        return Converted {
            body: body.to_string(),
            title: String::new(),
        };
    }
    let output_format = match format {
        BodyFormat::Text => OutputFormat::Plain,
        BodyFormat::Markdown => OutputFormat::Markdown,
        BodyFormat::Html => unreachable!(),
    };
    convert_with_crate(body, output_format)
}

// Returns `(slice, total_chars, next_offset)`. `next_offset = None` means the
// slice reaches the end of `text` (caller treats this as `complete`).
//
// Single `char_indices()` pass: locates the start/end byte boundaries for the
// requested char window, counts total chars, and computes `next_offset` without
// re-iterating the slice or copying into a `Vec<char>`. Byte-indexing into
// `text` is safe because `char_indices()` yields char-boundary offsets.
fn slice_body(text: &str, offset: usize, len: usize) -> (String, usize, Option<usize>) {
    let end_char = offset.saturating_add(len);
    let mut start_byte: Option<usize> = None;
    let mut end_byte: Option<usize> = None;
    let mut total_chars: usize = 0;
    for (char_idx, (byte_idx, _)) in text.char_indices().enumerate() {
        if char_idx == offset {
            start_byte = Some(byte_idx);
        }
        if char_idx == end_char {
            end_byte = Some(byte_idx);
        }
        total_chars = char_idx + 1;
    }
    let Some(s) = start_byte else {
        // `offset >= total_chars`: nothing to return, and we are complete.
        return (String::new(), total_chars, None);
    };
    let e = end_byte.unwrap_or(text.len());
    let slice = text[s..e].to_string();
    let next_offset = if e < text.len() { Some(end_char) } else { None };
    (slice, total_chars, next_offset)
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
            "Fetch a URL and return its body. HTML responses are converted to ",
            "the requested format; non-HTML responses (JSON, plain text, etc.) ",
            "pass through unchanged. Bodies are clamped to 30 KiB by default; ",
            "for more, call again with `offset` set to the previous ",
            "`next_offset`. Rate-limited to 1 request/second per host."
        ))
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch."
                },
                "format": {
                    "type": "string",
                    "enum": ["text", "markdown", "html"],
                    "description": "Body format. `text` (default) is visible text only — smallest token cost, no document structure. `markdown` keeps headings, lists, tables, and link targets. `html` returns the response body unchanged.",
                    "default": "text"
                },
                "offset": {
                    "type": "integer",
                    "description": "Character offset into the body. Use 0 for the first call; on subsequent calls pass the `next_offset` returned previously.",
                    "default": 0,
                    "minimum": 0
                },
                "length": {
                    "type": "integer",
                    "description": "Maximum number of characters returned in this call. Default 30720 (30 KiB). Hard cap 61440.",
                    "default": 30720,
                    "minimum": 256,
                    "maximum": 61440
                }
            },
            "required": ["url"]
        }))
        .build()
}

async fn fetch_one(
    state: WebFetchState,
    url_str: String,
    offset: usize,
    length: usize,
    format: BodyFormat,
) -> Value {
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

    rate_limit_for(&state, &host).await;

    let (body, content_type, status, final_url) = match download(&state, &url_str).await {
        Ok(t) => t,
        Err(e) => return crate::to_value!({"url": url_str, "error": e}),
    };
    let Converted {
        body: readable,
        title,
    } = convert(&body, &content_type, format);
    let (slice, total_chars, next_offset) = slice_body(&readable, offset, length);
    let complete = next_offset.is_none();
    let next_offset = match next_offset {
        Some(n) => Value::from(n as i64),
        None => Value::Null,
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
/// is shared across calls, matching `web_search`.
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
            // Unknown `format` values silently fall back to "text" so a
            // mistyped value still returns something useful.
            let format = args
                .pointer("/format")
                .and_then(|v| v.as_str())
                .and_then(BodyFormat::parse)
                .unwrap_or(BodyFormat::Text);

            let url_str = match args.pointer("/url").and_then(|v| v.as_str()) {
                Some(u) => u.to_string(),
                None => {
                    return crate::to_value!({
                        "error": "missing required parameter: `url`"
                    });
                }
            };
            fetch_one(state, url_str, offset, length, format).await
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_body_basic() {
        let s = "abcdefghij";
        let (out, total, next) = slice_body(s, 0, 5);
        assert_eq!(out, "abcde");
        assert_eq!(total, 10);
        assert_eq!(next, Some(5));
        let (out2, _, next2) = slice_body(s, 5, 5);
        assert_eq!(out2, "fghij");
        assert_eq!(next2, None);
        let (out3, _, next3) = slice_body(s, 100, 5);
        assert_eq!(out3, "");
        assert_eq!(next3, None);
    }

    #[test]
    fn slice_body_unicode_is_char_based() {
        let s = "한국어테스트";
        let (out, total, next) = slice_body(s, 0, 3);
        assert_eq!(out, "한국어");
        assert_eq!(total, 6);
        assert_eq!(next, Some(3));
        let (out2, _, next2) = slice_body(s, 3, 3);
        assert_eq!(out2, "테스트");
        assert_eq!(next2, None);
    }

    #[test]
    fn slice_body_next_offset_matches_chars_count() {
        // Freezes the contract that callers can rely on `next_offset` instead
        // of calling `.chars().count()` themselves. If this ever drifts,
        // `fetch_one` would emit a wrong pagination cursor.
        let s = "한국어테스트0123456789";
        let (slice, _, next) = slice_body(s, 2, 4);
        assert_eq!(next, Some(2 + slice.chars().count()));
    }

    #[test]
    fn body_format_parse_accepts_known_values() {
        assert_eq!(BodyFormat::parse("text"), Some(BodyFormat::Text));
        assert_eq!(BodyFormat::parse("plain"), Some(BodyFormat::Text));
        assert_eq!(BodyFormat::parse("Markdown"), Some(BodyFormat::Markdown));
        assert_eq!(BodyFormat::parse("md"), Some(BodyFormat::Markdown));
        assert_eq!(BodyFormat::parse("HTML"), Some(BodyFormat::Html));
        assert_eq!(BodyFormat::parse("raw"), Some(BodyFormat::Html));
        assert_eq!(BodyFormat::parse("xml"), None);
    }

    /// `last_hit` self-trim invariants. The retain expression is replicated
    /// here exactly as `rate_limit_for` uses it; if either drifts, this test
    /// catches it before the table starts growing unbounded again.
    #[test]
    fn last_hit_retain_keeps_fresh_and_future_drops_stale() {
        let now = Instant::now();
        let mut m: HashMap<String, Instant> = HashMap::new();
        m.insert("fresh".into(), now);
        m.insert("stale".into(), now - PER_HOST_MIN_INTERVAL * 2);
        // Mirrors `rate_limit_for`: the just-inserted entry uses `now + wait`,
        // i.e. a future timestamp when wait > 0.
        m.insert("future".into(), now + PER_HOST_MIN_INTERVAL);

        m.retain(|_, when| now.duration_since(*when) < PER_HOST_MIN_INTERVAL);

        assert!(m.contains_key("fresh"));
        assert!(m.contains_key("future"));
        assert!(!m.contains_key("stale"));
    }

    #[test]
    fn convert_text_strips_script_and_style() {
        let html = "<html><head><title>T</title></head><body>\
                    <script>var x = 1;</script>\
                    <style>body { color: red; }</style>\
                    <h1>Hello</h1>\
                    <p>World</p>\
                    </body></html>";
        let out = convert(html, "text/html", BodyFormat::Text);
        assert!(!out.body.contains("var x"), "{}", out.body);
        assert!(!out.body.contains("color: red"), "{}", out.body);
        assert!(out.body.contains("Hello"), "{}", out.body);
        assert!(out.body.contains("World"), "{}", out.body);
    }

    #[test]
    fn convert_markdown_preserves_structure() {
        let html = "<html><body><h1>Title</h1><p>See <a href=\"https://example.com/x\">docs</a></p></body></html>";
        let out = convert(html, "text/html", BodyFormat::Markdown);
        // Markdown should keep the heading sigil and the anchor URL.
        assert!(out.body.contains("# Title"), "{}", out.body);
        assert!(out.body.contains("(https://example.com/x)"), "{}", out.body);
    }

    #[test]
    fn convert_html_returns_body_verbatim() {
        let html = "<html><body><script>var x = 1;</script>\
                    <a href=\"https://example.com\">link</a></body></html>";
        let out = convert(html, "text/html", BodyFormat::Html);
        assert_eq!(out.body, html);
        // `format=html` skips conversion entirely, so no title is extracted.
        assert_eq!(out.title, "");
    }

    #[test]
    fn convert_non_html_passes_through() {
        let body = "{\"key\":\"value\"}";
        let out_text = convert(body, "application/json", BodyFormat::Text);
        assert_eq!(out_text.body, body);
        assert_eq!(out_text.title, "");
        let out_md = convert(body, "application/json", BodyFormat::Markdown);
        assert_eq!(out_md.body, body);
    }

    #[test]
    fn convert_extracts_title_from_title_tag() {
        let html = "<html><head><title>Hello World</title></head><body><p>x</p></body></html>";
        let out = convert(html, "text/html", BodyFormat::Text);
        assert_eq!(out.title, "Hello World");
    }

    #[test]
    fn convert_falls_back_to_og_title() {
        let html = "<html><head><meta property=\"og:title\" content=\"OG Title\"></head>\
                    <body><p>x</p></body></html>";
        let out = convert(html, "text/html", BodyFormat::Text);
        assert_eq!(out.title, "OG Title");
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
            "https://www.accuweather.com/ko/kr/bundang-gu/2330398/current-weather/2330398"
                .to_string(),
            0,
            DEFAULT_BODY_CHARS,
            BodyFormat::Text,
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
            body.to_ascii_lowercase().contains("accuweather") || body.contains("분당구"),
            "body should mention AccuWeather or 분당구, got first 200 chars: {:?}",
            &body.chars().take(200).collect::<String>()
        );
        assert!(retrieved_at.ends_with('Z'), "retrieved_at: {retrieved_at}");
    }

    /// `format="html"` against the same endpoint should return raw markup —
    /// `<html`, `<title>`, etc. — not the converted text form.
    #[tokio::test]
    #[ignore = "requires network"]
    async fn network_single_fetch_html_returns_raw_markup() {
        let state = WebFetchState::new();
        let result = fetch_one(
            state,
            "https://example.com/".to_string(),
            0,
            DEFAULT_BODY_CHARS,
            BodyFormat::Html,
        )
        .await;
        let body = result
            .pointer("/body")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(
            body.contains("<html") || body.contains("<HTML"),
            "html format should preserve markup, got first 200 chars: {:?}",
            &body.chars().take(200).collect::<String>()
        );
    }
}
