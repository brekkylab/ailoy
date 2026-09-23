use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use html_to_markdown_rs::{ConversionOptions, OutputFormat};
use parking_lot::Mutex;
use url::Url;
use wreq::{Client, ClientBuilder};
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
const MAX_REDIRECTS: usize = 10;

#[derive(Clone)]
struct WebFetchState {
    client: Client,
    last_hit: Arc<Mutex<HashMap<String, Instant>>>,
}

impl WebFetchState {
    fn new() -> Self {
        Self::from_builder(Self::client_builder())
    }

    /// Client configuration shared by production and tests. `web_fetch` runs in
    /// the host process, so the two settings that keep it inside the same
    /// network boundary the sandbox enforces live here:
    ///
    /// - the DNS resolver drops every non-public answer, which covers both a
    ///   name that points inward and one that re-resolves inward mid-request;
    /// - the redirect policy re-checks each hop, because a public start URL can
    ///   `302` to an internal address and a `Location` holding an IP literal
    ///   never reaches the resolver at all.
    fn client_builder() -> ClientBuilder {
        Client::builder()
            .emulation(Emulation::Firefox135)
            .dns_resolver(net_guard::PublicOnlyResolver::default())
            .redirect(wreq::redirect::Policy::custom(|attempt| {
                let target = attempt.uri.clone();
                if let Err(e) = net_guard::check_redirect_target(target.scheme_str(), target.host())
                {
                    log::warn!("web_fetch: refused redirect to {target}: {e}");
                    return attempt.error(e);
                }
                wreq::redirect::Policy::limited(MAX_REDIRECTS).redirect(attempt)
            }))
            .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
    }

    fn from_builder(builder: ClientBuilder) -> Self {
        let client = builder
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
        // A refusal from the resolver arrives wrapped in a connect error whose
        // `Display` drops it, so it has to be recovered rather than formatted.
        // Without this a blocked name reports a plain connection failure and is
        // indistinguishable from a target that is merely down.
        .map_err(|e| {
            net_guard::blocked_reason(&e).unwrap_or_else(|| format!("request failed: {e}"))
        })?;
    let status = resp.status().as_u16();
    let final_url = resp.uri().to_string();
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
    // Checked on the parsed host, not the raw input: `url` normalizes the
    // decimal, octal, and hex spellings of an address into a dotted quad while
    // parsing, so `http://2130706433/` arrives here as `127.0.0.1`.
    if let Err(e) = net_guard::check_host(&host) {
        log::warn!("web_fetch: refused {url_str}: {e}");
        return crate::to_value!({"url": url_str, "error": e.to_string()});
    }
    log::debug!("web_fetch: fetching {url_str}");

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

mod net_guard {
    //! Egress guard for host-side tools.
    //!
    //! Tools registered as *pure* run in the ailoy host process rather than inside
    //! the sandbox VM, so the guest network policy never sees their requests. A
    //! model — or a prompt-injected page a model reads — can otherwise aim such a
    //! tool at loopback, the LAN, or the cloud-metadata address and read the answer
    //! back into the conversation. This module decides whether a destination is on
    //! the public internet.
    //!
    //! There are two entry points because a URL names its destination in two ways:
    //! [`check_host`] for a host written as an IP literal, where the address is
    //! known before connecting, and [`PublicOnlyResolver`] for a host written as a
    //! name, where the addresses are known only once DNS answers.

    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

    use wreq::dns::{Addrs, GaiResolver, Name, Resolve, Resolving};

    /// A destination this module refused.
    ///
    /// A concrete type rather than a string because a refusal raised inside the
    /// resolver has to be recognized again after the connector has wrapped it,
    /// which [`blocked_reason`] does by downcast. Matching on the message text
    /// would work today and break the moment someone rewords it.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Blocked(String);

    impl std::fmt::Display for Blocked {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "blocked: {}", self.0)
        }
    }

    impl std::error::Error for Blocked {}

    /// Find a [`Blocked`] in an error's source chain.
    ///
    /// `Display` on a client error stops one level in, while a refusal from
    /// [`PublicOnlyResolver`] sits three deep — under the connect error and the
    /// DNS error the connector wraps it in. Formatting the outer error therefore
    /// reports a plain connection failure, which reads as a target that happens to
    /// be down rather than one that will never be reachable. A model told the first
    /// retries; told the second it stops. Walking the chain is what keeps a name
    /// refused by the resolver reporting the same way as an IP literal refused
    /// before the connection.
    pub fn blocked_reason(err: &(dyn std::error::Error + 'static)) -> Option<String> {
        let mut source = Some(err);
        while let Some(e) = source {
            if let Some(blocked) = e.downcast_ref::<Blocked>() {
                return Some(blocked.to_string());
            }
            source = e.source();
        }
        None
    }

    /// Reject a host that is written as an IP literal pointing anywhere other than
    /// the public internet. A host written as a name returns `Ok` here and is
    /// gated later by [`PublicOnlyResolver`], which is the only place its actual
    /// addresses are known.
    ///
    /// Strips the brackets an IPv6 literal is written with. Both callers hand them
    /// over — `url::Url::host_str` and `http::Uri::host` each return `[::1]` rather
    /// than `::1` — and `IpAddr` parses neither spelling with them attached.
    pub fn check_host(host: &str) -> Result<(), Blocked> {
        let bare = host
            .strip_prefix('[')
            .and_then(|h| h.strip_suffix(']'))
            .unwrap_or(host);
        match bare.parse::<IpAddr>() {
            Ok(ip) if !is_public(ip) => Err(Blocked(format!("{ip} is not a public address"))),
            _ => Ok(()),
        }
    }

    /// Reject a redirect hop by scheme and host. Separate from [`check_host`] so
    /// the per-hop rule — including the scheme, which a `Location` header controls
    /// just as freely as the host — is one testable function.
    pub fn check_redirect_target(scheme: Option<&str>, host: Option<&str>) -> Result<(), Blocked> {
        match scheme {
            Some("http") | Some("https") => {}
            Some(other) => return Err(Blocked(format!("unsupported redirect scheme: {other}"))),
            None => return Err(Blocked("redirect target has no scheme".to_string())),
        }
        match host {
            Some(h) => check_host(h),
            None => Err(Blocked("redirect target has no host".to_string())),
        }
    }

    /// Keep only the globally routable addresses in `addrs`. Returns `Err` when
    /// nothing survives, which the connector buries under two layers of its own
    /// error; [`blocked_reason`] is what digs it back out so the caller reports a
    /// blocked host rather than an opaque connection failure.
    fn filter_public(
        host: &str,
        addrs: impl Iterator<Item = SocketAddr>,
    ) -> Result<Vec<SocketAddr>, Blocked> {
        let (kept, dropped): (Vec<_>, Vec<_>) = addrs.partition(|addr| is_public(addr.ip()));
        if !dropped.is_empty() {
            let ips: Vec<IpAddr> = dropped.iter().map(|addr| addr.ip()).collect();
            log::warn!("net_guard: dropped non-public address(es) for '{host}': {ips:?}");
        }
        if kept.is_empty() {
            return Err(Blocked(format!(
                "'{host}' resolves only to non-public addresses"
            )));
        }
        Ok(kept)
    }

    /// DNS resolver that drops every answer outside the public internet before the
    /// connector can use it.
    ///
    /// Filtering here rather than checking the addresses up front is what closes
    /// the DNS-rebinding window: the connector reaches only the addresses this
    /// filter returned, so there is no gap between a check and the connect for a
    /// second, inward-pointing answer to land in.
    ///
    /// One case stays uncovered: with an HTTP proxy configured, the connector
    /// resolves the proxy's host and the proxy resolves the target, so the target's
    /// addresses never pass through here. That configuration comes from the host
    /// environment rather than from anything a model can set, and [`check_host`]
    /// still applies to the URL itself.
    #[derive(Clone, Default)]
    pub struct PublicOnlyResolver {
        inner: GaiResolver,
    }

    impl Resolve for PublicOnlyResolver {
        fn resolve(&self, name: Name) -> Resolving {
            let host = name.as_str().to_string();
            let resolving = self.inner.resolve(name);
            Box::pin(async move {
                let addrs = resolving.await?;
                let kept = filter_public(&host, addrs)?;
                Ok(Box::new(kept.into_iter()) as Addrs)
            })
        }
    }

    /// Is `ip` a globally routable unicast address, i.e. somewhere on the public
    /// internet rather than on this host, this network, or in a special-use range?
    ///
    /// Written as a blocklist of the IANA special-purpose ranges because
    /// `Ipv4Addr::is_global` is still unstable. Reserved ranges count as
    /// non-public, which errs toward refusing a fetch rather than allowing an
    /// internal one.
    fn is_public(ip: IpAddr) -> bool {
        match ip {
            IpAddr::V4(v4) => is_public_v4(v4),
            IpAddr::V6(v6) => {
                // An IPv4 address embedded in an IPv6 one is still that IPv4
                // address at the far end: `::ffff:127.0.0.1` reaches loopback, and
                // `64:ff9b::127.0.0.1` does too wherever a NAT64 gateway sits in
                // the path. Judge the embedded address, not the wrapper.
                if let Some(v4) = v6.to_ipv4_mapped() {
                    return is_public_v4(v4);
                }
                if let Some(v4) = nat64_embedded_v4(v6) {
                    return is_public_v4(v4);
                }
                is_public_v6(v6)
            }
        }
    }

    fn is_public_v4(ip: Ipv4Addr) -> bool {
        // The ranges std already has a predicate for. `is_link_local` is the one
        // that matters most here: 169.254.0.0/16 is where the cloud-metadata
        // address 169.254.169.254 lives.
        if ip.is_unspecified()
            || ip.is_loopback()
            || ip.is_private()
            || ip.is_link_local()
            || ip.is_multicast()
            || ip.is_broadcast()
            || ip.is_documentation()
        {
            return false;
        }
        // The remaining special-purpose ranges, spelled out because their
        // predicates are unstable. In order: 0.0.0.0/8 "this network", 100.64/10
        // carrier-grade NAT, 192.0.0/24 IETF protocol assignments, 192.88.99/24
        // 6to4 relay anycast, 198.18/15 benchmarking, and 240/4 reserved.
        let [a, b, c, _] = ip.octets();
        !(a == 0
            || (a == 100 && (64..128).contains(&b))
            || (a == 192 && b == 0 && c == 0)
            || (a == 192 && b == 88 && c == 99)
            || (a == 198 && (b == 18 || b == 19))
            || a >= 240)
    }

    fn is_public_v6(ip: Ipv6Addr) -> bool {
        if ip.is_unspecified() || ip.is_loopback() || ip.is_multicast() {
            return false;
        }
        // In order: the whole of ::/64, which also covers the deprecated
        // IPv4-compatible form `::a.b.c.d`; fc00::/7 unique-local; fe80::/10
        // link-local; fec0::/10 deprecated site-local; 2001:db8::/32
        // documentation; 2001::/23 IETF protocol assignments; and 2002::/16 6to4.
        // The last two are worth blocking because Teredo and 6to4 each embed an
        // IPv4 address, so an internal target can ride inside one.
        //
        // 64:ff9b:1::/48 is here for the same embedding reason but is refused
        // outright rather than unwrapped. RFC 8215 reserves it for network-specific
        // NAT64 prefixes of any length up to /96, so the embedded address does not
        // sit at a fixed offset the way it does under the well-known prefix, and
        // there is no address in the range worth reaching anyway.
        //
        // Then three ranges that carry no embedded address and reach no service:
        // 100::/64 discards whatever is sent to it, and 3fff::/20 and 5f00::/16 are
        // reserved for documentation and for SRv6 segment identifiers. None is
        // routable, so none belongs on the allowed side of a predicate that answers
        // "is this on the public internet".
        //
        // 2001:20::/28 (ORCHIDv2) and 2001:30::/28 (DRIP) need no entry of their
        // own — both sit inside 2001::/23 above.
        let s = ip.segments();
        !((s[0] == 0 && s[1] == 0 && s[2] == 0 && s[3] == 0)
            || (s[0] & 0xfe00) == 0xfc00
            || (s[0] & 0xffc0) == 0xfe80
            || (s[0] & 0xffc0) == 0xfec0
            || (s[0] == 0x2001 && s[1] == 0x0db8)
            || (s[0] == 0x2001 && s[1] < 0x0200)
            || s[0] == 0x2002
            || (s[0] == 0x0064 && s[1] == 0xff9b && s[2] == 0x0001)
            || (s[0] == 0x0100 && s[1] == 0 && s[2] == 0 && s[3] == 0)
            || (s[0] == 0x3fff && (s[1] & 0xf000) == 0)
            || s[0] == 0x5f00)
    }

    /// The IPv4 address embedded in a NAT64 well-known-prefix address
    /// (`64:ff9b::/96`, RFC 6052), if this is one.
    fn nat64_embedded_v4(ip: Ipv6Addr) -> Option<Ipv4Addr> {
        let s = ip.segments();
        let is_nat64 =
            s[0] == 0x0064 && s[1] == 0xff9b && s[2] == 0 && s[3] == 0 && s[4] == 0 && s[5] == 0;
        is_nat64.then(|| Ipv4Addr::from(((s[6] as u32) << 16) | s[7] as u32))
    }

    #[cfg(test)]
    mod tests {
        use url::Url;

        use super::*;

        /// The addresses an SSRF attempt actually aims at. Every one of these must
        /// be refused before a connection is opened.
        #[test]
        fn check_host_rejects_non_public_literals() {
            for host in [
                "127.0.0.1",
                "0.0.0.0",
                "10.0.0.1",
                "172.16.0.1",
                "192.168.1.1",
                "169.254.169.254", // IMDSv1
                "100.64.0.1",
                "192.0.2.1",
                "198.18.0.1",
                "224.0.0.1",
                "240.0.0.1",
                "255.255.255.255",
                "[::1]",
                "::1",
                "[::ffff:127.0.0.1]",
                "[64:ff9b::7f00:1]", // NAT64-wrapped 127.0.0.1
                "[fd00::1]",
                "[fe80::1]",
                "[fec0::1]",
                "[2001:db8::1]",
                "[2001::1]",        // Teredo
                "[2002:7f00:1::1]", // 6to4-wrapped 127.0.0.1
                "[::7f00:1]",       // IPv4-compatible loopback
                // RFC 8215 network-specific NAT64. The prefix may be anywhere up to
                // /96, so the embedded address moves; the whole /48 is refused
                // rather than unwrapped, which these two spellings pin.
                "[64:ff9b:1::7f00:1]",
                "[64:ff9b:1:ffff::a9fe:a9fe]",
                "[100::1]",           // RFC 6666 discard-only
                "[3fff::1]",          // RFC 9637 documentation
                "[3fff:fff:ffff::1]", // top of that /20
                "[5f00::1]",          // RFC 9602 SRv6 SIDs
                "[2001:20::1]",       // ORCHIDv2, covered by 2001::/23
            ] {
                assert!(
                    check_host(host).is_err(),
                    "{host} must be refused as non-public"
                );
            }
        }

        /// The addresses just outside the narrower reserved ranges are included so
        /// a mask that is one bit too wide fails here rather than silently refusing
        /// traffic that should have gone out.
        #[test]
        fn check_host_allows_public_literals() {
            for host in [
                "1.1.1.1",
                "8.8.8.8",
                "[2606:4700:4700::1111]",
                "[100:0:0:1::1]", // outside 100::/64
                "[3fff:1000::1]", // outside 3fff::/20
                "[5f01::1]",      // outside 5f00::/16
            ] {
                assert!(check_host(host).is_ok(), "{host} must be allowed");
            }
        }

        /// A name passes this stage even when it is known to resolve inward —
        /// `PublicOnlyResolver` is what refuses it, once the answer is in hand.
        /// Freezing the split here keeps a future edit from moving the check to a
        /// place where a rebinding answer would have a window.
        #[test]
        fn check_host_defers_names_to_the_resolver() {
            for host in ["localhost", "example.com", "metadata.google.internal"] {
                assert!(check_host(host).is_ok(), "{host} must reach the resolver");
            }
        }

        /// Obfuscated spellings of loopback. `std`'s `IpAddr` parser rejects all of
        /// these outright, so the literal check only recognizes them because `url`
        /// normalizes them to a dotted quad first and `fetch_one` checks the parsed
        /// host rather than the raw input. That ordering is what makes them safe.
        #[test]
        fn obfuscated_loopback_urls_are_rejected_after_parsing() {
            for raw in [
                "http://2130706433/", // decimal
                "http://0177.0.0.1/", // octal
                "http://0x7f.0.0.1/", // hex
                "http://127.1/",      // shorthand
                "http://127.0.0.1.:8080/",
            ] {
                let bare = raw.trim_start_matches("http://").trim_end_matches('/');
                assert!(
                    bare.parse::<IpAddr>().is_err(),
                    "{bare} parses as an address on its own, so this case does not \
                     exercise the normalization path it exists to cover"
                );
                let url = Url::parse(raw).unwrap_or_else(|e| panic!("parse {raw}: {e}"));
                let host = url
                    .host_str()
                    .unwrap_or_else(|| panic!("{raw} has no host"));
                assert!(
                    check_host(host).is_err(),
                    "{raw} normalized to host {host}, which must be refused"
                );
            }
        }

        #[test]
        fn redirect_target_requires_http_scheme_and_public_host() {
            assert!(check_redirect_target(Some("https"), Some("example.com")).is_ok());
            assert!(check_redirect_target(Some("http"), Some("1.1.1.1")).is_ok());
            assert!(check_redirect_target(Some("http"), Some("127.0.0.1")).is_err());
            assert!(check_redirect_target(Some("file"), Some("example.com")).is_err());
            assert!(check_redirect_target(None, Some("example.com")).is_err());
            assert!(check_redirect_target(Some("https"), None).is_err());
        }

        /// A split answer — one public address, one internal — must connect only to
        /// the public one rather than being refused outright or, worse, allowed
        /// through wholesale.
        #[test]
        fn filter_public_keeps_public_and_drops_internal() {
            let addrs = vec![
                "1.1.1.1:443".parse::<SocketAddr>().unwrap(),
                "127.0.0.1:443".parse::<SocketAddr>().unwrap(),
                "169.254.169.254:80".parse::<SocketAddr>().unwrap(),
            ];
            let kept =
                filter_public("split.example", addrs.into_iter()).expect("public addr survives");
            assert_eq!(kept, vec!["1.1.1.1:443".parse::<SocketAddr>().unwrap()]);
        }

        #[test]
        fn filter_public_errors_when_every_answer_is_internal() {
            let addrs = vec![
                "127.0.0.1:80".parse::<SocketAddr>().unwrap(),
                "10.1.2.3:80".parse::<SocketAddr>().unwrap(),
            ];
            let err = filter_public("rebind.example", addrs.into_iter())
                .expect_err("all-internal answer must be refused");
            let text = err.to_string();
            assert!(text.contains("rebind.example"), "error text: {text}");
        }

        /// The recovery `web_fetch` depends on: a refusal wrapped by layers that do
        /// not print their source has to stay findable. Two synthetic wrappers stand
        /// in for the connector's connect-and-DNS pair, so this holds the contract
        /// even if wreq restructures its errors.
        #[test]
        fn blocked_reason_digs_a_refusal_out_of_a_source_chain() {
            #[derive(Debug)]
            struct Opaque(Box<dyn std::error::Error + Send + Sync>);
            impl std::fmt::Display for Opaque {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    // Deliberately drops the source, which is what hides the
                    // refusal in the real chain.
                    write!(f, "client error (Connect)")
                }
            }
            impl std::error::Error for Opaque {
                fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                    Some(self.0.as_ref())
                }
            }

            let refusal = check_host("127.0.0.1").expect_err("loopback must be refused");
            let wrapped = Opaque(Box::new(Opaque(Box::new(refusal))));
            assert_eq!(
                blocked_reason(&wrapped).as_deref(),
                Some("blocked: 127.0.0.1 is not a public address")
            );
            assert_eq!(wrapped.to_string(), "client error (Connect)");
        }

        /// An unrelated failure must not be reported as a policy refusal — a target
        /// that is genuinely down is worth retrying, and a blocked one is not.
        #[test]
        fn blocked_reason_ignores_an_ordinary_error() {
            let err = std::io::Error::other("connection reset");
            assert_eq!(blocked_reason(&err), None);
        }

        #[test]
        fn nat64_prefix_unwraps_to_its_ipv4() {
            let ip: Ipv6Addr = "64:ff9b::7f00:1".parse().unwrap();
            assert_eq!(nat64_embedded_v4(ip), Some(Ipv4Addr::new(127, 0, 0, 1)));
            let ip: Ipv6Addr = "64:ff9b::808:808".parse().unwrap();
            assert_eq!(nat64_embedded_v4(ip), Some(Ipv4Addr::new(8, 8, 8, 8)));
            // A public address that merely starts with 0064 is not the NAT64 prefix.
            let ip: Ipv6Addr = "64:ff9c::1".parse().unwrap();
            assert_eq!(nat64_embedded_v4(ip), None);
        }

        /// The NAT64 unwrap must not turn a public embedded address into a block.
        /// Only the well-known prefix unwraps, so this stays scoped to it: the
        /// neighbouring `64:ff9b:1::/48` is refused wholesale, and `64:ff9c::/32`
        /// is an ordinary public range that must not be caught by either rule.
        #[test]
        fn nat64_wrapping_a_public_address_stays_allowed() {
            assert!(check_host("[64:ff9b::808:808]").is_ok());
            assert!(check_host("[64:ff9c::1]").is_ok());
        }
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

    /// A loopback target must be refused before a connection happens, not just
    /// have its body withheld. The server here stands in for an internal
    /// service: if it records a hit, the guard ran too late to matter.
    #[tokio::test]
    async fn fetch_one_refuses_loopback_before_connecting() {
        use std::sync::{Arc, Mutex};

        use axum::{Router, routing::get};

        let hits = Arc::new(Mutex::new(0u32));
        let count = hits.clone();
        let app = Router::new().route(
            "/admin",
            get(move || {
                let count = count.clone();
                async move {
                    *count.lock().unwrap() += 1;
                    "INTERNAL-ONLY db_password=hunter2"
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.ok() });

        let result = fetch_one(
            WebFetchState::new(),
            format!("http://{addr}/admin"),
            0,
            DEFAULT_BODY_CHARS,
            BodyFormat::Text,
        )
        .await;

        let error = result
            .pointer("/error")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        assert!(
            error.contains("blocked"),
            "expected a blocked error, got {result:?}"
        );
        assert!(
            result.pointer("/body").is_none(),
            "no body may reach the model: {result:?}"
        );
        assert_eq!(
            *hits.lock().unwrap(),
            0,
            "the request must never reach the service"
        );
    }

    /// The same refusal reached by name instead of by literal. It travels a
    /// different path — the resolver rather than the up-front check — and the
    /// connector wraps it in an error whose `Display` drops it, so without the
    /// recovery in `download` this reports a bare connect failure. The two must
    /// read alike: a model retries a connection that failed and gives up on one
    /// that policy refused, and this refusal will never succeed.
    #[tokio::test]
    async fn fetch_one_refuses_a_name_resolving_inward_with_the_same_error() {
        use std::sync::{Arc, Mutex};

        use axum::{Router, routing::get};

        let hits = Arc::new(Mutex::new(0u32));
        let count = hits.clone();
        let app = Router::new().route(
            "/admin",
            get(move || {
                let count = count.clone();
                async move {
                    *count.lock().unwrap() += 1;
                    "INTERNAL-ONLY db_password=hunter2"
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move { axum::serve(listener, app).await.ok() });

        // `localhost` clears the literal check and is refused by the resolver,
        // which is the whole point of routing it by name.
        let result = fetch_one(
            WebFetchState::new(),
            format!("http://localhost:{port}/admin"),
            0,
            DEFAULT_BODY_CHARS,
            BodyFormat::Text,
        )
        .await;

        let error = result
            .pointer("/error")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        assert!(
            error.starts_with("blocked: "),
            "a name refused by the resolver must report like a refused literal, \
             got {result:?}"
        );
        assert_eq!(
            *hits.lock().unwrap(),
            0,
            "the request must never reach the service"
        );
    }

    /// A public-looking start URL that redirects inward must fail at the hop.
    /// The start host is a name so it clears the literal check, and the resolve
    /// override aims it at the local server — the only way to exercise the
    /// redirect path without leaving the machine.
    #[tokio::test]
    async fn fetch_one_refuses_redirect_into_loopback() {
        use std::sync::{Arc, Mutex};

        use axum::{Router, response::Redirect, routing::get};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let go_hits = Arc::new(Mutex::new(0u32));
        let meta_hits = Arc::new(Mutex::new(0u32));

        let count = meta_hits.clone();
        let started = go_hits.clone();
        let app = Router::new()
            .route(
                "/go",
                get(move || {
                    let started = started.clone();
                    async move {
                        *started.lock().unwrap() += 1;
                        Redirect::temporary(&format!("http://127.0.0.1:{}/meta", addr.port()))
                    }
                }),
            )
            .route(
                "/meta",
                get(move || {
                    let count = count.clone();
                    async move {
                        *count.lock().unwrap() += 1;
                        "METADATA iam-role-cred=AKIA_EXAMPLE"
                    }
                }),
            );
        tokio::spawn(async move { axum::serve(listener, app).await.ok() });

        // A DNS override ignores the port in the address it is given, so the
        // port has to be in the URL as well.
        let state = WebFetchState::from_builder(
            WebFetchState::client_builder()
                .no_proxy()
                .resolve_to_addrs("first-hop.test", [addr]),
        );
        let result = fetch_one(
            state,
            format!("http://first-hop.test:{}/go", addr.port()),
            0,
            DEFAULT_BODY_CHARS,
            BodyFormat::Text,
        )
        .await;

        // Without this the test could pass for the wrong reason: a failure to
        // reach the first hop at all looks identical to a refused second hop.
        assert_eq!(
            *go_hits.lock().unwrap(),
            1,
            "the first hop must be served, otherwise this test proves nothing"
        );
        let error = result
            .pointer("/error")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        assert!(
            error.starts_with("blocked: "),
            "following the hop must fail as a refusal, not as a bare transport \
             error, got {result:?}"
        );
        assert_eq!(
            *meta_hits.lock().unwrap(),
            0,
            "the redirect target must never be fetched"
        );
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
            body.chars().take(200).collect::<String>()
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
            body.chars().take(200).collect::<String>()
        );
    }
}
