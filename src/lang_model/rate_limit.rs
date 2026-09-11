//! Rate-limit headroom, read off response headers per wire schema.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::header::HeaderMap;

use crate::{
    lang_model::LangModelAPISchema,
    message::{RateLimitInfo, RateLimitWindow},
};

/// The headroom `headers` report, or `None` when the schema reports none (Gemini, Bedrock)
/// or the headers are absent. `now` anchors relative resets (OpenAI's `6m0s`).
pub(crate) fn parse_rate_limit(
    schema: &LangModelAPISchema,
    headers: &HeaderMap,
    now: SystemTime,
) -> Option<RateLimitInfo> {
    let info = match schema {
        LangModelAPISchema::Anthropic => parse_anthropic(headers),
        LangModelAPISchema::OpenAI | LangModelAPISchema::ChatCompletion => {
            parse_openai(headers, now)
        }
        LangModelAPISchema::Gemini | LangModelAPISchema::Bedrock => return None,
    };
    (!info.is_empty()).then_some(info)
}

fn header_str<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name).and_then(|v| v.to_str().ok())
}

fn header_u64(headers: &HeaderMap, name: &str) -> Option<u64> {
    header_str(headers, name).and_then(|s| s.trim().parse().ok())
}

fn epoch_ms(t: SystemTime) -> Option<u64> {
    t.duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_millis() as u64)
}

fn window(
    limit: Option<u64>,
    remaining: Option<u64>,
    reset_at_ms: Option<u64>,
) -> Option<RateLimitWindow> {
    let w = RateLimitWindow {
        limit,
        remaining,
        reset_at_ms,
    };
    (w.limit.is_some() || w.remaining.is_some() || w.reset_at_ms.is_some()).then_some(w)
}

fn parse_anthropic(headers: &HeaderMap) -> RateLimitInfo {
    // `anthropic-ratelimit-<kind>-{limit,remaining,reset}`, reset in RFC 3339.
    let win = |kind: &str| {
        let get = |suffix: &str| format!("anthropic-ratelimit-{kind}-{suffix}");
        let reset = header_str(headers, &get("reset"))
            .and_then(|s| humantime::parse_rfc3339_weak(s.trim()).ok())
            .and_then(epoch_ms);
        window(
            header_u64(headers, &get("limit")),
            header_u64(headers, &get("remaining")),
            reset,
        )
    };
    RateLimitInfo {
        requests: win("requests"),
        tokens: win("tokens"),
        input_tokens: win("input-tokens"),
        output_tokens: win("output-tokens"),
    }
}

fn parse_openai(headers: &HeaderMap, now: SystemTime) -> RateLimitInfo {
    // `x-ratelimit-{limit,remaining,reset}-<kind>`, reset as a Go-style duration ("6m0s").
    let win = |kind: &str| {
        let reset = header_str(headers, &format!("x-ratelimit-reset-{kind}"))
            .and_then(parse_reset_duration)
            .and_then(|d| epoch_ms(now + d));
        window(
            header_u64(headers, &format!("x-ratelimit-limit-{kind}")),
            header_u64(headers, &format!("x-ratelimit-remaining-{kind}")),
            reset,
        )
    };
    RateLimitInfo {
        requests: win("requests"),
        tokens: win("tokens"),
        input_tokens: None,
        output_tokens: None,
    }
}

/// `"1h2m3s"`, `"6m0s"`, `"1.5s"`, `"120ms"` — number/unit pairs, concatenated.
pub(crate) fn parse_reset_duration(s: &str) -> Option<Duration> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let mut total = Duration::ZERO;
    let mut rest = s;
    while !rest.is_empty() {
        let num_end = rest
            .find(|c: char| !(c.is_ascii_digit() || c == '.'))
            .unwrap_or(rest.len());
        let (num, tail) = rest.split_at(num_end);
        let value: f64 = num.parse().ok()?;
        let unit_end = tail
            .find(|c: char| !c.is_ascii_alphabetic())
            .unwrap_or(tail.len());
        let (unit, tail) = tail.split_at(unit_end);
        let secs = match unit {
            "h" => value * 3600.0,
            "m" => value * 60.0,
            "s" => value,
            "ms" => value / 1000.0,
            _ => return None,
        };
        total += Duration::from_secs_f64(secs);
        rest = tail;
    }
    Some(total)
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, UNIX_EPOCH};

    use reqwest::header::{HeaderMap, HeaderValue};

    use super::*;
    use crate::lang_model::LangModelAPISchema;

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(*k, HeaderValue::from_str(v).unwrap());
        }
        h
    }

    #[test]
    fn anthropic_headers_fill_all_four_windows() {
        let h = headers(&[
            ("anthropic-ratelimit-requests-limit", "4000"),
            ("anthropic-ratelimit-requests-remaining", "3999"),
            ("anthropic-ratelimit-requests-reset", "2026-09-11T10:00:00Z"),
            ("anthropic-ratelimit-tokens-limit", "10400000"),
            ("anthropic-ratelimit-tokens-remaining", "10399000"),
            ("anthropic-ratelimit-input-tokens-limit", "10000000"),
            ("anthropic-ratelimit-input-tokens-remaining", "9999000"),
            ("anthropic-ratelimit-output-tokens-limit", "400000"),
            ("anthropic-ratelimit-output-tokens-remaining", "399000"),
            (
                "anthropic-ratelimit-output-tokens-reset",
                "2026-09-11T10:00:30Z",
            ),
        ]);
        let info = parse_rate_limit(&LangModelAPISchema::Anthropic, &h, UNIX_EPOCH).unwrap();
        assert_eq!(info.requests.as_ref().unwrap().limit, Some(4000));
        assert_eq!(info.requests.as_ref().unwrap().remaining, Some(3999));
        // 2026-09-11T10:00:00Z
        assert_eq!(
            info.requests.as_ref().unwrap().reset_at_ms,
            Some(1_789_120_800_000)
        );
        assert_eq!(info.tokens.as_ref().unwrap().remaining, Some(10_399_000));
        assert_eq!(info.input_tokens.as_ref().unwrap().limit, Some(10_000_000));
        assert_eq!(
            info.output_tokens.as_ref().unwrap().reset_at_ms,
            Some(1_789_120_830_000)
        );
    }

    #[test]
    fn openai_headers_fill_requests_and_tokens_with_relative_reset() {
        let now = UNIX_EPOCH + Duration::from_secs(1_000);
        let h = headers(&[
            ("x-ratelimit-limit-requests", "10000"),
            ("x-ratelimit-remaining-requests", "9999"),
            ("x-ratelimit-reset-requests", "6m0s"),
            ("x-ratelimit-limit-tokens", "30000000"),
            ("x-ratelimit-remaining-tokens", "29999500"),
            ("x-ratelimit-reset-tokens", "1.5s"),
        ]);
        for schema in [
            LangModelAPISchema::OpenAI,
            LangModelAPISchema::ChatCompletion,
        ] {
            let info = parse_rate_limit(&schema, &h, now).unwrap();
            assert_eq!(
                info.requests.as_ref().unwrap().reset_at_ms,
                Some(1_000_000 + 360_000)
            );
            assert_eq!(
                info.tokens.as_ref().unwrap().reset_at_ms,
                Some(1_000_000 + 1_500)
            );
            assert_eq!(info.tokens.as_ref().unwrap().remaining, Some(29_999_500));
            assert!(info.input_tokens.is_none());
        }
    }

    #[test]
    fn no_headers_is_none_not_empty() {
        assert!(
            parse_rate_limit(&LangModelAPISchema::OpenAI, &HeaderMap::new(), UNIX_EPOCH).is_none()
        );
        assert!(
            parse_rate_limit(
                &LangModelAPISchema::Gemini,
                &headers(&[("x-ratelimit-limit-requests", "1")]),
                UNIX_EPOCH
            )
            .is_none()
        );
    }

    #[test]
    fn duration_parser_handles_openai_shapes() {
        assert_eq!(parse_reset_duration("6m0s"), Some(Duration::from_secs(360)));
        assert_eq!(
            parse_reset_duration("1.5s"),
            Some(Duration::from_millis(1500))
        );
        assert_eq!(
            parse_reset_duration("120ms"),
            Some(Duration::from_millis(120))
        );
        assert_eq!(
            parse_reset_duration("1h2m3s"),
            Some(Duration::from_secs(3723))
        );
        assert_eq!(parse_reset_duration("abc"), None);
    }
}
