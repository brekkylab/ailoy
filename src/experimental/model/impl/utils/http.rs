//! Sending requests and reading streamed responses.

use futures::{StreamExt as _, stream::BoxStream};
use reqwest::header::{HeaderMap, HeaderValue};

use super::framing;
use crate::message::{FinishReason, MessageDeltaOutput, Role};

/// A vendor's streaming response, one event at a time. Each implementation is stateful
/// for one response, so a fresh one is made per request.
pub(crate) trait EventParser: Send + 'static {
    /// One framed event's data as a delta; `Ok(None)` for events that carry none.
    fn parse(&mut self, data: &str) -> anyhow::Result<Option<MessageDeltaOutput>>;

    /// Every complete event in `buf`, leaving a partial one for the next chunk. Server-sent
    /// events unless overridden.
    fn drain(&self, buf: &mut Vec<u8>) -> anyhow::Result<Vec<String>> {
        Ok(framing::sse_drain(buf))
    }

    /// The events in what is left of the body at EOF.
    fn flush(&self, buf: &[u8]) -> Vec<String> {
        framing::sse_flush(buf)
    }
}

/// Tells a 429 that reports quota exhausted for good, which is not retried, from a rate
/// limit, which is. Takes the response body.
pub(crate) type QuotaCheck = fn(&str) -> bool;

/// For vendors whose 429 is always a rate limit.
pub(crate) fn rate_limit_only(_body: &str) -> bool {
    false
}

/// Sends a non-streaming request and returns its JSON response.
pub(crate) async fn request_json(
    url: &str,
    headers: HeaderMap,
    body: &serde_json::Value,
    is_permanent_quota: QuotaCheck,
) -> anyhow::Result<serde_json::Value> {
    Ok(send(url, headers, body, is_permanent_quota)
        .await?
        .json()
        .await?)
}

/// Sends a request built up front, so the stream borrows nothing from the caller, and
/// yields its events as deltas. The stream runs to the end of the body, and a message the
/// vendor leaves without a `finish_reason` gets a terminal `Stop`. A request that failed to
/// build becomes a stream of that one error.
pub(crate) fn request_stream(
    url: String,
    request: anyhow::Result<(HeaderMap, serde_json::Value)>,
    is_permanent_quota: QuotaCheck,
    mut parser: impl EventParser,
) -> BoxStream<'static, anyhow::Result<MessageDeltaOutput>> {
    let (headers, body) = match request {
        Ok(request) => request,
        Err(e) => return Box::pin(futures::stream::once(async move { Err(e) })),
    };
    Box::pin(async_stream::try_stream! {
        let response = send(&url, headers, &body, is_permanent_quota).await?;

        // Network chunks don't align with event boundaries, so events are framed out of
        // a buffer; role/finish tracking closes the message at EOF.
        let mut seen_role: Option<Role> = None;
        let mut saw_finish = false;
        let mut byte_stream = response.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        let mut eof = false;
        while !eof {
            let events = match byte_stream.next().await {
                Some(chunk) => {
                    buf.extend_from_slice(&chunk?);
                    parser.drain(&mut buf)?
                }
                // Whatever is left may be a last event without its terminator.
                None => {
                    eof = true;
                    parser.flush(&buf)
                }
            };
            for data in events {
                if let Some(output) = parser.parse(&data)? {
                    if seen_role.is_none() {
                        seen_role = output.delta.role.clone();
                    }
                    saw_finish |= output.finish_reason.is_some();
                    yield output;
                }
            }
        }

        // A mid-stream error ends the generator before here, so a failed turn never
        // gets a fake Stop. Not a let-chain: `try_stream!` rejects them.
        #[allow(clippy::collapsible_if)]
        if !saw_finish {
            if let Some(role) = seen_role {
                let mut closer = MessageDeltaOutput::new();
                closer.delta.role = Some(role);
                closer.finish_reason = Some(FinishReason::Stop {});
                yield closer;
            }
        }
    })
}

/// POSTs `body`, retrying a transient 429 with backoff, and returns a 2xx response
/// unread. Bails on any other status, or on a 429 that `is_permanent_quota` calls permanent.
async fn send(
    url: &str,
    headers: HeaderMap,
    body: &serde_json::Value,
    is_permanent_quota: QuotaCheck,
) -> anyhow::Result<reqwest::Response> {
    const MAX_RETRIES: u32 = 3;
    const MAX_WAIT_SECS: u64 = 10;
    let client = reqwest::Client::new();
    let mut attempt = 0;
    loop {
        let response = client
            .post(url)
            .headers(headers.clone())
            .json(body)
            .send()
            .await?;
        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }
        let retry_after = response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok());
        let text = response.text().await.unwrap_or_default();
        if status.as_u16() != 429 || attempt == MAX_RETRIES || is_permanent_quota(&text) {
            anyhow::bail!("API request failed with status {status}: {text}");
        }
        let wait_secs = retry_after.unwrap_or(1 << attempt).min(MAX_WAIT_SECS);
        attempt += 1;
        log::warn!("Rate limited (429); retry {attempt}/{MAX_RETRIES} in {wait_secs}s: {text}");
        tokio::time::sleep(std::time::Duration::from_secs(wait_secs)).await;
    }
}

/// `value` as a header marked sensitive, so it stays out of logs.
pub(crate) fn secret_header(value: &str) -> anyhow::Result<HeaderValue> {
    let mut value = HeaderValue::from_str(value)?;
    value.set_sensitive(true);
    Ok(value)
}

/// `Bearer <token>`, marked sensitive.
pub(crate) fn bearer(token: &str) -> anyhow::Result<HeaderValue> {
    secret_header(&format!("Bearer {token}"))
}
