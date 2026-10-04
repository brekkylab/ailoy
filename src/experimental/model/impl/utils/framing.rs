//! Cuts a streamed response body into events: server-sent events for plain HTTP APIs, the
//! AWS binary event stream (`application/vnd.amazon.eventstream`) for Bedrock. Each event
//! comes out as the JSON text an [`EventParser`](super::http::EventParser) takes.

use anyhow::{Context as _, bail};

/// Every complete server-sent event in `buf`, as its `data:` payload; a partial one stays
/// behind for the next chunk.
pub(crate) fn sse_drain(buf: &mut Vec<u8>) -> Vec<String> {
    let mut out = Vec::new();
    while let Some(data) = sse_next(buf) {
        // Empty is a keep-alive or comment.
        if !data.is_empty() {
            out.push(data);
        }
    }
    out
}

/// The event in what is left at EOF: a server may close right after the last event
/// without its blank line, and that event may be the only one carrying the finish reason.
pub(crate) fn sse_flush(buf: &[u8]) -> Vec<String> {
    Some(sse_data(buf))
        .filter(|d| !d.is_empty())
        .into_iter()
        .collect()
}

/// Every complete event-stream frame in `buf`, each event as `{"<event-type>": payload}`;
/// a partial frame stays behind for the next chunk.
pub(crate) fn eventstream_drain(buf: &mut Vec<u8>) -> anyhow::Result<Vec<String>> {
    let mut out = Vec::new();
    while let Some(frame) = next_frame(buf)? {
        out.extend(frame.into_event()?);
    }
    Ok(out)
}

/// A frame is length-prefixed, so a leftover at EOF is a truncated frame and is dropped.
pub(crate) fn eventstream_flush(buf: &[u8]) -> Vec<String> {
    if !buf.is_empty() {
        log::warn!("event stream ended inside a frame ({} bytes)", buf.len());
    }
    vec![]
}

/// The next event terminated by a blank line (LF or CRLF), as its `data:` payload.
fn sse_next(buf: &mut Vec<u8>) -> Option<String> {
    let (pos, len) = buf
        .windows(2)
        .position(|w| w == b"\n\n")
        .map(|p| (p, 2))
        .or_else(|| {
            buf.windows(4)
                .position(|w| w == b"\r\n\r\n")
                .map(|p| (p, 4))
        })?;
    let raw: Vec<u8> = buf.drain(..pos + len).collect();
    Some(sse_data(&raw))
}

/// An event's `data:` lines joined by newlines. `event:`, `id:` and comment lines are
/// dropped: every vendor here repeats the event type inside the JSON.
fn sse_data(raw: &[u8]) -> String {
    String::from_utf8_lossy(raw)
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim)
        .collect::<Vec<_>>()
        .join("\n")
}

const PRELUDE_LEN: usize = 12;
const CRC_LEN: usize = 4;

/// One event-stream frame: its string headers and raw payload.
struct Frame {
    headers: Vec<(String, String)>,
    payload: Vec<u8>,
}

impl Frame {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    /// An event frame as `{"<event-type>": payload}`, the union encoding the AWS SDKs
    /// use, so the event name travels with its body. An exception or error frame fails.
    fn into_event(self) -> anyhow::Result<Option<String>> {
        match self.header(":message-type").unwrap_or("event") {
            "event" => {}
            "exception" => {
                let ty = self.header(":exception-type").unwrap_or("unknown");
                let message = serde_json::from_slice::<serde_json::Value>(&self.payload)
                    .ok()
                    .and_then(|v| v["message"].as_str().map(str::to_owned))
                    .unwrap_or_else(|| String::from_utf8_lossy(&self.payload).into_owned());
                bail!("stream exception ({ty}): {message}");
            }
            other => {
                let code = self.header(":error-code").unwrap_or(other);
                let message = self.header(":error-message").unwrap_or("(no message)");
                bail!("stream error ({code}): {message}");
            }
        }
        let Some(event_type) = self.header(":event-type") else {
            return Ok(None);
        };
        let payload: serde_json::Value = serde_json::from_slice(&self.payload)
            .with_context(|| format!("event `{event_type}` payload is not JSON"))?;
        Ok(Some(serde_json::json!({ event_type: payload }).to_string()))
    }
}

/// The next whole frame in `buf`, or `None` until one is buffered. A CRC mismatch or
/// malformed header fails, since the stream cannot be resynchronized after either.
///
/// Layout: a 12-byte prelude (total length, headers length, prelude CRC), the headers,
/// the payload, and a CRC over everything before it. Integers are big-endian; both CRCs
/// are CRC-32 (IEEE).
fn next_frame(buf: &mut Vec<u8>) -> anyhow::Result<Option<Frame>> {
    if buf.len() < PRELUDE_LEN {
        return Ok(None);
    }
    let be32 = |b: &[u8]| u32::from_be_bytes(b.try_into().unwrap());
    let total_len = be32(&buf[0..4]) as usize;
    let headers_len = be32(&buf[4..8]) as usize;
    if be32(&buf[8..12]) != crc32(&buf[0..8]) {
        bail!("event stream prelude CRC mismatch");
    }
    if total_len < PRELUDE_LEN + headers_len + CRC_LEN {
        bail!("event stream frame lengths are inconsistent");
    }
    if buf.len() < total_len {
        return Ok(None);
    }
    let frame: Vec<u8> = buf.drain(..total_len).collect();
    let body_end = total_len - CRC_LEN;
    if be32(&frame[body_end..]) != crc32(&frame[..body_end]) {
        bail!("event stream message CRC mismatch");
    }
    Ok(Some(Frame {
        headers: parse_headers(&frame[PRELUDE_LEN..PRELUDE_LEN + headers_len])?,
        payload: frame[PRELUDE_LEN + headers_len..body_end].to_vec(),
    }))
}

/// Headers are `name_len:u8, name, type:u8, value`. Only string values (type 7) are
/// kept; the routing headers are all strings. The rest are skipped by their length.
fn parse_headers(mut raw: &[u8]) -> anyhow::Result<Vec<(String, String)>> {
    let mut out = Vec::new();
    while let Some((&name_len, rest)) = raw.split_first() {
        let name_len = name_len as usize;
        let name = rest.get(..name_len).context("truncated header name")?;
        let name = std::str::from_utf8(name)?.to_owned();
        let (&ty, mut rest) = rest[name_len..]
            .split_first()
            .context("truncated header type")?;
        let value_len = match ty {
            0 | 1 => 0,
            2 => 1,
            3 => 2,
            4 => 4,
            5 | 8 => 8,
            6 | 7 => {
                let len = rest.get(..2).context("truncated header length")?;
                rest = &rest[2..];
                u16::from_be_bytes(len.try_into().unwrap()) as usize
            }
            9 => 16,
            other => bail!("unknown event stream header type {other}"),
        };
        let value = rest.get(..value_len).context("truncated header value")?;
        if ty == 7 {
            out.push((name, std::str::from_utf8(value)?.to_owned()));
        }
        raw = &rest[value_len..];
    }
    Ok(out)
}

/// CRC-32 (IEEE 802.3), bitwise; frames are small enough that a table is not worth it.
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xEDB8_8320 & (crc & 1).wrapping_neg());
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_frames_on_blank_lines_and_keeps_the_tail() {
        let mut buf = b"data: a\n\nevent: x\r\ndata: b\r\ndata: c\r\n\r\ndata: d".to_vec();
        assert_eq!(sse_drain(&mut buf), ["a", "b\nc"]);
        assert_eq!(buf, b"data: d");
        assert_eq!(sse_flush(&buf), ["d"]);
    }

    /// A frame built the way the service sends one.
    fn frame(headers: &[(&str, &str)], payload: &[u8]) -> Vec<u8> {
        let mut h = Vec::new();
        for (name, value) in headers {
            h.push(name.len() as u8);
            h.extend_from_slice(name.as_bytes());
            h.push(7);
            h.extend_from_slice(&(value.len() as u16).to_be_bytes());
            h.extend_from_slice(value.as_bytes());
        }
        let total = (PRELUDE_LEN + h.len() + payload.len() + CRC_LEN) as u32;
        let mut out = total.to_be_bytes().to_vec();
        out.extend_from_slice(&(h.len() as u32).to_be_bytes());
        out.extend_from_slice(&crc32(&out).to_be_bytes());
        out.extend_from_slice(&h);
        out.extend_from_slice(payload);
        out.extend_from_slice(&crc32(&out).to_be_bytes());
        out
    }

    #[test]
    fn event_stream_wraps_events_and_waits_for_whole_frames() {
        let bytes = frame(
            &[(":message-type", "event"), (":event-type", "messageStart")],
            br#"{"role":"assistant"}"#,
        );
        let mut buf = bytes[..10].to_vec();
        assert!(eventstream_drain(&mut buf).unwrap().is_empty());
        buf.extend_from_slice(&bytes[10..]);
        assert_eq!(
            eventstream_drain(&mut buf).unwrap(),
            [r#"{"messageStart":{"role":"assistant"}}"#]
        );
        assert!(buf.is_empty());
    }

    #[test]
    fn event_stream_exception_fails() {
        let mut buf = frame(
            &[
                (":message-type", "exception"),
                (":exception-type", "throttlingException"),
            ],
            br#"{"message":"slow down"}"#,
        );
        let err = eventstream_drain(&mut buf).unwrap_err();
        assert!(err.to_string().contains("slow down"));
    }
}
