//! Decoder for `application/vnd.amazon.eventstream`, the binary framing Amazon
//! Bedrock uses for streamed responses (`InvokeModelWithResponseStream` and
//! `ConverseStream`) in place of SSE.
//!
//! Frame layout: a 12-byte prelude (`total_len`, `headers_len`, prelude CRC),
//! then headers, payload, and a trailing CRC over everything before it. All
//! integers are big-endian; both CRCs are CRC-32 (IEEE).

use anyhow::{Context as _, bail};

use crate::datatype::Value;

const PRELUDE_LEN: usize = 12;
const CRC_LEN: usize = 4;

/// One decoded frame: string-valued headers plus the raw payload. Non-string
/// header values are dropped; the routing headers Bedrock uses (`:message-type`,
/// `:event-type`, `:exception-type`, `:error-code`) are all strings.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Frame {
    pub headers: Vec<(String, String)>,
    pub payload: Vec<u8>,
}

impl Frame {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

/// Drains the next complete frame from `buf`. Returns `Ok(None)` when the
/// buffer does not yet hold a whole frame; the partial bytes stay for the next
/// network chunk. Fails on a CRC mismatch or malformed header, since the
/// stream cannot be resynchronised after either.
pub(crate) fn drain_next_frame(buf: &mut Vec<u8>) -> anyhow::Result<Option<Frame>> {
    if buf.len() < PRELUDE_LEN {
        return Ok(None);
    }
    let total_len = u32::from_be_bytes(buf[0..4].try_into().unwrap()) as usize;
    let headers_len = u32::from_be_bytes(buf[4..8].try_into().unwrap()) as usize;
    let prelude_crc = u32::from_be_bytes(buf[8..12].try_into().unwrap());
    if prelude_crc != crc32(&buf[0..8]) {
        bail!("eventstream prelude CRC mismatch");
    }
    if total_len < PRELUDE_LEN + headers_len + CRC_LEN {
        bail!("eventstream frame lengths are inconsistent");
    }
    if buf.len() < total_len {
        return Ok(None);
    }

    let frame: Vec<u8> = buf.drain(..total_len).collect();
    let message_crc = u32::from_be_bytes(frame[total_len - CRC_LEN..].try_into().unwrap());
    if message_crc != crc32(&frame[..total_len - CRC_LEN]) {
        bail!("eventstream message CRC mismatch");
    }

    let headers = parse_headers(&frame[PRELUDE_LEN..PRELUDE_LEN + headers_len])?;
    let payload = frame[PRELUDE_LEN + headers_len..total_len - CRC_LEN].to_vec();
    Ok(Some(Frame { headers, payload }))
}

/// Header wire format: `name_len:u8, name, value_type:u8, value`. Only the
/// string type (7) is kept; the others are skipped by their fixed or prefixed
/// length.
fn parse_headers(mut raw: &[u8]) -> anyhow::Result<Vec<(String, String)>> {
    let mut out = Vec::new();
    while !raw.is_empty() {
        let name_len = raw[0] as usize;
        raw = &raw[1..];
        let name =
            std::str::from_utf8(raw.get(..name_len).context("truncated header name")?)?.to_owned();
        raw = &raw[name_len..];
        let ty = *raw.first().context("truncated header type")?;
        raw = &raw[1..];
        let value_len = match ty {
            0 | 1 => 0,
            2 => 1,
            3 => 2,
            4 => 4,
            5 | 8 => 8,
            6 | 7 => {
                let len = u16::from_be_bytes(
                    raw.get(..2)
                        .context("truncated header length")?
                        .try_into()
                        .unwrap(),
                ) as usize;
                raw = &raw[2..];
                len
            }
            9 => 16,
            other => bail!("unknown eventstream header type {other}"),
        };
        let value = raw.get(..value_len).context("truncated header value")?;
        if ty == 7 {
            out.push((name, std::str::from_utf8(value)?.to_owned()));
        }
        raw = &raw[value_len..];
    }
    Ok(out)
}

/// Turns a frame into the JSON text a provider's `unmarshal_event` expects, or
/// fails for a server-reported exception/error frame. Events are returned as
/// `{"<event-type>": payload}`, the same union encoding the AWS SDKs use, so
/// the event name travels with its body.
pub(crate) fn frame_to_event_data(frame: &Frame) -> anyhow::Result<Option<String>> {
    match frame.header(":message-type").unwrap_or("event") {
        "event" => {}
        "exception" => {
            let ty = frame.header(":exception-type").unwrap_or("unknown");
            bail!(
                "Bedrock stream exception ({ty}): {}",
                payload_message(frame)
            );
        }
        other => {
            let code = frame.header(":error-code").unwrap_or(other);
            let message = frame.header(":error-message").unwrap_or("(no message)");
            bail!("Bedrock stream error ({code}): {message}");
        }
    }
    let Some(event_type) = frame.header(":event-type") else {
        return Ok(None);
    };
    let payload: Value = serde_json::from_slice(&frame.payload)
        .with_context(|| format!("eventstream `{event_type}` payload is not JSON"))?;
    let wrapped: serde_json::Value = Value::object([(event_type, payload)]).into();
    Ok(Some(wrapped.to_string()))
}

fn payload_message(frame: &Frame) -> String {
    serde_json::from_slice::<serde_json::Value>(&frame.payload)
        .ok()
        .and_then(|v| v.get("message").and_then(|m| m.as_str()).map(str::to_owned))
        .unwrap_or_else(|| String::from_utf8_lossy(&frame.payload).into_owned())
}

/// CRC-32 (IEEE 802.3), bitwise; frames are small enough that a table is not
/// worth a dependency.
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Builds a well-formed frame with string headers; the inverse of
    /// [`drain_next_frame`], used by the Bedrock stream tests as well.
    pub(crate) fn encode_frame(headers: &[(&str, &str)], payload: &[u8]) -> Vec<u8> {
        let mut hdr = Vec::new();
        for (k, v) in headers {
            hdr.push(k.len() as u8);
            hdr.extend_from_slice(k.as_bytes());
            hdr.push(7);
            hdr.extend_from_slice(&(v.len() as u16).to_be_bytes());
            hdr.extend_from_slice(v.as_bytes());
        }
        let total = PRELUDE_LEN + hdr.len() + payload.len() + CRC_LEN;
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&(total as u32).to_be_bytes());
        out.extend_from_slice(&(hdr.len() as u32).to_be_bytes());
        out.extend_from_slice(&crc32(&out[..8]).to_be_bytes());
        out.extend_from_slice(&hdr);
        out.extend_from_slice(payload);
        let crc = crc32(&out);
        out.extend_from_slice(&crc.to_be_bytes());
        out
    }

    pub(crate) fn event_frame(event_type: &str, payload: &str) -> Vec<u8> {
        encode_frame(
            &[
                (":message-type", "event"),
                (":event-type", event_type),
                (":content-type", "application/json"),
            ],
            payload.as_bytes(),
        )
    }

    #[test]
    fn crc32_matches_reference_vector() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn drains_frames_across_chunk_boundaries() {
        let a = event_frame("messageStart", r#"{"role":"assistant"}"#);
        let b = event_frame("messageStop", r#"{"stopReason":"end_turn"}"#);
        let mut wire = a.clone();
        wire.extend_from_slice(&b);

        // Feed in a split that lands mid-prelude of the second frame.
        let split = a.len() + 5;
        let mut buf = wire[..split].to_vec();
        let first = drain_next_frame(&mut buf).unwrap().unwrap();
        assert_eq!(first.header(":event-type"), Some("messageStart"));
        assert_eq!(first.payload, br#"{"role":"assistant"}"#);
        assert!(drain_next_frame(&mut buf).unwrap().is_none());

        buf.extend_from_slice(&wire[split..]);
        let second = drain_next_frame(&mut buf).unwrap().unwrap();
        assert_eq!(second.header(":event-type"), Some("messageStop"));
        assert!(buf.is_empty());
    }

    #[test]
    fn corrupt_crc_is_an_error() {
        let mut wire = event_frame("messageStart", r#"{"role":"assistant"}"#);
        let n = wire.len();
        wire[n - 1] ^= 0xFF;
        assert!(drain_next_frame(&mut wire).is_err());
    }

    #[test]
    fn skips_non_string_headers() {
        // A bool header (type 0) and an int header (type 4) precede the string.
        let mut hdr = vec![1u8, b'b', 0, 1, b'i', 4, 0, 0, 0, 7];
        hdr.extend_from_slice(&[
            11, b':', b'e', b'v', b'e', b'n', b't', b'-', b't', b'y', b'p', b'e',
        ]);
        hdr.push(7);
        hdr.extend_from_slice(&5u16.to_be_bytes());
        hdr.extend_from_slice(b"chunk");
        let headers = parse_headers(&hdr).unwrap();
        assert_eq!(
            headers,
            vec![(":event-type".to_string(), "chunk".to_string())]
        );
    }

    #[test]
    fn converse_event_is_wrapped_by_type() {
        let mut wire = event_frame("contentBlockDelta", r#"{"delta":{"text":"hi"}}"#);
        let frame = drain_next_frame(&mut wire).unwrap().unwrap();
        let data = frame_to_event_data(&frame).unwrap().unwrap();
        let v: serde_json::Value = serde_json::from_str(&data).unwrap();
        assert_eq!(v["contentBlockDelta"]["delta"]["text"], "hi");
    }

    #[test]
    fn exception_frame_surfaces_type_and_message() {
        let mut wire = encode_frame(
            &[
                (":message-type", "exception"),
                (":exception-type", "throttlingException"),
            ],
            br#"{"message":"Too many requests"}"#,
        );
        let frame = drain_next_frame(&mut wire).unwrap().unwrap();
        let err = frame_to_event_data(&frame).unwrap_err().to_string();
        assert!(err.contains("throttlingException"), "{err}");
        assert!(err.contains("Too many requests"), "{err}");
    }
}
