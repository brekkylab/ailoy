//! How a streamed response body is cut into events before a provider's
//! `unmarshal_event` sees them: SSE for plain HTTP APIs, the AWS binary event
//! stream for Bedrock.

pub(crate) mod eventstream;
mod sse;

use crate::lang_model::LangModelAPISchema;

/// Framing for one streamed response. Each drained item is the JSON text
/// handed to the provider's `unmarshal_event`.
pub(crate) enum Framing {
    Sse,
    EventStream,
}

impl Framing {
    pub(crate) fn for_schema(schema: &LangModelAPISchema) -> Self {
        match schema {
            LangModelAPISchema::Bedrock => Self::EventStream,
            _ => Self::Sse,
        }
    }

    /// Every complete event currently in `buf`; partial bytes stay behind.
    pub(crate) fn drain(&mut self, buf: &mut Vec<u8>) -> anyhow::Result<Vec<String>> {
        let mut out = Vec::new();
        match self {
            Self::Sse => {
                while let Some(data) = sse::drain_next_event(buf) {
                    if !data.is_empty() {
                        out.push(data); // empty = keep-alive / comment line
                    }
                }
            }
            Self::EventStream => {
                while let Some(frame) = eventstream::drain_next_frame(buf)? {
                    if let Some(data) = eventstream::frame_to_event_data(&frame)? {
                        out.push(data);
                    }
                }
            }
        }
        Ok(out)
    }

    /// Events recoverable from the bytes left at EOF. Some SSE servers close
    /// the connection right after the last event with no trailing blank line,
    /// so `drain` never framed it — and it may be the only copy of the terminal
    /// event (e.g. Gemini's finish_reason + usage chunk). Event-stream frames
    /// are length-prefixed, so a leftover there is a truncated frame and is
    /// dropped.
    pub(crate) fn flush(&mut self, buf: &[u8]) -> anyhow::Result<Vec<String>> {
        match self {
            Self::Sse => {
                let data = sse::extract_event_data(buf);
                Ok(if data.is_empty() { vec![] } else { vec![data] })
            }
            Self::EventStream => {
                if !buf.is_empty() {
                    log::warn!(
                        "Bedrock stream ended with {} bytes of a truncated frame",
                        buf.len()
                    );
                }
                Ok(vec![])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The event-stream framing hands each frame's JSON to the provider and
    /// tolerates frames split across network chunks; a truncated tail at EOF is
    /// dropped rather than parsed.
    #[test]
    fn event_stream_framing_drains_whole_frames_only() {
        use eventstream::tests::event_frame;

        let mut wire = event_frame("messageStart", r#"{"role":"assistant"}"#);
        wire.extend(event_frame("messageStop", r#"{"stopReason":"end_turn"}"#));
        let cut = wire.len() - 7;

        let mut framing = Framing::EventStream;
        let mut buf = wire[..cut].to_vec();
        let first = framing.drain(&mut buf).unwrap();
        assert_eq!(first.len(), 1);
        assert!(first[0].starts_with(r#"{"messageStart""#), "{}", first[0]);

        buf.extend_from_slice(&wire[cut..]);
        let second = framing.drain(&mut buf).unwrap();
        assert_eq!(second.len(), 1);
        assert!(second[0].starts_with(r#"{"messageStop""#), "{}", second[0]);
        assert!(buf.is_empty());

        // A partial prelude left at EOF yields nothing instead of an error.
        assert!(framing.flush(&wire[..5]).unwrap().is_empty());
    }

    /// SSE keep-alive comments produce no events; an EOF-terminated final event
    /// is recovered by `flush`.
    #[test]
    fn sse_framing_skips_comments_and_flushes_tail() {
        let mut framing = Framing::Sse;
        let mut buf = b": keep-alive\n\ndata: a\n\ndata: tail".to_vec();
        assert_eq!(framing.drain(&mut buf).unwrap(), vec!["a".to_string()]);
        assert_eq!(framing.flush(&buf).unwrap(), vec!["tail".to_string()]);
    }
}
