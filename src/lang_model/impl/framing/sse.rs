//! Server-sent events framing: events separated by a blank line, payload in
//! `data:` lines.

/// Drains the next complete SSE event from `buf`, returning its concatenated
/// `data:` payload. Returns `None` if no event (terminated by a blank line) is
/// fully buffered yet; the partial bytes stay in `buf` for the next chunk.
pub(super) fn drain_next_event(buf: &mut Vec<u8>) -> Option<String> {
    // Events are separated by a blank line: "\n\n" (LF) or "\r\n\r\n" (CRLF).
    let (sep_pos, sep_len) = buf
        .windows(2)
        .position(|w| w == b"\n\n")
        .map(|p| (p, 2))
        .or_else(|| {
            buf.windows(4)
                .position(|w| w == b"\r\n\r\n")
                .map(|p| (p, 4))
        })?;

    let raw: Vec<u8> = buf.drain(..sep_pos + sep_len).collect();
    Some(extract_event_data(&raw))
}

/// Extracts the concatenated `data:` payload from one raw SSE event's bytes.
/// SSE permits multiple `data:` lines per event; they are joined with newlines.
/// Non-`data:` lines (`event:`, `id:`, comments) are dropped — every provider
/// here carries its event type inside the `data:` JSON.
pub(super) fn extract_event_data(raw: &[u8]) -> String {
    String::from_utf8_lossy(raw)
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(|rest| rest.trim())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drain_next_event_frames_on_blank_line() {
        // Two complete LF-separated events plus a partial third still buffered.
        let mut buf = b"data: a\n\ndata: b\n\ndata: c".to_vec();
        assert_eq!(drain_next_event(&mut buf).as_deref(), Some("a"));
        assert_eq!(drain_next_event(&mut buf).as_deref(), Some("b"));
        // The unterminated tail is not framed; it stays for the next chunk.
        assert_eq!(drain_next_event(&mut buf), None);
        assert_eq!(buf, b"data: c");
    }

    #[test]
    fn test_drain_next_event_handles_crlf_and_multi_data() {
        // CRLF separators, and an event with two `data:` lines (joined by \n).
        let mut buf = b"data: x\r\ndata: y\r\n\r\nrest".to_vec();
        assert_eq!(drain_next_event(&mut buf).as_deref(), Some("x\ny"));
        assert_eq!(drain_next_event(&mut buf), None);
        assert_eq!(buf, b"rest");
    }

    #[test]
    fn test_extract_event_data_recovers_eof_terminated_event() {
        // The run_stream EOF flush relies on this: a final event left in the
        // buffer without a trailing blank line must still yield its payload.
        assert_eq!(extract_event_data(b"data: {\"k\":1}"), "{\"k\":1}");
        // Non-`data:` lines (comments / event:) are dropped.
        assert_eq!(extract_event_data(b": keep-alive"), "");
        assert_eq!(extract_event_data(b"event: done\ndata: tail"), "tail");
    }
}
