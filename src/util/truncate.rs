/// Truncate `s` to at most `max_chars` characters, keeping equal-sized head and
/// tail and inserting an omission notice in the middle.
///
/// If the string is already within the limit it is returned unchanged.
pub fn middle_truncate(s: String, max_chars: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max_chars {
        return s;
    }
    let head = max_chars / 2;
    let tail = max_chars - head;
    let omitted = chars.len() - head - tail;
    let head_str: String = chars[..head].iter().collect();
    let tail_str: String = chars[chars.len() - tail..].iter().collect();
    format!("{head_str}\n\n... [{omitted} characters omitted] ...\n\n{tail_str}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_string_unchanged() {
        let s = "hello".to_string();
        assert_eq!(middle_truncate(s.clone(), 10), s);
    }

    #[test]
    fn exact_limit_unchanged() {
        let s = "hello".to_string();
        assert_eq!(middle_truncate(s.clone(), 5), s);
    }

    #[test]
    fn long_string_is_truncated() {
        let s = "a".repeat(100);
        let result = middle_truncate(s, 10);
        assert!(result.contains("characters omitted"));
        // head = 5, tail = 5
        assert!(result.starts_with("aaaaa"));
        assert!(result.ends_with("aaaaa"));
    }
}
