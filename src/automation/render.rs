//! Rendering a prompt template against a task's inputs.
//!
//! `{{ path }}` where `path` is `name(.key|[index])*`, looked up in the inputs object:
//! a string is written as itself, anything else as JSON, and a path that resolves to
//! nothing stays visible rather than vanishing. Nothing else is appended: what the
//! agent sees is exactly what the template says.

use serde_json::Value;

pub fn render(template: &str, inputs: &Value) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(start) = rest.find("{{") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        match after.find("}}") {
            Some(end) => {
                let path = after[..end].trim();
                match lookup(inputs, path) {
                    Some(Value::String(s)) => out.push_str(s),
                    Some(v) => out.push_str(&v.to_string()),
                    None => out.push_str(&rest[start..start + 2 + end + 2]),
                }
                rest = &after[end + 2..];
            }
            None => {
                out.push_str(&rest[start..]);
                rest = "";
            }
        }
    }
    out.push_str(rest);
    out
}

fn lookup<'a>(root: &'a Value, path: &str) -> Option<&'a Value> {
    if path.is_empty() {
        return None;
    }
    let mut current = root;
    for segment in path.split('.') {
        // `key[0][1]` is a key followed by any number of indexes.
        let (key, indexes) = match segment.find('[') {
            Some(at) => (&segment[..at], &segment[at..]),
            None => (segment, ""),
        };
        if !key.is_empty() {
            current = current.get(key)?;
        }
        let mut rest = indexes;
        while let Some(stripped) = rest.strip_prefix('[') {
            let close = stripped.find(']')?;
            let index: usize = stripped[..close].trim().parse().ok()?;
            current = current.get(index)?;
            rest = &stripped[close + 1..];
        }
        if !rest.is_empty() {
            return None;
        }
    }
    Some(current)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn substitutes_dotted_paths_and_indexes() {
        let inputs = json!({
            "fetch": {"title": "disk full", "tags": ["a", "b"], "count": 3,
                      "rows": [{"id": 7}]},
            "event": {"id": 42}
        });
        let s = render(
            "T={{ fetch.title }} tags={{fetch.tags}} first={{ fetch.tags[0] }} \
             id={{fetch.rows[0].id}} n={{ fetch.count }} ev={{ event.id }} {{ missing.x }}",
            &inputs,
        );
        assert_eq!(
            s,
            r#"T=disk full tags=["a","b"] first=a id=7 n=3 ev=42 {{ missing.x }}"#
        );
    }

    #[test]
    fn nothing_is_appended() {
        let s = render("hello", &json!({"a": 1}));
        assert_eq!(s, "hello");
    }

    #[test]
    fn unterminated_brace_is_kept() {
        assert_eq!(render("a {{ b", &json!({})), "a {{ b");
    }
}
