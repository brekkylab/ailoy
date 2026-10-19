//! A page as text: its accessibility tree in document order, with refs on what can be acted on.

use std::collections::HashMap;

use serde_json::Value;

/// Roles the agent acts on, which get a ref.
const INTERACTIVE: &[&str] = &[
    "button",
    "checkbox",
    "combobox",
    "link",
    "listbox",
    "menuitem",
    "menuitemcheckbox",
    "menuitemradio",
    "option",
    "radio",
    "searchbox",
    "slider",
    "spinbutton",
    "switch",
    "tab",
    "textbox",
    "treeitem",
];

/// Roles that only group: shown only when they have a name, their children either way.
const GROUPING: &[&str] = &[
    "generic",
    "none",
    "presentation",
    "GenericContainer",
    "RootWebArea",
    "WebArea",
    "paragraph",
    "div",
    "section",
    "group",
    "LayoutTable",
    "LayoutTableRow",
    "LayoutTableCell",
];

/// Roles that repeat their parent's text or draw nothing: left out with all below them.
const NOISE: &[&str] = &["InlineTextBox", "ListMarker", "LineBreak"];

/// Longest name kept, in characters.
const NAME_CHARS: usize = 150;

/// A rendered page and what its refs name: `e1` to a backend DOM node id.
pub(super) struct Snapshot {
    pub text: String,
    pub refs: HashMap<String, i64>,
}

/// Renders `nodes`, from `Accessibility.getFullAXTree`. `interactive` keeps only what has a ref,
/// one per line, with the heading it sits under.
pub(super) fn render(nodes: &[Value], interactive: bool) -> Snapshot {
    let by_id: HashMap<&str, &Value> = nodes
        .iter()
        .filter_map(|n| Some((n["nodeId"].as_str()?, n)))
        .collect();
    let root = nodes.iter().find(|n| n.get("parentId").is_none());
    let mut out = Render {
        by_id: &by_id,
        interactive,
        lines: Vec::new(),
        refs: HashMap::new(),
    };
    if let Some(root) = root {
        out.walk(root, 0, None);
    }
    Snapshot {
        text: out.lines.join("\n"),
        refs: out.refs,
    }
}

struct Render<'a> {
    by_id: &'a HashMap<&'a str, &'a Value>,
    interactive: bool,
    lines: Vec<String>,
    refs: HashMap<String, i64>,
}

impl Render<'_> {
    /// `parent_name` is the name its nearest shown ancestor already carries, which a text child
    /// would only repeat.
    fn walk(&mut self, node: &Value, depth: usize, parent_name: Option<&str>) {
        let role = node["role"]["value"].as_str().unwrap_or_default();
        if NOISE.contains(&role) {
            return;
        }
        let name = clip(node["name"]["value"].as_str().unwrap_or_default().trim());
        let ignored = node["ignored"].as_bool().unwrap_or(false);
        let backend = node["backendDOMNodeId"].as_i64();

        let shown = !ignored && self.shows(role, &name, parent_name);
        let (depth, name_below) = if shown {
            self.line(node, role, &name, depth, backend);
            (depth + 1, Some(name.as_str()).filter(|n| !n.is_empty()))
        } else {
            (depth, parent_name)
        };
        let name_below = name_below.map(str::to_string);
        for child in node["childIds"].as_array().into_iter().flatten() {
            if let Some(child) = child.as_str().and_then(|id| self.by_id.get(id)) {
                self.walk(child, depth, name_below.as_deref());
            }
        }
    }

    fn shows(&self, role: &str, name: &str, parent_name: Option<&str>) -> bool {
        if self.interactive {
            return INTERACTIVE.contains(&role) || role == "heading";
        }
        match role {
            "StaticText" => !name.is_empty() && parent_name.is_none_or(|p| !p.contains(name)),
            _ if GROUPING.contains(&role) => !name.is_empty() && role != "RootWebArea",
            "" => false,
            _ => true,
        }
    }

    fn line(&mut self, node: &Value, role: &str, name: &str, depth: usize, backend: Option<i64>) {
        let indent = if self.interactive {
            String::new()
        } else {
            "  ".repeat(depth)
        };
        let role = if role == "StaticText" { "text" } else { role };
        let mut line = format!("{indent}- {role}");
        if !name.is_empty() {
            line.push_str(&format!(" {name:?}"));
        }
        if let Some(value) = node["value"]["value"].as_str().filter(|v| !v.is_empty())
            && value != name
        {
            line.push_str(&format!(": {:?}", clip(value)));
        }
        for property in node["properties"].as_array().into_iter().flatten() {
            let value = &property["value"]["value"];
            let attr = match (property["name"].as_str().unwrap_or_default(), value) {
                ("level", Value::Number(n)) if role == "heading" => format!("level={n}"),
                ("checked" | "pressed", Value::String(s)) if s != "false" => {
                    format!("{}={s}", property["name"].as_str().unwrap_or_default())
                }
                (
                    "checked" | "pressed" | "selected" | "expanded" | "disabled" | "required",
                    Value::Bool(true),
                ) => property["name"].as_str().unwrap_or_default().to_string(),
                ("expanded", Value::Bool(false)) => "collapsed".to_string(),
                _ => continue,
            };
            line.push_str(&format!(" [{attr}]"));
        }
        if INTERACTIVE.contains(&role)
            && let Some(backend) = backend
        {
            let r#ref = format!("e{}", self.refs.len() + 1);
            line.push_str(&format!(" [ref={ref}]"));
            self.refs.insert(r#ref, backend);
        }
        self.lines.push(line);
    }
}

/// `name` cut to [`NAME_CHARS`], whitespace runs as one space.
fn clip(name: &str) -> String {
    let joined = name.split_whitespace().collect::<Vec<_>>().join(" ");
    match joined.char_indices().nth(NAME_CHARS) {
        Some((at, _)) => format!("{}…", &joined[..at]),
        None => joined,
    }
}

/// `text` cut to `max` characters at a line end, saying how much was left out.
pub(super) fn cut(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut kept = 0;
    let mut end = 0;
    for line in text.split_inclusive('\n') {
        let n = line.chars().count();
        if kept + n > max {
            break;
        }
        kept += n;
        end += line.len();
    }
    let left = text[end..].lines().count();
    format!(
        "{}\n… {left} more lines. Narrow it with `interactive`, or read a part with `browser_eval`.",
        text[..end].trim_end()
    )
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn node(id: &str, parent: Option<&str>, role: &str, name: &str, children: &[&str]) -> Value {
        let mut n = json!({
            "nodeId": id,
            "role": { "value": role },
            "name": { "value": name },
            "childIds": children,
            "backendDOMNodeId": id.parse::<i64>().unwrap() + 100,
        });
        if let Some(parent) = parent {
            n["parentId"] = json!(parent);
        }
        n
    }

    fn page() -> Vec<Value> {
        let mut heading = node("2", Some("1"), "heading", "Flights", &["3"]);
        heading["properties"] = json!([{ "name": "level", "value": { "value": 1 } }]);
        let mut input = node("5", Some("4"), "textbox", "From", &[]);
        input["value"] = json!({ "value": "Seoul" });
        vec![
            node("1", None, "RootWebArea", "Flights", &["2", "4", "7"]),
            heading,
            node("3", Some("2"), "StaticText", "Flights", &[]),
            node("4", Some("1"), "generic", "", &["5", "6"]),
            input,
            node("6", Some("4"), "link", "Deals", &["8"]),
            node("7", Some("1"), "StaticText", "Prices change daily.", &[]),
            node("8", Some("6"), "StaticText", "Deals", &["9"]),
            node("9", Some("8"), "InlineTextBox", "Deals", &[]),
        ]
    }

    #[test]
    fn a_page_reads_in_document_order_with_refs_on_what_acts() {
        let snapshot = render(&page(), false);
        assert_eq!(
            snapshot.text,
            "- heading \"Flights\" [level=1]\n\
             - textbox \"From\": \"Seoul\" [ref=e1]\n\
             - link \"Deals\" [ref=e2]\n\
             - text \"Prices change daily.\""
        );
        assert_eq!(snapshot.refs["e1"], 105);
        assert_eq!(snapshot.refs["e2"], 106);
    }

    #[test]
    fn interactive_keeps_headings_and_what_acts() {
        let snapshot = render(&page(), true);
        assert_eq!(
            snapshot.text,
            "- heading \"Flights\" [level=1]\n\
             - textbox \"From\": \"Seoul\" [ref=e1]\n\
             - link \"Deals\" [ref=e2]"
        );
    }

    #[test]
    fn a_long_page_is_cut_at_a_line_and_says_so() {
        let text = "- a\n- b\n- c\n- d";
        let cut = cut(text, 8);
        assert!(cut.starts_with("- a\n- b\n"), "{cut}");
        assert!(cut.contains("2 more lines"), "{cut}");
        assert_eq!(super::cut("- a", 8), "- a");
    }
}
