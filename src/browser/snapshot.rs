//! A page as text: its accessibility tree in document order, with refs on what can be acted on.
//! What can be acted on, and what is hidden or covered, follows rules over the page's DOM
//! snapshot.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::Arc,
};

use serde_json::Value;

/// Roles the agent acts on when the page's DOM could not be read; Chromium names the date, time
/// and color inputs' roles in its own casing.
const INTERACTIVE: &[&str] = &[
    "ColorWell",
    "Date",
    "DateTime",
    "InputTime",
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

/// The computed styles the DOM snapshot is asked for, in this order.
pub(super) const STYLES: [&str; 10] = [
    "display",
    "visibility",
    "opacity",
    "overflow",
    "overflow-x",
    "overflow-y",
    "cursor",
    "pointer-events",
    "position",
    "background-color",
];

/// The attributes shown on a line with a ref.
const ATTRIBUTES: &[&str] = &[
    "title",
    "type",
    "id",
    "name",
    "role",
    "value",
    "placeholder",
    "data-date-format",
    "alt",
    "aria-label",
    "aria-expanded",
    "data-state",
    "aria-checked",
    "aria-valuemin",
    "aria-valuemax",
    "aria-valuenow",
    "aria-placeholder",
    "pattern",
    "min",
    "max",
    "minlength",
    "maxlength",
    "step",
    "accept",
    "multiple",
    "inputmode",
    "autocomplete",
    "aria-autocomplete",
    "list",
    "data-mask",
    "data-inputmask",
    "data-datepicker",
    "contenteditable",
];

/// An axis-aligned box: x, y, width, height.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl Rect {
    fn of(v: &Value) -> Option<Rect> {
        let n = |i: usize| v.get(i).and_then(Value::as_f64);
        Some(Rect {
            x: n(0)?,
            y: n(1)?,
            w: n(2)?,
            h: n(3)?,
        })
    }

    /// How much of this box lies inside `outer`, from 0 to 1.
    fn inside(&self, outer: &Rect) -> f64 {
        let w = ((self.x + self.w).min(outer.x + outer.w) - self.x.max(outer.x)).max(0.0);
        let h = ((self.y + self.h).min(outer.y + outer.h) - self.y.max(outer.y)).max(0.0);
        let area = self.w * self.h;
        if area == 0.0 { 0.0 } else { w * h / area }
    }
}

/// What the DOM says of a node.
#[derive(Clone, Debug, Default)]
pub(super) struct Facts {
    pub element: bool,
    /// Lower case.
    pub tag: String,
    pub attrs: Vec<(String, String)>,
    pub bounds: Option<Rect>,
    /// Laid out, not `display: none`, `visibility: hidden` or transparent.
    pub visible: bool,
    pub pointer: bool,
    /// A click, mouse or pointer listener is on it.
    pub listener: bool,
    /// An input, select or textarea is its child or grandchild.
    pub control_below: bool,
    /// Pages above and below, for a box that scrolls.
    pub scroll: Option<(f64, f64)>,
    /// Painted over by something opaque.
    pub covered: bool,
}

impl Facts {
    fn attr(&self, name: &str) -> Option<&str> {
        self.attrs
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

/// A process's DOM, by backend id.
pub(super) type Dom = HashMap<i64, Facts>;

/// The DOM in a `DOMSnapshot.captureSnapshot` asked for [`STYLES`], paint orders and rects.
pub(super) fn read_dom(snapshot: &Value) -> Dom {
    let strings: Vec<&str> = snapshot["strings"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|s| s.as_str().unwrap_or_default())
        .collect();
    let string = |i: &Value| {
        i.as_i64()
            .filter(|i| *i >= 0)
            .and_then(|i| strings.get(i as usize).copied())
    };
    let list = |v: &Value| v.as_array().cloned().unwrap_or_default();
    let mut dom = Dom::new();
    for document in snapshot["documents"].as_array().into_iter().flatten() {
        let nodes = &document["nodes"];
        let backends = list(&nodes["backendNodeId"]);
        let parents = list(&nodes["parentIndex"]);
        let types = list(&nodes["nodeType"]);
        let names = list(&nodes["nodeName"]);
        let attributes = list(&nodes["attributes"]);
        let mut facts: Vec<Facts> = (0..backends.len())
            .map(|i| {
                let pairs = list(attributes.get(i).unwrap_or(&Value::Null));
                Facts {
                    element: types.get(i).and_then(Value::as_i64) == Some(1),
                    tag: names
                        .get(i)
                        .and_then(&string)
                        .unwrap_or_default()
                        .to_lowercase(),
                    attrs: pairs
                        .chunks(2)
                        .filter_map(|kv| {
                            Some((string(&kv[0])?.to_string(), string(kv.get(1)?)?.to_string()))
                        })
                        .collect(),
                    ..Facts::default()
                }
            })
            .collect();
        let parent = |i: usize| {
            parents
                .get(i)
                .and_then(Value::as_i64)
                .filter(|p| *p >= 0)
                .map(|p| p as usize)
        };
        for i in 0..facts.len() {
            if facts[i].element && matches!(facts[i].tag.as_str(), "input" | "select" | "textarea")
            {
                let mut up = parent(i);
                for _ in 0..2 {
                    let Some(p) = up.filter(|p| *p < facts.len()) else {
                        break;
                    };
                    facts[p].control_below = true;
                    up = parent(p);
                }
            }
        }
        let layout = &document["layout"];
        let mut painted = Vec::new();
        for (l, index) in list(&layout["nodeIndex"]).iter().enumerate() {
            let Some(i) = index
                .as_u64()
                .map(|i| i as usize)
                .filter(|i| *i < facts.len())
            else {
                continue;
            };
            let styles = list(&layout["styles"][l]);
            let style = |name: &str| {
                STYLES
                    .iter()
                    .position(|s| *s == name)
                    .and_then(|p| styles.get(p))
                    .and_then(&string)
                    .unwrap_or_default()
                    .to_lowercase()
            };
            let bounds = Rect::of(&layout["bounds"][l]);
            let opacity: f64 = style("opacity").parse().unwrap_or(1.0);
            let f = &mut facts[i];
            f.bounds = bounds;
            f.visible = bounds.is_some()
                && style("display") != "none"
                && style("visibility") != "hidden"
                && opacity > 0.0;
            f.pointer = style("cursor") == "pointer";
            if let (Some(scroll), Some(client)) = (
                Rect::of(&layout["scrollRects"][l]),
                Rect::of(&layout["clientRects"][l]),
            ) {
                let overflows = scroll.h > client.h + 1.0 || scroll.w > client.w + 1.0;
                let overflow = style("overflow");
                let allows = [overflow.clone(), style("overflow-x"), style("overflow-y")]
                    .iter()
                    .any(|o| matches!(o.as_str(), "auto" | "scroll" | "overlay"));
                if overflows && allows && client.h > 0.0 {
                    // A scroll rect's x and y are how far it has scrolled.
                    let above = scroll.y / client.h;
                    let below = ((scroll.h - client.h - scroll.y) / client.h).max(0.0);
                    f.scroll = Some((above, below));
                }
            }
            if let (Some(order), Some(bounds)) = (layout["paintOrders"][l].as_i64(), bounds) {
                let background = style("background-color");
                let opaque =
                    !background.is_empty() && background != "rgba(0, 0, 0, 0)" && opacity >= 0.8;
                painted.push((order, i, bounds, opaque));
            }
        }
        for i in covered(&painted) {
            facts[i].covered = true;
        }
        for (backend, f) in backends.iter().zip(facts) {
            if let Some(backend) = backend.as_i64() {
                dom.insert(backend, f);
            }
        }
    }
    dom
}

/// Of `painted` (paint order, node, box, opaque), the nodes an opaque box painted later covers
/// whole.
fn covered(painted: &[(i64, usize, Rect, bool)]) -> Vec<usize> {
    let mut by_order: BTreeMap<i64, Vec<(usize, Rect, bool)>> = BTreeMap::new();
    for &(order, i, rect, opaque) in painted {
        by_order.entry(order).or_default().push((i, rect, opaque));
    }
    let mut union = Union::default();
    let mut out = Vec::new();
    for group in by_order.values().rev() {
        for &(i, rect, _) in group {
            if union.contains(Area::of(rect)) {
                out.push(i);
            }
        }
        for &(_, rect, opaque) in group {
            if opaque {
                union.add(Area::of(rect));
            }
        }
    }
    out
}

/// A closed box by its corners.
#[derive(Clone, Copy)]
struct Area {
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
}

impl Area {
    fn of(r: Rect) -> Area {
        Area {
            x1: r.x,
            y1: r.y,
            x2: r.x + r.w,
            y2: r.y + r.h,
        }
    }

    fn intersects(&self, o: &Area) -> bool {
        !(self.x2 <= o.x1 || o.x2 <= self.x1 || self.y2 <= o.y1 || o.y2 <= self.y1)
    }

    fn contains(&self, o: &Area) -> bool {
        self.x1 <= o.x1 && self.y1 <= o.y1 && self.x2 >= o.x2 && self.y2 >= o.y2
    }

    /// `self` less `b`, in up to four boxes; `self` intersects `b`.
    fn minus(&self, b: &Area) -> Vec<Area> {
        let a = self;
        let mut parts = Vec::new();
        if a.y1 < b.y1 {
            parts.push(Area { y2: b.y1, ..*a });
        }
        if b.y2 < a.y2 {
            parts.push(Area { y1: b.y2, ..*a });
        }
        let (y1, y2) = (a.y1.max(b.y1), a.y2.min(b.y2));
        if a.x1 < b.x1 {
            parts.push(Area {
                x1: a.x1,
                y1,
                x2: b.x1,
                y2,
            });
        }
        if b.x2 < a.x2 {
            parts.push(Area {
                x1: b.x2,
                y1,
                x2: a.x2,
                y2,
            });
        }
        parts
    }
}

/// Disjoint boxes; past [`Union::MAX`] it grows no more, and covers less than it could.
#[derive(Default)]
struct Union {
    areas: Vec<Area>,
}

impl Union {
    const MAX: usize = 5000;

    fn contains(&self, r: Area) -> bool {
        if self.areas.is_empty() {
            return false;
        }
        let mut left = vec![r];
        for s in &self.areas {
            let mut next = Vec::new();
            for piece in left {
                if s.contains(&piece) {
                    continue;
                }
                if piece.intersects(s) {
                    next.extend(piece.minus(s));
                } else {
                    next.push(piece);
                }
            }
            if next.is_empty() {
                return true;
            }
            left = next;
        }
        false
    }

    fn add(&mut self, r: Area) {
        if self.areas.len() >= Self::MAX || self.contains(r) {
            return;
        }
        let mut pending = vec![r];
        for s in &self.areas {
            pending = pending
                .into_iter()
                .flat_map(|p| if p.intersects(s) { p.minus(s) } else { vec![p] })
                .collect();
        }
        self.areas.extend(pending);
    }
}

/// Whether the element `f`, whose accessibility node is `node`, is one to act on.
fn interactive(node: &Value, role: &str, f: &Facts) -> bool {
    if !f.element || matches!(f.tag.as_str(), "html" | "body") {
        return false;
    }
    if f.listener {
        return true;
    }
    if matches!(f.tag.as_str(), "iframe" | "frame")
        && f.bounds.is_some_and(|b| b.w > 100.0 && b.h > 100.0)
    {
        return true;
    }
    if f.tag == "label" {
        if f.attr("for").is_some_and(|v| !v.is_empty()) {
            return false;
        }
        if f.control_below {
            return true;
        }
    }
    if f.tag == "span" && f.control_below {
        return true;
    }
    const SEARCH: [&str; 10] = [
        "search",
        "magnify",
        "glass",
        "lookup",
        "find",
        "query",
        "search-icon",
        "search-btn",
        "search-button",
        "searchbox",
    ];
    let has_search = |v: &str| {
        let v = v.to_lowercase();
        SEARCH.iter().any(|s| v.contains(s))
    };
    if f.attr("class").is_some_and(has_search)
        || f.attr("id").is_some_and(has_search)
        || f.attrs
            .iter()
            .any(|(k, v)| k.starts_with("data-") && has_search(v))
    {
        return true;
    }
    for property in node["properties"].as_array().into_iter().flatten() {
        let value = &property["value"]["value"];
        let truthy = match value {
            Value::Bool(b) => *b,
            Value::String(s) => !s.is_empty() && s != "false",
            Value::Null => false,
            _ => true,
        };
        match property["name"].as_str().unwrap_or_default() {
            "disabled" | "hidden" if truthy => return false,
            "focusable" | "editable" | "settable" if truthy => return true,
            "checked" | "expanded" | "pressed" | "selected" => return true,
            "required" | "autocomplete" | "keyshortcuts" if truthy => return true,
            _ => {}
        }
    }
    if matches!(
        f.tag.as_str(),
        "button"
            | "input"
            | "select"
            | "textarea"
            | "a"
            | "details"
            | "summary"
            | "option"
            | "optgroup"
    ) {
        return true;
    }
    if [
        "onclick",
        "onmousedown",
        "onmouseup",
        "onkeydown",
        "onkeyup",
        "tabindex",
    ]
    .iter()
    .any(|a| f.attr(a).is_some())
    {
        return true;
    }
    const ROLES: [&str; 16] = [
        "button",
        "link",
        "menuitem",
        "option",
        "radio",
        "checkbox",
        "tab",
        "textbox",
        "combobox",
        "slider",
        "spinbutton",
        "search",
        "searchbox",
        "row",
        "cell",
        "gridcell",
    ];
    if f.attr("role").is_some_and(|r| ROLES.contains(&r))
        || ROLES.contains(&role)
        || role == "listbox"
    {
        return true;
    }
    if let Some(b) = f.bounds
        && (10.0..=50.0).contains(&b.w)
        && (10.0..=50.0).contains(&b.h)
        && ["class", "role", "onclick", "data-action", "aria-label"]
            .iter()
            .any(|a| f.attr(a).is_some())
    {
        return true;
    }
    f.pointer
}

/// Whether `f` passes its box on to what is inside it, which then gets no ref of its own.
fn propagates(f: &Facts) -> bool {
    let role = f.attr("role").unwrap_or_default();
    match f.tag.as_str() {
        "a" | "button" => true,
        "div" | "span" => matches!(role, "button" | "combobox"),
        "input" => role == "combobox",
        _ => false,
    }
}

/// Whether `f` lies inside `outer`, a box that [`propagates`], as one of its parts.
fn part_of(f: &Facts, outer: Option<Rect>) -> bool {
    let (Some(outer), Some(bounds)) = (outer, f.bounds) else {
        return false;
    };
    f.element
        && bounds.inside(&outer) >= 0.99
        && !matches!(f.tag.as_str(), "input" | "select" | "textarea" | "label")
        && !propagates(f)
        && f.attr("onclick").is_none()
        && f.attr("aria-label").is_none_or(|l| l.trim().is_empty())
        && !f.attr("role").is_some_and(|r| {
            matches!(
                r,
                "button" | "link" | "checkbox" | "radio" | "tab" | "menuitem" | "option"
            )
        })
}

/// Whether a box that scrolls is a dropdown's list, which gets a ref though what is in it has
/// refs too.
fn dropdown_list(f: &Facts) -> bool {
    let role = f.attr("role").unwrap_or_default().to_lowercase();
    let class = f.attr("class").unwrap_or_default().to_lowercase();
    let classes: Vec<&str> = class.split_whitespace().collect();
    matches!(
        role.as_str(),
        "listbox" | "menu" | "combobox" | "menubar" | "tree" | "grid"
    ) || f.tag == "select"
        || classes
            .iter()
            .any(|c| matches!(*c, "dropdown" | "dropdown-menu" | "select-menu"))
        || (classes.contains(&"ui") && class.contains("dropdown"))
}

/// The format a date or time input's value is typed in.
fn input_format(f: &Facts) -> Option<&'static str> {
    if f.tag != "input" {
        return None;
    }
    Some(match f.attr("type")?.to_lowercase().as_str() {
        "date" => "YYYY-MM-DD",
        "time" => "HH:MM",
        "datetime-local" => "YYYY-MM-DDTHH:MM",
        "month" => "YYYY-MM",
        "week" => "YYYY-W##",
        _ => return None,
    })
}

/// One document's accessibility tree: the page's, or a frame's.
pub(super) struct Doc {
    pub nodes: Vec<Value>,
    /// The DOM of the process it is in; empty when that could not be read.
    pub dom: Arc<Dom>,
    /// Put before a backend id in a ref, to tell a frame in a process of its own apart.
    pub prefix: String,
}

impl Doc {
    pub(super) fn new(nodes: Vec<Value>) -> Doc {
        Doc {
            nodes,
            dom: Arc::default(),
            prefix: String::new(),
        }
    }
}

/// Where a ref points: the document it is in, and the node there.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct At {
    pub doc: usize,
    pub backend: i64,
}

/// How much of the page to render.
#[derive(Default)]
pub(super) struct Options {
    /// Only what has a ref, one per line, with the headings it sits under.
    pub interactive: bool,
    /// How deep the tree goes, counting shown lines.
    pub depth: Option<usize>,
    /// The page's own nodes to render, and everything below them; all when `None`.
    pub scope: Option<HashSet<i64>>,
    /// A link's address, shown after it.
    pub urls: HashMap<At, String>,
    /// Refs an earlier snapshot of this document gave; others are marked new.
    pub seen: HashSet<String>,
}

/// A rendered page and what its refs name: `e<backend id>` to a node in one of its documents.
pub(super) struct Snapshot {
    pub text: String,
    pub refs: HashMap<String, At>,
}

/// Renders `docs` (the page first), from `Accessibility.getFullAXTree`. `frames` puts a frame's
/// document below the node that holds it: the iframe at `At` holds document `usize`.
pub(super) fn render(docs: &[Doc], frames: &HashMap<At, usize>, options: &Options) -> Snapshot {
    let indexed: Vec<HashMap<&str, &Value>> = docs
        .iter()
        .map(|doc| {
            doc.nodes
                .iter()
                .filter_map(|n| Some((n["nodeId"].as_str()?, n)))
                .collect()
        })
        .collect();
    let mut out = Render {
        docs,
        indexed: &indexed,
        frames,
        options,
        lines: Vec::new(),
        refs: HashMap::new(),
    };
    if !docs.is_empty() {
        out.walk_doc(0, 0, options.scope.is_none());
    }
    Snapshot {
        text: out.lines.join("\n"),
        refs: out.refs,
    }
}

struct Render<'a> {
    docs: &'a [Doc],
    indexed: &'a [HashMap<&'a str, &'a Value>],
    frames: &'a HashMap<At, usize>,
    options: &'a Options,
    lines: Vec<String>,
    refs: HashMap<String, At>,
}

impl<'a> Render<'a> {
    fn walk_doc(&mut self, doc: usize, depth: usize, in_scope: bool) {
        if let Some(root) = self.docs[doc]
            .nodes
            .iter()
            .find(|n| n.get("parentId").is_none())
        {
            self.walk(doc, root, depth, None, in_scope, None);
        }
    }

    fn children(&self, doc: usize, node: &Value) -> Vec<&'a Value> {
        let indexed = &self.indexed[doc];
        node["childIds"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|id| id.as_str().and_then(|id| indexed.get(id).copied()))
            .collect()
    }

    /// `parent_name` is the name its nearest shown ancestor already carries, which a text child
    /// would only repeat. `outer` is the box of the nearest ancestor that [`propagates`].
    fn walk(
        &mut self,
        doc: usize,
        node: &Value,
        depth: usize,
        parent_name: Option<&str>,
        in_scope: bool,
        outer: Option<Rect>,
    ) {
        let role = node["role"]["value"].as_str().unwrap_or_default();
        if NOISE.contains(&role) {
            return;
        }
        let backend = node["backendDOMNodeId"].as_i64();
        let docs = self.docs;
        let facts = backend.and_then(|b| docs[doc].dom.get(&b));
        // Text hidden, or under something painted over it, is left out.
        if role == "StaticText" && facts.is_some_and(|f| !f.visible || f.covered) {
            return;
        }
        let name = clip(node["name"]["value"].as_str().unwrap_or_default().trim());
        let ignored = node["ignored"].as_bool().unwrap_or(false);
        let in_scope = in_scope
            || (doc == 0
                && backend.is_some_and(|b| {
                    self.options
                        .scope
                        .as_ref()
                        .is_some_and(|scope| scope.contains(&b))
                }));
        let acts = match facts {
            Some(f) => {
                f.element
                    && !f.covered
                    && !part_of(f, outer)
                    && (f.visible || is_file(f))
                    && interactive(node, role, f)
            }
            None => INTERACTIVE.contains(&role),
        };
        let scroll = facts
            .filter(|f| f.element && !matches!(f.tag.as_str(), "html" | "body"))
            .and_then(|f| f.scroll);
        let shown = in_scope
            && (acts || scroll.is_some() || (!ignored && self.shows(role, &name, parent_name)));
        if shown && self.options.depth.is_some_and(|max| depth >= max) {
            return;
        }
        let dropdown = facts.is_some_and(dropdown_list);
        let mut deferred = None;
        let (depth, name_below) = if shown {
            let refs_now = acts || (scroll.is_some() && dropdown);
            let (at, shown_name) =
                self.line(doc, node, role, &name, depth, facts, refs_now, scroll);
            if scroll.is_some() && !refs_now {
                deferred = Some((at, self.refs.len()));
            }
            (depth + 1, Some(shown_name).filter(|n| !n.is_empty()))
        } else {
            (depth, parent_name.map(str::to_string))
        };
        if let Some(&frame) = backend.and_then(|backend| self.frames.get(&At { doc, backend })) {
            self.walk_doc(frame, depth, in_scope);
        }
        // A form control's own parts (a date's fields, a select's options) are on its line.
        let folds =
            facts.is_some_and(|f| matches!(f.tag.as_str(), "input" | "select" | "textarea"));
        if !folds {
            let outer = match facts {
                Some(f) if propagates(f) && f.bounds.is_some() => f.bounds,
                _ => outer,
            };
            for child in self.children(doc, node) {
                self.walk(doc, child, depth, name_below.as_deref(), in_scope, outer);
            }
        }
        // A box that scrolls has a ref when nothing in it has one, to scroll it by.
        if let (Some((at, before)), Some(backend)) = (deferred, backend)
            && self.refs.len() == before
        {
            let mut line = std::mem::take(&mut self.lines[at]);
            self.give_ref(&mut line, doc, backend);
            self.lines[at] = line;
        }
    }

    fn shows(&self, role: &str, name: &str, parent_name: Option<&str>) -> bool {
        if self.options.interactive {
            return role == "heading";
        }
        match role {
            "StaticText" => !name.is_empty() && parent_name.is_none_or(|p| !p.contains(name)),
            _ if GROUPING.contains(&role) => !name.is_empty() && role != "RootWebArea",
            "" => false,
            _ => true,
        }
    }

    /// Adds `node`'s line; says where it is, and the name it shows.
    #[allow(clippy::too_many_arguments)]
    fn line(
        &mut self,
        doc: usize,
        node: &Value,
        role: &str,
        name: &str,
        depth: usize,
        facts: Option<&Facts>,
        gives_ref: bool,
        scroll: Option<(f64, f64)>,
    ) -> (usize, String) {
        let indent = if self.options.interactive {
            String::new()
        } else {
            "  ".repeat(depth)
        };
        let element = facts.filter(|f| f.element);
        let ignored = node["ignored"].as_bool().unwrap_or(false);
        let label = match element {
            Some(f) if ignored || role.is_empty() || GROUPING.contains(&role) => f.tag.as_str(),
            _ if role == "StaticText" => "text",
            _ => role,
        };
        let name = if name.is_empty() && gives_ref {
            clip(&self.text_below(doc, node))
        } else {
            name.to_string()
        };
        let mut line = format!("{indent}- {label}");
        if !name.is_empty() {
            line.push_str(&format!(" {name:?}"));
        }
        let value = node["value"]["value"].as_str().filter(|v| !v.is_empty());
        if let Some(value) = value
            && value != name
        {
            line.push_str(&format!(": {:?}", clip(value)));
        }
        if let Some(f) = element.filter(|_| gives_ref) {
            line.push_str(&attributes(f, &name, value, role));
        }
        if element.is_some_and(|f| f.tag == "select") {
            let options = self.options_below(doc, node);
            let mut first: Vec<&str> = options.iter().take(4).map(String::as_str).collect();
            if options.len() > 4 {
                first.push("…");
            }
            line.push_str(&format!(
                " [{} options: {}]",
                options.len(),
                first.join("|")
            ));
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
        if let Some((above, below)) = scroll {
            line.push_str(&format!(
                " [scroll: {above:.1} pages above, {below:.1} below]"
            ));
        }
        if let Some(backend) = node["backendDOMNodeId"].as_i64() {
            if gives_ref {
                self.give_ref(&mut line, doc, backend);
            }
            if let Some(url) = self.options.urls.get(&At { doc, backend }) {
                line.push_str(&format!(" -> {}", clip(url)));
            }
        }
        self.lines.push(line);
        (self.lines.len() - 1, name)
    }

    fn give_ref(&mut self, line: &mut String, doc: usize, backend: i64) {
        let r#ref = format!("e{}{backend}", self.docs[doc].prefix);
        if !self.options.seen.is_empty() && !self.options.seen.contains(&r#ref) {
            line.push_str(" [new]");
        }
        line.push_str(&format!(" [ref={ref}]"));
        self.refs.insert(r#ref, At { doc, backend });
    }

    /// The text below `node`, for a line that has no name of its own.
    fn text_below(&self, doc: usize, node: &Value) -> String {
        let mut words = Vec::new();
        let mut stack = self.children(doc, node);
        stack.reverse();
        while let Some(n) = stack.pop() {
            let role = n["role"]["value"].as_str().unwrap_or_default();
            if NOISE.contains(&role) {
                continue;
            }
            if role == "StaticText" {
                if let Some(text) = n["name"]["value"]
                    .as_str()
                    .map(str::trim)
                    .filter(|t| !t.is_empty())
                {
                    words.push(text.to_string());
                }
                continue;
            }
            let mut below = self.children(doc, n);
            below.reverse();
            stack.extend(below);
        }
        words.join(" ")
    }

    /// The names of the options below a select.
    fn options_below(&self, doc: usize, node: &Value) -> Vec<String> {
        let mut found = Vec::new();
        let mut stack = self.children(doc, node);
        stack.reverse();
        while let Some(n) = stack.pop() {
            let role = n["role"]["value"].as_str().unwrap_or_default();
            if matches!(role, "option" | "MenuListOption") {
                found.push(clip(n["name"]["value"].as_str().unwrap_or_default()));
                continue;
            }
            let mut below = self.children(doc, n);
            below.reverse();
            stack.extend(below);
        }
        found
    }
}

fn is_file(f: &Facts) -> bool {
    f.tag == "input"
        && f.attr("type")
            .is_some_and(|t| t.eq_ignore_ascii_case("file"))
}

/// `f`'s attributes from [`ATTRIBUTES`] as ` key=value …`, leaving out what the line says
/// already; a date or time input also says the format it takes.
fn attributes(f: &Facts, name: &str, value: Option<&str>, role: &str) -> String {
    let mut shown: Vec<String> = Vec::new();
    let format = input_format(f);
    if let Some(format) = format {
        shown.push(format!("format={format}"));
        if f.attr("placeholder").is_none() {
            shown.push(format!("placeholder={format}"));
        }
    }
    for key in ATTRIBUTES {
        let Some(v) = f.attr(key) else {
            continue;
        };
        let v = v.trim();
        if v.is_empty() {
            if matches!(*key, "multiple" | "contenteditable") {
                shown.push(key.to_string());
            }
            continue;
        }
        if v == name || Some(v) == value || (*key == "role" && v == role) {
            continue;
        }
        let v = clip(v);
        if v.chars().any(|c| c.is_whitespace() || c == '"') {
            shown.push(format!("{key}={v:?}"));
        } else {
            shown.push(format!("{key}={v}"));
        }
    }
    if shown.is_empty() {
        String::new()
    } else {
        format!(" {}", shown.join(" "))
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
        "{}\n… {left} more lines. Narrow it with `interactive`, `depth` or `selector`, or search it with `browser_search_page`.",
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
        let snapshot = render(&[Doc::new(page())], &HashMap::new(), &Options::default());
        assert_eq!(
            snapshot.text,
            "- heading \"Flights\" [level=1]\n\
             - textbox \"From\": \"Seoul\" [ref=e105]\n\
             - link \"Deals\" [ref=e106]\n\
             - text \"Prices change daily.\""
        );
        assert_eq!(
            snapshot.refs["e105"],
            At {
                doc: 0,
                backend: 105
            }
        );
        assert_eq!(
            snapshot.refs["e106"],
            At {
                doc: 0,
                backend: 106
            }
        );
    }

    #[test]
    fn interactive_keeps_headings_and_what_acts() {
        let options = Options {
            interactive: true,
            ..Options::default()
        };
        let snapshot = render(&[Doc::new(page())], &HashMap::new(), &options);
        assert_eq!(
            snapshot.text,
            "- heading \"Flights\" [level=1]\n\
             - textbox \"From\": \"Seoul\" [ref=e105]\n\
             - link \"Deals\" [ref=e106]"
        );
    }

    #[test]
    fn a_frame_reads_below_the_iframe_holding_it() {
        let mut page = vec![
            node("1", None, "RootWebArea", "Outer", &["2", "3"]),
            node("2", Some("1"), "heading", "Outer", &[]),
            node("3", Some("1"), "Iframe", "", &[]),
        ];
        page[1]["properties"] = json!([{ "name": "level", "value": { "value": 1 } }]);
        let frame = vec![
            node("10", None, "RootWebArea", "Inner", &["11"]),
            node("11", Some("10"), "button", "Inner", &[]),
        ];
        let docs = [Doc::new(page), Doc::new(frame)];
        let frames = HashMap::from([(
            At {
                doc: 0,
                backend: 103,
            },
            1,
        )]);
        let snapshot = render(&docs, &frames, &Options::default());
        assert_eq!(
            snapshot.text,
            "- heading \"Outer\" [level=1]\n- Iframe\n  - button \"Inner\" [ref=e111]"
        );
        assert_eq!(
            snapshot.refs["e111"],
            At {
                doc: 1,
                backend: 111
            }
        );

        let shallow = Options {
            depth: Some(1),
            ..Options::default()
        };
        let snapshot = render(&docs, &frames, &shallow);
        assert!(!snapshot.text.contains("Inner"), "{}", snapshot.text);
    }

    #[test]
    fn a_scope_keeps_its_subtree_and_a_link_shows_where_it_goes() {
        let docs = [Doc::new(page())];
        let scoped = Options {
            scope: Some([104, 105, 106, 108].into()),
            urls: HashMap::from([(
                At {
                    doc: 0,
                    backend: 106,
                },
                "https://x.test/deals".into(),
            )]),
            ..Options::default()
        };
        let snapshot = render(&docs, &HashMap::new(), &scoped);
        assert_eq!(
            snapshot.text,
            "- textbox \"From\": \"Seoul\" [ref=e105]\n- link \"Deals\" [ref=e106] -> https://x.test/deals"
        );
    }

    #[test]
    fn the_dom_gives_refs_to_what_acts_and_folds_a_select() {
        let element = |tag: &str, attrs: &[(&str, &str)]| Facts {
            element: true,
            tag: tag.into(),
            attrs: attrs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            bounds: Some(Rect {
                x: 0.0,
                y: 0.0,
                w: 80.0,
                h: 20.0,
            }),
            visible: true,
            ..Facts::default()
        };
        let nodes = vec![
            node("1", None, "RootWebArea", "Shop", &["2", "4", "7"]),
            node("2", Some("1"), "generic", "", &["3"]),
            node("3", Some("2"), "StaticText", "Add to cart", &[]),
            node("4", Some("1"), "combobox", "Sort", &["5"]),
            node("5", Some("4"), "MenuListPopup", "", &["6"]),
            node("6", Some("5"), "option", "Price", &[]),
            node("7", Some("1"), "StaticText", "Under the banner", &[]),
        ];
        let mut listened = element("div", &[("id", "buy")]);
        listened.listener = true;
        let covered_text = Facts {
            visible: true,
            covered: true,
            ..Facts::default()
        };
        let dom = Dom::from([
            (102, listened),
            (
                103,
                Facts {
                    visible: true,
                    ..Facts::default()
                },
            ),
            (104, element("select", &[("name", "sort")])),
            (107, covered_text),
        ]);
        let docs = [Doc {
            nodes,
            dom: Arc::new(dom),
            prefix: String::new(),
        }];
        let options = Options {
            seen: ["e104".to_string()].into(),
            ..Options::default()
        };
        let snapshot = render(&docs, &HashMap::new(), &options);
        assert_eq!(
            snapshot.text,
            "- div \"Add to cart\" id=buy [new] [ref=e102]\n\
             - combobox \"Sort\" name=sort [1 options: Price] [ref=e104]"
        );
    }
}
