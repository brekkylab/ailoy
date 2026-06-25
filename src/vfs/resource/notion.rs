use async_trait::async_trait;
use serde_json::{Value, json};

use crate::vfs::{
    accessor::{NotionAccessor, NotionConfig},
    path::VPath,
    resource::{DirEntry, FileKind, FileStat, Resource},
};

const NOTION_PROMPT: &str = "\
Notion (read + write). Mirrors the workspace page tree:
  pages/<title>__<page-id>/page.json     — page metadata, markdown body, raw blocks
  pages/<title>__<page-id>/<child>__<id>/ — nested sub-pages, recursively
`pages/` lists only top-level (workspace) pages; descend a page dir to find its
sub-pages. The <page-id> is the part after the last `__`.
Domain writes (write JSON to the control path):
  echo '{\"parent\":{\"page_id\":\"ID\"},\"properties\":{\"title\":[{\"text\":{\"content\":\"T\"}}]}}' > .cmd/page-create
  echo '{\"block_id\":\"ID\",\"children\":[{\"object\":\"block\",\"type\":\"paragraph\",\"paragraph\":{\"rich_text\":[{\"type\":\"text\",\"text\":{\"content\":\"hi\"}}]}}]}' > .cmd/block-append";

pub struct NotionResource {
    accessor: NotionAccessor,
}

impl NotionResource {
    pub fn new(config: &NotionConfig) -> anyhow::Result<Self> {
        Ok(Self {
            accessor: NotionAccessor::new(config)?,
        })
    }

    /// Top-level (workspace) pages as `<title>__<id>` directory names.
    async fn top_level_page_dirs(&self) -> anyhow::Result<Vec<String>> {
        let pages = self.accessor.search_pages().await?;
        Ok(pages
            .iter()
            .filter(|p| {
                p.get("parent")
                    .and_then(|x| x.get("type"))
                    .and_then(|t| t.as_str())
                    == Some("workspace")
            })
            .map(page_dirname)
            .collect())
    }

    /// Contents of a page directory: its `page.json` plus a subdirectory per
    /// `child_page` block.
    async fn page_dir_entries(&self, page_id: &str) -> anyhow::Result<Vec<DirEntry>> {
        let blocks = self.accessor.list_children(page_id).await?;
        let mut out = vec![file("page.json", 0)];
        for b in &blocks {
            if b.get("type").and_then(|t| t.as_str()) != Some("child_page") {
                continue;
            }
            let child_title = b
                .get("child_page")
                .and_then(|c| c.get("title"))
                .and_then(|t| t.as_str())
                .unwrap_or("untitled");
            let child_id = b.get("id").and_then(|i| i.as_str()).unwrap_or("");
            out.push(dir(&format!("{}__{}", sanitize_name(child_title), child_id)));
        }
        Ok(out)
    }

    async fn render_page_json(&self, id: &str) -> anyhow::Result<Vec<u8>> {
        let page = self.accessor.get_page(id).await?;
        let blocks = self.accessor.list_block_tree(id).await?;
        let normalized = normalize_page(&page, &blocks);
        Ok(serde_json::to_vec_pretty(&normalized)?)
    }
}

#[async_trait]
impl Resource for NotionResource {
    async fn read_bytes(
        &self,
        path: &VPath,
        range: Option<std::ops::Range<u64>>,
    ) -> anyhow::Result<Vec<u8>> {
        let segs = segments(path);
        if segs.len() >= 3 && segs[0] == "pages" && segs.last().map(String::as_str) == Some("page.json")
        {
            let id = page_id(&segs[segs.len() - 2]);
            let data = self.render_page_json(&id).await?;
            return Ok(slice(data, range));
        }
        anyhow::bail!("not a readable notion file: {}", path.as_str())
    }

    async fn write_bytes(&self, path: &VPath, _data: Vec<u8>) -> anyhow::Result<()> {
        anyhow::bail!(
            "notion is read-only for file writes; use the .cmd/ control path \
             (path was {})",
            path.as_str()
        )
    }

    async fn readdir(&self, path: &VPath) -> anyhow::Result<Vec<DirEntry>> {
        let segs = segments(path);
        match segs.as_slice() {
            [] => Ok(vec![dir("pages")]),
            [p] if p == "pages" => Ok(self
                .top_level_page_dirs()
                .await?
                .into_iter()
                .map(|n| dir(&n))
                .collect()),
            [p, rest @ ..] if p == "pages" && !rest.is_empty() => {
                let last = rest.last().unwrap();
                if last == "page.json" {
                    anyhow::bail!("not a notion directory: {}", path.as_str());
                }
                self.page_dir_entries(&page_id(last)).await
            }
            _ => anyhow::bail!("not a notion directory: {}", path.as_str()),
        }
    }

    async fn stat(&self, path: &VPath) -> anyhow::Result<FileStat> {
        let segs = segments(path);
        match segs.as_slice() {
            [] | [_] => Ok(FileStat {
                kind: FileKind::Dir,
                size: 0,
            }),
            [p, rest @ ..] if p == "pages" && !rest.is_empty() => {
                if rest.last().map(String::as_str) == Some("page.json") {
                    let id = page_id(&rest[rest.len() - 2]);
                    let size = self.render_page_json(&id).await?.len() as u64;
                    Ok(FileStat {
                        kind: FileKind::File,
                        size,
                    })
                } else {
                    Ok(FileStat {
                        kind: FileKind::Dir,
                        size: 0,
                    })
                }
            }
            _ => anyhow::bail!("notion path not found: {}", path.as_str()),
        }
    }

    async fn command(&self, name: &str, body: &[u8]) -> anyhow::Result<Vec<u8>> {
        let v: Value = serde_json::from_slice(body)
            .map_err(|e| anyhow::anyhow!("notion {name}: invalid JSON body: {e}"))?;
        let result = match name {
            "page-create" => self.accessor.create_page(v).await?,
            "block-append" => {
                let block_id = v
                    .get("block_id")
                    .and_then(|x| x.as_str())
                    .ok_or_else(|| anyhow::anyhow!("block-append: missing block_id"))?;
                let children = v
                    .get("children")
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("block-append: missing children"))?;
                self.accessor.append_blocks(block_id, children).await?
            }
            "comment-add" => self.accessor.add_comment(v).await?,
            other => anyhow::bail!("unknown notion command: {other}"),
        };
        Ok(serde_json::to_vec(&result)?)
    }

    fn prompt(&self) -> &str {
        NOTION_PROMPT
    }
}

fn segments(path: &VPath) -> Vec<String> {
    path.as_str()
        .trim_matches('/')
        .split('/')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

/// Page id encoded as the part after the last `__` in a directory name.
fn page_id(dir_name: &str) -> String {
    dir_name
        .rsplit_once("__")
        .map(|(_, id)| id)
        .unwrap_or(dir_name)
        .to_string()
}

/// Directory name for a page: `<sanitized-title>__<id>`, falling back to
/// `untitled` when the page has no title.
fn page_dirname(page: &Value) -> String {
    let title = extract_title(page);
    let id = page.get("id").and_then(|v| v.as_str()).unwrap_or("");
    let label = if title.is_empty() {
        "untitled".to_string()
    } else {
        sanitize_name(&title)
    };
    format!("{label}__{id}")
}

/// Concatenated plain text of the page's `title` property (returns "" when
/// there is no title property).
fn extract_title(page: &Value) -> String {
    let props = match page.get("properties").and_then(|p| p.as_object()) {
        Some(p) => p,
        None => return String::new(),
    };
    for prop in props.values() {
        if prop.get("type").and_then(|t| t.as_str()) == Some("title") {
            return prop
                .get("title")
                .and_then(|t| t.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|t| t.get("plain_text").and_then(|p| p.as_str()))
                        .collect::<String>()
                })
                .unwrap_or_default();
        }
    }
    String::new()
}

/// Page metadata + markdown body + raw blocks. `child_page`/`child_database`
/// blocks are excluded from both `markdown` and `blocks` (they surface as
/// subdirectories instead).
fn normalize_page(page: &Value, blocks: &[Value]) -> Value {
    let parent = page.get("parent").cloned().unwrap_or_else(|| json!({}));
    let parent_type = parent.get("type").and_then(|t| t.as_str()).unwrap_or("");
    let parent_id = parent
        .get(parent_type)
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let content_blocks: Vec<Value> = blocks
        .iter()
        .filter(|b| {
            let t = b.get("type").and_then(|t| t.as_str()).unwrap_or("");
            t != "child_page" && t != "child_database"
        })
        .cloned()
        .collect();
    json!({
        "page_id": page.get("id").and_then(|v| v.as_str()).unwrap_or(""),
        "title": extract_title(page),
        "url": page.get("url").and_then(|v| v.as_str()).unwrap_or(""),
        "created_time": page.get("created_time").and_then(|v| v.as_str()).unwrap_or(""),
        "last_edited_time": page.get("last_edited_time").and_then(|v| v.as_str()).unwrap_or(""),
        "parent_type": parent_type,
        "parent_id": parent_id,
        "archived": page.get("archived").and_then(|v| v.as_bool()).unwrap_or(false),
        "created_by": page.get("created_by").and_then(|c| c.get("id")).and_then(|v| v.as_str()).unwrap_or(""),
        "last_edited_by": page.get("last_edited_by").and_then(|c| c.get("id")).and_then(|v| v.as_str()).unwrap_or(""),
        "markdown": blocks_to_markdown(&content_blocks),
        "blocks": content_blocks,
    })
}

/// Sanitize a name for use in a virtual path segment: keep word chars /
/// whitespace / `-` / `.`, replace the rest with `_`, fold spaces and runs of
/// `_`, strip, cap at 100 chars.
fn sanitize_name(name: &str) -> String {
    if name.trim().is_empty() {
        return "unknown".to_string();
    }
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c.is_whitespace() || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let cleaned = cleaned.replace(' ', "_");
    // Collapse runs of '_' into a single '_'.
    let mut folded = String::with_capacity(cleaned.len());
    let mut prev_underscore = false;
    for c in cleaned.chars() {
        if c == '_' {
            if !prev_underscore {
                folded.push(c);
            }
            prev_underscore = true;
        } else {
            folded.push(c);
            prev_underscore = false;
        }
    }
    let trimmed = folded.trim_matches('_');
    trimmed.chars().take(100).collect()
}

fn rich_text_to_md(list: &[Value]) -> String {
    let mut parts = String::new();
    for rt in list {
        let mut text = rt
            .get("plain_text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let flag = |k: &str| {
            rt.get("annotations")
                .and_then(|a| a.get(k))
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        };
        if flag("code") {
            text = format!("`{text}`");
        }
        if flag("bold") {
            text = format!("**{text}**");
        }
        if flag("italic") {
            text = format!("*{text}*");
        }
        if flag("strikethrough") {
            text = format!("~~{text}~~");
        }
        if let Some(href) = rt.get("href").and_then(|v| v.as_str())
            && !href.is_empty()
        {
            text = format!("[{text}]({href})");
        }
        parts.push_str(&text);
    }
    parts
}

fn block_to_md(block: &Value, indent: usize) -> String {
    let btype = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
    let content = block.get(btype).cloned().unwrap_or_else(|| json!({}));
    let rich_text = content
        .get("rich_text")
        .and_then(|r| r.as_array())
        .cloned()
        .unwrap_or_default();
    let text = rich_text_to_md(&rich_text);
    let prefix = "  ".repeat(indent);

    match btype {
        "paragraph" => format!("{prefix}{text}"),
        "heading_1" => format!("# {text}"),
        "heading_2" => format!("## {text}"),
        "heading_3" => format!("### {text}"),
        "bulleted_list_item" => format!("{prefix}- {text}"),
        "numbered_list_item" => format!("{prefix}1. {text}"),
        "to_do" => {
            let checked = content
                .get("checked")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let marker = if checked { "x" } else { " " };
            format!("{prefix}- [{marker}] {text}")
        }
        "toggle" => format!("{prefix}<details><summary>{text}</summary></details>"),
        "code" => {
            let language = content.get("language").and_then(|v| v.as_str()).unwrap_or("");
            format!("```{language}\n{text}\n```")
        }
        "quote" => format!("{prefix}> {text}"),
        "callout" => {
            let icon = content.get("icon");
            let emoji = if icon.and_then(|i| i.get("type")).and_then(|t| t.as_str()) == Some("emoji")
            {
                icon.and_then(|i| i.get("emoji"))
                    .and_then(|e| e.as_str())
                    .unwrap_or("")
            } else {
                ""
            };
            format!("{prefix}> {emoji} {text}")
        }
        "divider" => "---".to_string(),
        "image" => {
            let inner = content.get("type").and_then(|t| t.as_str()).unwrap_or("");
            let img = content.get(inner).cloned().unwrap_or_else(|| json!({}));
            let url = img.get("url").and_then(|v| v.as_str()).unwrap_or("");
            let caption = rich_text_to_md(
                &content
                    .get("caption")
                    .and_then(|c| c.as_array())
                    .cloned()
                    .unwrap_or_default(),
            );
            format!("![{caption}]({url})")
        }
        "bookmark" => {
            let url = content.get("url").and_then(|v| v.as_str()).unwrap_or("");
            let caption = rich_text_to_md(
                &content
                    .get("caption")
                    .and_then(|c| c.as_array())
                    .cloned()
                    .unwrap_or_default(),
            );
            let label = if caption.is_empty() {
                url.to_string()
            } else {
                caption
            };
            format!("[{label}]({url})")
        }
        "equation" => {
            let expr = content.get("expression").and_then(|v| v.as_str()).unwrap_or("");
            format!("$${expr}$$")
        }
        "table_of_contents" => "[TOC]".to_string(),
        "child_page" | "child_database" => String::new(),
        _ => {
            if text.is_empty() {
                String::new()
            } else {
                format!("{prefix}{text}")
            }
        }
    }
}

fn walk_block(block: &Value, indent: usize, lines: &mut Vec<String>) {
    let line = block_to_md(block, indent);
    let btype = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
    if !line.is_empty() || btype == "paragraph" {
        lines.push(line);
    }
    if let Some(children) = block.get("children").and_then(|c| c.as_array()) {
        for child in children {
            walk_block(child, indent + 1, lines);
        }
    }
}

fn blocks_to_markdown(blocks: &[Value]) -> String {
    let mut lines: Vec<String> = Vec::new();
    for b in blocks {
        walk_block(b, 0, &mut lines);
    }
    if lines.is_empty() {
        String::new()
    } else {
        format!("{}\n", lines.join("\n\n"))
    }
}

fn slice(data: Vec<u8>, range: Option<std::ops::Range<u64>>) -> Vec<u8> {
    match range {
        Some(r) => {
            let start = (r.start as usize).min(data.len());
            let end = (r.end as usize).min(data.len());
            data[start..end].to_vec()
        }
        None => data,
    }
}

fn dir(name: &str) -> DirEntry {
    DirEntry {
        name: name.to_string(),
        kind: FileKind::Dir,
        size: 0,
    }
}

fn file(name: &str, size: u64) -> DirEntry {
    DirEntry {
        name: name.to_string(),
        kind: FileKind::File,
        size,
    }
}
