use async_trait::async_trait;
use serde_json::{Value, json};

use crate::vfs::accessor::{NotionAccessor, NotionConfig};
use crate::vfs::path::VPath;
use crate::vfs::resource::{DirEntry, FileKind, FileStat, Resource};

const NOTION_PROMPT: &str = "\
Notion (read + write). Layout:
  pages/<title>__<page-id>/page.json   — page metadata, markdown body, raw blocks
List `pages/` to discover pages; the <page-id> is the part after the last `__`.
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

    async fn page_dir_names(&self) -> anyhow::Result<Vec<String>> {
        let pages = self.accessor.search_pages().await?;
        Ok(pages
            .iter()
            .filter_map(|p| {
                let id = p.get("id")?.as_str()?;
                Some(format!("{}__{}", sanitize(&page_title(p)), id))
            })
            .collect())
    }

    async fn render_page_json(&self, id: &str) -> anyhow::Result<Vec<u8>> {
        let page = self.accessor.get_page(id).await?;
        let blocks = self.accessor.list_children(id).await?;
        let markdown = blocks_to_markdown(&blocks);
        let out = json!({
            "title": page_title(&page),
            "id": id,
            "url": page.get("url").and_then(|v| v.as_str()).unwrap_or(""),
            "markdown": markdown,
            "blocks": blocks,
        });
        Ok(serde_json::to_vec_pretty(&out)?)
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
        if segs.len() == 3 && segs[0] == "pages" && segs[2] == "page.json" {
            let id = page_id(&segs[1]);
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
                .page_dir_names()
                .await?
                .into_iter()
                .map(|n| dir(&n))
                .collect()),
            [p, _page] if p == "pages" => Ok(vec![file("page.json", 0)]),
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
            [p, _page] if p == "pages" => Ok(FileStat {
                kind: FileKind::Dir,
                size: 0,
            }),
            [p, _page, f] if p == "pages" && f == "page.json" => {
                let id = page_id(&segs[1]);
                let size = self.render_page_json(&id).await?.len() as u64;
                Ok(FileStat {
                    kind: FileKind::File,
                    size,
                })
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

fn page_id(dir_name: &str) -> String {
    dir_name.rsplit_once("__").map(|(_, id)| id).unwrap_or(dir_name).to_string()
}

fn page_title(page: &Value) -> String {
    let props = match page.get("properties").and_then(|p| p.as_object()) {
        Some(p) => p,
        None => return "untitled".to_string(),
    };
    for prop in props.values() {
        if prop.get("type").and_then(|t| t.as_str()) == Some("title") {
            if let Some(arr) = prop.get("title").and_then(|t| t.as_array()) {
                let s: String = arr
                    .iter()
                    .filter_map(|t| t.get("plain_text").and_then(|p| p.as_str()))
                    .collect();
                if !s.is_empty() {
                    return s;
                }
            }
        }
    }
    "untitled".to_string()
}

fn sanitize(title: &str) -> String {
    let s: String = title
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' { c } else { '_' })
        .collect();
    let trimmed = s.trim_matches('_');
    if trimmed.is_empty() {
        "untitled".to_string()
    } else {
        trimmed.to_string()
    }
}

fn blocks_to_markdown(blocks: &[Value]) -> String {
    let mut out = String::new();
    for b in blocks {
        let kind = b.get("type").and_then(|t| t.as_str()).unwrap_or("");
        let text = b
            .get(kind)
            .and_then(|inner| inner.get("rich_text"))
            .and_then(|rt| rt.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|t| t.get("plain_text").and_then(|p| p.as_str()))
                    .collect::<String>()
            })
            .unwrap_or_default();
        let line = match kind {
            "heading_1" => format!("# {text}"),
            "heading_2" => format!("## {text}"),
            "heading_3" => format!("### {text}"),
            "bulleted_list_item" | "numbered_list_item" => format!("- {text}"),
            _ => text,
        };
        out.push_str(&line);
        out.push('\n');
    }
    out
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
