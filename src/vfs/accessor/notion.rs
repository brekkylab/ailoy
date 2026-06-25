use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const API: &str = "https://api.notion.com/v1";
const NOTION_VERSION: &str = "2022-06-28";
/// Recursion ceiling for `list_block_tree`, mirroring mirage's `MAX_BLOCK_DEPTH`.
const MAX_BLOCK_DEPTH: usize = 10;

#[derive(Clone, Serialize, Deserialize)]
pub struct NotionConfig {
    pub api_key: String,
}

/// Holds the Notion API client (one token). Mirrors mirage `accessor/notion.py`.
pub struct NotionAccessor {
    client: reqwest::Client,
    api_key: String,
}

impl NotionAccessor {
    pub fn new(config: &NotionConfig) -> anyhow::Result<Self> {
        Ok(Self {
            client: reqwest::Client::new(),
            api_key: config.api_key.clone(),
        })
    }

    async fn send(&self, req: reqwest::RequestBuilder) -> anyhow::Result<Value> {
        // Content-Type is set by `.json(...)` on the builder; do not add a
        // second one here (duplicate Content-Type makes Notion ignore the body).
        let resp = req
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Notion-Version", NOTION_VERSION)
            .send()
            .await?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            anyhow::bail!("notion API {status}: {body}");
        }
        Ok(serde_json::from_str(&body).unwrap_or(Value::Null))
    }

    /// Pages shared with the integration (search, filtered to pages), paging
    /// through every result. Mirrors mirage `paginate_post("/search", ...)`.
    pub async fn search_pages(&self) -> anyhow::Result<Vec<Value>> {
        let mut results = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            let mut body = json!({
                "filter": {"property": "object", "value": "page"},
                "page_size": 100,
            });
            if let Some(c) = &cursor {
                body["start_cursor"] = json!(c);
            }
            let v = self
                .send(self.client.post(format!("{API}/search")).json(&body))
                .await?;
            if let Some(arr) = v.get("results").and_then(|r| r.as_array()) {
                results.extend(arr.iter().cloned());
            }
            if !v.get("has_more").and_then(|h| h.as_bool()).unwrap_or(false) {
                break;
            }
            match v.get("next_cursor").and_then(|c| c.as_str()) {
                Some(c) => cursor = Some(c.to_string()),
                None => break,
            }
        }
        Ok(results)
    }

    pub async fn get_page(&self, id: &str) -> anyhow::Result<Value> {
        self.send(self.client.get(format!("{API}/pages/{id}")))
            .await
    }

    /// All immediate block children of `id`, paging through every result.
    /// Mirrors mirage `list_block_children` / `paginate_list`.
    pub async fn list_children(&self, id: &str) -> anyhow::Result<Vec<Value>> {
        let mut results = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            let mut url = format!("{API}/blocks/{id}/children?page_size=100");
            if let Some(c) = &cursor {
                url.push_str("&start_cursor=");
                url.push_str(c);
            }
            let v = self.send(self.client.get(url)).await?;
            if let Some(arr) = v.get("results").and_then(|r| r.as_array()) {
                results.extend(arr.iter().cloned());
            }
            if !v.get("has_more").and_then(|h| h.as_bool()).unwrap_or(false) {
                break;
            }
            match v.get("next_cursor").and_then(|c| c.as_str()) {
                Some(c) => cursor = Some(c.to_string()),
                None => break,
            }
        }
        Ok(results)
    }

    /// List block children recursively, embedding nested blocks under a
    /// `children` key. Blocks of type `child_page`/`child_database` are not
    /// descended into (their children belong to a different page). Recursion
    /// stops at [`MAX_BLOCK_DEPTH`]. Mirrors mirage `list_block_tree`.
    pub async fn list_block_tree(&self, id: &str) -> anyhow::Result<Vec<Value>> {
        self.list_block_tree_depth(id.to_string(), 0).await
    }

    fn list_block_tree_depth(
        &self,
        id: String,
        depth: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<Vec<Value>>> + Send + '_>>
    {
        Box::pin(async move {
            let mut blocks = self.list_children(&id).await?;
            if depth >= MAX_BLOCK_DEPTH {
                return Ok(blocks);
            }
            for block in &mut blocks {
                let btype = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
                if btype == "child_page" || btype == "child_database" {
                    continue;
                }
                let has_children = block
                    .get("has_children")
                    .and_then(|h| h.as_bool())
                    .unwrap_or(false);
                if has_children {
                    let child_id = block
                        .get("id")
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string();
                    let children = self.list_block_tree_depth(child_id, depth + 1).await?;
                    block["children"] = Value::Array(children);
                }
            }
            Ok(blocks)
        })
    }

    pub async fn create_page(&self, body: Value) -> anyhow::Result<Value> {
        self.send(self.client.post(format!("{API}/pages")).json(&body))
            .await
    }

    pub async fn append_blocks(&self, block_id: &str, children: Value) -> anyhow::Result<Value> {
        let body = json!({ "children": children });
        self.send(
            self.client
                .patch(format!("{API}/blocks/{block_id}/children"))
                .json(&body),
        )
        .await
    }

    pub async fn add_comment(&self, body: Value) -> anyhow::Result<Value> {
        self.send(self.client.post(format!("{API}/comments")).json(&body))
            .await
    }
}
