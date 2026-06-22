use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const API: &str = "https://api.notion.com/v1";
const NOTION_VERSION: &str = "2022-06-28";

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

    /// Pages shared with the integration (search, filtered to pages).
    pub async fn search_pages(&self) -> anyhow::Result<Vec<Value>> {
        let body = json!({"filter": {"property": "object", "value": "page"}});
        let v = self
            .send(self.client.post(format!("{API}/search")).json(&body))
            .await?;
        Ok(v.get("results")
            .and_then(|r| r.as_array())
            .cloned()
            .unwrap_or_default())
    }

    pub async fn get_page(&self, id: &str) -> anyhow::Result<Value> {
        self.send(self.client.get(format!("{API}/pages/{id}")))
            .await
    }

    pub async fn list_children(&self, id: &str) -> anyhow::Result<Vec<Value>> {
        let v = self
            .send(
                self.client
                    .get(format!("{API}/blocks/{id}/children?page_size=100")),
            )
            .await?;
        Ok(v.get("results")
            .and_then(|r| r.as_array())
            .cloned()
            .unwrap_or_default())
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
