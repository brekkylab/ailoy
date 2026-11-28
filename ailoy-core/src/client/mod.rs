//! The two clients of an Ailoy server: [`Client`], for the code that manages it, the agents it
//! keeps and their sessions, and [`AppClient`], for an app's user, who reaches the one session
//! they start with its grant alone. Each reaches a server in this process (`embed`, the `embed`
//! feature) or one at a URL (`connect`); a `Client` also a child process (`spawn`).

mod admin;
mod app;

pub use admin::Client;
pub use app::AppClient;
use futures::{Stream, StreamExt as _};
use http_over_x::{Error, Members, full, http, refused};
use sse_stream::SseStream;

use crate::message::{Message, MessageDeltaOutput};

/// Run one turn of the session at `path` on `query`, through `inner`, and stream what it
/// produces, one delta each, ending on the error a turn that fails partway ends with.
fn run_stream<'a>(
    inner: &'a http_over_x::Client,
    path: String,
    query: &Message,
) -> impl Stream<Item = Result<MessageDeltaOutput, Error>> + Send + 'a {
    let body = serde_json::to_vec(query);
    async_stream::try_stream! {
        let body = body.map_err(|e| Error::Client(e.to_string()))?;
        let request = http::Request::post(&path)
            .header(http::header::ACCEPT, "text/event-stream")
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(full(body))
            .map_err(|e| Error::Client(format!("not a request: {e}")))?;
        let response = inner.send(request).await?;
        let response = match response.status().is_success() {
            true => response,
            false => Err(refused(response).await)?,
        };
        let mut events = SseStream::new(response.into_body());
        while let Some(event) = events.next().await {
            let event = event.map_err(|e| Error::Transport(format!("reading {path}: {e}")))?;
            let Some(data) = event.data else {
                continue;
            };
            if event.event.as_deref() == Some("error") {
                Err(failed(&data))?;
            }
            yield serde_json::from_str(&data).map_err(|e| Error::Client(e.to_string()))?;
        }
    }
}

/// The error a turn ended on partway, from its `error` event's data, the
/// `{"error": {"code", "message"}}` a refusal answers with.
fn failed(data: &str) -> Error {
    let mut error: Members = serde_json::from_str::<serde_json::Value>(data)
        .ok()
        .and_then(|value| value.get("error")?.as_object().cloned())
        .unwrap_or_default();
    let mut field = |name| match error.remove(name) {
        Some(serde_json::Value::String(value)) => value,
        _ => String::new(),
    };
    let (code, message) = (field("code"), field("message"));
    Error::Status(500, format!("{code}: {message}"), error)
}
