//! Telling the outside about a run's status change: an event of type `runs:<trigger>`,
//! and the outbound webhooks in `daemon.toml`.

use serde_json::{Value, json};

use crate::{sources, state::AppState};

pub struct Transition<'a> {
    pub run_id: &'a str,
    pub trigger: &'a str,
    pub status: &'a str,
    pub output: Option<&'a Value>,
    pub error: Option<&'a str>,
}

pub async fn notify(state: &AppState, t: Transition<'_>) {
    let envelope = json!({
        "trigger": t.trigger,
        "run_id": t.run_id,
        "status": t.status,
        "output": t.output,
        "error": t.error,
        "artifacts_url": format!("http://{}/runs/{}/artifacts", state.config.listen, t.run_id),
    });

    if let Err(e) = state
        .publisher
        .publish(&sources::run_type(t.trigger), envelope.clone())
    {
        tracing::error!(run = t.run_id, "publishing run status: {e}");
    }

    for hook in &state.config.hooks {
        if !hook.statuses.is_empty() && !hook.statuses.iter().any(|s| s == t.status) {
            continue;
        }
        if !hook.triggers.is_empty() && !hook.triggers.iter().any(|s| s == t.trigger) {
            continue;
        }
        let body = serde_json::to_vec(&envelope).unwrap_or_default();
        match state
            .http
            .post(&hook.url)
            .header("content-type", "application/json")
            .body(body)
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {}
            Ok(resp) => tracing::warn!(url = %hook.url, status = %resp.status(), "hook refused"),
            Err(e) => tracing::warn!(url = %hook.url, "hook failed: {e}"),
        }
    }
}
