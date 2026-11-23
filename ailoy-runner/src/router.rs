use std::sync::Arc;

use ailoy_core::{
    agent::{AgentListOutputItem, AgentPutArgs, SessionCreateArgs},
    tool::ToolDesc,
};
use axum::{
    Json, Router,
    body::Bytes,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde_json::json;

use crate::{AgentProvider, Refusal};

/// What `provider` serves to the code that runs its agents, `ailoy_core::Client`: the
/// sessions of each agent kept.
pub fn router(provider: Arc<AgentProvider>) -> Router {
    Router::new()
        .route(
            "/agents/{name}/sessions/",
            post(create_session).get(list_sessions),
        )
        .route(
            "/agents/{name}/sessions/{id}",
            get(get_session).delete(remove_session),
        )
        .with_state(provider)
}

/// What `provider` serves to the code that manages it, `ailoy_core::AdminClient`: the tools
/// provided and the agents kept. Its paths are none of [`router`]'s, so one server may serve
/// both merged, or each to its own listener.
pub fn admin_router(provider: Arc<AgentProvider>) -> Router {
    Router::new()
        .route("/tools", get(list_tools))
        .route("/agents/", get(list_agents))
        .route(
            "/agents/{name}",
            get(get_agent).put(put_agent).delete(remove_agent),
        )
        .with_state(provider)
}

/// A refusal as the protocol answers one: its status, and `{"error": {"code", "message"}}`.
fn refuse(status: StatusCode, code: &str, message: String) -> Response {
    let body = json!({"error": {"code": code, "message": message}});
    (status, Json(body)).into_response()
}

fn refuse_as(refusal: Refusal) -> Response {
    let (status, code) = match refusal {
        Refusal::NoAgent(_) | Refusal::NoSession(_) => (StatusCode::NOT_FOUND, "not_found"),
        Refusal::NoTool(_) => (StatusCode::UNPROCESSABLE_ENTITY, "unknown_tool"),
    };
    refuse(status, code, refusal.to_string())
}

async fn list_tools(State(provider): State<Arc<AgentProvider>>) -> Json<Vec<ToolDesc>> {
    Json(provider.tool_descs())
}

async fn list_agents(State(provider): State<Arc<AgentProvider>>) -> Json<Vec<AgentListOutputItem>> {
    Json(provider.list_agents())
}

async fn put_agent(
    State(provider): State<Arc<AgentProvider>>,
    Path(name): Path<String>,
    body: Bytes,
) -> Response {
    let args: AgentPutArgs = match http_over_x::from_body(&body) {
        Ok(args) => args,
        Err(why) => return refuse(StatusCode::BAD_REQUEST, "invalid_args", why),
    };
    match provider.put_agent(&name, args) {
        Ok(agent) => Json(agent).into_response(),
        Err(refusal) => refuse_as(refusal),
    }
}

async fn get_agent(
    State(provider): State<Arc<AgentProvider>>,
    Path(name): Path<String>,
) -> Response {
    match provider.get_agent(&name) {
        Ok(agent) => Json(agent).into_response(),
        Err(refusal) => refuse_as(refusal),
    }
}

async fn remove_agent(
    State(provider): State<Arc<AgentProvider>>,
    Path(name): Path<String>,
) -> StatusCode {
    provider.remove_agent(&name);
    StatusCode::NO_CONTENT
}

async fn create_session(
    State(provider): State<Arc<AgentProvider>>,
    Path(agent): Path<String>,
    body: Bytes,
) -> Response {
    let args: SessionCreateArgs = match http_over_x::from_body(&body) {
        Ok(args) => args,
        Err(why) => return refuse(StatusCode::BAD_REQUEST, "invalid_args", why),
    };
    match provider.create_session(&agent, args) {
        Ok(session) => (StatusCode::CREATED, Json(session)).into_response(),
        Err(refusal) => refuse_as(refusal),
    }
}

async fn list_sessions(
    State(provider): State<Arc<AgentProvider>>,
    Path(agent): Path<String>,
) -> Response {
    match provider.list_sessions(&agent) {
        Ok(sessions) => Json(sessions).into_response(),
        Err(refusal) => refuse_as(refusal),
    }
}

async fn get_session(
    State(provider): State<Arc<AgentProvider>>,
    Path((agent, id)): Path<(String, String)>,
) -> Response {
    match provider.get_session(&agent, &id) {
        Ok(session) => Json(session).into_response(),
        Err(refusal) => refuse_as(refusal),
    }
}

async fn remove_session(
    State(provider): State<Arc<AgentProvider>>,
    Path((agent, id)): Path<(String, String)>,
) -> StatusCode {
    provider.remove_session(&agent, &id);
    StatusCode::NO_CONTENT
}

#[cfg(test)]
mod tests {
    use ailoy_core::{
        AdminClient, Client,
        agent::AgentSpec,
        message::{Message, Part, Role},
        model::ModelKeyProvider,
        tool::{ToolDescBuilder, ToolFunc, ToolProvider},
        tool_func,
    };

    use super::*;

    /// Provides one tool, `echo`.
    struct Echo;

    impl ToolProvider for Echo {
        fn descs(&self) -> Vec<ToolDesc> {
            vec![
                ToolDescBuilder::new("echo")
                    .description("Say it back")
                    .build(),
            ]
        }

        fn make(&self, _name: &str) -> anyhow::Result<ToolFunc> {
            Ok(tool_func!(|args: Value| -> Value { args }))
        }
    }

    /// The two clients of a provider with `echo`, served in this process.
    fn clients() -> (AdminClient, Client) {
        let provider = Arc::new(AgentProvider::new().tool(Echo));
        let admin = AdminClient::embed(admin_router(provider.clone()));
        (admin, Client::embed(router(provider)))
    }

    /// The two clients of a provider that keeps the agent `a`.
    async fn clients_with_agent() -> (AdminClient, Client) {
        let (admin, client) = clients();
        let args = AgentPutArgs::new(AgentSpec::new("anthropic"));
        admin.put_agent("a", &args).await.unwrap();
        (admin, client)
    }

    #[tokio::test]
    async fn test_the_tools_provided_are_listed() {
        let (admin, _) = clients();
        let tools = admin.list_tool().await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo");
        assert_eq!(tools[0].description.as_deref(), Some("Say it back"));
    }

    #[tokio::test]
    async fn test_an_agent_is_put_listed_and_removed() {
        let (admin, _) = clients();
        let spec = AgentSpec::new("anthropic/claude-sonnet-5-5").tool("echo");
        let args = AgentPutArgs::new(spec).model_key_provider(ModelKeyProvider::Client);

        let put = admin.put_agent("image editor", &args).await.unwrap();
        assert_eq!(put.name, "image editor");
        assert_eq!(put.model_key_provider, ModelKeyProvider::Client);

        let got = admin.get_agent("image editor").await.unwrap();
        assert_eq!(got.spec.model, "anthropic/claude-sonnet-5-5");
        assert_eq!(got.spec.tools, ["echo"]);
        let agents = admin.list_agent().await.unwrap();
        let names: Vec<&str> = agents.iter().map(|agent| agent.name.as_str()).collect();
        assert_eq!(names, ["image editor"]);

        admin.remove_agent("image editor").await.unwrap();
        assert!(admin.list_agent().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_an_agent_put_again_replaces_the_one_there() {
        let (admin, _) = clients();
        let first = AgentPutArgs::new(AgentSpec::new("anthropic"));
        let second = AgentPutArgs::new(AgentSpec::new("openai"));
        admin.put_agent("a", &first).await.unwrap();
        admin.put_agent("a", &second).await.unwrap();

        let agents = admin.list_agent().await.unwrap();
        assert_eq!(agents.len(), 1);
        assert_eq!(agents[0].spec.model, "openai");
    }

    #[tokio::test]
    async fn test_an_agent_naming_a_tool_nothing_provides_is_refused() {
        let (admin, _) = clients();
        let args = AgentPutArgs::new(AgentSpec::new("anthropic").tools(["echo", "nowhere"]));
        let error = admin.put_agent("a", &args).await.err().unwrap();
        assert_eq!(error.status(), Some(422));
        assert_eq!(error.code(), Some("unknown_tool"));
        assert!(error.to_string().contains("nowhere"), "{error}");
        assert!(admin.list_agent().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_an_agent_not_kept_is_refused() {
        let (admin, _) = clients();
        let error = admin.get_agent("none").await.err().unwrap();
        assert_eq!(error.status(), Some(404));
        assert_eq!(error.code(), Some("not_found"));
    }

    #[tokio::test]
    async fn test_removing_an_agent_not_kept_does_nothing() {
        let (admin, _) = clients();
        admin.remove_agent("none").await.unwrap();
    }

    #[tokio::test]
    async fn test_a_session_is_started_listed_and_removed() {
        let (_admin, client) = clients_with_agent().await;
        let hello = Message::new(Role::User).with_contents([Part::text("hello")]);

        let empty = client
            .create_session("a", &SessionCreateArgs::new())
            .await
            .unwrap();
        assert!(empty.history.is_empty());
        let seeded = client
            .create_session("a", &SessionCreateArgs::new().history([hello]))
            .await
            .unwrap();
        assert_ne!(empty.id, seeded.id);

        let got = client.get_session("a", &seeded.id).await.unwrap();
        assert_eq!(got.history.len(), 1);
        assert_eq!(got.history[0].contents[0].as_text(), Some("hello"));
        let listed = client.list_session("a").await.unwrap();
        let counts: Vec<(&str, usize)> = listed
            .iter()
            .map(|session| (session.id.as_str(), session.message_count))
            .collect();
        assert_eq!(counts, [(empty.id.as_str(), 0), (seeded.id.as_str(), 1)]);

        client.remove_session("a", &empty.id).await.unwrap();
        let error = client.get_session("a", &empty.id).await.err().unwrap();
        assert_eq!(error.status(), Some(404));
        assert_eq!(client.list_session("a").await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_a_session_of_an_agent_not_kept_is_refused() {
        let (_admin, client) = clients();
        let args = SessionCreateArgs::new();
        let error = client.create_session("none", &args).await.err().unwrap();
        assert_eq!(error.status(), Some(404));
        assert_eq!(error.code(), Some("not_found"));
        let error = client.list_session("none").await.err().unwrap();
        assert_eq!(error.status(), Some(404));
    }

    #[tokio::test]
    async fn test_sessions_stay_when_their_agent_is_put_again() {
        let (admin, client) = clients_with_agent().await;
        let session = client
            .create_session("a", &SessionCreateArgs::new())
            .await
            .unwrap();

        let args = AgentPutArgs::new(AgentSpec::new("openai"));
        admin.put_agent("a", &args).await.unwrap();
        let state = client.get_session("a", &session.id).await.unwrap();
        assert_eq!(state.id, session.id);
    }

    #[tokio::test]
    async fn test_sessions_go_with_their_agent() {
        let (admin, client) = clients_with_agent().await;
        let session = client
            .create_session("a", &SessionCreateArgs::new())
            .await
            .unwrap();

        admin.remove_agent("a").await.unwrap();
        let args = AgentPutArgs::new(AgentSpec::new("anthropic"));
        admin.put_agent("a", &args).await.unwrap();
        assert!(client.list_session("a").await.unwrap().is_empty());
        let error = client.get_session("a", &session.id).await.err().unwrap();
        assert_eq!(error.status(), Some(404));
    }

    /// Each router serves its own client's paths and none of the other's.
    #[tokio::test]
    async fn test_the_routers_share_no_path() {
        let provider = Arc::new(AgentProvider::new());
        let admin = AdminClient::embed(router(provider.clone()));
        assert_eq!(admin.list_agent().await.err().unwrap().status(), Some(404));
        let client = Client::embed(admin_router(provider));
        let error = client.list_session("a").await.err().unwrap();
        assert_eq!(error.status(), Some(404));
    }
}
