use std::{
    collections::HashSet,
    convert::Infallible,
    sync::{Arc, Mutex, RwLock},
};

use ailoy_core::{
    agent::{AgentListOutputItem, AgentProvider, AgentPutArgs, Refusal, SessionCreateArgs},
    message::Message,
    tool::ToolDesc,
};
use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::{Path, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use futures::StreamExt as _;
use serde::Serialize;
use serde_json::json;

use crate::agent::AgentRuntime;

/// What [`router`] serves: the provider, the virtx client its sessions' sandboxes are reached
/// through, and the sessions with a turn running, as their agent and id.
struct Served {
    provider: Arc<RwLock<AgentProvider>>,
    virtx: Option<virtx::Client>,
    running: Mutex<HashSet<(String, String)>>,
}

/// What `provider` serves to the code that runs its agents, `ailoy_core::Client`: the
/// sessions of each agent kept, and the turns run in them. A session's sandbox is reached
/// through `virtx`, on the virtx server the caller made it on; without one, only agents that
/// use no sandbox run.
pub fn router(provider: Arc<RwLock<AgentProvider>>, virtx: Option<virtx::Client>) -> Router {
    let served = Arc::new(Served {
        provider,
        virtx,
        running: Mutex::new(HashSet::new()),
    });
    Router::new()
        .route(
            "/agents/{name}/sessions/",
            post(create_session).get(list_sessions),
        )
        .route(
            "/agents/{name}/sessions/{id}",
            get(get_session).post(run).delete(remove_session),
        )
        .with_state(served)
}

/// What `provider` serves to the code that manages it, `ailoy_core::AdminClient`: the tools
/// provided and the agents kept. Its paths are none of [`router`]'s, so one server may serve
/// both merged, or each to its own listener.
pub fn admin_router(provider: Arc<RwLock<AgentProvider>>) -> Router {
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

async fn list_tools(State(provider): State<Arc<RwLock<AgentProvider>>>) -> Json<Vec<ToolDesc>> {
    Json(provider.read().unwrap().tools.descs())
}

async fn list_agents(
    State(provider): State<Arc<RwLock<AgentProvider>>>,
) -> Json<Vec<AgentListOutputItem>> {
    Json(provider.read().unwrap().list_agents())
}

async fn put_agent(
    State(provider): State<Arc<RwLock<AgentProvider>>>,
    Path(name): Path<String>,
    body: Bytes,
) -> Response {
    let args: AgentPutArgs = match http_over_x::from_body(&body) {
        Ok(args) => args,
        Err(why) => return refuse(StatusCode::BAD_REQUEST, "invalid_args", why),
    };
    match provider.write().unwrap().put_agent(&name, args) {
        Ok(agent) => Json(agent).into_response(),
        Err(refusal) => refuse_as(refusal),
    }
}

async fn get_agent(
    State(provider): State<Arc<RwLock<AgentProvider>>>,
    Path(name): Path<String>,
) -> Response {
    match provider.read().unwrap().get_agent(&name) {
        Ok(agent) => Json(agent).into_response(),
        Err(refusal) => refuse_as(refusal),
    }
}

async fn remove_agent(
    State(provider): State<Arc<RwLock<AgentProvider>>>,
    Path(name): Path<String>,
) -> StatusCode {
    provider.write().unwrap().remove_agent(&name);
    StatusCode::NO_CONTENT
}

async fn create_session(
    State(served): State<Arc<Served>>,
    Path(agent): Path<String>,
    body: Bytes,
) -> Response {
    let args: SessionCreateArgs = match http_over_x::from_body(&body) {
        Ok(args) => args,
        Err(why) => return refuse(StatusCode::BAD_REQUEST, "invalid_args", why),
    };
    match served
        .provider
        .write()
        .unwrap()
        .create_session(&agent, args)
    {
        Ok(session) => (StatusCode::CREATED, Json(session)).into_response(),
        Err(refusal) => refuse_as(refusal),
    }
}

async fn list_sessions(State(served): State<Arc<Served>>, Path(agent): Path<String>) -> Response {
    match served.provider.read().unwrap().list_sessions(&agent) {
        Ok(sessions) => Json(sessions).into_response(),
        Err(refusal) => refuse_as(refusal),
    }
}

async fn get_session(
    State(served): State<Arc<Served>>,
    Path((agent, id)): Path<(String, String)>,
) -> Response {
    match served.provider.read().unwrap().get_session(&agent, &id) {
        Ok(session) => Json(session).into_response(),
        Err(refusal) => refuse_as(refusal),
    }
}

async fn remove_session(
    State(served): State<Arc<Served>>,
    Path((agent, id)): Path<(String, String)>,
) -> StatusCode {
    served.provider.write().unwrap().remove_session(&agent, &id);
    StatusCode::NO_CONTENT
}

/// Run one turn of the session `id` of the agent `agent` on the query the body holds, and answer
/// with what it produces as an event stream, one `MessageDeltaOutput` per event. The turn runs on
/// a task of its own, so it ends, and the session keeps its history, whether or not the stream
/// is read to its end. A failure partway is an `error` event, `{"error": {"code", "message"}}`.
async fn run(
    State(served): State<Arc<Served>>,
    Path((agent, session)): Path<(String, String)>,
    body: Bytes,
) -> Response {
    let query: Message = match http_over_x::from_body(&body) {
        Ok(query) => query,
        Err(why) => return refuse(StatusCode::BAD_REQUEST, "invalid_args", why),
    };
    let (spec, kept) = {
        let provider = served.provider.read().unwrap();
        let spec = match provider.get_agent(&agent) {
            Ok(agent) => agent.spec,
            Err(refusal) => return refuse_as(refusal),
        };
        match provider.get_session(&agent, &session) {
            Ok(kept) => (spec, kept),
            Err(refusal) => return refuse_as(refusal),
        }
    };
    let key = (agent.clone(), session.clone());
    if !served.running.lock().unwrap().insert(key.clone()) {
        let why = format!("the session '{session}' has a turn running");
        return refuse(StatusCode::CONFLICT, "busy", why);
    }

    let virtx = served.virtx.as_ref();
    let mut runtime = match AgentRuntime::try_in(&served.provider, virtx, spec, kept).await {
        Ok(runtime) => runtime,
        Err(e) => {
            served.running.lock().unwrap().remove(&key);
            return refuse(
                StatusCode::UNPROCESSABLE_ENTITY,
                "unusable",
                format!("{e:#}"),
            );
        }
    };

    let (events, body) = futures::channel::mpsc::unbounded::<Result<Bytes, Infallible>>();
    tokio::spawn(async move {
        let mut failed = None;
        {
            let mut turn = runtime.run_stream(query);
            while let Some(delta) = turn.next().await {
                let event = match delta {
                    Ok(delta) => event(None, &delta),
                    Err(e) => {
                        let error =
                            json!({"error": {"code": "turn_failed", "message": format!("{e:#}")}});
                        failed = Some(event(Some("error"), &error));
                        continue;
                    }
                };
                // A reader gone leaves the turn to finish.
                let _ = events.unbounded_send(Ok(event));
            }
        }
        // A session removed while its turn ran stays removed.
        let kept = runtime.state.clone();
        let _ = served.provider.write().unwrap().set_session(&key.0, kept);
        served.running.lock().unwrap().remove(&key);
        if let Some(failed) = failed {
            let _ = events.unbounded_send(Ok(failed));
        }
    });
    let body = Body::from_stream(body);
    ([(header::CONTENT_TYPE, "text/event-stream")], body).into_response()
}

/// `data` as one server-sent event, of the type `kind` if given.
fn event(kind: Option<&str>, data: &impl Serialize) -> Bytes {
    let data = serde_json::to_string(data).expect("an event serializes");
    match kind {
        Some(kind) => Bytes::from(format!("event: {kind}\ndata: {data}\n\n")),
        None => Bytes::from(format!("data: {data}\n\n")),
    }
}

#[cfg(test)]
mod tests {
    use ailoy_core::{
        AdminClient, Client,
        agent::AgentSpec,
        message::{FinishReason, MessageDeltaOutput, MessageOutput, Part, Role, into_messages},
        model::{InferLangModel, LangModelOptions, ModelKeyProvider},
        tool::{ToolDescBuilder, ToolFunc, ToolSet},
        tool_func,
    };
    use futures::{future::BoxFuture, stream::BoxStream};

    use super::*;

    /// Provides one tool, `echo`.
    struct Echo;

    impl ToolSet for Echo {
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
        let mut provider = AgentProvider::new();
        provider.tools.set(Echo);
        let provider = Arc::new(RwLock::new(provider));
        let admin = AdminClient::embed(admin_router(provider.clone()));
        (admin, Client::embed(router(provider, None)))
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

    /// Answers `pong`, after a pause if `slow`, or fails, whatever it is asked.
    struct Parrot {
        fails: bool,
        slow: bool,
    }

    impl InferLangModel for Parrot {
        fn infer(
            &self,
            _: &[Message],
            _: &[ToolDesc],
            _: &LangModelOptions,
        ) -> BoxFuture<'static, anyhow::Result<MessageOutput>> {
            unimplemented!("a turn streams")
        }

        fn infer_stream(
            &self,
            _: &[Message],
            _: &[ToolDesc],
            _: &LangModelOptions,
        ) -> BoxStream<'static, anyhow::Result<MessageDeltaOutput>> {
            let slow = self.slow;
            let output = match self.fails {
                true => Err(anyhow::anyhow!("no words today")),
                false => Ok(MessageDeltaOutput::from(MessageOutput {
                    message: Message::new(Role::Assistant).with_contents([Part::text("pong")]),
                    finish_reason: FinishReason::Stop {},
                    usage: None,
                })),
            };
            Box::pin(futures::stream::once(async move {
                if slow {
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                }
                output
            }))
        }
    }

    /// A client of a provider whose `parrot` org's models are [`Parrot`]s, failing ones under
    /// `parrot/fails` and slow ones under `parrot/slow`, and whose agent `p` runs `model`, with a
    /// session started.
    async fn parrot_session(model: &str) -> (Client, String) {
        parrot_session_of(AgentSpec::new(model), None).await
    }

    /// As [`parrot_session`], of the agent `spec`, the session working in `sandbox`.
    async fn parrot_session_of(
        spec: AgentSpec,
        sandbox: Option<&virtx::Sandbox>,
    ) -> (Client, String) {
        let mut provider = AgentProvider::new();
        provider.models.set(
            "parrot",
            |_: &str, name: Option<&str>| -> anyhow::Result<Arc<dyn InferLangModel>> {
                Ok(Arc::new(Parrot {
                    fails: name == Some("fails"),
                    slow: name == Some("slow"),
                }))
            },
        );
        // The tools a sandbox brings.
        provider.tools.set(crate::tool::Console);
        provider.put_agent("p", AgentPutArgs::new(spec)).unwrap();
        let virtx = sandbox.map(|sandbox| sandbox.client().clone());
        let client = Client::embed(router(Arc::new(RwLock::new(provider)), virtx));
        let mut args = SessionCreateArgs::new();
        if let Some(sandbox) = sandbox {
            args = args.sandbox_id(sandbox.id());
        }
        let session = client.create_session("p", &args).await.unwrap();
        (client, session.id)
    }

    fn ping() -> Message {
        Message::new(Role::User).with_contents([Part::text("ping")])
    }

    #[tokio::test]
    async fn test_a_turn_streams_its_deltas_and_the_session_keeps_it() {
        let (client, id) = parrot_session("parrot/x").await;
        let deltas = client.run_stream("p", &id, &ping());
        let deltas = deltas.map(|delta| delta.map_err(anyhow::Error::from));
        let messages: Vec<_> = into_messages(deltas).collect().await;
        let messages: Vec<_> = messages.into_iter().map(Result::unwrap).collect();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].message.contents[0].as_text(), Some("pong"));

        let state = client.get_session("p", &id).await.unwrap();
        let roles: Vec<Role> = state.history.iter().map(|m| m.role.clone()).collect();
        assert_eq!(roles, [Role::User, Role::Assistant]);
    }

    /// An agent that works in a sandbox runs in the one its session names, the caller's.
    #[tokio::test]
    async fn test_a_turn_works_in_the_sandbox_its_session_names() {
        let sandbox = crate::test_sandbox().await;
        let spec = AgentSpec::new("parrot/x").sandbox();
        let (client, id) = parrot_session_of(spec, Some(&sandbox)).await;
        let outcomes: Vec<_> = client.run_stream("p", &id, &ping()).collect().await;
        assert!(outcomes.iter().all(Result::is_ok), "{outcomes:?}");
        let state = client.get_session("p", &id).await.unwrap();
        assert_eq!(state.sandbox_id.as_deref(), Some(sandbox.id()));
        assert_eq!(state.history.len(), 2);
    }

    #[tokio::test]
    async fn test_a_turn_that_fails_partway_ends_on_its_error() {
        let (client, id) = parrot_session("parrot/fails").await;
        let outcomes: Vec<_> = client.run_stream("p", &id, &ping()).collect().await;
        let error = outcomes.last().unwrap().as_ref().err().unwrap();
        assert_eq!(error.code(), Some("turn_failed"));
        assert!(error.to_string().contains("no words today"), "{error}");
    }

    #[tokio::test]
    async fn test_a_session_runs_one_turn_at_a_time() {
        let (client, id) = parrot_session("parrot/slow").await;
        let client = Arc::new(client);
        let first = tokio::spawn({
            let (client, id) = (client.clone(), id.clone());
            async move {
                client
                    .run_stream("p", &id, &ping())
                    .collect::<Vec<_>>()
                    .await
            }
        });
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let outcomes: Vec<_> = client.run_stream("p", &id, &ping()).collect().await;
        let error = outcomes[0].as_ref().err().unwrap();
        assert_eq!(error.status(), Some(409));
        assert_eq!(error.code(), Some("busy"));
        assert!(first.await.unwrap().iter().all(Result::is_ok));
    }

    #[tokio::test]
    async fn test_a_turn_of_a_model_nothing_makes_is_refused() {
        let (client, id) = parrot_session("nobody/x").await;
        let outcomes: Vec<_> = client.run_stream("p", &id, &ping()).collect().await;
        let error = outcomes[0].as_ref().err().unwrap();
        assert_eq!(error.status(), Some(422));
        assert_eq!(error.code(), Some("unusable"));
        // The session is free again for the next turn.
        let outcomes: Vec<_> = client.run_stream("p", &id, &ping()).collect().await;
        assert_eq!(outcomes[0].as_ref().err().unwrap().code(), Some("unusable"));
    }

    /// Each router serves its own client's paths and none of the other's.
    #[tokio::test]
    async fn test_the_routers_share_no_path() {
        let provider = Arc::new(RwLock::new(AgentProvider::new()));
        let admin = AdminClient::embed(router(provider.clone(), None));
        assert_eq!(admin.list_agent().await.err().unwrap().status(), Some(404));
        let client = Client::embed(admin_router(provider));
        let error = client.list_session("a").await.err().unwrap();
        assert_eq!(error.status(), Some(404));
    }
}
