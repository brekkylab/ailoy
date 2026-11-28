use std::{
    convert::Infallible,
    sync::{Arc, RwLock},
};

use ailoy_core::{
    agent::{
        AgentListOutputItem, AgentProvider, AgentPutArgs, AgentState, AgentStore,
        AppSessionCreateArgs, AppSessionCreateOutput, Refusal, SessionCreateArgs,
        SessionListOutputItem, SessionRecord, SessionSandbox,
    },
    message::Message,
    model::{MODEL_KEY_HEADER, ModelKeyProvider, ModelListOutputItem},
    tool::ToolDesc,
};
use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::{Path, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use futures::StreamExt as _;
use serde::Serialize;
use serde_json::json;

use crate::agent::AgentRuntime;

/// What `provider` serves to an app, `ailoy_core::AppClient`: a session started of an agent
/// kept, its sandbox made of a virtx package as the app's own credentials let in, and that
/// session alone reached with the grant it is given. Its paths are none of [`admin_router`]'s,
/// so one server may serve both merged, or each to its own listener.
pub fn router(provider: Arc<RwLock<AgentProvider>>) -> Router {
    Router::new()
        .route("/sessions/", post(create_app_session))
        .route(
            "/sessions/{id}",
            get(get_app_session)
                .post(run_app)
                .delete(remove_app_session),
        )
        .with_state(provider)
}

/// What `provider` serves to the code that manages it, `ailoy_core::Client`: the tools
/// provided, the agents kept, and their sessions, each with the sandbox made for it on the
/// provider's virtx, as one of its members.
pub fn admin_router(provider: Arc<RwLock<AgentProvider>>) -> Router {
    Router::new()
        .route("/tools", get(list_tools))
        .route("/models", get(list_models))
        .route("/agents/", get(list_agents))
        .route(
            "/agents/{name}",
            get(get_agent).put(put_agent).delete(remove_agent),
        )
        .route(
            "/agents/{name}/sessions/",
            post(create_session).get(list_sessions),
        )
        .route(
            "/agents/{name}/sessions/{id}",
            get(get_session).post(run).delete(remove_session),
        )
        .with_state(provider)
}

/// A request not done as asked, as the protocol answers it: its status, and
/// `{"error": {"code", "message"}}`.
struct Refused(Box<Response>);

impl Refused {
    fn new(status: StatusCode, code: &str, message: String) -> Self {
        let body = json!({"error": {"code": code, "message": message}});
        Refused(Box::new((status, Json(body)).into_response()))
    }
}

impl IntoResponse for Refused {
    fn into_response(self) -> Response {
        *self.0
    }
}

impl From<Refusal> for Refused {
    fn from(refusal: Refusal) -> Self {
        let (status, code) = match refusal {
            Refusal::NoAgent(_) | Refusal::NoSession(_) => (StatusCode::NOT_FOUND, "not_found"),
            Refusal::NoTool(_) => (StatusCode::UNPROCESSABLE_ENTITY, "unknown_tool"),
            Refusal::NoModel(_) => (StatusCode::UNPROCESSABLE_ENTITY, "unknown_model"),
            Refusal::NoClientKey(_) => (StatusCode::UNPROCESSABLE_ENTITY, "no_client_key"),
            Refusal::Busy(_) => (StatusCode::CONFLICT, "busy"),
            Refusal::Sandbox(_) => (StatusCode::UNPROCESSABLE_ENTITY, "unusable"),
        };
        Refused::new(status, code, refusal.to_string())
    }
}

/// What virtx did not do with a session's sandbox: virtx's own status and code where it
/// refused, `502` where it failed otherwise.
impl From<virtx::Error> for Refused {
    fn from(error: virtx::Error) -> Self {
        let status = (error.status()).and_then(|status| StatusCode::from_u16(status).ok());
        let code = error.code().unwrap_or("virtx").to_owned();
        let message = format!("virtx: {error}");
        Refused::new(status.unwrap_or(StatusCode::BAD_GATEWAY), &code, message)
    }
}

/// The store's own failure.
impl From<anyhow::Error> for Refused {
    fn from(error: anyhow::Error) -> Self {
        let message = format!("the store: {error:#}");
        Refused::new(StatusCode::INTERNAL_SERVER_ERROR, "store", message)
    }
}

/// The args a request's `body` holds.
fn args<T: serde::de::DeserializeOwned>(body: &Bytes) -> Result<T, Refused> {
    http_over_x::from_body(body)
        .map_err(|why| Refused::new(StatusCode::BAD_REQUEST, "invalid_args", why))
}

/// Where `provider` keeps its agents and sessions, taken out of the lock to be awaited.
fn store(provider: &RwLock<AgentProvider>) -> Arc<dyn AgentStore> {
    provider.read().unwrap().store.clone()
}

/// The bearer token `headers` carry, if they do.
fn bearer(headers: &HeaderMap) -> Option<&str> {
    let authorization = headers.get(header::AUTHORIZATION)?.to_str().ok()?;
    authorization.strip_prefix("Bearer ")
}

async fn list_models(
    State(provider): State<Arc<RwLock<AgentProvider>>>,
) -> Json<Vec<ModelListOutputItem>> {
    let provider = provider.read().unwrap();
    let models = provider
        .models
        .list()
        .map(|(org, set)| ModelListOutputItem {
            org: org.to_owned(),
            backend: set.name(),
            takes_key: set.takes_key(org),
        });
    Json(models.collect())
}

async fn list_tools(State(provider): State<Arc<RwLock<AgentProvider>>>) -> Json<Vec<ToolDesc>> {
    Json(provider.read().unwrap().tools.descs())
}

async fn list_agents(
    State(provider): State<Arc<RwLock<AgentProvider>>>,
) -> Result<Json<Vec<AgentListOutputItem>>, Refused> {
    let agents = store(&provider).list_agents().await?;
    let agents = agents
        .into_iter()
        .map(|(name, args)| AgentListOutputItem::new(name, args));
    Ok(Json(agents.collect()))
}

async fn put_agent(
    State(provider): State<Arc<RwLock<AgentProvider>>>,
    Path(name): Path<String>,
    body: Bytes,
) -> Result<Json<AgentListOutputItem>, Refused> {
    let args: AgentPutArgs = args(&body)?;
    provider.read().unwrap().check_agent(&args)?;
    store(&provider).put_agent(&name, &args).await?;
    Ok(Json(AgentListOutputItem::new(name, args)))
}

async fn get_agent(
    State(provider): State<Arc<RwLock<AgentProvider>>>,
    Path(name): Path<String>,
) -> Result<Json<AgentListOutputItem>, Refused> {
    let args = store(&provider).get_agent(&name).await?;
    let args = args.ok_or_else(|| Refusal::NoAgent(name.clone()))?;
    Ok(Json(AgentListOutputItem::new(name, args)))
}

async fn remove_agent(
    State(provider): State<Arc<RwLock<AgentProvider>>>,
    Path(name): Path<String>,
) -> Result<StatusCode, Refused> {
    let sessions = store(&provider).remove_agent(&name).await?;
    for session in sessions {
        remove_sandbox(&provider, &session).await;
    }
    Ok(StatusCode::NO_CONTENT)
}

/// The sandbox of `session`, if it has one: the one held for it, or, since this process began
/// without it, reached again on virtx, with the session's sandbox grant where it has one.
async fn reach_sandbox(
    provider: &RwLock<AgentProvider>,
    session: &SessionRecord,
) -> Result<Option<virtx::Sandbox>, Refused> {
    let Some(sandbox_id) = &session.state.sandbox_id else {
        return Ok(None);
    };
    let id = &session.state.id;
    if let Some(held) = provider.read().unwrap().sandbox(id) {
        return Ok(Some(held));
    }
    let unusable =
        |why: &str| Refused::new(StatusCode::UNPROCESSABLE_ENTITY, "unusable", why.into());
    let client = match &session.sandbox_grant {
        Some(grant) => {
            let url = provider.read().unwrap().virtx_url.clone();
            let url = url.ok_or_else(|| unusable("this server reaches no app's sandbox"))?;
            let auth = virtx::Auth::Bearer {
                token: grant.clone(),
            };
            virtx::Client::connect(&url, auth, None)?
        }
        None => {
            let virtx = provider.read().unwrap().virtx.clone();
            virtx.ok_or_else(|| unusable("this server has no virtx to reach a sandbox on"))?
        }
    };
    let sandbox = client.get_sandbox(sandbox_id).await?;
    provider.write().unwrap().hold_sandbox(id, sandbox.clone());
    Ok(Some(sandbox))
}

/// Remove the sandbox of `session`, if it has one. One already gone, or not reached, is as good
/// as removed.
async fn remove_sandbox(provider: &RwLock<AgentProvider>, session: &SessionRecord) {
    let held = provider.write().unwrap().release_sandbox(&session.state.id);
    let sandbox = match held {
        Some(held) => Some(held),
        None => reach_sandbox(provider, session).await.ok().flatten(),
    };
    provider.write().unwrap().release_sandbox(&session.state.id);
    if let Some(sandbox) = sandbox {
        let _ = sandbox.remove().await;
    }
}

/// Refused unless the agent `agent` is kept, and uses a sandbox exactly when `given` one.
async fn check_sandbox(
    provider: &RwLock<AgentProvider>,
    agent: &str,
    given: bool,
) -> Result<(), Refused> {
    let args = store(provider).get_agent(agent).await?;
    let args = args.ok_or_else(|| Refusal::NoAgent(agent.to_owned()))?;
    match args.spec.use_sandbox == given {
        true => Ok(()),
        false => Err(Refusal::Sandbox(args.spec.use_sandbox).into()),
    }
}

/// Keep `session` as a new session of the agent `agent`, working in `sandbox`; or remove the
/// sandbox, where it is not kept.
async fn keep_session(
    provider: &RwLock<AgentProvider>,
    agent: &str,
    session: &SessionRecord,
    sandbox: Option<virtx::Sandbox>,
) -> Result<(), Refused> {
    let kept = store(provider).create_session(agent, session).await;
    if let (Ok(true), Some(sandbox)) = (&kept, &sandbox) {
        let id = &session.state.id;
        provider.write().unwrap().hold_sandbox(id, sandbox.clone());
    }
    if !matches!(kept, Ok(true))
        && let Some(sandbox) = sandbox
    {
        let _ = sandbox.remove().await;
    }
    match kept? {
        true => Ok(()),
        false => Err(Refusal::NoAgent(agent.to_owned()).into()),
    }
}

/// A new session's record, starting with `history` and working in `sandbox`.
fn new_session(history: Vec<Message>, sandbox: Option<&virtx::Sandbox>) -> SessionRecord {
    let mut state = AgentState::new().with_history(history);
    state.sandbox_id = sandbox.map(|sandbox| sandbox.id().to_owned());
    SessionRecord::new(state)
}

async fn create_session(
    State(provider): State<Arc<RwLock<AgentProvider>>>,
    Path(agent): Path<String>,
    body: Bytes,
) -> Result<Response, Refused> {
    let args: SessionCreateArgs = args(&body)?;
    check_sandbox(&provider, &agent, args.sandbox.is_some()).await?;
    let sandbox = match args.sandbox {
        None => None,
        Some(how) => {
            let virtx = provider.read().unwrap().virtx.clone();
            let Some(virtx) = virtx else {
                let why = "this server has no virtx to make a sandbox on".to_owned();
                return Err(Refused::new(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "unusable",
                    why,
                ));
            };
            Some(match how {
                SessionSandbox::Package(id) => {
                    let args = virtx::PackageSandboxCreateArgs::default();
                    virtx.create_package_sandbox(id, &args).await?.sandbox
                }
                SessionSandbox::Create(args) => virtx.create_sandbox(&args).await?,
            })
        }
    };
    let session = new_session(args.history, sandbox.as_ref());
    keep_session(&provider, &agent, &session, sandbox).await?;
    Ok((StatusCode::CREATED, Json(session.state)).into_response())
}

/// A sandbox made of the package `package` on the virtx server at `url`, as `auth`, an app's own
/// credentials, lets in; reached with its grant from then on, which reaches nothing else, and
/// which is answered beside it.
async fn app_sandbox(
    url: &str,
    package: &str,
    auth: virtx::Auth,
) -> Result<(virtx::Sandbox, String), virtx::Error> {
    let client = virtx::Client::connect(url, auth, None)?;
    let args = virtx::PackageSandboxCreateArgs::default();
    let made = client.create_package_sandbox(package, &args).await?;
    let Some(token) = made.grant else {
        let why = "the server gave the sandbox no grant to reach it with";
        return Err(virtx::Error::Call(virtx::Source::Server, why.into()));
    };
    client.set_auth(virtx::Auth::Bearer {
        token: token.clone(),
    })?;
    Ok((made.sandbox, token))
}

async fn create_app_session(
    State(provider): State<Arc<RwLock<AgentProvider>>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, Refused> {
    let args: AppSessionCreateArgs = args(&body)?;
    check_sandbox(&provider, &args.agent, args.package.is_some()).await?;
    let made = match &args.package {
        None => None,
        Some(package) => {
            let url = provider.read().unwrap().virtx_url.clone();
            let Some(url) = url else {
                let why = "this server makes no sandbox for an app".to_owned();
                return Err(Refused::new(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "unusable",
                    why,
                ));
            };
            // The app's own credentials, which virtx lets in, or not, to the package.
            let auth = match bearer(&headers) {
                Some(token) => virtx::Auth::Bearer {
                    token: token.to_owned(),
                },
                None => virtx::Auth::None,
            };
            Some(app_sandbox(&url, package, auth).await?)
        }
    };
    let (sandbox, sandbox_grant) = made.unzip();
    let mut session = new_session(args.history, sandbox.as_ref());
    session.sandbox_grant = sandbox_grant;
    let grant = session.grant();
    keep_session(&provider, &args.agent, &session, sandbox).await?;
    let made = AppSessionCreateOutput {
        session: session.state,
        grant,
    };
    Ok((StatusCode::CREATED, Json(made)).into_response())
}

/// The agent of the session `id`, where `headers` carry its grant: `401` without a grant this
/// server gave, `403` with another session's.
async fn granted(
    provider: &RwLock<AgentProvider>,
    headers: &HeaderMap,
    id: &str,
) -> Result<String, Refused> {
    let found = match bearer(headers) {
        Some(grant) => {
            let hash = SessionRecord::hash_grant(grant);
            store(provider).find_grant(&hash).await?
        }
        None => None,
    };
    match found {
        Some((agent, session)) if session == id => Ok(agent),
        Some(_) => {
            let why = format!("the grant does not reach the session '{id}'");
            Err(Refused::new(StatusCode::FORBIDDEN, "forbidden", why))
        }
        None => {
            let why = "no grant this server gave is sent".to_owned();
            Err(Refused::new(StatusCode::UNAUTHORIZED, "unauthorized", why))
        }
    }
}

async fn list_sessions(
    State(provider): State<Arc<RwLock<AgentProvider>>>,
    Path(agent): Path<String>,
) -> Result<Json<Vec<SessionListOutputItem>>, Refused> {
    let sessions = store(&provider).list_sessions(&agent).await?;
    let sessions = sessions.ok_or(Refusal::NoAgent(agent))?;
    let sessions = sessions.into_iter().map(|session| SessionListOutputItem {
        message_count: session.state.history.len(),
        id: session.state.id,
    });
    Ok(Json(sessions.collect()))
}

/// The session `id` of the agent `agent`.
async fn session(
    provider: &RwLock<AgentProvider>,
    agent: &str,
    id: &str,
) -> Result<SessionRecord, Refused> {
    let session = store(provider).get_session(agent, id).await?;
    Ok(session.ok_or_else(|| Refusal::NoSession(id.to_owned()))?)
}

async fn get_session(
    State(provider): State<Arc<RwLock<AgentProvider>>>,
    Path((agent, id)): Path<(String, String)>,
) -> Result<Json<AgentState>, Refused> {
    Ok(Json(session(&provider, &agent, &id).await?.state))
}

async fn get_app_session(
    State(provider): State<Arc<RwLock<AgentProvider>>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<AgentState>, Refused> {
    let agent = granted(&provider, &headers, &id).await?;
    Ok(Json(session(&provider, &agent, &id).await?.state))
}

/// Keep the session `id` of the agent `agent` no longer, and remove its sandbox.
async fn remove(
    provider: &RwLock<AgentProvider>,
    agent: &str,
    id: &str,
) -> Result<StatusCode, Refused> {
    if let Some(session) = store(provider).remove_session(agent, id).await? {
        remove_sandbox(provider, &session).await;
    }
    Ok(StatusCode::NO_CONTENT)
}

async fn remove_session(
    State(provider): State<Arc<RwLock<AgentProvider>>>,
    Path((agent, id)): Path<(String, String)>,
) -> Result<StatusCode, Refused> {
    remove(&provider, &agent, &id).await
}

async fn remove_app_session(
    State(provider): State<Arc<RwLock<AgentProvider>>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<StatusCode, Refused> {
    let agent = granted(&provider, &headers, &id).await?;
    remove(&provider, &agent, &id).await
}

async fn run(
    State(provider): State<Arc<RwLock<AgentProvider>>>,
    headers: HeaderMap,
    Path((agent, id)): Path<(String, String)>,
    body: Bytes,
) -> Result<Response, Refused> {
    turn(provider, &headers, agent, id, body).await
}

async fn run_app(
    State(provider): State<Arc<RwLock<AgentProvider>>>,
    headers: HeaderMap,
    Path(id): Path<String>,
    body: Bytes,
) -> Result<Response, Refused> {
    let agent = granted(&provider, &headers, &id).await?;
    turn(provider, &headers, agent, id, body).await
}

/// The runtime of a turn of the session `id` of the agent `agent`, with the session as it is
/// kept; its model called with the key `headers` carry, where the agent's model is to be called
/// with the client's, and refused without one.
async fn runtime(
    provider: &RwLock<AgentProvider>,
    headers: &HeaderMap,
    agent: &str,
    id: &str,
) -> Result<(AgentRuntime, SessionRecord), Refused> {
    let args = store(provider).get_agent(agent).await?;
    let args = args.ok_or_else(|| Refusal::NoAgent(agent.to_owned()))?;
    let model_key = match args.model_key_provider {
        ModelKeyProvider::Server => None,
        ModelKeyProvider::Client => {
            let key = headers
                .get(MODEL_KEY_HEADER)
                .and_then(|key| key.to_str().ok());
            let why = format!(
                "the agent's model is called with the client's key, sent as `{MODEL_KEY_HEADER}`"
            );
            Some(key.ok_or_else(|| Refused::new(StatusCode::BAD_REQUEST, "no_model_key", why))?)
        }
    };
    let session = session(provider, agent, id).await?;
    let sandbox = reach_sandbox(provider, &session).await?;
    let state = session.state.clone();
    match AgentRuntime::try_in(provider, args.spec, state, sandbox, model_key).await {
        Ok(runtime) => Ok((runtime, session)),
        Err(e) => {
            let why = format!("{e:#}");
            Err(Refused::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "unusable",
                why,
            ))
        }
    }
}

/// Run one turn of the session `id` of the agent `agent` on the query `body` holds, and answer
/// with what it produces as an event stream, one `MessageDeltaOutput` per event. The turn runs on
/// a task of its own, so it ends, and the session keeps its history, whether or not the stream
/// is read to its end; the history is kept as each message is added to it, and once more as the
/// turn ends. A failure partway is an `error` event, `{"error": {"code", "message"}}`.
async fn turn(
    provider: Arc<RwLock<AgentProvider>>,
    headers: &HeaderMap,
    agent: String,
    id: String,
    body: Bytes,
) -> Result<Response, Refused> {
    let query: Message = args(&body)?;
    provider.write().unwrap().begin_turn(&id)?;
    let (mut runtime, mut session) = match runtime(&provider, headers, &agent, &id).await {
        Ok(begun) => begun,
        Err(refused) => {
            provider.write().unwrap().end_turn(&id);
            return Err(refused);
        }
    };

    // Each state to keep, kept in order on a task of its own, as the turn does not wait on it.
    let (keeps, mut kept) = futures::channel::mpsc::unbounded::<AgentState>();
    let keeper = {
        let store = store(&provider);
        let agent = agent.clone();
        tokio::spawn(async move {
            while let Some(state) = kept.next().await {
                session.state = state;
                match store.update_session(&agent, &session).await {
                    Ok(true) => {}
                    // Removed while its turn ran: it stays removed.
                    Ok(false) => break,
                    Err(e) => log::warn!("keeping the session '{}': {e:#}", session.state.id),
                }
            }
        })
    };
    let noted = keeps.clone();
    runtime.on_message(move |state| {
        let _ = noted.unbounded_send(state.clone());
    });

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
        let _ = keeps.unbounded_send(runtime.state.clone());
        drop((keeps, runtime));
        let _ = keeper.await;
        provider.write().unwrap().end_turn(&id);
        if let Some(failed) = failed {
            let _ = events.unbounded_send(Ok(failed));
        }
    });
    let body = Body::from_stream(body);
    Ok(([(header::CONTENT_TYPE, "text/event-stream")], body).into_response())
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
        AppClient, Client,
        agent::AgentSpec,
        message::{FinishReason, MessageDeltaOutput, MessageOutput, Part, Role, into_messages},
        model::{InferLangModel, LangModelOptions, LangModelSet, ModelKeyProvider},
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

    /// A client of a provider with `echo`, served in this process.
    fn client() -> Client {
        let mut provider = AgentProvider::new();
        provider.tools.set(Echo);
        // Models are never made here: an agent is only kept.
        for org in ["anthropic", "openai"] {
            provider.models.set(org, crate::model::Backend::Api);
        }
        Client::embed(admin_router(Arc::new(RwLock::new(provider))))
    }

    /// A client of a provider that keeps the agent `a`.
    async fn client_with_agent() -> Client {
        let client = client();
        let args = AgentPutArgs::new(AgentSpec::new("anthropic"));
        client.put_agent("a", &args).await.unwrap();
        client
    }

    #[tokio::test]
    async fn test_the_tools_provided_are_listed() {
        let tools = client().list_tool().await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo");
        assert_eq!(tools[0].description.as_deref(), Some("Say it back"));
    }

    #[tokio::test]
    async fn test_an_agent_is_put_listed_and_removed() {
        let client = client();
        let spec = AgentSpec::new("anthropic/claude-sonnet-5-5").tool("echo");
        let args = AgentPutArgs::new(spec).model_key_provider(ModelKeyProvider::Client);

        let put = client.put_agent("image editor", &args).await.unwrap();
        assert_eq!(put.name, "image editor");
        assert_eq!(put.model_key_provider, ModelKeyProvider::Client);

        let got = client.get_agent("image editor").await.unwrap();
        assert_eq!(got.spec.model, "anthropic/claude-sonnet-5-5");
        assert_eq!(got.spec.tools, ["echo"]);
        let agents = client.list_agent().await.unwrap();
        let names: Vec<&str> = agents.iter().map(|agent| agent.name.as_str()).collect();
        assert_eq!(names, ["image editor"]);

        client.remove_agent("image editor").await.unwrap();
        assert!(client.list_agent().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_an_agent_put_again_replaces_the_one_there() {
        let client = client();
        let first = AgentPutArgs::new(AgentSpec::new("anthropic"));
        let second = AgentPutArgs::new(AgentSpec::new("openai"));
        client.put_agent("a", &first).await.unwrap();
        client.put_agent("a", &second).await.unwrap();

        let agents = client.list_agent().await.unwrap();
        assert_eq!(agents.len(), 1);
        assert_eq!(agents[0].spec.model, "openai");
    }

    #[tokio::test]
    async fn test_an_agent_naming_a_tool_nothing_provides_is_refused() {
        let client = client();
        let args = AgentPutArgs::new(AgentSpec::new("anthropic").tools(["echo", "nowhere"]));
        let error = client.put_agent("a", &args).await.err().unwrap();
        assert_eq!(error.status(), Some(422));
        assert_eq!(error.code(), Some("unknown_tool"));
        assert!(error.to_string().contains("nowhere"), "{error}");
        assert!(client.list_agent().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_an_agent_of_a_model_nothing_makes_is_refused() {
        let args = AgentPutArgs::new(AgentSpec::new("nobody/x"));
        let error = client().put_agent("a", &args).await.err().unwrap();
        assert_eq!(error.status(), Some(422));
        assert_eq!(error.code(), Some("unknown_model"));
        assert!(error.to_string().contains("nobody"), "{error}");
    }

    #[tokio::test]
    async fn test_an_agent_not_kept_is_refused() {
        let error = client().get_agent("none").await.err().unwrap();
        assert_eq!(error.status(), Some(404));
        assert_eq!(error.code(), Some("not_found"));
    }

    #[tokio::test]
    async fn test_removing_an_agent_not_kept_does_nothing() {
        client().remove_agent("none").await.unwrap();
    }

    #[tokio::test]
    async fn test_a_session_is_started_listed_and_removed() {
        let client = client_with_agent().await;
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
        let client = client();
        let args = SessionCreateArgs::new();
        let error = client.create_session("none", &args).await.err().unwrap();
        assert_eq!(error.status(), Some(404));
        assert_eq!(error.code(), Some("not_found"));
        let error = client.list_session("none").await.err().unwrap();
        assert_eq!(error.status(), Some(404));
    }

    #[tokio::test]
    async fn test_sessions_stay_when_their_agent_is_put_again() {
        let client = client_with_agent().await;
        let session = client
            .create_session("a", &SessionCreateArgs::new())
            .await
            .unwrap();

        let args = AgentPutArgs::new(AgentSpec::new("openai"));
        client.put_agent("a", &args).await.unwrap();
        let state = client.get_session("a", &session.id).await.unwrap();
        assert_eq!(state.id, session.id);
    }

    #[tokio::test]
    async fn test_sessions_go_with_their_agent() {
        let client = client_with_agent().await;
        let session = client
            .create_session("a", &SessionCreateArgs::new())
            .await
            .unwrap();

        client.remove_agent("a").await.unwrap();
        let args = AgentPutArgs::new(AgentSpec::new("anthropic"));
        client.put_agent("a", &args).await.unwrap();
        assert!(client.list_session("a").await.unwrap().is_empty());
        let error = client.get_session("a", &session.id).await.err().unwrap();
        assert_eq!(error.status(), Some(404));
    }

    /// A session has a sandbox exactly when its agent's spec uses one, and one is made on the
    /// provider's virtx alone.
    #[tokio::test]
    async fn test_a_session_has_a_sandbox_as_its_spec_uses_one() {
        let client = client();
        let spec = AgentSpec::new("anthropic").sandbox();
        client
            .put_agent("s", &AgentPutArgs::new(spec))
            .await
            .unwrap();
        let error = (client.create_session("s", &SessionCreateArgs::new()).await)
            .err()
            .unwrap();
        assert_eq!(error.status(), Some(422));
        assert_eq!(error.code(), Some("unusable"));

        let args = SessionCreateArgs::new().sandbox(SessionSandbox::Package("p".into()));
        let error = client.create_session("s", &args).await.err().unwrap();
        assert_eq!(error.status(), Some(422));
        assert!(error.to_string().contains("no virtx"), "{error}");

        let client = client_with_agent().await;
        let error = client.create_session("a", &args).await.err().unwrap();
        assert_eq!(error.status(), Some(422));
        assert!(error.to_string().contains("uses none"), "{error}");
    }

    /// Answers `pong`, after a pause if `slow`, or fails, whatever it is asked.
    struct Parrot {
        fails: bool,
        slow: bool,
        /// What it answers: `pong`, unless given.
        says: Option<String>,
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
                    message: Message::new(Role::Assistant)
                        .with_contents([Part::text(self.says.as_deref().unwrap_or("pong"))]),
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

    /// A provider whose `parrot` org's models are [`Parrot`]s, failing ones under
    /// `parrot/fails` and slow ones under `parrot/slow`, whose agent `p` is of `spec`, and whose
    /// sandboxes are made through `virtx`.
    fn parrots(spec: AgentSpec, virtx: Option<virtx::Client>) -> Arc<RwLock<AgentProvider>> {
        let mut provider = AgentProvider::new();
        provider.models.set(
            "parrot",
            |_: &str, name: Option<&str>| -> anyhow::Result<Arc<dyn InferLangModel>> {
                Ok(Arc::new(Parrot {
                    fails: name == Some("fails"),
                    slow: name == Some("slow"),
                    says: None,
                }))
            },
        );
        // The tools a sandbox brings.
        provider.tools.set(crate::tool::Console);
        provider.virtx = virtx;
        futures::executor::block_on(provider.store.put_agent("p", &AgentPutArgs::new(spec)))
            .unwrap();
        Arc::new(RwLock::new(provider))
    }

    /// A client of [`parrots`] whose agent `p` runs `model`, with a session started.
    async fn parrot_session(model: &str) -> (Client, String) {
        let client = Client::embed(admin_router(parrots(AgentSpec::new(model), None)));
        let args = SessionCreateArgs::new();
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

    /// Makes [`Parrot`]s with a caller's key alone, each answering the key it was made with.
    struct Keyed;

    impl LangModelSet for Keyed {
        fn name(&self) -> String {
            "keyed".to_owned()
        }

        fn make(
            &self,
            _: &str,
            _: Option<&str>,
        ) -> BoxFuture<'static, anyhow::Result<Arc<dyn InferLangModel>>> {
            Box::pin(async { anyhow::bail!("no key of its own") })
        }

        fn takes_key(&self, _: &str) -> bool {
            true
        }

        fn make_with_key(
            &self,
            _: &str,
            _: Option<&str>,
            key: &str,
        ) -> BoxFuture<'static, anyhow::Result<Arc<dyn InferLangModel>>> {
            let says = Some(key.to_owned());
            let parrot = Parrot {
                fails: false,
                slow: false,
                says,
            };
            Box::pin(async move { Ok(Arc::new(parrot) as Arc<dyn InferLangModel>) })
        }
    }

    /// [`parrots`], with the `keyed` org's models made by [`Keyed`], and its agent `k` of
    /// `keyed/x` on the client's key.
    fn keyed() -> Arc<RwLock<AgentProvider>> {
        let provider = parrots(AgentSpec::new("parrot/x"), None);
        provider.write().unwrap().models.set("keyed", Keyed);
        let args = AgentPutArgs::new(AgentSpec::new("keyed/x"))
            .model_key_provider(ModelKeyProvider::Client);
        let store = provider.read().unwrap().store.clone();
        futures::executor::block_on(store.put_agent("k", &args)).unwrap();
        provider
    }

    #[tokio::test]
    async fn test_the_orgs_whose_models_are_made_are_listed() {
        let client = Client::embed(admin_router(keyed()));
        let models = client.list_model().await.unwrap();
        let listed: Vec<_> = (models.iter())
            .map(|model| (model.org.as_str(), model.backend.as_str(), model.takes_key))
            .collect();
        assert_eq!(
            listed,
            [("parrot", "custom", false), ("keyed", "keyed", true)]
        );
    }

    #[tokio::test]
    async fn test_an_agent_on_the_client_s_key_needs_models_made_with_one() {
        let client = Client::embed(admin_router(keyed()));
        let args = AgentPutArgs::new(AgentSpec::new("parrot/x"))
            .model_key_provider(ModelKeyProvider::Client);
        let error = client.put_agent("c", &args).await.err().unwrap();
        assert_eq!(error.status(), Some(422));
        assert_eq!(error.code(), Some("no_client_key"));
    }

    /// A turn of an agent on the client's key runs with the key the client sends, and is
    /// refused without one.
    #[tokio::test]
    async fn test_a_turn_on_the_client_s_key_calls_the_model_with_the_key_sent() {
        let provider = keyed();
        let args = AppSessionCreateArgs::new("k");
        let bare = AppClient::embed(router(provider.clone()), &args)
            .await
            .unwrap();
        let outcomes: Vec<_> = bare.run_stream(&ping()).collect().await;
        let error = outcomes[0].as_ref().err().unwrap();
        assert_eq!(error.status(), Some(400));
        assert_eq!(error.code(), Some("no_model_key"));

        let raw = http_over_x::Client::embed(router(provider.clone()));
        let key = header::HeaderValue::from_static("sk-secret");
        raw.set_header(header::HeaderName::from_static(MODEL_KEY_HEADER), Some(key));
        let keyed = AppClient::over(raw, &args).await.unwrap();
        let deltas = keyed.run_stream(&ping());
        let deltas = deltas.map(|delta| delta.map_err(anyhow::Error::from));
        let messages: Vec<_> = into_messages(deltas).collect().await;
        let said = messages[0].as_ref().unwrap().message.contents[0].as_text();
        assert_eq!(said, Some("sk-secret"));
    }

    /// A session kept goes on in a process that did not start it, sharing its store: its
    /// history as it was kept, and its sandbox reached again on virtx, and removed with it.
    #[tokio::test]
    async fn test_a_session_kept_goes_on_where_its_sandbox_is_not_held() {
        let virtx = crate::test_sandbox().await.client().clone();
        let spec = AgentSpec::new("parrot/x").sandbox();
        let first = parrots(spec.clone(), Some(virtx.clone()));
        let client = Client::embed(admin_router(first.clone()));
        let mut made = virtx::SandboxCreateArgs::new("python:3.12-slim");
        made.memory_mib = Some(1024);
        let args = SessionCreateArgs::new().sandbox(SessionSandbox::Create(made));
        let session = client.create_session("p", &args).await.unwrap();
        let sandbox = session.sandbox_id.clone().unwrap();
        let outcomes: Vec<_> = client.run_stream("p", &session.id, &ping()).collect().await;
        assert!(outcomes.iter().all(Result::is_ok), "{outcomes:?}");

        let second = parrots(spec, Some(virtx.clone()));
        second.write().unwrap().store = first.read().unwrap().store.clone();
        let client = Client::embed(admin_router(second));
        let outcomes: Vec<_> = client.run_stream("p", &session.id, &ping()).collect().await;
        assert!(outcomes.iter().all(Result::is_ok), "{outcomes:?}");
        let state = client.get_session("p", &session.id).await.unwrap();
        assert_eq!(state.history.len(), 4);

        client.remove_session("p", &session.id).await.unwrap();
        let gone = virtx.get_sandbox(&sandbox).await.err().unwrap();
        assert_eq!(gone.status(), Some(404));
    }

    /// What is kept of an app's grant is its hash, which reaches no session as a grant.
    #[tokio::test]
    async fn test_a_grant_is_kept_as_its_hash_alone() {
        let provider = parrots(AgentSpec::new("parrot/x"), None);
        let served = router(provider.clone());
        let raw = http_over_x::Client::embed(served);
        let args = AppSessionCreateArgs::new("p");
        let made: AppSessionCreateOutput = raw.post("/sessions/", &args).await.unwrap();

        let store = provider.read().unwrap().store.clone();
        let kept = store
            .get_session("p", &made.session.id)
            .await
            .unwrap()
            .unwrap();
        let hash = kept.grant_hash.unwrap();
        assert_ne!(hash, made.grant);
        assert_eq!(hash, SessionRecord::hash_grant(&made.grant));

        let path = format!("/sessions/{}", made.session.id);
        let as_grant = format!("Bearer {hash}").parse().unwrap();
        raw.set_header(header::AUTHORIZATION, Some(as_grant));
        let error = raw.get::<AgentState>(&path).await.err().unwrap();
        assert_eq!(error.status(), Some(401));
    }

    /// A session's sandbox is made for it on the provider's virtx, its turns work in it, and it
    /// goes with the session.
    #[tokio::test]
    async fn test_a_session_works_in_the_sandbox_made_for_it_until_it_goes() {
        let virtx = crate::test_sandbox().await.client().clone();
        let provider = parrots(AgentSpec::new("parrot/x").sandbox(), Some(virtx.clone()));
        let client = Client::embed(admin_router(provider));
        let mut made = virtx::SandboxCreateArgs::new("python:3.12-slim");
        made.memory_mib = Some(1024);
        let args = SessionCreateArgs::new().sandbox(SessionSandbox::Create(made));
        let session = client.create_session("p", &args).await.unwrap();
        let sandbox = session.sandbox_id.clone().unwrap();
        assert!(virtx.get_sandbox(&sandbox).await.is_ok());

        let outcomes: Vec<_> = client.run_stream("p", &session.id, &ping()).collect().await;
        assert!(outcomes.iter().all(Result::is_ok), "{outcomes:?}");
        let state = client.get_session("p", &session.id).await.unwrap();
        assert_eq!(state.sandbox_id.as_deref(), Some(sandbox.as_str()));
        assert_eq!(state.history.len(), 2);

        client.remove_session("p", &session.id).await.unwrap();
        let gone = virtx.get_sandbox(&sandbox).await.err().unwrap();
        assert_eq!(gone.status(), Some(404));
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
    async fn test_a_turn_of_an_agent_that_cannot_be_made_is_refused() {
        // It drives a browser, and its session has no sandbox to have one in.
        let mut spec = AgentSpec::new("parrot/x");
        spec.use_browser = true;
        let client = Client::embed(admin_router(parrots(spec, None)));
        let args = SessionCreateArgs::new();
        let id = client.create_session("p", &args).await.unwrap().id;
        let outcomes: Vec<_> = client.run_stream("p", &id, &ping()).collect().await;
        let error = outcomes[0].as_ref().err().unwrap();
        assert_eq!(error.status(), Some(422));
        assert_eq!(error.code(), Some("unusable"));
        // The session is free again for the next turn.
        let outcomes: Vec<_> = client.run_stream("p", &id, &ping()).collect().await;
        assert_eq!(outcomes[0].as_ref().err().unwrap().code(), Some("unusable"));
    }

    /// An app's session runs its turns and goes with its grant alone.
    #[tokio::test]
    async fn test_an_app_reaches_its_session_and_removes_it() {
        let provider = parrots(AgentSpec::new("parrot/x"), None);
        let client = Client::embed(admin_router(provider.clone()));
        let args = AppSessionCreateArgs::new("p");
        let app = AppClient::embed(router(provider), &args).await.unwrap();

        let outcomes: Vec<_> = app.run_stream(&ping()).collect().await;
        assert!(outcomes.iter().all(Result::is_ok), "{outcomes:?}");
        let state = app.get_session().await.unwrap();
        assert_eq!(state.history.len(), 2);
        assert_eq!(
            client
                .get_session("p", app.id())
                .await
                .unwrap()
                .history
                .len(),
            2
        );

        let id = app.id().to_owned();
        app.close().await.unwrap();
        let error = client.get_session("p", &id).await.err().unwrap();
        assert_eq!(error.status(), Some(404));
    }

    /// A session is reached with its own grant, and with none or another's is refused.
    #[tokio::test]
    async fn test_an_app_session_is_reached_with_its_own_grant_alone() {
        let served = router(parrots(AgentSpec::new("parrot/x"), None));
        let raw = http_over_x::Client::embed(served);
        let args = AppSessionCreateArgs::new("p");
        let mine: AppSessionCreateOutput = raw.post("/sessions/", &args).await.unwrap();
        let theirs: AppSessionCreateOutput = raw.post("/sessions/", &args).await.unwrap();

        let path = format!("/sessions/{}", theirs.session.id);
        let error = raw.get::<AgentState>(&path).await.err().unwrap();
        assert_eq!(error.status(), Some(401));
        let grant = format!("Bearer {}", mine.grant).parse().unwrap();
        raw.set_header(header::AUTHORIZATION, Some(grant));
        let error = raw.get::<AgentState>(&path).await.err().unwrap();
        assert_eq!(error.status(), Some(403));
        let path = format!("/sessions/{}", mine.session.id);
        let state: AgentState = raw.get(&path).await.unwrap();
        assert_eq!(state.id, mine.session.id);
    }

    /// An app's session of an agent that uses a sandbox names a package, which a server
    /// without a virtx URL, as over stdio, makes none of.
    #[tokio::test]
    async fn test_an_app_session_needs_a_package_where_its_agent_uses_a_sandbox() {
        let provider = parrots(AgentSpec::new("parrot/x").sandbox(), None);
        let args = AppSessionCreateArgs::new("p");
        let error = AppClient::embed(router(provider.clone()), &args).await;
        let error = error.err().unwrap();
        assert_eq!(error.status(), Some(422));
        assert!(error.to_string().contains("uses a sandbox"), "{error}");

        let args = args.package("somewhere");
        let error = AppClient::embed(router(provider), &args)
            .await
            .err()
            .unwrap();
        assert_eq!(error.status(), Some(422));
        assert!(
            error.to_string().contains("no sandbox for an app"),
            "{error}"
        );
    }

    /// Each router serves its own client's paths and none of the other's.
    #[tokio::test]
    async fn test_the_routers_share_no_path() {
        let provider = parrots(AgentSpec::new("parrot/x"), None);
        let client = Client::embed(router(provider.clone()));
        assert_eq!(client.list_agent().await.err().unwrap().status(), Some(404));
        let args = AppSessionCreateArgs::new("p");
        let error = AppClient::embed(admin_router(provider), &args).await;
        assert_eq!(error.err().unwrap().status(), Some(404));
    }
}
