//! The HTTP surface: events, triggers, runs, health.

use std::{collections::HashMap, sync::Arc};

use axum::{
    Json, Router,
    body::Bytes,
    extract::{Path, Query, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{console, db, dispatcher, state::AppState, trigger::TriggerConfig};

type S = State<Arc<AppState>>;
type ApiResult = Result<Json<Value>, ApiError>;

pub struct ApiError(StatusCode, Value);

impl ApiError {
    fn bad(problems: Vec<String>) -> Self {
        ApiError(StatusCode::BAD_REQUEST, json!({ "errors": problems }))
    }
    fn not_found(what: &str) -> Self {
        ApiError(
            StatusCode::NOT_FOUND,
            json!({ "errors": [format!("{what} not found")] }),
        )
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        ApiError(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({ "errors": [format!("{e:#}")] }),
        )
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(self.1)).into_response()
    }
}

/// The listener a request arrived on decides what it may ask for, so the routes are
/// mounted twice: everything on the socket, event publishing alone on TCP.
pub fn socket_router(state: Arc<AppState>) -> Router {
    routes().with_state(state)
}

/// `POST /events/{type}` and nothing else, behind the token check.
pub fn tcp_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/events/{type}", post(publish))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            authorize,
        ))
        .fallback(|| async {
            ApiError(
                StatusCode::NOT_FOUND,
                json!({ "errors": ["this address publishes events; administration is on the daemon's socket"] }),
            )
        })
        .with_state(state)
}

fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/health", get(health))
        .route("/events", get(list_events))
        .route("/events/{type}", post(publish))
        .route("/triggers", get(list_triggers))
        .route(
            "/triggers/{name}",
            get(get_trigger).put(put_trigger).delete(delete_trigger),
        )
        .route("/triggers/{name}/fire", post(fire_trigger))
        .route("/triggers/{name}/runs", get(trigger_runs))
        .route("/runs", get(list_runs))
        .route("/runs/{id}", get(get_run))
        .route("/runs/{id}/artifacts", get(run_artifacts))
        .route("/tokens", get(list_tokens).post(issue_token))
        .route("/tokens/{id}", axum::routing::delete(revoke_token))
}

/// What the TCP listener asks of a request. Two credentials pass: the value of
/// [`TOKEN_ENV`](crate::config::TOKEN_ENV), and a token issued for the event type the
/// path names. Without the variable the address asks for nothing.
/// The socket asks nothing either: opening the file is the authorization.
async fn authorize(
    State(s): S,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Result<Response, ApiError> {
    let Ok(expected) = std::env::var(crate::config::TOKEN_ENV) else {
        return Ok(next.run(req).await);
    };
    let offered = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or_default();

    // Digests rather than the tokens: equal length, so the comparison tells an
    // attacker nothing about where a guess went wrong.
    let admin = Sha256::digest(offered.as_bytes()) == Sha256::digest(expected.as_bytes());
    if admin || posts_events_of(&req).is_some_and(|kind| issued_for(&s, offered, kind)) {
        return Ok(next.run(req).await);
    }
    Err(ApiError(
        StatusCode::UNAUTHORIZED,
        json!({ "errors": ["bad or missing token"] }),
    ))
}

/// The event type a request appends to, for `POST /events/{type}` alone.
fn posts_events_of(req: &axum::extract::Request) -> Option<&str> {
    (req.method() == axum::http::Method::POST)
        .then(|| req.uri().path().strip_prefix("/events/"))
        .flatten()
        .filter(|rest| !rest.is_empty() && !rest.contains('/'))
}

/// Whether an issued token stands for this event type.
fn issued_for(s: &AppState, offered: &str, kind: &str) -> bool {
    if offered.is_empty() {
        return false;
    }
    let digest = hex::encode(Sha256::digest(offered.as_bytes()));
    matches!(s.db.token_by_digest(&digest), Ok(Some(t)) if t.kind == kind)
}

// ----- tokens -----

/// `POST /tokens`: issue a credential for posting events of one type. The secret is
/// in this answer and nowhere else.
async fn issue_token(State(s): S, Json(body): Json<Value>) -> ApiResult {
    let kind = body
        .get("type")
        .and_then(Value::as_str)
        .filter(|k| !k.is_empty() && !k.starts_with("runs:"))
        .ok_or_else(|| {
            ApiError::bad(vec![
                "type: a non-empty event type not starting with `runs:` is required".into(),
            ])
        })?;
    let secret = format!(
        "aly_{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    );
    let id = format!("tok_{}", &uuid::Uuid::new_v4().simple().to_string()[..8]);
    let row =
        s.db.issue_token(&id, kind, &hex::encode(Sha256::digest(secret.as_bytes())))?;
    Ok(Json(json!({
        "id": row.id,
        "type": row.kind,
        "created_at": row.created_at,
        "token": secret,
    })))
}

async fn list_tokens(State(s): S) -> ApiResult {
    Ok(Json(json!(s.db.list_tokens()?)))
}

async fn revoke_token(State(s): S, Path(id): Path<String>) -> ApiResult {
    if !s.db.revoke_token(&id)? {
        return Err(ApiError::not_found("token"));
    }
    Ok(Json(json!({ "revoked": id })))
}

fn limit(q: &HashMap<String, String>) -> u32 {
    q.get("limit").and_then(|s| s.parse().ok()).unwrap_or(50)
}

// ----- health -----

async fn health(State(s): S) -> ApiResult {
    let triggers = s.db.list_triggers()?;
    let failing: Vec<&str> = triggers
        .iter()
        .filter(|t| t.last_error.is_some())
        .map(|t| t.name.as_str())
        .collect();
    let console_up = s.console.lock().await.is_up();
    Ok(Json(json!({
        "ok": true,
        "triggers": triggers.len(),
        "failing_triggers": failing,
        "trigger_console_up": console_up,
        "active_runs": s.db.count_active_runs()?,
    })))
}

// ----- events -----

/// `POST /events/{type}`: append one event.
async fn publish(State(s): S, Path(kind): Path<String>, body: Bytes) -> ApiResult {
    if kind.is_empty() || kind.starts_with("runs:") {
        return Err(ApiError::bad(vec![
            "event type must be non-empty and not start with `runs:`".into(),
        ]));
    }
    let payload: Value = if body.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&body)
            .map_err(|e| ApiError::bad(vec![format!("body is not JSON: {e}")]))?
    };
    let id = s.publisher.publish(&kind, payload)?;
    Ok(Json(json!({ "event_id": id, "type": kind })))
}

async fn list_events(State(s): S, Query(q): Query<HashMap<String, String>>) -> ApiResult {
    Ok(Json(json!(s.db.list_events(
        q.get("type").map(String::as_str),
        limit(&q)
    )?)))
}

// ----- triggers -----

async fn list_triggers(State(s): S) -> ApiResult {
    Ok(Json(json!(s.db.list_triggers()?)))
}

async fn get_trigger(State(s): S, Path(name): Path<String>) -> ApiResult {
    let t =
        s.db.get_trigger(&name)?
            .ok_or_else(|| ApiError::not_found("trigger"))?;
    Ok(Json(json!({
        "trigger": t,
        "config_hash": t.config.hash(),
        "runs": {
            "pending": s.db.count_runs(&name, db::PENDING)?,
            "running": s.db.count_runs(&name, db::RUNNING)?,
            "done": s.db.count_runs(&name, db::DONE)?,
            "failed": s.db.count_runs(&name, db::FAILED)?,
        }
    })))
}

async fn put_trigger(
    State(s): S,
    Path(name): Path<String>,
    Json(config): Json<TriggerConfig>,
) -> ApiResult {
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
    {
        return Err(ApiError::bad(vec![
            "trigger name must be letters, digits, - or _".into(),
        ]));
    }
    let problems = config.validate();
    if !problems.is_empty() {
        return Err(ApiError::bad(problems));
    }
    s.db.upsert_trigger(&name, &config)?;
    s.sources.lock().await.start(&name, &config);
    s.console.lock().await.invalidate();
    s.publisher.wake.notify_one();
    Ok(Json(
        json!({ "name": name, "config_hash": config.hash(), "config": config }),
    ))
}

async fn delete_trigger(State(s): S, Path(name): Path<String>) -> ApiResult {
    if !s.db.delete_trigger(&name)? {
        return Err(ApiError::not_found("trigger"));
    }
    s.sources.lock().await.stop(&name);
    dispatcher::clear_trigger_dir(&s.trigger_dir(&name));
    s.console.lock().await.invalidate();
    Ok(Json(json!({ "deleted": name })))
}

async fn fire_trigger(State(s): S, Path(name): Path<String>, body: Bytes) -> ApiResult {
    let t =
        s.db.get_trigger(&name)?
            .ok_or_else(|| ApiError::not_found("trigger"))?;
    let payload: Value = if body.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&body)
            .map_err(|e| ApiError::bad(vec![format!("body is not JSON: {e}")]))?
    };
    let _ = t;
    let id = uuid::Uuid::new_v4().simple().to_string();
    s.db.insert_run(&id, &name, &payload)?;
    s.worker_wake.notify_one();
    Ok(Json(json!({ "run_id": id })))
}

async fn trigger_runs(
    State(s): S,
    Path(name): Path<String>,
    Query(q): Query<HashMap<String, String>>,
) -> ApiResult {
    Ok(Json(json!(s.db.list_runs(
        Some(&name),
        q.get("status").map(String::as_str),
        limit(&q)
    )?)))
}

// ----- runs -----

async fn list_runs(State(s): S, Query(q): Query<HashMap<String, String>>) -> ApiResult {
    Ok(Json(json!(s.db.list_runs(
        q.get("trigger").map(String::as_str),
        q.get("status").map(String::as_str),
        limit(&q)
    )?)))
}

async fn get_run(State(s): S, Path(id): Path<String>) -> ApiResult {
    let run =
        s.db.get_run(&id)?
            .ok_or_else(|| ApiError::not_found("run"))?;
    let record: Option<Value> = std::fs::read(
        s.run_dir(&run.trigger, &run.id)
            .join(ailoy::automation::RECORD_FILE),
    )
    .ok()
    .and_then(|b| serde_json::from_slice(&b).ok());
    let summary = record.as_ref().map(|r| {
        json!({
            "outcome": r.get("outcome"),
            "output": r.get("output"),
            "tasks": r.get("tasks").and_then(Value::as_array).map(|tasks| {
                tasks.iter().map(|t| json!({
                    "name": t.get("name"), "outcome": t.get("outcome"),
                })).collect::<Vec<_>>()
            }),
        })
    });
    Ok(Json(json!({ "run": run, "record": summary })))
}

async fn run_artifacts(State(s): S, Path(id): Path<String>) -> Result<Response, ApiError> {
    let run =
        s.db.get_run(&id)?
            .ok_or_else(|| ApiError::not_found("run"))?;
    let dir = s.run_dir(&run.trigger, &run.id).join("artifacts");
    if !dir.is_dir() {
        return Err(ApiError::not_found("run directory"));
    }
    let mut tar = tar::Builder::new(Vec::new());
    tar.append_dir_all(".", &dir)
        .map_err(|e| anyhow::anyhow!("packing artifacts: {e}"))?;
    let bytes = tar.into_inner().map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(([(header::CONTENT_TYPE, "application/x-tar")], bytes).into_response())
}

// Keep the console module linked even when only the dispatcher uses it directly.
#[allow(dead_code)]
fn _uses_console(_: &console::SharedConsole) {}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use reqwest::Method;

    use super::*;
    use crate::{cli::send_over_socket, config::Config, db::Db};

    /// `TOKEN_ENV` is one variable for the whole process, so the tests that set it
    /// take turns.
    static ENV: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    /// Hold the turn and put `TOKEN_ENV` where the test needs it.
    async fn with_token(token: Option<&str>) -> tokio::sync::MutexGuard<'static, ()> {
        let guard = ENV.lock().await;
        unsafe {
            match token {
                Some(t) => std::env::set_var(crate::config::TOKEN_ENV, t),
                None => std::env::remove_var(crate::config::TOKEN_ENV),
            }
        }
        guard
    }

    fn state_with(root: PathBuf) -> Arc<AppState> {
        AppState::new(root, Config::default(), Db::open_in_memory().unwrap())
    }

    /// The TCP listener, on an ephemeral port: event publishing alone.
    async fn serve_tcp(state: Arc<AppState>) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, tcp_router(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    /// The socket listener: every route, no credential.
    async fn serve_socket(state: Arc<AppState>, dir: &Path) -> PathBuf {
        let path = dir.join("daemon.sock");
        let listener = tokio::net::UnixListener::bind(&path).unwrap();
        tokio::spawn(async move {
            axum::serve(listener, socket_router(state)).await.unwrap();
        });
        path
    }

    async fn get(base: &str, path: &str, token: Option<&str>) -> u16 {
        let req = reqwest::Client::new().get(format!("{base}{path}"));
        let req = match token {
            Some(t) => req.bearer_auth(t),
            None => req,
        };
        req.send().await.unwrap().status().as_u16()
    }

    async fn post_event(base: &str, kind: &str, token: Option<&str>) -> u16 {
        let req = reqwest::Client::new()
            .post(format!("{base}/events/{kind}"))
            .json(&json!({ "hello": 1 }));
        let req = match token {
            Some(t) => req.bearer_auth(t),
            None => req,
        };
        req.send().await.unwrap().status().as_u16()
    }

    #[tokio::test]
    async fn tcp_publishes_events_and_serves_nothing_else() {
        let _turn = with_token(None).await;
        let dir = tempfile::tempdir().unwrap();
        let base = serve_tcp(state_with(dir.path().to_path_buf())).await;
        assert_eq!(post_event(&base, "tickets", None).await, 200);
        assert_eq!(get(&base, "/health", None).await, 404);
        assert_eq!(get(&base, "/triggers", None).await, 404);
    }

    #[tokio::test]
    async fn a_token_in_the_environment_makes_tcp_ask_for_it() {
        let _turn = with_token(Some("s3cret")).await;
        let dir = tempfile::tempdir().unwrap();
        let base = serve_tcp(state_with(dir.path().to_path_buf())).await;
        assert_eq!(post_event(&base, "tickets", None).await, 401);
        assert_eq!(post_event(&base, "tickets", Some("wrong")).await, 401);
        assert_eq!(post_event(&base, "tickets", Some("s3cret")).await, 200);
    }

    #[tokio::test]
    async fn the_socket_serves_everything_without_a_credential() {
        let _turn = with_token(Some("admin")).await;
        let dir = tempfile::tempdir().unwrap();
        let state = state_with(dir.path().to_path_buf());
        let sock = serve_socket(state, dir.path()).await;

        let (status, body) = send_over_socket(&sock, Method::GET, "/health", Vec::new())
            .await
            .unwrap();
        assert_eq!(status.as_u16(), 200, "{}", String::from_utf8_lossy(&body));
        let (status, _) = send_over_socket(&sock, Method::GET, "/triggers", Vec::new())
            .await
            .unwrap();
        assert_eq!(status.as_u16(), 200);
    }

    #[tokio::test]
    async fn an_issued_token_posts_its_own_type_and_nothing_else() {
        let _turn = with_token(Some("admin")).await;
        let dir = tempfile::tempdir().unwrap();
        let state = state_with(dir.path().to_path_buf());
        let sock = serve_socket(state.clone(), dir.path()).await;
        let base = serve_tcp(state).await;

        let (status, body) = send_over_socket(
            &sock,
            Method::POST,
            "/tokens",
            serde_json::to_vec(&json!({ "type": "tickets" })).unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(status.as_u16(), 200, "{}", String::from_utf8_lossy(&body));
        let issued: Value = serde_json::from_slice(&body).unwrap();
        let token = issued["token"].as_str().unwrap().to_string();

        assert_eq!(post_event(&base, "tickets", Some(&token)).await, 200);
        assert_eq!(post_event(&base, "deploys", Some(&token)).await, 401);
        assert_eq!(get(&base, "/tokens", Some(&token)).await, 404, "not on tcp");

        let id = issued["id"].as_str().unwrap();
        let (status, _) =
            send_over_socket(&sock, Method::DELETE, &format!("/tokens/{id}"), Vec::new())
                .await
                .unwrap();
        assert_eq!(status.as_u16(), 200);
        assert_eq!(post_event(&base, "tickets", Some(&token)).await, 401);
    }
}
