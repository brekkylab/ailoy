//! The thin client: every command is one API call, printed as JSON.

use std::path::{Path, PathBuf};

use clap::Subcommand;
use reqwest::{Method, StatusCode};
use serde_json::{Value, json};

use crate::trigger;

#[derive(Subcommand, Debug)]
pub enum Events {
    /// Post one event of a type; the payload is `--json` or stdin.
    Publish {
        #[arg(value_name = "EVENT_TYPE")]
        kind: String,
        #[arg(long)]
        json: Option<String>,
    },
    List {
        #[arg(long = "type")]
        kind: Option<String>,
        #[arg(long, default_value_t = 50)]
        limit: u32,
    },
}

#[derive(Subcommand, Debug)]
pub enum Triggers {
    List,
    Show {
        name: String,
    },
    /// Upload an automation directory's trigger.json as a registration.
    Register {
        dir: PathBuf,
        /// Defaults to the directory's name.
        #[arg(long)]
        name: Option<String>,
        /// Draft file; defaults to `<dir>/trigger.json`.
        #[arg(long)]
        file: Option<PathBuf>,
    },
    Delete {
        name: String,
    },
    /// Create one run directly from a payload (`--json` or stdin), bypassing the script.
    Fire {
        name: String,
        #[arg(long)]
        json: Option<String>,
    },
}

#[derive(Subcommand, Debug)]
pub enum Runs {
    List {
        #[arg(long)]
        trigger: Option<String>,
        #[arg(long)]
        status: Option<String>,
        #[arg(long, default_value_t = 50)]
        limit: u32,
    },
    Show {
        id: String,
    },
    /// Save the run's artifacts as a tar file.
    Artifacts {
        id: String,
        #[arg(long, default_value = "artifacts.tar")]
        out: PathBuf,
    },
}

#[derive(Subcommand, Debug)]
pub enum Tokens {
    /// Issue a credential that may post events of one type. The secret is printed
    /// once and is not stored.
    Issue {
        #[arg(value_name = "EVENT_TYPE")]
        kind: String,
    },
    List,
    Revoke {
        id: String,
    },
}

/// Where the daemon is: its administration socket, or a TCP address that carries
/// event publishing alone.
#[derive(Clone, Debug)]
pub enum Target {
    Socket(PathBuf),
    Tcp(String),
}

pub struct Client {
    target: Target,
    token: Option<String>,
    http: reqwest::Client,
}

impl Client {
    pub fn new(target: Target, token: Option<String>) -> Self {
        Self {
            target,
            token,
            http: reqwest::Client::new(),
        }
    }

    /// One request, answered as its status and body. The socket carries no credential:
    /// opening the file is the authorization.
    async fn send(
        &self,
        method: Method,
        path: &str,
        body: Option<Value>,
    ) -> anyhow::Result<(StatusCode, Vec<u8>)> {
        let body = match body {
            Some(v) => serde_json::to_vec(&v)?,
            None => Vec::new(),
        };
        match &self.target {
            Target::Tcp(base) => {
                let mut req = self
                    .http
                    .request(method, format!("{}{path}", base.trim_end_matches('/')));
                if let Some(t) = &self.token {
                    req = req.bearer_auth(t);
                }
                if !body.is_empty() {
                    req = req.header("content-type", "application/json").body(body);
                }
                let resp = req.send().await?;
                let status = resp.status();
                Ok((status, resp.bytes().await?.to_vec()))
            }
            Target::Socket(sock) => send_over_socket(sock, method, path, body).await,
        }
    }

    async fn call(&self, method: Method, path: &str, body: Option<Value>) -> anyhow::Result<Value> {
        let (status, bytes) = self.send(method, path, body).await?;
        let value: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        if !status.is_success() {
            anyhow::bail!("{status}: {}", serde_json::to_string_pretty(&value)?);
        }
        Ok(value)
    }

    pub async fn events(&self, cmd: Events) -> anyhow::Result<Value> {
        match cmd {
            Events::Publish { kind, json } => {
                let payload = json_arg(json, None, "payload").or_else(|_| stdin_json())?;
                self.call(Method::POST, &format!("/events/{kind}"), Some(payload))
                    .await
            }
            Events::List { kind, limit } => {
                let mut q = vec![format!("limit={limit}")];
                if let Some(k) = kind {
                    q.push(format!("type={k}"));
                }
                self.call(Method::GET, &format!("/events?{}", q.join("&")), None)
                    .await
            }
        }
    }

    pub async fn triggers(&self, cmd: Triggers) -> anyhow::Result<Value> {
        match cmd {
            Triggers::List => self.call(Method::GET, "/triggers", None).await,
            Triggers::Show { name } => {
                self.call(Method::GET, &format!("/triggers/{name}"), None)
                    .await
            }
            Triggers::Register { dir, name, file } => {
                let name = name.unwrap_or(dir_name(&dir)?);
                let config = trigger::read_draft(&dir, file.as_deref())?;
                self.call(
                    Method::PUT,
                    &format!("/triggers/{name}"),
                    Some(serde_json::to_value(config)?),
                )
                .await
            }
            Triggers::Delete { name } => {
                self.call(Method::DELETE, &format!("/triggers/{name}"), None)
                    .await
            }
            Triggers::Fire { name, json } => {
                let payload = json_arg(json, None, "payload").or_else(|_| stdin_json())?;
                self.call(
                    Method::POST,
                    &format!("/triggers/{name}/fire"),
                    Some(payload),
                )
                .await
            }
        }
    }

    pub async fn tokens(&self, cmd: Tokens) -> anyhow::Result<Value> {
        match cmd {
            Tokens::Issue { kind } => {
                self.call(Method::POST, "/tokens", Some(json!({ "type": kind })))
                    .await
            }
            Tokens::List => self.call(Method::GET, "/tokens", None).await,
            Tokens::Revoke { id } => {
                self.call(Method::DELETE, &format!("/tokens/{id}"), None)
                    .await
            }
        }
    }

    pub async fn runs(&self, cmd: Runs) -> anyhow::Result<Value> {
        match cmd {
            Runs::List {
                trigger,
                status,
                limit,
            } => {
                let mut q = vec![format!("limit={limit}")];
                if let Some(t) = trigger {
                    q.push(format!("trigger={t}"));
                }
                if let Some(s) = status {
                    q.push(format!("status={s}"));
                }
                self.call(Method::GET, &format!("/runs?{}", q.join("&")), None)
                    .await
            }
            Runs::Show { id } => self.call(Method::GET, &format!("/runs/{id}"), None).await,
            Runs::Artifacts { id, out } => {
                let (status, bytes) = self
                    .send(Method::GET, &format!("/runs/{id}/artifacts"), None)
                    .await?;
                if !status.is_success() {
                    anyhow::bail!("{status}: {}", String::from_utf8_lossy(&bytes));
                }
                std::fs::write(&out, &bytes)?;
                Ok(json!({ "saved": out, "bytes": bytes.len() }))
            }
        }
    }
}

fn dir_name(dir: &Path) -> anyhow::Result<String> {
    std::path::absolute(dir)?
        .file_name()
        .and_then(|n| n.to_str())
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("{} has no usable name; pass --name", dir.display()))
}

fn json_arg(json: Option<String>, file: Option<PathBuf>, what: &str) -> anyhow::Result<Value> {
    match (json, file) {
        (Some(j), _) => Ok(serde_json::from_str(&j)?),
        (None, Some(f)) => Ok(serde_json::from_slice(&std::fs::read(&f)?)?),
        (None, None) => anyhow::bail!("give the {what} with --json or --file"),
    }
}

fn stdin_json() -> anyhow::Result<Value> {
    let mut s = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut s)?;
    Ok(serde_json::from_str(&s)?)
}

/// One HTTP/1 exchange over a unix socket. `reqwest` has no unix transport, so the
/// connection is made by hand and dropped with the answer.
pub(crate) async fn send_over_socket(
    path: &Path,
    method: Method,
    target: &str,
    body: Vec<u8>,
) -> anyhow::Result<(StatusCode, Vec<u8>)> {
    use http_body_util::{BodyExt, Full};

    let stream = tokio::net::UnixStream::connect(path)
        .await
        .map_err(|e| anyhow::anyhow!("connecting to {}: {e}", path.display()))?;
    let (mut sender, conn) =
        hyper::client::conn::http1::handshake(hyper_util::rt::TokioIo::new(stream)).await?;
    tokio::spawn(conn);

    let mut req = hyper::Request::builder()
        .method(method.as_str())
        .uri(target)
        .header(hyper::header::HOST, "localhost");
    if !body.is_empty() {
        req = req.header(hyper::header::CONTENT_TYPE, "application/json");
    }
    let resp = sender
        .send_request(req.body(Full::new(bytes::Bytes::from(body)))?)
        .await?;
    let status = StatusCode::from_u16(resp.status().as_u16())?;
    Ok((
        status,
        resp.into_body().collect().await?.to_bytes().to_vec(),
    ))
}
