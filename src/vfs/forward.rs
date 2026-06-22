use std::{net::SocketAddr, sync::Arc};

use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
    runtime::Handle,
    task::JoinHandle,
};

use crate::vfs::{Vfs, path::VPath, resource::FileKind};

/// Host-side forward server exposing a [`Vfs`] over a tiny HTTP/1.1 API for the
/// in-guest FUSE forwarder. Bound to an OS-assigned ephemeral port; requests
/// must carry the session token in the `x-vfs-token` header. Aborts on drop.
///
/// Routes: `GET /readdir|/stat|/read?path=…[&offset=&size=]`, `PUT /write?path=…`.
pub struct VfsForward {
    addr: SocketAddr,
    token: String,
    task: JoinHandle<()>,
}

impl Drop for VfsForward {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl VfsForward {
    pub fn spawn(vfs: Arc<Vfs>, rt: &Handle) -> anyhow::Result<Self> {
        let listener = std::net::TcpListener::bind(("0.0.0.0", 0))?;
        listener.set_nonblocking(true)?;
        let addr = listener.local_addr()?;

        let mut raw = [0u8; 24];
        getrandom::fill(&mut raw).map_err(|e| anyhow::anyhow!("token rng: {e}"))?;
        let token = hex::encode(raw);

        let task_token = token.clone();
        let task = rt.spawn(async move {
            let listener = match tokio::net::TcpListener::from_std(listener) {
                Ok(l) => l,
                Err(e) => {
                    log::error!("vfs forward: listener init failed: {e}");
                    return;
                }
            };
            loop {
                match listener.accept().await {
                    Ok((stream, _)) => {
                        let vfs = vfs.clone();
                        let token = task_token.clone();
                        tokio::spawn(async move {
                            if let Err(e) = handle_conn(stream, vfs, token).await {
                                log::debug!("vfs forward: connection error: {e}");
                            }
                        });
                    }
                    Err(e) => {
                        log::debug!("vfs forward: accept error: {e}");
                    }
                }
            }
        });

        Ok(Self { addr, token, task })
    }

    pub fn port(&self) -> u16 {
        self.addr.port()
    }

    pub fn token(&self) -> &str {
        &self.token
    }
}

struct Req {
    method: String,
    path: String,
    query: String,
    token: Option<String>,
    content_length: usize,
    body_prefix: Vec<u8>,
}

async fn read_request(stream: &mut TcpStream) -> anyhow::Result<Req> {
    let mut buf = Vec::with_capacity(1024);
    let mut tmp = [0u8; 4096];
    let header_end = loop {
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            break pos;
        }
        let n = stream.read(&mut tmp).await?;
        if n == 0 {
            anyhow::bail!("connection closed before headers");
        }
        buf.extend_from_slice(&tmp[..n]);
        if buf.len() > 64 * 1024 {
            anyhow::bail!("request header too large");
        }
    };

    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut lines = head.split("\r\n");
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split(' ');
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("/");
    let (path, query) = match target.split_once('?') {
        Some((p, q)) => (p.to_string(), q.to_string()),
        None => (target.to_string(), String::new()),
    };

    let mut token = None;
    let mut content_length = 0usize;
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            let key = k.trim().to_ascii_lowercase();
            let val = v.trim();
            match key.as_str() {
                "x-vfs-token" => token = Some(val.to_string()),
                "content-length" => content_length = val.parse().unwrap_or(0),
                _ => {}
            }
        }
    }

    let body_prefix = buf[header_end + 4..].to_vec();
    Ok(Req {
        method,
        path,
        query,
        token,
        content_length,
        body_prefix,
    })
}

async fn handle_conn(mut stream: TcpStream, vfs: Arc<Vfs>, token: String) -> anyhow::Result<()> {
    let req = read_request(&mut stream).await?;

    if req.token.as_deref() != Some(token.as_str()) {
        return respond(&mut stream, 403, "text/plain", b"forbidden".to_vec()).await;
    }

    let params = parse_query(&req.query);
    let path = params
        .get("path")
        .cloned()
        .unwrap_or_else(|| "/".to_string());

    let result = match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/readdir") => readdir_json(&vfs, &path).await,
        ("GET", "/stat") => stat_json(&vfs, &path).await,
        ("GET", "/read") => {
            let offset = params.get("offset").and_then(|s| s.parse::<u64>().ok());
            let size = params.get("size").and_then(|s| s.parse::<u64>().ok());
            return match read_bytes(&vfs, &path, offset, size).await {
                Ok(data) => respond(&mut stream, 200, "application/octet-stream", data).await,
                Err(e) => respond(&mut stream, 500, "text/plain", e.to_string().into_bytes()).await,
            };
        }
        ("PUT", "/write") => {
            let body = read_body(&mut stream, &req).await?;
            write_bytes(&vfs, &path, body)
                .await
                .map(|_| b"{\"ok\":true}".to_vec())
        }
        _ => return respond(&mut stream, 404, "text/plain", b"not found".to_vec()).await,
    };

    match result {
        Ok(json) => respond(&mut stream, 200, "application/json", json).await,
        Err(e) => respond(&mut stream, 500, "text/plain", e.to_string().into_bytes()).await,
    }
}

async fn read_body(stream: &mut TcpStream, req: &Req) -> anyhow::Result<Vec<u8>> {
    let mut body = req.body_prefix.clone();
    while body.len() < req.content_length {
        let mut tmp = [0u8; 8192];
        let n = stream.read(&mut tmp).await?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tmp[..n]);
    }
    body.truncate(req.content_length);
    Ok(body)
}

async fn readdir_json(vfs: &Vfs, path: &str) -> anyhow::Result<Vec<u8>> {
    let entries: Vec<(String, FileKind, u64)> = if path == "/" {
        vfs.mount_names()
            .into_iter()
            .map(|n| (n, FileKind::Dir, 0))
            .collect()
    } else {
        let (res, vp) = vfs
            .route(path)
            .ok_or_else(|| anyhow::anyhow!("no mount for {path}"))?;
        res.readdir(&vp)
            .await?
            .into_iter()
            .map(|e| (e.name, e.kind, e.size))
            .collect()
    };
    let items: Vec<serde_json::Value> = entries
        .into_iter()
        .map(|(name, kind, size)| {
            serde_json::json!({"name": name, "is_dir": matches!(kind, FileKind::Dir), "size": size})
        })
        .collect();
    Ok(serde_json::to_vec(
        &serde_json::json!({ "entries": items }),
    )?)
}

async fn stat_json(vfs: &Vfs, path: &str) -> anyhow::Result<Vec<u8>> {
    if path == "/" {
        return Ok(serde_json::to_vec(
            &serde_json::json!({"exists": true, "is_dir": true, "size": 0}),
        )?);
    }
    let Some((res, vp)) = vfs.route(path) else {
        return Ok(serde_json::to_vec(&serde_json::json!({"exists": false}))?);
    };
    match res.stat(&vp).await {
        Ok(s) => Ok(serde_json::to_vec(&serde_json::json!({
            "exists": true,
            "is_dir": matches!(s.kind, FileKind::Dir),
            "size": s.size,
        }))?),
        Err(_) => Ok(serde_json::to_vec(&serde_json::json!({"exists": false}))?),
    }
}

async fn read_bytes(
    vfs: &Vfs,
    path: &str,
    offset: Option<u64>,
    size: Option<u64>,
) -> anyhow::Result<Vec<u8>> {
    let (res, vp) = vfs
        .route(path)
        .ok_or_else(|| anyhow::anyhow!("no mount for {path}"))?;
    let range = match (offset, size) {
        (Some(o), Some(s)) => Some(o..o + s),
        _ => None,
    };
    res.read_bytes(&vp, range).await
}

async fn write_bytes(vfs: &Vfs, path: &str, data: Vec<u8>) -> anyhow::Result<()> {
    let (res, vp): (_, VPath) = vfs
        .route(path)
        .ok_or_else(|| anyhow::anyhow!("no mount for {path}"))?;
    if let Some(op) = vp.as_str().strip_prefix("/.cmd/") {
        res.command(op, &data).await?;
        Ok(())
    } else {
        res.write_bytes(&vp, data).await
    }
}

async fn respond(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: Vec<u8>,
) -> anyhow::Result<()> {
    let reason = match status {
        200 => "OK",
        403 => "Forbidden",
        404 => "Not Found",
        _ => "Internal Server Error",
    };
    let header = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes()).await?;
    stream.write_all(&body).await?;
    stream.flush().await?;
    Ok(())
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn parse_query(query: &str) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    for pair in query.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        map.insert(percent_decode(k), percent_decode(v));
    }
    map
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hi = (bytes[i + 1] as char).to_digit(16);
                let lo = (bytes[i + 2] as char).to_digit(16);
                if let (Some(hi), Some(lo)) = (hi, lo) {
                    out.push((hi * 16 + lo) as u8);
                    i += 3;
                    continue;
                }
                out.push(bytes[i]);
                i += 1;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).to_string()
}
