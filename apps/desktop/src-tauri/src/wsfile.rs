//! The app's own URI scheme for workspace files.
//!
//! `wsfile://localhost/<url-encoded workspace path>` hands the window a file's bytes.
//!
//! A scheme rather than a command, because the alternative is base64 over the IPC bridge:
//! a third again in size, a copy at each end, and the whole document resident as a JSON
//! string before anything can look at it. A URL is also what the elements that render
//! these formats actually want — `<img src>`, `<object data>` — so the webview hands the
//! address to its own loader instead of the app building a blob for it.
//!
//! It reads through the engine, not off the disk, which is what makes it work for every
//! source: an S3 object and a Notion page have no host path, and the workspace has none at
//! all while the FUSE mount is down. The engine's tree is the one place all of them exist.

use std::sync::Arc;

use ailoy_desktop_core::Engine;
use tauri::{
    http::{Request, Response, StatusCode},
    Manager, UriSchemeContext,
};

/// The scheme name. Also in `tauri.conf.json`'s CSP, which has to admit it by name.
pub const SCHEME: &str = "wsfile";

/// What to tell the webview a file is, from its name.
///
/// Only the formats a viewer opens, plus the few a browser will render on its own. The
/// fallback is deliberately `application/octet-stream`: a type guessed wrong is worse than
/// no type, because the webview acts on it — an HTML file served as `text/html` from this
/// scheme would be a document running with the app's own privileges.
fn content_type(path: &str) -> &'static str {
    let ext = path.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "avif" => "image/avif",
        "bmp" => "image/bmp",
        "ico" => "image/x-icon",
        // Rendered in an `<img>`, which runs none of what makes an SVG active content.
        "svg" => "image/svg+xml",
        "pdf" => "application/pdf",
        _ => "application/octet-stream",
    }
}

/// The workspace path a request names, or `None` when the URL does not name one.
///
/// The path is the URL's own path component, percent-decoded: a file called `q3 (final).pdf`
/// has to survive the round trip, and the window builds these with `encodeURIComponent`.
///
/// `/`-rooted on the way out, because that is the shape the engine's tree is walked in and
/// what every other command is handed. The authority eats the leading slash on the way in,
/// so putting it back is not cosmetic: without it the engine is asked about a path that
/// resolves against nothing and answers `NotFound` for a file that is there.
fn requested_path(uri: &str) -> Option<String> {
    // `wsfile://localhost/<path>` — everything after the authority, which is the third `/`.
    let rest = uri.split_once("://")?.1;
    let path = rest.split_once('/').map(|(_, p)| p)?;
    let path = path.split(['?', '#']).next().unwrap_or("");
    if path.is_empty() {
        return None;
    }
    let decoded = percent_decode(path)?;
    Some(format!("/{}", decoded.trim_start_matches('/')))
}

/// Percent-decoding, over bytes so a multi-byte character split across escapes survives.
fn percent_decode(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = bytes.get(i + 1..i + 3)?;
            let hex = std::str::from_utf8(hex).ok()?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

fn reply(status: StatusCode, body: Vec<u8>, content_type: &str) -> Response<Vec<u8>> {
    Response::builder()
        .status(status)
        .header("content-type", content_type)
        // The webview is the only client and the engine is the only source; saying so
        // keeps a file that changed on disk from being served from a stale cache.
        .header("cache-control", "no-store")
        .body(body)
        .expect("a response with a valid status and header")
}

/// Serves one request. Registered as an asynchronous handler so the read does not block
/// the webview's thread while a connector answers over the network.
pub fn handle<R: tauri::Runtime>(
    ctx: UriSchemeContext<'_, R>,
    request: Request<Vec<u8>>,
    responder: tauri::UriSchemeResponder,
) {
    let Some(engine) = ctx.app_handle().try_state::<Arc<Engine>>() else {
        responder.respond(reply(
            StatusCode::SERVICE_UNAVAILABLE,
            b"the engine is not running".to_vec(),
            "text/plain",
        ));
        return;
    };
    let engine = engine.inner().clone();
    let uri = request.uri().to_string();

    tauri::async_runtime::spawn(async move {
        let Some(path) = requested_path(&uri) else {
            responder.respond(reply(
                StatusCode::BAD_REQUEST,
                b"no path in the request".to_vec(),
                "text/plain",
            ));
            return;
        };
        match engine.fs_read_bytes(&path).await {
            Ok(bytes) => responder.respond(reply(StatusCode::OK, bytes, content_type(&path))),
            Err(e) => {
                // The window can only say "this could not be opened"; the reason belongs in
                // the log, at a level the default filter actually shows — a viewer failing
                // with nothing written down is the hard kind of bug to be told about.
                tracing::warn!("{SCHEME}: {path}: {e}");
                responder.respond(reply(
                    StatusCode::NOT_FOUND,
                    e.to_string().into_bytes(),
                    "text/plain",
                ))
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_names_the_workspace_path_it_asked_for() {
        assert_eq!(
            requested_path("wsfile://localhost/reports/q3.pdf").as_deref(),
            Some("/reports/q3.pdf")
        );
        // Percent-encoded, because the window builds these with `encodeURIComponent` and
        // a real file is allowed spaces, parentheses and Hangul.
        assert_eq!(
            requested_path("wsfile://localhost/a%20b/%ED%95%9C%EA%B8%80.png").as_deref(),
            Some("/a b/한글.png")
        );
        // A query or fragment is not part of the name.
        assert_eq!(
            requested_path("wsfile://localhost/x.png?v=2#top").as_deref(),
            Some("/x.png")
        );
        assert_eq!(requested_path("wsfile://localhost/"), None);
        // The shape every other command is handed, so a file reached this way and a file
        // reached through `fs_read` name the same thing.
        assert!(requested_path("wsfile://localhost/a.png").unwrap().starts_with('/'));
        assert_eq!(requested_path("wsfile://localhost"), None);
    }

    #[test]
    fn only_the_formats_a_viewer_opens_get_a_type() {
        assert_eq!(content_type("a/b/photo.JPG"), "image/jpeg");
        assert_eq!(content_type("doc.pdf"), "application/pdf");
        // Everything else is bytes. In particular a document that would *run* if the
        // webview were told what it is.
        assert_eq!(content_type("page.html"), "application/octet-stream");
        assert_eq!(content_type("script.js"), "application/octet-stream");
        assert_eq!(content_type("noextension"), "application/octet-stream");
    }
}
