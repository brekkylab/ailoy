use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use futures::stream::{self, StreamExt};
use serde_json::{Value, json};

use crate::vfs::{
    accessor::{GmailAccessor, GmailConfig, encode_b64url},
    path::VPath,
    resource::{DirEntry, FileKind, FileStat, Resource},
};

/// Listing TTL for the label set (cheap, changes rarely).
const LABEL_TTL: Duration = Duration::from_secs(60);
/// Raw-message cache TTL — dedups the per-message fetches that readdir, stat,
/// read, and unlink all need for the same id.
const MSG_TTL: Duration = Duration::from_secs(30);
/// Concurrency for the per-message fetches a directory listing fans out.
const FETCH_CONCURRENCY: usize = 6;
/// Messages pulled per label / per date dir.
const MAX_MESSAGES: u32 = 50;
/// Over-estimate reported by `stat` for an `.gmail.json` whose processed length
/// isn't known without fetching. The guest kernel clamps reads at the reported
/// size even under direct_io, so this must exceed any real email JSON; reads
/// return the real bytes then empty at EOF (mirrors the gdrive workspace-doc
/// sentinel). Attachments report their exact size (known from the part).
const MSG_SENTINEL_SIZE: u64 = 16 * 1024 * 1024;

const GMAIL_SUFFIX: &str = ".gmail.json";

pub struct GmailResource {
    accessor: GmailAccessor,
    /// `(display_name, label_id)` for every label, cached briefly.
    label_cache: tokio::sync::Mutex<Option<(Instant, Vec<(String, String)>)>>,
    /// Full raw message JSON by id, cached briefly.
    msg_cache: tokio::sync::Mutex<HashMap<String, (Instant, Value)>>,
    /// Dates (`yyyy-mm-dd`) per label that have been visited via a direct
    /// `ls <label>/<date>` and turned out to have mail. The label listing is
    /// capped at the newest 50 messages, so older dates don't appear on their
    /// own; folding these visited dates back in makes the listing grow
    /// incrementally (and lets tab-completion offer them).
    seen_dates: tokio::sync::Mutex<HashMap<String, std::collections::BTreeSet<String>>>,
}

impl GmailResource {
    pub fn new(config: &GmailConfig) -> anyhow::Result<Self> {
        Ok(Self {
            accessor: GmailAccessor::new(config)?,
            label_cache: tokio::sync::Mutex::new(None),
            msg_cache: tokio::sync::Mutex::new(HashMap::new()),
            seen_dates: tokio::sync::Mutex::new(HashMap::new()),
        })
    }

    // ---- labels -----------------------------------------------------------

    /// `(display_name, id)` for every label. System labels display as their id
    /// (INBOX, SENT, …); user labels display as their name.
    async fn labels(&self) -> anyhow::Result<Vec<(String, String)>> {
        {
            let c = self.label_cache.lock().await;
            if let Some((at, v)) = c.as_ref()
                && at.elapsed() < LABEL_TTL
            {
                return Ok(v.clone());
            }
        }
        let raw = self.accessor.list_labels().await?;
        let mut out = Vec::new();
        for lb in &raw {
            let id = lb.get("id").and_then(|x| x.as_str()).unwrap_or("");
            if id.is_empty() {
                continue;
            }
            let display = if lb.get("type").and_then(|t| t.as_str()) == Some("system") {
                id.to_string()
            } else {
                lb.get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or(id)
                    .to_string()
            };
            out.push((display, id.to_string()));
        }
        *self.label_cache.lock().await = Some((Instant::now(), out.clone()));
        Ok(out)
    }

    async fn label_id(&self, display: &str) -> anyhow::Result<Option<String>> {
        Ok(self
            .labels()
            .await?
            .into_iter()
            .find(|(d, _)| d == display)
            .map(|(_, id)| id))
    }

    // ---- message fetch (cached) ------------------------------------------

    async fn message_full(&self, id: &str) -> anyhow::Result<Value> {
        {
            let c = self.msg_cache.lock().await;
            if let Some((at, v)) = c.get(id)
                && at.elapsed() < MSG_TTL
            {
                return Ok(v.clone());
            }
        }
        let v = self.accessor.get_message_full(id).await?;
        self.msg_cache
            .lock()
            .await
            .insert(id.to_string(), (Instant::now(), v.clone()));
        Ok(v)
    }

    async fn fetch_full_many(&self, ids: &[String]) -> Vec<Value> {
        stream::iter(ids.iter().cloned())
            .map(|id| async move { self.message_full(&id).await.ok() })
            .buffer_unordered(FETCH_CONCURRENCY)
            .filter_map(|x| async move { x })
            .collect()
            .await
    }

    // ---- readdir levels ---------------------------------------------------

    async fn readdir_labels(&self) -> anyhow::Result<Vec<DirEntry>> {
        Ok(self
            .labels()
            .await?
            .into_iter()
            .map(|(display, _)| dir_entry(display))
            .collect())
    }

    /// Date dirs present in a label: list the label's messages, bucket by the
    /// (UTC) received date.
    async fn readdir_dates(&self, label: &str) -> anyhow::Result<Vec<DirEntry>> {
        let Some(label_id) = self.label_id(label).await? else {
            anyhow::bail!("no such label: {label}");
        };
        let stubs = self
            .accessor
            .list_messages(Some(&label_id), None, MAX_MESSAGES)
            .await?;
        let ids: Vec<String> = stubs
            .iter()
            .filter_map(|m| m.get("id").and_then(|i| i.as_str()).map(String::from))
            .collect();
        // Only internalDate is needed here — use the minimal fetch.
        let dates: Vec<String> = stream::iter(ids)
            .map(|id| async move {
                self.accessor
                    .get_message_minimal(&id)
                    .await
                    .ok()
                    .and_then(|v| {
                        v.get("internalDate")
                            .and_then(|d| d.as_str())
                            .map(String::from)
                    })
                    .map(|ms| epoch_ms_to_date(&ms))
            })
            .buffer_unordered(FETCH_CONCURRENCY)
            .filter_map(|x| async move { x })
            .collect()
            .await;
        let mut set: std::collections::BTreeSet<String> = dates.into_iter().collect();
        // Fold in older dates the user has already visited (capped listing only
        // covers the newest ~50 messages).
        if let Some(seen) = self.seen_dates.lock().await.get(label) {
            set.extend(seen.iter().cloned());
        }
        let mut uniq: Vec<String> = set.into_iter().collect();
        uniq.sort_by(|a, b| b.cmp(a)); // newest first
        Ok(uniq.into_iter().map(dir_entry).collect())
    }

    /// Messages (and per-message attachment dirs) within `<label>/<date>`.
    async fn readdir_messages(&self, label: &str, date: &str) -> anyhow::Result<Vec<DirEntry>> {
        let Some(query) = date_dir_to_gmail_query(date) else {
            anyhow::bail!("invalid date dir: {date}");
        };
        let Some(label_id) = self.label_id(label).await? else {
            anyhow::bail!("no such label: {label}");
        };
        let stubs = self
            .accessor
            .list_messages(Some(&label_id), Some(&query), MAX_MESSAGES)
            .await?;
        let ids: Vec<String> = stubs
            .iter()
            .filter_map(|m| m.get("id").and_then(|i| i.as_str()).map(String::from))
            .collect();
        // Remember a visited date that actually has mail so the (capped) label
        // listing folds it in next time it's rebuilt.
        if !ids.is_empty() {
            self.seen_dates
                .lock()
                .await
                .entry(label.to_string())
                .or_default()
                .insert(date.to_string());
        }
        let raws = self.fetch_full_many(&ids).await;
        let mut out = Vec::new();
        for raw in &raws {
            let id = raw.get("id").and_then(|i| i.as_str()).unwrap_or("");
            if id.is_empty() {
                continue;
            }
            let subject = header(raw, "Subject");
            let subject = if subject.is_empty() {
                "No Subject".to_string()
            } else {
                subject
            };
            let size = raw.get("sizeEstimate").and_then(|s| s.as_u64());
            out.push(DirEntry {
                name: msg_filename(&subject, id),
                kind: FileKind::File,
                size: size.unwrap_or(0),
                mtime: None,
            });
            if !attachments(raw).is_empty() {
                out.push(dir_entry(attach_dir_name(&subject, id)));
            }
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }

    /// Attachment files within a message's attachment dir.
    async fn readdir_attachments(&self, msg_id: &str) -> anyhow::Result<Vec<DirEntry>> {
        let raw = self.message_full(msg_id).await?;
        Ok(attachments(&raw)
            .into_iter()
            .map(|a| DirEntry {
                name: a.filename,
                kind: FileKind::File,
                size: a.size,
                mtime: None,
            })
            .collect())
    }
}

#[async_trait]
impl Resource for GmailResource {
    async fn read_bytes(
        &self,
        path: &VPath,
        range: Option<std::ops::Range<u64>>,
    ) -> anyhow::Result<Vec<u8>> {
        let seg = segments(path);
        let data = match seg.as_slice() {
            // <label>/<date>/<file>.gmail.json -> processed email JSON
            [_label, _date, file] if file.ends_with(GMAIL_SUFFIX) => {
                let id = id_from_name(file.trim_end_matches(GMAIL_SUFFIX));
                let raw = self.message_full(&id).await?;
                serde_json::to_vec(&process_message(&raw))?
            }
            // <label>/<date>/<subject>__<id>/<filename> -> attachment bytes
            [_label, _date, dir, fname] => {
                let id = id_from_name(dir);
                let raw = self.message_full(&id).await?;
                let att = attachments(&raw)
                    .into_iter()
                    .find(|a| &a.filename == fname)
                    .ok_or_else(|| anyhow::anyhow!("no such attachment: {fname}"))?;
                self.accessor
                    .get_attachment(&id, &att.attachment_id)
                    .await?
            }
            _ => anyhow::bail!("is a directory or not a file: {}", path.as_str()),
        };
        Ok(slice(data, range))
    }

    async fn write_bytes(&self, path: &VPath, _data: Vec<u8>) -> anyhow::Result<()> {
        anyhow::bail!(
            "gmail file writes not supported; use .cmd/send|reply|reply-all|forward (path was {})",
            path.as_str()
        )
    }

    async fn readdir(&self, path: &VPath) -> anyhow::Result<Vec<DirEntry>> {
        let seg = segments(path);
        match seg.as_slice() {
            [] => self.readdir_labels().await,
            [label] => self.readdir_dates(label).await,
            [label, date] => self.readdir_messages(label, date).await,
            // attachment dir: <label>/<date>/<subject>__<id>
            [_label, _date, dir] if !dir.ends_with(GMAIL_SUFFIX) => {
                self.readdir_attachments(&id_from_name(dir)).await
            }
            _ => anyhow::bail!("not a directory: {}", path.as_str()),
        }
    }

    async fn stat(&self, path: &VPath) -> anyhow::Result<FileStat> {
        let seg = segments(path);
        match seg.as_slice() {
            [] => Ok(dir_stat()),
            // a label: confirm it exists (cheap, cached) — else ENOENT
            [label] => {
                if self.label_id(label).await?.is_some() {
                    Ok(dir_stat())
                } else {
                    anyhow::bail!("no such label: {label}")
                }
            }
            // a date dir: a well-formed yyyy-mm-dd is a (possibly empty) dir
            [_label, date] => {
                if date_dir_to_gmail_query(date).is_some() {
                    Ok(dir_stat())
                } else {
                    anyhow::bail!("invalid date dir: {date}")
                }
            }
            // a message file (sentinel size — don't fetch just to size it) or
            // an attachment dir
            [_label, _date, name] => {
                if name.ends_with(GMAIL_SUFFIX) {
                    Ok(FileStat {
                        kind: FileKind::File,
                        size: MSG_SENTINEL_SIZE,
                        ..Default::default()
                    })
                } else {
                    Ok(dir_stat())
                }
            }
            // an attachment file: exact size from the message's part metadata
            [_label, _date, dir, fname] => {
                let raw = self.message_full(&id_from_name(dir)).await?;
                let att = attachments(&raw)
                    .into_iter()
                    .find(|a| &a.filename == fname)
                    .ok_or_else(|| anyhow::anyhow!("no such attachment: {fname}"))?;
                Ok(FileStat {
                    kind: FileKind::File,
                    size: att.size,
                    ..Default::default()
                })
            }
            _ => anyhow::bail!("not found: {}", path.as_str()),
        }
    }

    /// `rm <…>.gmail.json` moves the message to Trash.
    async fn unlink(&self, path: &VPath) -> anyhow::Result<()> {
        let seg = segments(path);
        match seg.as_slice() {
            [_label, _date, file] if file.ends_with(GMAIL_SUFFIX) => {
                let id = id_from_name(file.trim_end_matches(GMAIL_SUFFIX));
                self.accessor.trash(&id).await?;
                self.msg_cache.lock().await.remove(&id);
                Ok(())
            }
            _ => anyhow::bail!("only .gmail.json files can be removed: {}", path.as_str()),
        }
    }

    async fn command(&self, name: &str, body: &[u8]) -> anyhow::Result<Vec<u8>> {
        let v: Value = serde_json::from_slice(body)
            .map_err(|e| anyhow::anyhow!("{name}: invalid JSON: {e}"))?;
        let s = |k: &str| v.get(k).and_then(|x| x.as_str()).map(String::from);
        let result = match name {
            "send" => {
                let to = s("to").ok_or_else(|| anyhow::anyhow!("send: missing to"))?;
                let subject = s("subject").unwrap_or_default();
                let body = s("body").unwrap_or_default();
                let raw = encode_b64url(&build_mime(&to, None, &subject, &body, &[]));
                self.accessor.send_raw(&raw, None).await?
            }
            "reply" | "reply-all" => {
                let mid =
                    s("message_id").ok_or_else(|| anyhow::anyhow!("{name}: missing message_id"))?;
                let body = s("body").unwrap_or_default();
                let orig = self.message_full(&mid).await?;
                let thread_id = orig.get("threadId").and_then(|t| t.as_str());
                let mut subject = header(&orig, "Subject");
                if !subject.to_lowercase().starts_with("re:") {
                    subject = format!("Re: {subject}");
                }
                let sender = header(&orig, "From");
                let to = if name == "reply-all" {
                    let orig_to = header(&orig, "To");
                    [sender.as_str(), orig_to.as_str()]
                        .iter()
                        .filter(|x| !x.is_empty())
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                } else {
                    sender
                };
                let cc = if name == "reply-all" {
                    let c = header(&orig, "Cc");
                    if c.is_empty() { None } else { Some(c) }
                } else {
                    None
                };
                let msg_id_hdr = header(&orig, "Message-ID");
                let mut extra: Vec<(&str, String)> = Vec::new();
                if let Some(cc) = &cc {
                    extra.push(("Cc", cc.clone()));
                }
                if !msg_id_hdr.is_empty() {
                    extra.push(("In-Reply-To", msg_id_hdr.clone()));
                    extra.push(("References", msg_id_hdr.clone()));
                }
                let raw = encode_b64url(&build_mime(&to, None, &subject, &body, &extra));
                self.accessor.send_raw(&raw, thread_id).await?
            }
            "forward" => {
                let mid = s("message_id")
                    .ok_or_else(|| anyhow::anyhow!("forward: missing message_id"))?;
                let to = s("to").ok_or_else(|| anyhow::anyhow!("forward: missing to"))?;
                let raw_msg = self.message_full(&mid).await?;
                let p = process_message(&raw_msg);
                let mut subject = p
                    .get("subject")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                if !subject.to_lowercase().starts_with("fwd:") {
                    subject = format!("Fwd: {subject}");
                }
                let from_email = p
                    .get("from")
                    .and_then(|f| f.get("email"))
                    .and_then(|e| e.as_str())
                    .unwrap_or("");
                let date = p.get("date").and_then(|d| d.as_str()).unwrap_or("");
                let orig_subject = p.get("subject").and_then(|s| s.as_str()).unwrap_or("");
                let body_text = p.get("body_text").and_then(|b| b.as_str()).unwrap_or("");
                let fwd = format!(
                    "---------- Forwarded message ----------\nFrom: {from_email}\nDate: {date}\nSubject: {orig_subject}\n\n{body_text}"
                );
                let raw = encode_b64url(&build_mime(&to, None, &subject, &fwd, &[]));
                self.accessor.send_raw(&raw, None).await?
            }
            other => anyhow::bail!("unknown gmail command: {other}"),
        };
        Ok(serde_json::to_vec(&result)?)
    }

    fn prompt(&self) -> &str {
        GMAIL_PROMPT
    }

    /// Gmail listings are capped (newest 50 per label/date), so a date or message
    /// missing from a listing may still exist — disable negative caching so the
    /// cache probes the adapter (e.g. `ls INBOX/2026-05-01` after `ls INBOX`).
    fn listings_complete(&self) -> bool {
        false
    }
}

// ---- path helpers ---------------------------------------------------------

/// Mount-relative path segments (`/INBOX/2026-05-03/x.gmail.json` -> 3).
fn segments(path: &VPath) -> Vec<String> {
    path.as_str()
        .trim_matches('/')
        .split('/')
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

/// The message id encoded in a `<subject>__<id>` name (last `__`-separated
/// field; sanitized subjects use single underscores, the separator is `__`).
fn id_from_name(name: &str) -> String {
    name.rsplit_once("__")
        .map(|(_, id)| id)
        .unwrap_or(name)
        .to_string()
}

fn msg_filename(subject: &str, id: &str) -> String {
    format!("{}__{id}{GMAIL_SUFFIX}", sanitize(subject))
}

fn attach_dir_name(subject: &str, id: &str) -> String {
    format!("{}__{id}", sanitize(subject))
}

const TITLE_MAX: usize = 80;

/// Sanitize a subject for use as a path segment:
/// keep word chars / spaces / `-._`, collapse the rest to `_`, spaces->`_`,
/// squeeze repeats, trim, cap length.
fn sanitize(text: &str) -> String {
    if text.trim().is_empty() {
        return "No_Subject".to_string();
    }
    let mut s = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' || ch == '-' || ch == '.' {
            s.push(ch);
        } else if ch.is_whitespace() {
            s.push('_');
        } else {
            s.push('_');
        }
    }
    // squeeze repeated underscores
    let mut squeezed = String::with_capacity(s.len());
    let mut prev_us = false;
    for ch in s.chars() {
        if ch == '_' {
            if !prev_us {
                squeezed.push(ch);
            }
            prev_us = true;
        } else {
            squeezed.push(ch);
            prev_us = false;
        }
    }
    let trimmed = squeezed.trim_matches('_');
    let mut out: String = trimmed.chars().collect();
    if out.chars().count() > TITLE_MAX {
        out = out.chars().take(TITLE_MAX - 3).collect::<String>() + "...";
    }
    if out.is_empty() {
        "No_Subject".to_string()
    } else {
        out
    }
}

// ---- message processing ---------------------------------------------------

struct Attach {
    filename: String,
    attachment_id: String,
    size: u64,
    mime_type: String,
}

fn header(raw: &Value, name: &str) -> String {
    raw.get("payload")
        .and_then(|p| p.get("headers"))
        .and_then(|h| h.as_array())
        .into_iter()
        .flatten()
        .find(|h| {
            h.get("name")
                .and_then(|n| n.as_str())
                .map(|n| n.eq_ignore_ascii_case(name))
                .unwrap_or(false)
        })
        .and_then(|h| h.get("value").and_then(|v| v.as_str()))
        .unwrap_or("")
        .to_string()
}

/// Recursively decode the first text/plain body part.
fn decode_body(payload: &Value) -> String {
    if payload.get("mimeType").and_then(|m| m.as_str()) == Some("text/plain") {
        if let Some(data) = payload
            .get("body")
            .and_then(|b| b.get("data"))
            .and_then(|d| d.as_str())
            && !data.is_empty()
        {
            let trimmed = data.trim_end_matches('=');
            if let Ok(bytes) =
                base64::Engine::decode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, trimmed)
            {
                return String::from_utf8_lossy(&bytes).into_owned();
            }
        }
    }
    if let Some(parts) = payload.get("parts").and_then(|p| p.as_array()) {
        for part in parts {
            let t = decode_body(part);
            if !t.is_empty() {
                return t;
            }
        }
    }
    String::new()
}

fn parse_address(raw: &str) -> Value {
    let raw = raw.trim();
    if let (Some(lt), Some(gt)) = (raw.find('<'), raw.find('>'))
        && lt < gt
    {
        let name = raw[..lt].trim().trim_matches('"').to_string();
        let email = raw[lt + 1..gt].trim().to_string();
        return json!({ "name": name, "email": email });
    }
    json!({ "name": "", "email": raw })
}

fn parse_address_list(raw: &str) -> Value {
    if raw.trim().is_empty() {
        return json!([]);
    }
    Value::Array(raw.split(',').map(|a| parse_address(a.trim())).collect())
}

fn attachments(raw: &Value) -> Vec<Attach> {
    let mut out = Vec::new();
    let mut push_part = |part: &Value| {
        let filename = part.get("filename").and_then(|f| f.as_str()).unwrap_or("");
        let body = part.get("body");
        let aid = body
            .and_then(|b| b.get("attachmentId"))
            .and_then(|a| a.as_str())
            .unwrap_or("");
        if !filename.is_empty() && !aid.is_empty() {
            out.push(Attach {
                filename: filename.to_string(),
                attachment_id: aid.to_string(),
                size: body
                    .and_then(|b| b.get("size"))
                    .and_then(|s| s.as_u64())
                    .unwrap_or(0),
                mime_type: part
                    .get("mimeType")
                    .and_then(|m| m.as_str())
                    .unwrap_or("")
                    .to_string(),
            });
        }
    };
    if let Some(parts) = raw
        .get("payload")
        .and_then(|p| p.get("parts"))
        .and_then(|p| p.as_array())
    {
        for part in parts {
            push_part(part);
            if let Some(subs) = part.get("parts").and_then(|p| p.as_array()) {
                for sub in subs {
                    push_part(sub);
                }
            }
        }
    }
    out
}

/// Build the processed email JSON (the `.gmail.json` content).
fn process_message(raw: &Value) -> Value {
    let payload = raw.get("payload").cloned().unwrap_or(Value::Null);
    let body_text = decode_body(&payload);
    let atts: Vec<Value> = attachments(raw)
        .into_iter()
        .map(|a| {
            json!({
                "id": a.attachment_id,
                "filename": a.filename,
                "mime_type": a.mime_type,
                "size": a.size,
            })
        })
        .collect();
    json!({
        "id": raw.get("id").and_then(|i| i.as_str()).unwrap_or(""),
        "thread_id": raw.get("threadId").and_then(|i| i.as_str()).unwrap_or(""),
        "from": parse_address(&header(raw, "From")),
        "to": parse_address_list(&header(raw, "To")),
        "cc": parse_address_list(&header(raw, "Cc")),
        "subject": header(raw, "Subject"),
        "date": header(raw, "Date"),
        "body_text": body_text,
        "snippet": raw.get("snippet").and_then(|s| s.as_str()).unwrap_or(""),
        "labels": raw.get("labelIds").cloned().unwrap_or(json!([])),
        "attachments": atts,
    })
}

// ---- MIME build (RFC 2822) ------------------------------------------------

/// Build a minimal text/plain RFC-2822 message. Non-ASCII subjects are RFC-2047
/// encoded so they survive transport.
fn build_mime(
    to: &str,
    from: Option<&str>,
    subject: &str,
    body: &str,
    extra_headers: &[(&str, String)],
) -> Vec<u8> {
    let mut h = String::new();
    if let Some(f) = from {
        h.push_str(&format!("From: {f}\r\n"));
    }
    h.push_str(&format!("To: {to}\r\n"));
    h.push_str(&format!("Subject: {}\r\n", encode_header(subject)));
    for (k, v) in extra_headers {
        h.push_str(&format!("{k}: {v}\r\n"));
    }
    h.push_str("MIME-Version: 1.0\r\n");
    h.push_str("Content-Type: text/plain; charset=\"utf-8\"\r\n");
    h.push_str("Content-Transfer-Encoding: 8bit\r\n");
    h.push_str("\r\n");
    let mut bytes = h.into_bytes();
    bytes.extend_from_slice(body.as_bytes());
    bytes
}

/// RFC-2047-encode a header value if it contains non-ASCII; else pass through.
fn encode_header(value: &str) -> String {
    if value.is_ascii() {
        return value.to_string();
    }
    let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, value.as_bytes());
    format!("=?UTF-8?B?{b64}?=")
}

// ---- date helpers (dependency-free civil calendar) ------------------------

/// Gmail `internalDate` (epoch ms, string) -> `YYYY-MM-DD` in UTC.
fn epoch_ms_to_date(ms: &str) -> String {
    let ms: i64 = ms.parse().unwrap_or(0);
    let days = (ms / 1000).div_euclid(86400);
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}")
}

/// `YYYY-MM-DD` date dir -> a one-day Gmail `after:/before:` query (cheap server
/// -side narrowing), or `None` if not a valid date.
fn date_dir_to_gmail_query(name: &str) -> Option<String> {
    let parts: Vec<&str> = name.split('-').collect();
    if parts.len() != 3 || parts[0].len() != 4 || parts[1].len() != 2 || parts[2].len() != 2 {
        return None;
    }
    let y: i64 = parts[0].parse().ok()?;
    let m: u32 = parts[1].parse().ok()?;
    let d: u32 = parts[2].parse().ok()?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    let days = days_from_civil(y, m as i64, d as i64);
    let (ny, nm, nd) = civil_from_days(days + 1);
    Some(format!(
        "after:{y}/{m:02}/{d:02} before:{ny}/{nm:02}/{nd:02}"
    ))
}

/// Days since 1970-01-01 -> (year, month, day). Howard Hinnant's algorithm.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = z - era * 146097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// (year, month, day) -> days since 1970-01-01. Howard Hinnant's algorithm.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400; // [0, 399]
    let mp = (m + 9) % 12; // [0, 11]
    let doy = (153 * mp + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719468
}

// ---- small builders -------------------------------------------------------

fn dir_entry(name: String) -> DirEntry {
    DirEntry {
        name,
        kind: FileKind::Dir,
        size: 0,
        mtime: None,
    }
}

fn dir_stat() -> FileStat {
    FileStat {
        kind: FileKind::Dir,
        ..Default::default()
    }
}

fn slice(data: Vec<u8>, range: Option<std::ops::Range<u64>>) -> Vec<u8> {
    match range {
        Some(r) => {
            let start = (r.start as usize).min(data.len());
            let end = (r.end as usize).min(data.len());
            data[start..end].to_vec()
        }
        None => data,
    }
}

#[cfg(test)]
mod tests {
    use base64::Engine as _;

    use super::*;

    #[test]
    fn sanitize_subject_rules() {
        assert_eq!(sanitize("Hello World"), "Hello_World");
        assert_eq!(sanitize("Re: [urgent] ping!"), "Re_urgent_ping");
        assert_eq!(sanitize("a / b \\ c"), "a_b_c");
        assert_eq!(sanitize("   "), "No_Subject");
        assert_eq!(sanitize(""), "No_Subject");
        // keeps word chars, dash, dot, underscore
        assert_eq!(sanitize("file-name.v2_final"), "file-name.v2_final");
        // length cap (80) with ellipsis
        let long = "x".repeat(100);
        let s = sanitize(&long);
        assert_eq!(s.chars().count(), 80);
        assert!(s.ends_with("..."));
    }

    #[test]
    fn id_parsed_from_last_double_underscore() {
        assert_eq!(id_from_name("Subject__abc123"), "abc123");
        // single underscores in the (sanitized) subject don't confuse it
        assert_eq!(id_from_name("a_b_c__99zz"), "99zz");
        assert_eq!(id_from_name("Re_urgent_ping__ff00ab"), "ff00ab");
        // no separator -> the whole name
        assert_eq!(id_from_name("noseparator"), "noseparator");
        // round-trips with the filename builders
        assert_eq!(
            id_from_name(&msg_filename("Hi there", "ID42").trim_end_matches(GMAIL_SUFFIX)),
            "ID42"
        );
        assert_eq!(id_from_name(&attach_dir_name("Hi there", "ID42")), "ID42");
    }

    #[test]
    fn epoch_ms_to_date_anchors() {
        assert_eq!(epoch_ms_to_date("0"), "1970-01-01");
        // 2026-05-03T10:00:00Z = 1777800000 s
        assert_eq!(epoch_ms_to_date("1777802400000"), "2026-05-03");
        assert_eq!(epoch_ms_to_date("bogus"), "1970-01-01");
    }

    #[test]
    fn date_query_narrowing_and_rollover() {
        assert_eq!(
            date_dir_to_gmail_query("2026-05-03").as_deref(),
            Some("after:2026/05/03 before:2026/05/04")
        );
        // month rollover
        assert_eq!(
            date_dir_to_gmail_query("2026-01-31").as_deref(),
            Some("after:2026/01/31 before:2026/02/01")
        );
        // year rollover
        assert_eq!(
            date_dir_to_gmail_query("2026-12-31").as_deref(),
            Some("after:2026/12/31 before:2027/01/01")
        );
        // invalid shapes -> None
        assert!(date_dir_to_gmail_query("2026-5-3").is_none());
        assert!(date_dir_to_gmail_query("not-a-date").is_none());
        assert!(date_dir_to_gmail_query("2026-13-01").is_none());
    }

    #[test]
    fn civil_calendar_roundtrips() {
        for &(y, m, d) in &[
            (1970, 1, 1),
            (2000, 2, 29),
            (2026, 5, 3),
            (2027, 1, 1),
            (1999, 12, 31),
        ] {
            let days = days_from_civil(y, m, d);
            assert_eq!(civil_from_days(days), (y, m as u32, d as u32));
        }
    }

    #[test]
    fn process_message_shapes_the_email() {
        // a minimal raw Gmail message with a plain-text body and one attachment
        let body_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b"hello body");
        let raw = json!({
            "id": "m1",
            "threadId": "t1",
            "snippet": "hello",
            "labelIds": ["INBOX", "IMPORTANT"],
            "payload": {
                "headers": [
                    {"name": "From", "value": "Alice <alice@example.com>"},
                    {"name": "To", "value": "bob@example.com, carol@example.com"},
                    {"name": "Subject", "value": "Hi"},
                    {"name": "Date", "value": "Mon, 3 May 2026 10:00:00 -0700"}
                ],
                "parts": [
                    {"mimeType": "text/plain", "body": {"data": body_b64}},
                    {"filename": "a.pdf", "mimeType": "application/pdf",
                     "body": {"attachmentId": "att1", "size": 12345}}
                ]
            }
        });
        let p = process_message(&raw);
        assert_eq!(p["id"], "m1");
        assert_eq!(p["thread_id"], "t1");
        assert_eq!(p["from"]["name"], "Alice");
        assert_eq!(p["from"]["email"], "alice@example.com");
        assert_eq!(p["to"].as_array().unwrap().len(), 2);
        assert_eq!(p["subject"], "Hi");
        assert_eq!(p["body_text"], "hello body");
        assert_eq!(p["labels"], json!(["INBOX", "IMPORTANT"]));
        let atts = p["attachments"].as_array().unwrap();
        assert_eq!(atts.len(), 1);
        assert_eq!(atts[0]["filename"], "a.pdf");
        assert_eq!(atts[0]["id"], "att1");
        assert_eq!(atts[0]["size"], 12345);
    }

    #[test]
    fn build_mime_encodes_nonascii_subject() {
        let bytes = build_mime("to@x.com", None, "안녕 hi", "body", &[]);
        let s = String::from_utf8_lossy(&bytes);
        assert!(s.contains("To: to@x.com\r\n"));
        assert!(
            s.contains("Subject: =?UTF-8?B?"),
            "non-ascii subject must be RFC-2047 encoded"
        );
        assert!(s.ends_with("body"));
        // ascii subject passes through
        let bytes = build_mime("to@x.com", None, "Plain", "b", &[]);
        assert!(String::from_utf8_lossy(&bytes).contains("Subject: Plain\r\n"));
    }
}

const GMAIL_PROMPT: &str = "\
Gmail (read + send/reply/forward, trash on delete). Layout:
  <label>/<yyyy-mm-dd>/<subject>__<message-id>.gmail.json   # the email (JSON)
  <label>/<yyyy-mm-dd>/<subject>__<message-id>/<filename>   # attachments (only if any)

  <label>       INBOX, SENT, DRAFT, IMPORTANT, STARRED, TRASH, SPAM, or a user label
  <yyyy-mm-dd>  received date; `ls <label>/2026-05-03/` narrows the Gmail query
                server-side (after:/before:) — far cheaper than scanning the label
  <subject>     sanitized subject (don't construct it; ls the date dir)
  <message-id>  Gmail message id (the field after the last `__`)

  cat <…>.gmail.json (keep the suffix) returns:
    {\"id\",\"thread_id\",\"from\":{\"name\",\"email\"},\"to\":[…],\"cc\":[…],
     \"subject\",\"date\",\"body_text\",\"snippet\",\"labels\":[…],
     \"attachments\":[{\"id\",\"filename\",\"mime_type\",\"size\"}]}
  The sibling dir (same name without .gmail.json) holds attachment bytes; cat a
  file inside to download it. ENOENT there means the message has no attachments.

  rm <…>.gmail.json    moves the message to Trash (only .gmail.json is removable).

  Write via control-path JSON (echo '{…}' > .cmd/<op>):
    .cmd/send        {\"to\":\"a@b.com\",\"subject\":\"Hi\",\"body\":\"…\"}
    .cmd/reply       {\"message_id\":\"<id>\",\"body\":\"…\"}
    .cmd/reply-all   {\"message_id\":\"<id>\",\"body\":\"…\"}
    .cmd/forward     {\"message_id\":\"<id>\",\"to\":\"a@b.com\"}";
