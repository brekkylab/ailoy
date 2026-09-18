//! SQLite persistence. One connection behind a mutex: every call is a few local
//! statements, so a lock is cheaper than a pool and keeps writes ordered.

use std::{path::Path, sync::Mutex};

use ailoy::message::{Message, TokenUsage};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};

use crate::{
    error::{EngineError, Result},
    types::{MountConfig, MountKind, StoredMessage, now_ms},
};

const MIGRATIONS: &[&str] = &[include_str!("migrations/0001_init.sql")];

/// Narrow a store file to its owner. Best-effort on purpose: a file that is not there yet
/// (the WAL sidecars before the first write) and a filesystem with no unix modes are both
/// normal, and neither is a reason to refuse to open the database. Nothing at all on
/// non-unix, where the mode has no meaning.
fn restrict_permissions(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        if path.exists()
            && let Err(e) = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        {
            tracing::warn!("could not restrict {} to 0600: {e}", path.display());
        }
    }
    #[cfg(not(unix))]
    let _ = path;
}

/// The content column: the ailoy message, versioned so a later shape can be read back.
#[derive(Serialize, Deserialize)]
struct Content {
    version: u32,
    message: Message,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionRow {
    pub id: String,
    pub title: String,
    pub model: String,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MountRow {
    pub id: String,
    pub path: String,
    pub kind: MountKind,
    pub label: String,
    pub config: MountConfig,
    pub writable: bool,
    pub created_at: i64,
}

pub struct NewMessage<'a> {
    pub depth: u8,
    pub source_agent: Option<&'a str>,
    pub message: &'a Message,
    pub usage: Option<&'a TokenUsage>,
}

pub struct Store {
    conn: Mutex<Connection>,
}

impl Store {
    pub fn open(path: &Path) -> Result<Store> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        // The database holds every provider API key and every connector's credentials in
        // plain text (see `MountConfig::Notion`/`S3`). SQLite creates it with the process
        // umask, which on a stock macOS account is world-readable — so it is narrowed the
        // moment it exists, before the first row is written.
        restrict_permissions(path);
        conn.execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000;",
        )?;
        // WAL mode creates the sidecars on that batch, and the rows in `-wal` are the same
        // secrets not yet checkpointed into the database — so they get the same treatment.
        for suffix in ["-wal", "-shm"] {
            let mut sidecar = path.as_os_str().to_os_string();
            sidecar.push(suffix);
            restrict_permissions(Path::new(&sidecar));
        }
        Self::migrate(&conn)?;
        Ok(Store {
            conn: Mutex::new(conn),
        })
    }

    pub fn open_in_memory() -> Result<Store> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("PRAGMA foreign_keys=ON;")?;
        Self::migrate(&conn)?;
        Ok(Store {
            conn: Mutex::new(conn),
        })
    }

    fn migrate(conn: &Connection) -> Result<()> {
        let current: u32 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        for (i, sql) in MIGRATIONS.iter().enumerate() {
            let version = i as u32 + 1;
            if version > current {
                conn.execute_batch(&format!(
                    "BEGIN; {sql} PRAGMA user_version = {version}; COMMIT;"
                ))?;
            }
        }
        Ok(())
    }

    fn with<T>(&self, f: impl FnOnce(&Connection) -> rusqlite::Result<T>) -> Result<T> {
        // A poisoned connection is still a usable connection: the panic that poisoned it
        // happened in the caller's closure, not inside SQLite, which either committed its
        // statement or did not. Honouring the poison would mean every later store call in
        // the process panicking — losing the whole session over one bad read.
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        Ok(f(&conn)?)
    }

    // ── sessions ────────────────────────────────────────────────────────────

    fn session_from_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<SessionRow> {
        Ok(SessionRow {
            id: r.get(0)?,
            title: r.get(1)?,
            model: r.get(2)?,
            created_at: r.get(3)?,
            updated_at: r.get(4)?,
        })
    }

    pub fn session_list(&self) -> Result<Vec<SessionRow>> {
        self.with(|c| {
            let mut st = c.prepare(
                "SELECT id, title, model, created_at, updated_at FROM sessions ORDER BY updated_at DESC",
            )?;
            st.query_map([], Self::session_from_row)?.collect()
        })
    }

    pub fn session_get(&self, id: &str) -> Result<SessionRow> {
        self.with(|c| {
            c.query_row(
                "SELECT id, title, model, created_at, updated_at FROM sessions WHERE id = ?1",
                params![id],
                Self::session_from_row,
            )
            .optional()
        })?
        .ok_or_else(|| EngineError::NotFound(format!("session {id}")))
    }

    pub fn session_create(&self, id: &str, title: &str, model: &str) -> Result<SessionRow> {
        let now = now_ms();
        self.with(|c| {
            c.execute(
                "INSERT INTO sessions (id, title, model, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?4)",
                params![id, title, model, now],
            )
        })?;
        Ok(SessionRow {
            id: id.into(),
            title: title.into(),
            model: model.into(),
            created_at: now,
            updated_at: now,
        })
    }

    pub fn session_rename(&self, id: &str, title: &str) -> Result<()> {
        let n = self.with(|c| {
            c.execute(
                "UPDATE sessions SET title = ?2, updated_at = ?3 WHERE id = ?1",
                params![id, title, now_ms()],
            )
        })?;
        if n == 0 {
            return Err(EngineError::NotFound(format!("session {id}")));
        }
        Ok(())
    }

    pub fn session_set_model(&self, id: &str, model: &str) -> Result<()> {
        let n = self.with(|c| {
            c.execute(
                "UPDATE sessions SET model = ?2 WHERE id = ?1",
                params![id, model],
            )
        })?;
        if n == 0 {
            return Err(EngineError::NotFound(format!("session {id}")));
        }
        Ok(())
    }

    pub fn session_touch(&self, id: &str) -> Result<()> {
        self.with(|c| {
            c.execute(
                "UPDATE sessions SET updated_at = ?2 WHERE id = ?1",
                params![id, now_ms()],
            )
        })?;
        Ok(())
    }

    pub fn session_delete(&self, id: &str) -> Result<()> {
        let n = self.with(|c| c.execute("DELETE FROM sessions WHERE id = ?1", params![id]))?;
        if n == 0 {
            return Err(EngineError::NotFound(format!("session {id}")));
        }
        Ok(())
    }

    // ── messages ────────────────────────────────────────────────────────────

    pub fn message_append(&self, session_id: &str, m: NewMessage<'_>) -> Result<i64> {
        let content = serde_json::to_string(&Content {
            version: 1,
            message: m.message.clone(),
        })
        .map_err(|e| EngineError::Other(e.into()))?;
        let usage = m
            .usage
            .map(serde_json::to_string)
            .transpose()
            .map_err(|e| EngineError::Other(e.into()))?;
        let role = m.message.role.to_string();
        self.with(|c| {
            let seq: i64 = c.query_row(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE session_id = ?1",
                params![session_id],
                |r| r.get(0),
            )?;
            c.execute(
                "INSERT INTO messages (session_id, seq, depth, source_agent, role, content, usage, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![session_id, seq, m.depth as i64, m.source_agent, role, content, usage, now_ms()],
            )?;
            Ok(seq)
        })
    }

    /// Attach (or replace) the accounting on a message already written. The ChatCompletion
    /// schema reports a turn's usage in a frame that arrives after the message is complete
    /// and persisted, so the row has to be revisited rather than written once.
    pub fn message_set_usage(&self, session_id: &str, seq: i64, usage: &TokenUsage) -> Result<()> {
        let usage = serde_json::to_string(usage).map_err(|e| EngineError::Other(e.into()))?;
        let updated = self.with(|c| {
            c.execute(
                "UPDATE messages SET usage = ?3 WHERE session_id = ?1 AND seq = ?2",
                params![session_id, seq, usage],
            )
        })?;
        if updated == 0 {
            return Err(EngineError::NotFound(format!(
                "message {seq} of session {session_id}"
            )));
        }
        Ok(())
    }

    fn stored_from_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<StoredMessage> {
        let content: String = r.get(3)?;
        let usage: Option<String> = r.get(4)?;
        let parsed: Content = serde_json::from_str(&content).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(3, rusqlite::types::Type::Text, Box::new(e))
        })?;
        let usage = usage
            .map(|u| serde_json::from_str::<TokenUsage>(&u))
            .transpose()
            .map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    4,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?;
        Ok(StoredMessage {
            seq: r.get(0)?,
            depth: r.get::<_, i64>(1)? as u8,
            source_agent: r.get(2)?,
            message: parsed.message,
            usage,
            created_at: r.get(5)?,
        })
    }

    pub fn message_list(&self, session_id: &str) -> Result<Vec<StoredMessage>> {
        self.with(|c| {
            let mut st = c.prepare(
                "SELECT seq, depth, source_agent, content, usage, created_at FROM messages WHERE session_id = ?1 ORDER BY seq",
            )?;
            st.query_map(params![session_id], Self::stored_from_row)?
                .collect()
        })
    }

    pub fn message_history(&self, session_id: &str) -> Result<Vec<Message>> {
        Ok(self
            .message_list(session_id)?
            .into_iter()
            .filter(|m| m.depth == 0)
            .map(|m| m.message)
            .collect())
    }

    pub fn message_usages(&self, session_id: &str) -> Result<Vec<TokenUsage>> {
        Ok(self
            .message_list(session_id)?
            .into_iter()
            .filter_map(|m| m.usage)
            .collect())
    }

    // ── mounts ──────────────────────────────────────────────────────────────

    pub fn mount_list(&self) -> Result<Vec<MountRow>> {
        self.with(|c| {
            let mut st = c.prepare(
                "SELECT id, path, kind, label, config, writable, created_at FROM mounts ORDER BY path",
            )?;
            st.query_map([], |r| {
                let kind: String = r.get(2)?;
                let config: String = r.get(4)?;
                Ok(MountRow {
                    id: r.get(0)?,
                    path: r.get(1)?,
                    kind: serde_json::from_value(serde_json::Value::String(kind)).map_err(|e| {
                        rusqlite::Error::FromSqlConversionFailure(
                            2,
                            rusqlite::types::Type::Text,
                            Box::new(e),
                        )
                    })?,
                    label: r.get(3)?,
                    config: serde_json::from_str(&config).map_err(|e| {
                        rusqlite::Error::FromSqlConversionFailure(
                            4,
                            rusqlite::types::Type::Text,
                            Box::new(e),
                        )
                    })?,
                    writable: r.get::<_, i64>(5)? != 0,
                    created_at: r.get(6)?,
                })
            })?
            .collect()
        })
    }

    pub fn mount_insert(&self, row: &MountRow) -> Result<()> {
        let kind =
            serde_json::to_value(row.kind.clone()).map_err(|e| EngineError::Other(e.into()))?;
        let kind = kind.as_str().unwrap_or("local").to_string();
        let config =
            serde_json::to_string(&row.config).map_err(|e| EngineError::Other(e.into()))?;
        self.with(|c| {
            c.execute(
                "INSERT INTO mounts (id, path, kind, label, config, writable, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![row.id, row.path, kind, row.label, config, row.writable as i64, row.created_at],
            )
        })?;
        Ok(())
    }

    pub fn mount_delete(&self, path: &str) -> Result<()> {
        self.with(|c| c.execute("DELETE FROM mounts WHERE path = ?1", params![path]))?;
        Ok(())
    }

    // ── settings ────────────────────────────────────────────────────────────

    pub fn setting_get(&self, key: &str) -> Result<Option<String>> {
        self.with(|c| {
            c.query_row(
                "SELECT value FROM settings WHERE key = ?1",
                params![key],
                |r| r.get(0),
            )
            .optional()
        })
    }

    pub fn setting_set(&self, key: &str, value: &str) -> Result<()> {
        self.with(|c| {
            c.execute(
                "INSERT INTO settings (key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                params![key, value],
            )
        })?;
        Ok(())
    }

    pub fn setting_delete(&self, key: &str) -> Result<()> {
        self.with(|c| c.execute("DELETE FROM settings WHERE key = ?1", params![key]))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ailoy::message::{Message, Part, Role, TokenUsage};

    use super::*;

    fn text(role: Role, t: &str) -> Message {
        Message::new(role).with_contents([Part::text(t)])
    }

    #[test]
    fn sessions_round_trip() {
        let s = Store::open_in_memory().unwrap();
        let row = s
            .session_create("s1", "First chat", "anthropic/claude-opus-5")
            .unwrap();
        assert_eq!(row.title, "First chat");
        s.session_rename("s1", "Renamed").unwrap();
        assert_eq!(s.session_get("s1").unwrap().title, "Renamed");
        assert_eq!(s.session_list().unwrap().len(), 1);
        s.session_delete("s1").unwrap();
        assert!(matches!(s.session_get("s1"), Err(EngineError::NotFound(_))));
    }

    #[test]
    fn messages_keep_order_depth_and_usage() {
        let s = Store::open_in_memory().unwrap();
        s.session_create("s1", "t", "m").unwrap();
        let u = TokenUsage {
            input_tokens: 10,
            output_tokens: 5,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: Some(3),
        };
        let seq1 = s
            .message_append(
                "s1",
                NewMessage {
                    depth: 0,
                    source_agent: None,
                    message: &text(Role::User, "hi"),
                    usage: None,
                },
            )
            .unwrap();
        let seq2 = s
            .message_append(
                "s1",
                NewMessage {
                    depth: 0,
                    source_agent: None,
                    message: &text(Role::Assistant, "hello"),
                    usage: Some(&u),
                },
            )
            .unwrap();
        let seq3 = s
            .message_append(
                "s1",
                NewMessage {
                    depth: 1,
                    source_agent: Some("sub"),
                    message: &text(Role::Assistant, "inner"),
                    usage: None,
                },
            )
            .unwrap();
        assert_eq!((seq1, seq2, seq3), (1, 2, 3));

        let all = s.message_list("s1").unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(all[2].depth, 1);
        assert_eq!(all[2].source_agent.as_deref(), Some("sub"));
        assert_eq!(
            all[1].usage.as_ref().unwrap().cache_read_input_tokens,
            Some(3)
        );

        let history = s.message_history("s1").unwrap();
        assert_eq!(history.len(), 2, "depth 1 is not replayed to the model");
        assert_eq!(history[1].contents[0].as_text(), Some("hello"));

        assert_eq!(s.message_usages("s1").unwrap().len(), 1);
        s.session_delete("s1").unwrap();
        assert!(s.message_list("s1").unwrap().is_empty(), "cascade");
    }

    #[test]
    fn usage_can_be_attached_after_the_message_is_written() {
        // The ChatCompletion schema sends a turn's usage in a frame *after* the one that
        // completed the message, by which time the row is already in the table.
        let s = Store::open_in_memory().unwrap();
        s.session_create("s1", "t", "m").unwrap();
        let seq = s
            .message_append(
                "s1",
                NewMessage {
                    depth: 0,
                    source_agent: None,
                    message: &text(Role::Assistant, "hello"),
                    usage: None,
                },
            )
            .unwrap();
        assert!(s.message_list("s1").unwrap()[0].usage.is_none());
        assert!(s.message_usages("s1").unwrap().is_empty());

        let u = TokenUsage {
            input_tokens: 7,
            output_tokens: 2,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        };
        s.message_set_usage("s1", seq, &u).unwrap();

        let stored = s.message_list("s1").unwrap();
        assert_eq!(stored[0].usage.as_ref().unwrap().input_tokens, 7);
        assert_eq!(stored[0].usage.as_ref().unwrap().output_tokens, 2);
        assert_eq!(s.message_usages("s1").unwrap().len(), 1);

        assert!(matches!(
            s.message_set_usage("s1", 99, &u),
            Err(EngineError::NotFound(_))
        ));
    }

    /// The database and its WAL sidecars hold API keys and connector credentials in plain
    /// text; on a shared machine the default umask would leave them world-readable.
    #[cfg(unix)]
    #[test]
    fn a_file_store_and_its_wal_are_readable_only_by_their_owner() {
        use std::os::unix::fs::PermissionsExt as _;

        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("ailoy.sqlite");
        let s = Store::open(&db).unwrap();
        // A write, so the WAL is not just created but has something in it.
        s.setting_set("anthropic_api_key", "sk-ant-secret").unwrap();

        let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&db), 0o600, "the database itself");
        let wal = dir.path().join("ailoy.sqlite-wal");
        assert!(wal.exists(), "WAL mode should have created the sidecar");
        assert_eq!(mode(&wal), 0o600, "the write-ahead log");
    }

    #[test]
    fn mounts_and_settings_round_trip() {
        let s = Store::open_in_memory().unwrap();
        let row = MountRow {
            id: "m1".into(),
            path: "/notion".into(),
            kind: MountKind::Notion,
            label: "notion".into(),
            config: MountConfig::Notion {
                api_key: "secret_x".into(),
            },
            writable: false,
            created_at: 1,
        };
        s.mount_insert(&row).unwrap();
        assert!(s.mount_insert(&row).is_err(), "unique path");
        let rows = s.mount_list().unwrap();
        assert!(
            matches!(&rows[0].config, MountConfig::Notion { api_key } if api_key == "secret_x")
        );
        s.mount_delete("/notion").unwrap();
        assert!(s.mount_list().unwrap().is_empty());

        assert_eq!(s.setting_get("default_model").unwrap(), None);
        s.setting_set("default_model", "openai/gpt-5").unwrap();
        s.setting_set("default_model", "anthropic/claude-opus-5")
            .unwrap();
        assert_eq!(
            s.setting_get("default_model").unwrap().as_deref(),
            Some("anthropic/claude-opus-5")
        );
        s.setting_delete("default_model").unwrap();
        assert_eq!(s.setting_get("default_model").unwrap(), None);
    }
}
