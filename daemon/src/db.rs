//! The daemon's state: sqlite, behind one connection and one lock.
//!
//! Four tables. `triggers` is what was registered; `events` is everything that
//! arrived, by type; `runs` is everything the triggers decided to do; `tokens` is the
//! credentials issued for posting events of one type.
//! Every method is synchronous and short; callers on the runtime hold the lock only
//! for the duration of a statement or a small transaction.

use std::{
    path::Path,
    sync::{Arc, Mutex, MutexGuard},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Context as _;
use rusqlite::{Connection, OptionalExtension, params, params_from_iter};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::trigger::TriggerConfig;

pub fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[derive(Clone)]
pub struct Db {
    conn: Arc<Mutex<Connection>>,
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT NOT NULL, at INTEGER NOT NULL, payload TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS events_type ON events(type, id);
CREATE TABLE IF NOT EXISTS triggers (
  name TEXT PRIMARY KEY, config TEXT NOT NULL, dirty INTEGER NOT NULL DEFAULT 1, last_error TEXT);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, trigger TEXT NOT NULL, event_id INTEGER, payload TEXT NOT NULL,
  status TEXT NOT NULL, error TEXT, created_at INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS runs_trigger ON runs(trigger, status);
CREATE TABLE IF NOT EXISTS tokens (
  id TEXT PRIMARY KEY, type TEXT NOT NULL, sha256 TEXT NOT NULL, created_at INTEGER NOT NULL);
CREATE UNIQUE INDEX IF NOT EXISTS tokens_sha ON tokens(sha256);
";

/// `triggers` rows woken by event type `?1`: it is one of their sources or watched.
const WAKES: &str = "(EXISTS (SELECT 1 FROM json_each(triggers.config, '$.watch') WHERE value = ?1)
   OR EXISTS (SELECT 1 FROM json_each(triggers.config, '$.sources') WHERE key = ?1))";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventRow {
    pub id: i64,
    #[serde(rename = "type")]
    pub kind: String,
    pub at: i64,
    pub payload: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TriggerRow {
    pub name: String,
    pub config: TriggerConfig,
    pub dirty: bool,
    /// Why the last script call or run start failed; `None` after a success.
    pub last_error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunRow {
    pub id: String,
    pub trigger: String,
    /// The event this run was made for, when the script said so.
    pub event_id: Option<i64>,
    pub payload: Value,
    pub status: String,
    pub error: Option<String>,
    pub created_at: i64,
}

/// A credential that may post events of one type, and nothing else. The secret
/// itself is not stored: `sha256` is all that is kept of it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenRow {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub created_at: i64,
}

/// What a script call decided: one run, or nothing to do.
#[derive(Clone, Debug)]
pub struct NewRun {
    pub id: String,
    pub payload: Value,
    pub event_id: Option<i64>,
}

pub const PENDING: &str = "pending";
pub const RUNNING: &str = "running";
pub const DONE: &str = "done";
pub const FAILED: &str = "failed";

fn json_col<T: serde::de::DeserializeOwned>(text: Option<String>) -> Option<T> {
    text.and_then(|t| serde_json::from_str(&t).ok())
}

fn placeholders(n: usize) -> String {
    std::iter::repeat_n("?", n).collect::<Vec<_>>().join(",")
}

impl Db {
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let conn = Connection::open(path).with_context(|| format!("opening {}", path.display()))?;
        conn.execute_batch("PRAGMA journal_mode=WAL;")?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    #[cfg(test)]
    pub fn open_in_memory() -> anyhow::Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    fn lock(&self) -> MutexGuard<'_, Connection> {
        self.conn.lock().unwrap_or_else(|e| e.into_inner())
    }

    // ----- events -----

    /// Append an event and mark every trigger it wakes dirty, in one transaction.
    /// Returns the event id and how many triggers were marked.
    pub fn publish(&self, kind: &str, payload: &Value, at: i64) -> anyhow::Result<(i64, usize)> {
        let mut conn = self.lock();
        let tx = conn.transaction()?;
        tx.execute(
            "INSERT INTO events (type, at, payload) VALUES (?1, ?2, ?3)",
            params![kind, at, serde_json::to_string(payload)?],
        )?;
        let id = tx.last_insert_rowid();
        let dirtied = tx.execute(
            &format!("UPDATE triggers SET dirty = 1 WHERE {WAKES}"),
            params![kind],
        )?;
        tx.commit()?;
        Ok((id, dirtied))
    }

    fn event_from_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<EventRow> {
        Ok(EventRow {
            id: r.get(0)?,
            kind: r.get(1)?,
            at: r.get(2)?,
            payload: json_col(r.get::<_, Option<String>>(3)?).unwrap_or(Value::Null),
        })
    }

    /// Newest first; `kind` narrows to one type.
    pub fn list_events(&self, kind: Option<&str>, limit: u32) -> anyhow::Result<Vec<EventRow>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT id, type, at, payload FROM events WHERE (?1 IS NULL OR type = ?1)
             ORDER BY id DESC LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![kind, limit], Self::event_from_row)?
            .filter_map(Result::ok)
            .collect();
        Ok(rows)
    }

    /// The oldest event of `types` that `trigger` has made no run for. What the
    /// identity logic turns into the next run.
    pub fn next_unhandled_event(
        &self,
        trigger: &str,
        types: &[String],
    ) -> anyhow::Result<Option<EventRow>> {
        if types.is_empty() {
            return Ok(None);
        }
        let conn = self.lock();
        let mut stmt = conn.prepare(&format!(
            "SELECT e.id, e.type, e.at, e.payload FROM events e
             WHERE e.type IN ({}) AND NOT EXISTS (
               SELECT 1 FROM runs r WHERE r.trigger = ? AND r.event_id = e.id)
             ORDER BY e.id LIMIT 1",
            placeholders(types.len())
        ))?;
        let mut args: Vec<rusqlite::types::Value> =
            types.iter().map(|t| t.clone().into()).collect();
        args.push(trigger.to_string().into());
        Ok(stmt
            .query_row(params_from_iter(args), Self::event_from_row)
            .optional()?)
    }

    /// The newest event id, for [`trigger_decided`](Self::trigger_decided).
    pub fn max_event_id(&self) -> anyhow::Result<i64> {
        Ok(self
            .lock()
            .query_row("SELECT COALESCE(MAX(id), 0) FROM events", [], |r| r.get(0))?)
    }

    pub fn prune_events(&self, before: i64) -> anyhow::Result<usize> {
        Ok(self
            .lock()
            .execute("DELETE FROM events WHERE at < ?1", params![before])?)
    }

    // ----- triggers -----

    /// Register or replace. A new or changed trigger starts dirty so its next dispatch
    /// sees whatever events of its types already exist.
    pub fn upsert_trigger(&self, name: &str, config: &TriggerConfig) -> anyhow::Result<()> {
        self.lock().execute(
            "INSERT INTO triggers (name, config) VALUES (?1, ?2)
             ON CONFLICT(name) DO UPDATE SET config = excluded.config, dirty = 1, last_error = NULL",
            params![name, serde_json::to_string(config)?],
        )?;
        Ok(())
    }

    fn trigger_from_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<Option<TriggerRow>> {
        let config: String = r.get(1)?;
        let Ok(config) = serde_json::from_str(&config) else {
            return Ok(None);
        };
        Ok(Some(TriggerRow {
            name: r.get(0)?,
            config,
            dirty: r.get::<_, i64>(2)? != 0,
            last_error: r.get(3)?,
        }))
    }

    const TRIGGER_COLS: &'static str = "name, config, dirty, last_error";

    pub fn get_trigger(&self, name: &str) -> anyhow::Result<Option<TriggerRow>> {
        Ok(self
            .lock()
            .query_row(
                &format!(
                    "SELECT {} FROM triggers WHERE name = ?1",
                    Self::TRIGGER_COLS
                ),
                params![name],
                Self::trigger_from_row,
            )
            .optional()?
            .flatten())
    }

    pub fn list_triggers(&self) -> anyhow::Result<Vec<TriggerRow>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(&format!(
            "SELECT {} FROM triggers ORDER BY name",
            Self::TRIGGER_COLS
        ))?;
        let rows = stmt
            .query_map([], Self::trigger_from_row)?
            .filter_map(Result::ok)
            .flatten()
            .collect();
        Ok(rows)
    }

    /// Dirty triggers, in name order.
    pub fn dirty_triggers(&self) -> anyhow::Result<Vec<TriggerRow>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(&format!(
            "SELECT {} FROM triggers WHERE dirty = 1 ORDER BY name",
            Self::TRIGGER_COLS
        ))?;
        let rows = stmt
            .query_map([], Self::trigger_from_row)?
            .filter_map(Result::ok)
            .flatten()
            .collect();
        Ok(rows)
    }

    pub fn delete_trigger(&self, name: &str) -> anyhow::Result<bool> {
        let n = self
            .lock()
            .execute("DELETE FROM triggers WHERE name = ?1", params![name])?;
        Ok(n > 0)
    }

    /// A script call or run start that failed: remember why and stop asking until the
    /// next event of its types.
    pub fn trigger_failed(&self, name: &str, error: &str) -> anyhow::Result<()> {
        self.lock().execute(
            "UPDATE triggers SET last_error = ?2, dirty = 0 WHERE name = ?1",
            params![name, error],
        )?;
        Ok(())
    }

    /// A script call that answered. With a run, the run goes in and the trigger stays
    /// dirty so it is asked again. With none, the trigger is clean unless an event of
    /// its types newer than `seen_event_id` (the newest when the call began) arrived
    /// meanwhile, in which case it stays dirty for the next pass. One transaction.
    pub fn trigger_decided(
        &self,
        name: &str,
        run: Option<&NewRun>,
        seen_event_id: i64,
    ) -> anyhow::Result<()> {
        let mut conn = self.lock();
        let tx = conn.transaction()?;
        if let Some(r) = run {
            tx.execute(
                "INSERT INTO runs (id, trigger, event_id, payload, status, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    r.id,
                    name,
                    r.event_id,
                    serde_json::to_string(&r.payload)?,
                    PENDING,
                    now()
                ],
            )?;
        }
        tx.execute(
            "UPDATE triggers SET last_error = NULL,
               dirty = CASE WHEN ?2 OR EXISTS (
                 SELECT 1 FROM events e WHERE e.id > ?3 AND (
                   e.type IN (SELECT value FROM json_each(triggers.config, '$.watch'))
                   OR e.type IN (SELECT key FROM json_each(triggers.config, '$.sources'))))
               THEN 1 ELSE 0 END
             WHERE name = ?1",
            params![name, run.is_some(), seen_event_id],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Write the read-only snapshot a trigger script queries: events of its types and
    /// its own runs. Rebuilds the file's contents, keeping the file itself.
    pub fn snapshot(&self, trigger: &str, types: &[String], path: &Path) -> anyhow::Result<()> {
        let conn = self.lock();
        let path_str = path.to_str().context("snapshot path is not UTF-8")?;
        conn.execute("ATTACH DATABASE ?1 AS snap", params![path_str])?;
        let result = (|| -> anyhow::Result<()> {
            // The file keeps its identity across calls: a console shares this tree with a
            // guest, and a name whose file was unlinked and remade reads back there as the
            // old one or as nothing. Rebuilt in place instead, and with no journal beside
            // it for the same reason.
            conn.execute_batch(
                "PRAGMA snap.journal_mode = OFF;
                 DROP TABLE IF EXISTS snap.events;
                 DROP TABLE IF EXISTS snap.runs;",
            )?;
            let ph = placeholders(types.len());
            let type_args: Vec<rusqlite::types::Value> =
                types.iter().map(|t| t.clone().into()).collect();
            conn.execute(
                &format!(
                    "CREATE TABLE snap.events AS SELECT id, type, at, payload FROM main.events
                     WHERE type IN ({ph}) ORDER BY id"
                ),
                params_from_iter(type_args.iter()),
            )?;
            conn.execute(
                "CREATE TABLE snap.runs AS SELECT id, event_id, created_at, status, payload
                 FROM main.runs WHERE trigger = ?1 ORDER BY created_at",
                params![trigger],
            )?;
            Ok(())
        })();
        conn.execute_batch("DETACH DATABASE snap")?;
        result
    }

    // ----- runs -----

    fn run_from_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<RunRow> {
        Ok(RunRow {
            id: r.get(0)?,
            trigger: r.get(1)?,
            event_id: r.get(2)?,
            payload: json_col(r.get::<_, Option<String>>(3)?).unwrap_or(Value::Null),
            status: r.get(4)?,
            error: r.get(5)?,
            created_at: r.get(6)?,
        })
    }

    const RUN_COLS: &'static str = "id, trigger, event_id, payload, status, error, created_at";

    /// A run made outside any script call (`fire`).
    pub fn insert_run(&self, id: &str, trigger: &str, payload: &Value) -> anyhow::Result<()> {
        self.lock().execute(
            "INSERT INTO runs (id, trigger, payload, status, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, trigger, serde_json::to_string(payload)?, PENDING, now()],
        )?;
        Ok(())
    }

    pub fn get_run(&self, id: &str) -> anyhow::Result<Option<RunRow>> {
        Ok(self
            .lock()
            .query_row(
                &format!("SELECT {} FROM runs WHERE id = ?1", Self::RUN_COLS),
                params![id],
                Self::run_from_row,
            )
            .optional()?)
    }

    pub fn list_runs(
        &self,
        trigger: Option<&str>,
        status: Option<&str>,
        limit: u32,
    ) -> anyhow::Result<Vec<RunRow>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(&format!(
            "SELECT {} FROM runs WHERE (?1 IS NULL OR trigger = ?1) AND (?2 IS NULL OR status = ?2)
             ORDER BY created_at DESC LIMIT ?3",
            Self::RUN_COLS
        ))?;
        let rows = stmt
            .query_map(params![trigger, status, limit], Self::run_from_row)?
            .filter_map(Result::ok)
            .collect();
        Ok(rows)
    }

    pub fn count_runs(&self, trigger: &str, status: &str) -> anyhow::Result<u32> {
        Ok(self.lock().query_row(
            "SELECT COUNT(*) FROM runs WHERE trigger = ?1 AND status = ?2",
            params![trigger, status],
            |r| r.get::<_, i64>(0),
        )? as u32)
    }

    /// Take the oldest pending run of a trigger that has nothing running, marking it
    /// running. Runs of one trigger execute one at a time.
    pub fn claim_run(&self) -> anyhow::Result<Option<RunRow>> {
        let mut conn = self.lock();
        let tx = conn.transaction()?;
        let candidate = tx
            .query_row(
                &format!(
                    "SELECT {} FROM runs r WHERE r.status = 'pending'
                       AND NOT EXISTS (SELECT 1 FROM runs x WHERE x.trigger = r.trigger AND x.status = 'running')
                       AND EXISTS (SELECT 1 FROM triggers t WHERE t.name = r.trigger)
                     ORDER BY r.created_at LIMIT 1",
                    Self::RUN_COLS
                ),
                [],
                Self::run_from_row,
            )
            .optional()?;
        let Some(mut run) = candidate else {
            tx.commit()?;
            return Ok(None);
        };
        tx.execute(
            "UPDATE runs SET status = 'running' WHERE id = ?1",
            params![run.id],
        )?;
        tx.commit()?;
        run.status = RUNNING.into();
        Ok(Some(run))
    }

    /// Move a run to a status; `error` belongs to the new status and is replaced.
    pub fn set_run_status(
        &self,
        id: &str,
        status: &str,
        error: Option<&str>,
    ) -> anyhow::Result<()> {
        self.lock().execute(
            "UPDATE runs SET status = ?2, error = ?3 WHERE id = ?1",
            params![id, status, error],
        )?;
        Ok(())
    }

    /// Runs left `running` by a previous process.
    pub fn running_runs(&self) -> anyhow::Result<Vec<RunRow>> {
        let conn = self.lock();
        let mut stmt = conn.prepare(&format!(
            "SELECT {} FROM runs WHERE status = 'running'",
            Self::RUN_COLS
        ))?;
        let rows = stmt
            .query_map([], Self::run_from_row)?
            .filter_map(Result::ok)
            .collect();
        Ok(rows)
    }

    pub fn count_active_runs(&self) -> anyhow::Result<u32> {
        Ok(self.lock().query_row(
            "SELECT COUNT(*) FROM runs WHERE status IN ('pending', 'running')",
            [],
            |r| r.get::<_, i64>(0),
        )? as u32)
    }

    // ----- tokens -----

    /// Store a credential for `kind` under `id`, keeping only the digest.
    pub fn issue_token(&self, id: &str, kind: &str, sha256: &str) -> anyhow::Result<TokenRow> {
        let created_at = now();
        self.lock().execute(
            "INSERT INTO tokens (id, type, sha256, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![id, kind, sha256, created_at],
        )?;
        Ok(TokenRow {
            id: id.to_string(),
            kind: kind.to_string(),
            created_at,
        })
    }

    /// The credential a digest belongs to, if it is still issued.
    pub fn token_by_digest(&self, sha256: &str) -> anyhow::Result<Option<TokenRow>> {
        Ok(self
            .lock()
            .query_row(
                "SELECT id, type, created_at FROM tokens WHERE sha256 = ?1",
                params![sha256],
                Self::token_from_row,
            )
            .optional()?)
    }

    pub fn list_tokens(&self) -> anyhow::Result<Vec<TokenRow>> {
        let conn = self.lock();
        let mut stmt =
            conn.prepare("SELECT id, type, created_at FROM tokens ORDER BY created_at")?;
        let rows = stmt
            .query_map([], Self::token_from_row)?
            .filter_map(Result::ok)
            .collect();
        Ok(rows)
    }

    pub fn revoke_token(&self, id: &str) -> anyhow::Result<bool> {
        let n = self
            .lock()
            .execute("DELETE FROM tokens WHERE id = ?1", params![id])?;
        Ok(n > 0)
    }

    fn token_from_row(row: &rusqlite::Row) -> rusqlite::Result<TokenRow> {
        Ok(TokenRow {
            id: row.get(0)?,
            kind: row.get(1)?,
            created_at: row.get(2)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn trigger(watch: &[&str]) -> TriggerConfig {
        TriggerConfig {
            automation: "/tmp/x".into(),
            sources: Default::default(),
            watch: watch.iter().map(|s| s.to_string()).collect(),
            script_timeout_secs: 30,
        }
    }

    fn new_run(id: &str, payload: Value, event_id: Option<i64>) -> NewRun {
        NewRun {
            id: id.into(),
            payload,
            event_id,
        }
    }

    #[test]
    fn publish_wakes_triggers_by_type() {
        let db = Db::open_in_memory().unwrap();
        db.upsert_trigger("t1", &trigger(&["a"])).unwrap();
        db.upsert_trigger("t2", &trigger(&["b"])).unwrap();
        db.trigger_decided("t1", None, 0).unwrap();
        db.trigger_decided("t2", None, 0).unwrap();

        let (id, dirtied) = db.publish("a", &json!({"x": 1}), 100).unwrap();
        assert_eq!((id, dirtied), (1, 1));
        let dirty: Vec<_> = db
            .dirty_triggers()
            .unwrap()
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert_eq!(dirty, ["t1"]);
        // A source counts as well as a watched type.
        let mut with_source = trigger(&[]);
        with_source.sources.insert(
            "tick".into(),
            crate::trigger::SourceConfig::Cron {
                schedule: "* * * * *".into(),
            },
        );
        db.upsert_trigger("t3", &with_source).unwrap();
        db.trigger_decided("t3", None, 0).unwrap();
        assert_eq!(db.publish("tick", &json!({}), 101).unwrap().1, 1);
        assert!(db.get_trigger("t3").unwrap().unwrap().dirty);
    }

    #[test]
    fn a_failure_stops_asking_and_a_decision_clears_it() {
        let db = Db::open_in_memory().unwrap();
        db.upsert_trigger("t", &trigger(&["a"])).unwrap();
        db.trigger_failed("t", "boom").unwrap();
        let t = db.get_trigger("t").unwrap().unwrap();
        assert_eq!((t.last_error.as_deref(), t.dirty), (Some("boom"), false));

        db.publish("a", &json!({}), 100).unwrap();
        assert!(
            db.get_trigger("t").unwrap().unwrap().dirty,
            "an event asks again"
        );
        db.trigger_decided("t", Some(&new_run("r1", json!({"k": 1}), Some(7))), 0)
            .unwrap();
        let t = db.get_trigger("t").unwrap().unwrap();
        assert!(t.last_error.is_none());
        assert!(t.dirty, "a run means: ask again");
        let run = db.get_run("r1").unwrap().unwrap();
        assert_eq!((run.status.as_str(), run.event_id), (PENDING, Some(7)));

        db.trigger_decided("t", None, db.max_event_id().unwrap())
            .unwrap();
        assert!(
            !db.get_trigger("t").unwrap().unwrap().dirty,
            "nothing to do means: clean"
        );
    }

    #[test]
    fn an_event_arriving_during_a_call_keeps_the_trigger_dirty() {
        let db = Db::open_in_memory().unwrap();
        db.upsert_trigger("t", &trigger(&["a"])).unwrap();
        let seen = db.max_event_id().unwrap();
        db.publish("a", &json!(1), 1).unwrap();
        db.trigger_decided("t", None, seen).unwrap();
        assert!(
            db.get_trigger("t").unwrap().unwrap().dirty,
            "the event must be seen next pass"
        );

        let seen = db.max_event_id().unwrap();
        db.publish("other", &json!(2), 2).unwrap();
        db.trigger_decided("t", None, seen).unwrap();
        assert!(
            !db.get_trigger("t").unwrap().unwrap().dirty,
            "an unwatched type changes nothing"
        );
    }

    #[test]
    fn identity_takes_the_oldest_event_without_a_run_per_trigger() {
        let db = Db::open_in_memory().unwrap();
        db.upsert_trigger("t", &trigger(&["a"])).unwrap();
        db.upsert_trigger("u", &trigger(&["a"])).unwrap();
        db.publish("a", &json!(1), 1).unwrap();
        db.publish("a", &json!(2), 2).unwrap();
        let types = vec!["a".to_string()];

        assert_eq!(db.next_unhandled_event("t", &types).unwrap().unwrap().id, 1);
        db.trigger_decided("t", Some(&new_run("r1", json!(1), Some(1))), 0)
            .unwrap();
        assert_eq!(db.next_unhandled_event("t", &types).unwrap().unwrap().id, 2);
        assert_eq!(db.next_unhandled_event("u", &types).unwrap().unwrap().id, 1);
        db.trigger_decided("t", Some(&new_run("r2", json!(2), Some(2))), 0)
            .unwrap();
        assert!(db.next_unhandled_event("t", &types).unwrap().is_none());
    }

    #[test]
    fn runs_of_one_trigger_execute_one_at_a_time() {
        let db = Db::open_in_memory().unwrap();
        db.upsert_trigger("t", &trigger(&["a"])).unwrap();
        db.insert_run("r1", "t", &json!(1)).unwrap();
        db.insert_run("r2", "t", &json!(2)).unwrap();

        let first = db.claim_run().unwrap().unwrap();
        assert_eq!(first.id, "r1");
        assert!(db.claim_run().unwrap().is_none(), "one at a time");

        db.set_run_status("r1", FAILED, Some("x")).unwrap();
        assert_eq!(
            db.claim_run().unwrap().unwrap().id,
            "r2",
            "a failed run stays put"
        );
        db.set_run_status("r2", DONE, None).unwrap();
        assert!(db.claim_run().unwrap().is_none());
    }

    #[test]
    fn snapshot_is_scoped_to_the_trigger() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(&dir.path().join("d.db")).unwrap();
        db.upsert_trigger("t", &trigger(&["a"])).unwrap();
        db.upsert_trigger("u", &trigger(&["b"])).unwrap();
        db.publish("a", &json!({"on": "a"}), 1).unwrap();
        db.publish("b", &json!({"on": "b"}), 2).unwrap();
        db.insert_run("rt", "t", &json!("mine")).unwrap();
        db.insert_run("ru", "u", &json!("theirs")).unwrap();

        let snap = dir.path().join("snapshot.sqlite");
        db.snapshot("t", &["a".into()], &snap).unwrap();
        let conn = Connection::open(&snap).unwrap();
        let col = |sql: &str| -> Vec<String> {
            conn.prepare(sql)
                .unwrap()
                .query_map([], |r| r.get(0))
                .unwrap()
                .map(Result::unwrap)
                .collect()
        };
        assert_eq!(col("SELECT type FROM events"), ["a"]);
        assert_eq!(col("SELECT payload FROM runs"), ["\"mine\""]);
        assert_eq!(
            col(
                "SELECT CAST(e.id AS TEXT) FROM events e LEFT JOIN runs r ON r.event_id = e.id WHERE r.id IS NULL"
            ),
            ["1"]
        );
        db.snapshot("t", &["a".into()], &snap).unwrap();
    }
}
