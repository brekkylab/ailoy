//! Turning dirty triggers into runs, one trigger at a time.
//!
//! An event marks the triggers of its type dirty; this loop takes the dirty triggers in name
//! order and asks each one's script (or the identity logic) for the next run. A
//! script answers one run or none. One means the trigger stays dirty and is asked
//! again on the next pass; none means it is clean until an event of its types arrives. A burst of
//! events costs one call per run made, plus one.

use std::{fs, path::Path, sync::Arc, time::Duration};

use ailoy::automation::AutomationDef;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    db::{NewRun, TriggerRow, now},
    state::AppState,
};

/// `output.json`: the next run's payload, or `null` for nothing to do.
#[derive(Deserialize)]
struct ScriptOutput {
    #[serde(default)]
    run: Option<Value>,
    #[serde(default)]
    event_id: Option<i64>,
}

pub async fn run_loop(state: Arc<AppState>) {
    loop {
        match pass(&state).await {
            Ok(n) if n > 0 => continue,
            Ok(_) => {}
            Err(e) => tracing::error!("dispatcher: {e:#}"),
        }
        tokio::select! {
            _ = state.publisher.wake.notified() => {}
            _ = tokio::time::sleep(Duration::from_secs(60)) => {}
        }
    }
}

/// One pass over the dirty triggers. Returns how many were asked.
pub async fn pass(state: &AppState) -> anyhow::Result<usize> {
    let retention = state.config.events_retention_secs as i64;
    state.db.prune_events(now() - retention)?;

    let dirty = state.db.dirty_triggers()?;
    for trigger in &dirty {
        if let Err(e) = process(state, trigger).await {
            tracing::error!(trigger = %trigger.name, "dispatch: {e:#}");
        }
    }
    Ok(dirty.len())
}

async fn process(state: &AppState, trigger: &TriggerRow) -> anyhow::Result<()> {
    let name = trigger.name.as_str();
    let cfg = &trigger.config;
    let seen_event_id = state.db.max_event_id()?;

    // The automation has to be there before anything is made from its events.
    if let Err(e) = AutomationDef::load(&cfg.automation) {
        state.db.trigger_failed(name, &e.to_string())?;
        tracing::warn!(trigger = name, "{e}");
        return Ok(());
    }

    let decision = match cfg.script_path() {
        None => identity(state, trigger),
        Some(path) => script(state, trigger, &path).await,
    };

    match decision {
        Ok(next) => {
            let run = next.map(|(payload, event_id)| NewRun {
                id: uuid::Uuid::new_v4().simple().to_string(),
                payload,
                event_id,
            });
            state
                .db
                .trigger_decided(name, run.as_ref(), seen_event_id)?;
            if let Some(r) = run {
                tracing::info!(trigger = name, run = r.id, "run created");
                state.worker_wake.notify_one();
            }
        }
        Err(e) => {
            state.db.trigger_failed(name, &format!("{e:#}"))?;
            tracing::warn!(trigger = name, "script failed: {e:#}");
        }
    }
    Ok(())
}

/// No script: the oldest event this trigger has made no run for.
fn identity(
    state: &AppState,
    trigger: &TriggerRow,
) -> anyhow::Result<Option<(Value, Option<i64>)>> {
    Ok(state
        .db
        .next_unhandled_event(&trigger.name, &trigger.config.types())?
        .map(|e| {
            (
                json!({ "type": e.kind, "at": e.at, "payload": e.payload }),
                Some(e.id),
            )
        }))
}

/// Snapshot, copy the script in, run it on the shared console, read what it answered.
async fn script(
    state: &AppState,
    trigger: &TriggerRow,
    script_path: &Path,
) -> anyhow::Result<Option<(Value, Option<i64>)>> {
    let name = &trigger.name;
    let cfg = &trigger.config;
    let code = fs::read_to_string(script_path)
        .map_err(|e| anyhow::anyhow!("reading {}: {e}", script_path.display()))?;

    // Every file below is rewritten in place and none is ever unlinked: the console
    // shares this tree with a guest, and a name whose file was removed and remade reads
    // back there as the old one or as nothing.
    let host_dir = state.trigger_dir(name);
    fs::create_dir_all(&host_dir)?;
    state
        .db
        .snapshot(name, &cfg.types(), &host_dir.join("snapshot.sqlite"))?;
    fs::write(host_dir.join("trigger.py"), code)?;
    let output_path = host_dir.join("output.json");
    // Emptied rather than removed, so what the script writes is not read as the last
    // call's answer and the file keeps its identity.
    fs::write(&output_path, b"")?;

    let mut console = state.console.lock().await;
    console.ensure(&state.db).await?;
    let guest_dir = console.guest_scratch().join(name);
    let guest = |file: &str| guest_dir.join(file).to_string_lossy().into_owned();

    let input = json!({ "trigger": name, "now": now(), "db": guest("snapshot.sqlite") });
    fs::write(
        host_dir.join("input.json"),
        serde_json::to_vec_pretty(&input)?,
    )?;

    let resp = console
        .exec(
            [
                "python3".to_string(),
                guest("trigger.py"),
                guest("input.json"),
                guest("output.json"),
            ],
            cfg.script_timeout_secs * 1000,
        )
        .await?;
    drop(console);

    if resp.code != 0 {
        let stderr = String::from_utf8_lossy(&resp.stderr);
        anyhow::bail!(
            "trigger.py exited {}: {}",
            resp.code,
            stderr
                .chars()
                .rev()
                .take(1000)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<String>()
                .trim()
        );
    }
    let bytes = fs::read(&output_path)
        .map_err(|e| anyhow::anyhow!("trigger.py wrote no output.json: {e}"))?;
    if bytes.is_empty() {
        anyhow::bail!("trigger.py wrote no output.json");
    }
    let out: ScriptOutput = serde_json::from_slice(&bytes)
        .map_err(|e| anyhow::anyhow!("output.json is not {{\"run\": ... | null}}: {e}"))?;
    Ok(out.run.map(|payload| (payload, out.event_id)))
}

/// Remove the leftovers of one trigger's script calls.
pub fn clear_trigger_dir(dir: &Path) {
    let _ = fs::remove_dir_all(dir);
}
