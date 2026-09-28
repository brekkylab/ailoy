//! Executing runs: claim a pending one, hand it to the ailoy runner, record how it
//! ended. A failed run stays failed; a new run is made with `fire`.

use std::{sync::Arc, time::Duration};

use ailoy::automation::{AutomationDef, Event, Outcome, Runner};
use serde_json::Value;

use crate::{
    db::{self, RunRow},
    hooks::{self, Transition},
    state::AppState,
};

pub async fn run_loop(state: Arc<AppState>) {
    loop {
        match claim_all(&state).await {
            Ok(n) if n > 0 => continue,
            Ok(_) => {}
            Err(e) => tracing::error!("worker: {e:#}"),
        }
        tokio::select! {
            _ = state.worker_wake.notified() => {}
            _ = tokio::time::sleep(Duration::from_secs(5)) => {}
        }
    }
}

/// Claim every run that can start now and spawn each. Returns how many started.
pub async fn claim_all(state: &Arc<AppState>) -> anyhow::Result<usize> {
    let mut started = 0;
    while let Some(run) = state.db.claim_run()? {
        started += 1;
        let state = state.clone();
        tokio::spawn(async move {
            let id = run.id.clone();
            if let Err(e) = execute(&state, run).await {
                tracing::error!(run = id, "execute: {e:#}");
            }
        });
    }
    Ok(started)
}

/// Runs left `running` by a process that died are failed.
pub async fn recover(state: &AppState) -> anyhow::Result<()> {
    for run in state.db.running_runs()? {
        tracing::warn!(run = run.id, "was running when the daemon stopped");
        fail(
            state,
            &run,
            "the daemon stopped while this run was in progress",
        )
        .await?;
    }
    Ok(())
}

async fn execute(state: &Arc<AppState>, run: RunRow) -> anyhow::Result<()> {
    let Some(trigger) = state.db.get_trigger(&run.trigger)? else {
        return fail(state, &run, "its trigger is gone").await;
    };
    let def = match AutomationDef::load(&trigger.config.automation) {
        Ok(def) => def,
        Err(e) => {
            state.db.trigger_failed(&run.trigger, &e.to_string())?;
            return fail(state, &run, &format!("loading the automation: {e}")).await;
        }
    };
    let program = state.config.console.program.clone();
    let runner = Runner::new(def, state.work_dir(&run.trigger));
    let runner = if state.config.console.host {
        runner.with_host_console(program)
    } else {
        runner.with_console_program(program)
    };
    let prepared = runner.prepare_with_id(&run.id, Event::new(run.payload.clone()))?;

    tracing::info!(run = run.id, trigger = run.trigger, "run starting");
    match runner.execute(prepared).await {
        Ok(record) if record.outcome.is_completed() => {
            state.db.set_run_status(&run.id, db::DONE, None)?;
            tracing::info!(run = run.id, "done");
            hooks::notify(
                state,
                Transition {
                    run_id: &run.id,
                    trigger: &run.trigger,
                    status: db::DONE,
                    output: record.output.as_ref(),
                    error: None,
                },
            )
            .await;
            Ok(())
        }
        Ok(record) => {
            let error = match &record.outcome {
                Outcome::Failed { error } => error.clone(),
                other => format!("{other:?}"),
            };
            fail(state, &run, &error).await
        }
        Err(e) => fail(state, &run, &format!("{e:#}")).await,
    }
}

/// Mark the run failed and tell the outside.
async fn fail(state: &AppState, run: &RunRow, error: &str) -> anyhow::Result<()> {
    state.db.set_run_status(&run.id, db::FAILED, Some(error))?;
    tracing::warn!(run = run.id, "{error}");
    hooks::notify(
        state,
        Transition {
            run_id: &run.id,
            trigger: &run.trigger,
            status: db::FAILED,
            output: None::<&Value>,
            error: Some(error),
        },
    )
    .await;
    Ok(())
}
