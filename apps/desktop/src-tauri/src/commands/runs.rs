use ailoy::message::Part;
use ailoy_desktop_core::{EngineError, RunEvent};
use tauri::ipc::Channel;
use tokio::sync::broadcast;

use super::Eng;

/// Pump a run's broadcast into `sink` until a terminal event or the run is gone. A lagged
/// receiver skips ahead: the UI reconciles from `message_list`, which every completed
/// message already reached.
pub async fn forward(mut rx: broadcast::Receiver<RunEvent>, mut sink: impl FnMut(RunEvent)) {
    loop {
        match rx.recv().await {
            Ok(ev) => {
                let terminal = matches!(
                    ev,
                    RunEvent::Done | RunEvent::Cancelled | RunEvent::Error { .. }
                );
                sink(ev);
                if terminal {
                    break;
                }
            }
            Err(broadcast::error::RecvError::Lagged(n)) => {
                tracing::warn!("run event channel lagged by {n}");
            }
            Err(broadcast::error::RecvError::Closed) => break,
        }
    }
}

#[tauri::command]
pub async fn run_start(
    engine: Eng<'_>,
    session_id: String,
    text: String,
    on_event: Channel<RunEvent>,
) -> Result<String, EngineError> {
    let handle = engine
        .run_start(&session_id, vec![Part::text(text)])
        .await?;
    let run_id = handle.run_id.clone();
    tauri::async_runtime::spawn(forward(handle.events, move |ev| {
        let _ = on_event.send(ev);
    }));
    Ok(run_id)
}

#[tauri::command]
pub async fn run_attach(
    engine: Eng<'_>,
    session_id: String,
    on_event: Channel<RunEvent>,
) -> Result<Option<String>, EngineError> {
    let Some((handle, partial)) = engine.run_attach(&session_id).await else {
        return Ok(None);
    };
    let run_id = handle.run_id.clone();
    let _ = on_event.send(RunEvent::Started {
        run_id: run_id.clone(),
    });
    if !partial.is_empty() {
        let _ = on_event.send(RunEvent::TextDelta { text: partial });
    }
    tauri::async_runtime::spawn(forward(handle.events, move |ev| {
        let _ = on_event.send(ev);
    }));
    Ok(Some(run_id))
}

#[tauri::command]
pub async fn run_cancel(engine: Eng<'_>, session_id: String) -> Result<bool, EngineError> {
    Ok(engine.run_cancel(&session_id).await)
}

#[cfg(test)]
mod tests {
    use ailoy_desktop_core::RunEvent;
    use tokio::sync::broadcast;

    use super::forward;

    #[tokio::test]
    async fn forward_stops_after_a_terminal_event() {
        let (tx, rx) = broadcast::channel(16);
        tx.send(RunEvent::TextDelta { text: "a".into() }).unwrap();
        tx.send(RunEvent::Done).unwrap();
        tx.send(RunEvent::TextDelta {
            text: "late".into(),
        })
        .unwrap();
        let mut seen = Vec::new();
        forward(rx, |ev| seen.push(ev)).await;
        assert_eq!(seen.len(), 2);
        assert!(matches!(seen[1], RunEvent::Done));
    }

    #[tokio::test]
    async fn forward_ends_when_the_sender_is_dropped() {
        let (tx, rx) = broadcast::channel(16);
        tx.send(RunEvent::TextDelta { text: "a".into() }).unwrap();
        drop(tx);
        let mut n = 0;
        forward(rx, |_| n += 1).await;
        assert_eq!(n, 1);
    }
}
