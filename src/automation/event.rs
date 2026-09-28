//! What starts a run: one event, as whoever produced it handed it over.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One thing that happened. The payload is the workflow's `event` input, verbatim;
/// what it means is between the trigger that produced it and the tasks that read it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub payload: Value,
    /// Unix seconds.
    pub received_at: u64,
}

impl Event {
    pub fn new(payload: Value) -> Self {
        Self {
            payload,
            received_at: unix_now(),
        }
    }
}

pub(super) fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
