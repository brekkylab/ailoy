use anyhow::{Result, anyhow};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tsify::Tsify;
use wasm_bindgen::prelude::*;

use crate::cache::{Cache, TryFromCache};

#[derive(Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct CacheProgress {
    pub comment: String,
    pub current: usize,
    pub total: usize,
}

pub async fn await_cache_result<T>(
    cache_key: impl Into<String>,
    progress_callback: Option<js_sys::Function>,
) -> Result<T>
where
    T: TryFromCache + std::fmt::Debug + 'static,
{
    let cache_key = cache_key.into();
    let mut strm = Box::pin(Cache::new().try_create::<T>(cache_key));
    while let Some(item) = strm.next().await {
        if item.is_err() {
            return Err(anyhow!(item.unwrap_err()));
        }

        let progress = item.unwrap();

        // Call progress_callback if exists
        if let Some(callback) = &progress_callback {
            let js_progress = CacheProgress {
                comment: progress.comment,
                current: progress.current_task,
                total: progress.total_task,
            };
            callback
                .call1(&JsValue::NULL, &js_progress.into())
                .map_err(|_| anyhow!("Failed to call progress callback"))?;
        }

        if progress.current_task < progress.total_task {
            // Continue if progress is not completed
            continue;
        }

        match progress.result {
            Some(inner) => return Ok(inner),
            None => {
                return Err(anyhow!("CacheProgress didn't return anything"));
            }
        }
    }

    Err(anyhow!("Unreachable"))
}
