use anyhow::{Result, anyhow};
use futures::StreamExt;
use js_sys::Function;
use serde::{Deserialize, Serialize};
use tsify::Tsify;
use wasm_bindgen::prelude::*;

use crate::utils::BoxStream;

#[derive(Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct CacheProgress {
    pub comment: String,
    pub current: usize,
    pub total: usize,
}

#[wasm_bindgen(typescript_custom_section)]
const TS_APPEND_CONTENT: &'static str = dedent::dedent!(
    r#"
    export type CacheProgressCallbackFn = (progress: CacheProgress) => void;
    "#
);

pub async fn await_cache_result<'a, T>(
    mut cache_strm: BoxStream<'a, anyhow::Result<crate::cache::CacheProgress<T>>>,
    progress_callback: Option<Function>,
) -> Result<T>
where
    T: std::fmt::Debug,
{
    while let Some(item) = cache_strm.next().await {
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
