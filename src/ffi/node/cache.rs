use futures::StreamExt;
use napi::{
    Error, Result, Status,
    threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode},
};
use napi_derive::napi;

use crate::cache::{Cache, TryFromCache};

#[napi(object, js_name = "CacheProgress")]
pub struct JsCacheProgress {
    pub comment: String,
    pub current: u32,
    pub total: u32,
}

pub async fn await_cache_result<T>(
    cache_key: impl Into<String>,
    cache_ctx: Option<std::collections::HashMap<String, crate::value::Value>>,
    progress_callback: Option<
        ThreadsafeFunction<JsCacheProgress, (), JsCacheProgress, Status, false>,
    >,
) -> Result<T>
where
    T: TryFromCache + 'static,
{
    let cache_key = cache_key.into();
    let mut strm = Box::pin(Cache::new().try_create::<T>(cache_key, cache_ctx));
    while let Some(item) = strm.next().await {
        if item.is_err() {
            // Exit the loop and return the error
            return item
                .err()
                .map(|e| Err(Error::new(Status::GenericFailure, e)))
                .unwrap();
        }

        let progress = item.unwrap();

        // Call progress_callback if exists
        if let Some(callback) = &progress_callback {
            let js_progress = JsCacheProgress {
                comment: progress.comment,
                current: progress.current_task as u32,
                total: progress.total_task as u32,
            };
            callback.call(js_progress, ThreadsafeFunctionCallMode::Blocking);
        }

        if progress.current_task < progress.total_task {
            // Continue if progress is not completed
            continue;
        }

        match progress.result {
            Some(inner) => return Ok(inner),
            None => {
                return Err(Error::new(
                    Status::GenericFailure,
                    "CacheProgress didn't return anything",
                ));
            }
        }
    }

    Err(Error::new(Status::GenericFailure, "Unreachable"))
}
