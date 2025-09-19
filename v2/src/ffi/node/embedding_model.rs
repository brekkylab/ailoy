use std::sync::Arc;

use futures::lock::Mutex;
use napi::{Status, bindgen_prelude::*, threadsafe_function::ThreadsafeFunction};
use napi_derive::napi;

use crate::{
    ffi::node::cache::{JsCacheProgress, await_cache_result},
    model::{EmbeddingModel, LocalEmbeddingModel},
};

type Embedding = Vec<f32>;

pub trait JsEmbeddingModelMethods<T: EmbeddingModel + 'static> {
    fn inner(&self) -> Arc<Mutex<LocalEmbeddingModel>>;

    async fn _run(&mut self, message: String) -> napi::Result<Embedding> {
        self.inner()
            .lock()
            .await
            .run(message)
            .await
            .map_err(|e| napi::Error::new(Status::GenericFailure, e.to_string()))
    }
}

#[napi(js_name = "LocalEmbeddingModel")]
pub struct JsLocalEmbeddingModel {
    inner: Arc<Mutex<LocalEmbeddingModel>>,
}

impl JsEmbeddingModelMethods<LocalEmbeddingModel> for JsLocalEmbeddingModel {
    fn inner(&self) -> Arc<Mutex<LocalEmbeddingModel>> {
        self.inner.clone()
    }
}

impl FromNapiValue for JsLocalEmbeddingModel {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> Result<Self> {
        let ci = unsafe { ClassInstance::<Self>::from_napi_value(env, napi_val) }?;
        let inner = ci.as_ref().inner.clone();
        Ok(Self { inner })
    }
}

#[napi]
impl JsLocalEmbeddingModel {
    #[napi]
    pub async fn create(
        model_name: String,
        progress_callback: Option<
            ThreadsafeFunction<JsCacheProgress, (), JsCacheProgress, Status, false>,
        >,
    ) -> napi::Result<JsLocalEmbeddingModel> {
        let inner = await_cache_result::<LocalEmbeddingModel>(model_name, progress_callback)
            .await
            .unwrap();
        Ok(JsLocalEmbeddingModel {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    #[napi]
    pub async unsafe fn run(&mut self, message: String) -> napi::Result<Embedding> {
        self._run(message).await
    }
}
