//! `ImageModel`, built from a desc and run on a prompt.

use std::sync::Arc;

use ailoy::model::{
    ImageModelOptions, ImageOptions, ImageOutput, InferImageModel, create_image_model,
};
use napi::{
    Env,
    bindgen_prelude::{PromiseRaw, Unknown},
};
use napi_derive::napi;
use tokio::runtime::Handle;

use crate::{
    agent::OnRuntime,
    convert::{Json, from_js},
    error::{self, promise, thrown},
};

/// An image model built from its desc, run on a prompt.
#[napi(js_name = "ImageModel")]
pub struct JsImageModel {
    model: OnRuntime<Arc<dyn InferImageModel>>,
}

impl JsImageModel {
    fn model(&self) -> Arc<dyn InferImageModel> {
        self.model
            .value
            .clone()
            .expect("a model is only let go of when dropped")
    }
}

#[napi]
impl JsImageModel {
    /// The image model `desc` names, `<org>/<name>`.
    #[napi(ts_return_type = "Promise<ImageModel>")]
    pub fn create<'env>(
        env: &'env Env,
        desc: String,
    ) -> napi::Result<PromiseRaw<'env, JsImageModel>> {
        promise(env, async move {
            let model = create_image_model(&desc).await.map_err(error::anyhow)?;
            Ok(JsImageModel {
                model: OnRuntime::new(model, Handle::current()),
            })
        })
    }

    /// Generate images from `prompt`, settling with at least one image, and rejecting for a
    /// call that ends with none.
    #[napi(ts_return_type = "Promise<ImageOutput>")]
    pub fn generate<'env>(
        &self,
        env: &'env Env,
        prompt: String,
        #[napi(ts_arg_type = "ImageModelOptions")] options: Option<Unknown<'_>>,
    ) -> napi::Result<PromiseRaw<'env, Json<ImageOutput>>> {
        let options: Option<ImageModelOptions> = options
            .map(|options| from_js(env, options, "image model options"))
            .transpose()
            .map_err(|e| thrown(env, e))?;
        let model = self.model();
        promise(env, async move {
            model
                .generate(&prompt, options.as_ref().map(|o| o as &dyn ImageOptions))
                .await
                .map(Json)
                .map_err(error::anyhow)
        })
    }
}
