//! `ImageModel`, built from a desc, or around an image model of JavaScript's own, and run on a
//! prompt.
//!
//! A model of JavaScript's own is any object with `generate(prompt, options)`, returning an
//! `ImageOutput` or a promise of one, called with `this` as the model. `options` is the object
//! `ImageModel.generate` was given, or `null`; its keys are the model's to read.

use std::{collections::BTreeMap, sync::Arc};

use ailoy::{
    datatype::Value,
    model::{ImageOptions, ImageOutput, InferImageModel, create_image_model},
};
use futures::future::BoxFuture;
use napi::{
    Env,
    bindgen_prelude::{FnArgs, Function, Object, PromiseRaw, Unknown, within_runtime_if_available},
};
use napi_derive::napi;
use tokio::runtime::Handle;

use crate::{
    agent::OnRuntime,
    convert::{Json, from_js},
    error::{self, Result, invalid, promise, thrown},
    model::{Tsfn, bound},
    tool::{Returned, reason},
};

/// An image model built from its desc, or around one of JavaScript's own, run on a prompt.
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

    /// An `ImageModel` that runs `model`, an object of JavaScript's own with `generate`, as the
    /// module says.
    #[napi(factory)]
    pub fn custom(#[napi(ts_arg_type = "CustomImageModel")] model: Object<'_>) -> Result<Self> {
        let generate: Function<GenerateArgs, Returned<ImageOutput>> = model
            .get("generate")
            .map_err(|e| invalid(e.reason))?
            .ok_or_else(|| invalid("an image model has a `generate` method"))?;
        let generate = bound(generate, model).map_err(|e| invalid(e.reason))?;
        let model: Arc<dyn InferImageModel> = Arc::new(JsModel(Arc::new(generate)));
        Ok(JsImageModel {
            model: OnRuntime::new(model, within_runtime_if_available(Handle::current)),
        })
    }

    /// Generate images from `prompt`, settling with at least one image, and rejecting for a
    /// call that ends with none.
    #[napi(ts_return_type = "Promise<ImageOutput>")]
    pub fn generate<'env>(
        &self,
        env: &'env Env,
        prompt: String,
        #[napi(ts_arg_type = "ImageModelOptions | Record<string, Value>")] options: Option<
            Unknown<'_>,
        >,
    ) -> napi::Result<PromiseRaw<'env, Json<ImageOutput>>> {
        // Any keys: which a model takes is the model's to say.
        let options: Option<BTreeMap<String, Value>> = options
            .map(|options| from_js(env, options, "image model options"))
            .transpose()
            .map_err(|e| thrown(env, e))?;
        let model = self.model();
        promise(env, async move {
            let options = options.as_ref().map(|o| o as &dyn ImageOptions);
            model
                .generate(&prompt, options)
                .await
                .map(Json)
                .map_err(error::anyhow)
        })
    }
}

/// The arguments a custom model's `generate` is called with.
type GenerateArgs = FnArgs<(String, Json<Option<Value>>)>;

/// An image model of JavaScript's own.
struct JsModel(Arc<Tsfn<GenerateArgs, Returned<ImageOutput>>>);

impl InferImageModel for JsModel {
    fn generate(
        &self,
        prompt: &str,
        options: Option<&dyn ImageOptions>,
    ) -> BoxFuture<'static, anyhow::Result<ImageOutput>> {
        let generate = self.0.clone();
        let prompt = prompt.to_owned();
        let options = options.map(|o| o.read::<Value>()).transpose();
        Box::pin(async move {
            let args = FnArgs {
                data: (prompt, Json(options?)),
            };
            let returned = generate
                .call_async_catch(args)
                .await
                .map_err(|e| anyhow::anyhow!(reason(e)))?;
            returned.settle().await.map_err(anyhow::Error::msg)
        })
    }
}
