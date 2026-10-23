//! `ImageModel`, and the per-org providers image models are built through, in a table apart
//! from the language models'.
//!
//! An org's provider is a backend (`'api'`, the org's own API, is the one image models run
//! through) or a factory of JavaScript's own: called with the org and the name after it
//! (`null` for a desc naming only the org), it returns a model. That is an `ImageModel`, used
//! as is, or any object with `generate(prompt, options)`, returning an `ImageOutput` or a
//! promise of one.
//!
//! `generate` is called with `this` as the model. The factory is called whenever
//! `ImageModel.create` builds a model of the org, from the runtime's thread, which waits for
//! the JavaScript thread to run it; it must return the model, not a promise of it.

use std::sync::Arc;

use ailoy::model::{
    Backend, ImageModelOptions, ImageModelProvider, ImageOutput, InferImageModel,
    create_image_model, get_image_model_provider as get_image_model_provider_rs,
    remove_image_model_provider as remove_image_model_provider_rs,
    set_image_model_provider as set_image_model_provider_rs,
};
use futures::future::BoxFuture;
use napi::{
    Env, JsValue, Status, ValueType,
    bindgen_prelude::{
        ClassInstance, FnArgs, FromNapiValue, Function, Object, PromiseRaw, Unknown,
    },
    sys,
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
    /// The image model `desc` names, `<org>/<name>`, built through the provider set for
    /// `<org>`.
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
        let options = match options {
            Some(options) => {
                from_js(env, options, "image model options").map_err(|e| thrown(env, e))?
            }
            None => ImageModelOptions::default(),
        };
        let model = self.model();
        promise(env, async move {
            model
                .generate(&prompt, &options)
                .await
                .map(Json)
                .map_err(error::anyhow)
        })
    }
}

/// The arguments a custom model's `generate` is called with.
type GenerateArgs = FnArgs<(String, Json<ImageModelOptions>)>;

/// An image model of JavaScript's own, as a factory returned it.
struct JsModel(Arc<Tsfn<GenerateArgs, Returned<ImageOutput>>>);

impl InferImageModel for JsModel {
    fn generate(
        &self,
        prompt: &str,
        options: &ImageModelOptions,
    ) -> BoxFuture<'static, anyhow::Result<ImageOutput>> {
        let generate = self.0.clone();
        let args = FnArgs {
            data: (prompt.to_owned(), Json(options.clone())),
        };
        Box::pin(async move {
            let returned = generate
                .call_async_catch(args)
                .await
                .map_err(|e| anyhow::anyhow!(reason(e)))?;
            returned.settle().await.map_err(anyhow::Error::msg)
        })
    }
}

/// What a factory returned: an `ImageModel`, or a model of JavaScript's own.
struct Made(Arc<dyn InferImageModel>);

impl FromNapiValue for Made {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> napi::Result<Self> {
        if let Ok(model) = unsafe { ClassInstance::<JsImageModel>::from_napi_value(env, napi_val) }
        {
            return Ok(Made(model.model()));
        }
        let model = unsafe { Object::from_napi_value(env, napi_val)? };
        let generate: Function<GenerateArgs, Returned<ImageOutput>> =
            model.get("generate")?.ok_or_else(|| {
                napi::Error::new(Status::InvalidArg, "an image model has a `generate` method")
            })?;
        Ok(Made(Arc::new(JsModel(Arc::new(bound(generate, model)?)))))
    }
}

type Factory = Tsfn<FnArgs<(String, Option<String>)>, Made>;

/// The image models of an org from what `factory` returns for it.
fn custom(factory: Factory) -> ImageModelProvider {
    ImageModelProvider::custom(move |org, name| {
        let args = FnArgs {
            data: (org.to_owned(), name.map(str::to_owned)),
        };
        // Called on the runtime's thread, never the JavaScript one, which is free to run it.
        futures::executor::block_on(factory.call_async_catch(args))
            .map(|made| made.0)
            .map_err(|e| anyhow::anyhow!(reason(e)))
    })
}

/// Build the image models of `org` through `provider`: a backend, or a factory, as the module
/// says. Replaces the provider the org had.
#[napi]
pub fn set_image_model_provider(
    env: &Env,
    org: String,
    #[napi(
        ts_arg_type = "'cli' | 'api' | 'bedrock' | 'openrouter' | 'vllm' | ((org: string, name: string | null) => ImageModel | CustomImageModel)"
    )]
    provider: Unknown<'_>,
) -> Result<()> {
    let provider = match provider.get_type().map_err(|e| invalid(e.reason))? {
        ValueType::String => {
            ImageModelProvider::from(from_js::<Backend>(env, provider, "a backend")?)
        }
        ValueType::Function => {
            let factory = unsafe { Factory::from_napi_value(env.raw(), provider.raw()) }
                .map_err(|e| invalid(e.reason))?;
            custom(factory)
        }
        _ => {
            return Err(invalid(
                "an image model provider is a backend's name or a factory of models",
            ));
        }
    };
    set_image_model_provider_rs(org, provider);
    Ok(())
}

/// How the image model `desc` names (`'<org>/<name>'`, or `'<org>'`) is built: a backend's
/// name, `'custom'` for a factory, or `null` when its org has no provider.
#[napi(ts_return_type = "'cli' | 'api' | 'bedrock' | 'openrouter' | 'vllm' | 'custom' | null")]
pub fn get_image_model_provider(desc: String) -> Option<String> {
    get_image_model_provider_rs(&desc).map(|provider| match provider {
        ImageModelProvider::Backend(backend) => backend.to_string(),
        ImageModelProvider::Custom(_) => "custom".to_string(),
    })
}

/// Stop building the image models of `org`. Answers whether it had a provider.
#[napi]
pub fn remove_image_model_provider(org: String) -> bool {
    remove_image_model_provider_rs(&org).is_some()
}
