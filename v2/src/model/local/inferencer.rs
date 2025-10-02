use anyhow::Context;
#[cfg(any(target_family = "unix", target_family = "windows"))]
pub use tvm_runtime::EmbeddingModelInferencer;
#[cfg(any(target_family = "unix", target_family = "windows"))]
pub use tvm_runtime::LanguageModelInferencer;
#[cfg(any(target_family = "wasm"))]
pub use tvmjs_runtime::EmbeddingModelInferencer;
#[cfg(any(target_family = "wasm"))]
pub use tvmjs_runtime::LanguageModelInferencer;

use crate::{
    cache::{Cache, CacheClaim, CacheEntry},
    utils::BoxFuture,
};

pub fn get_lib_extension() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "dylib"
    }
    #[cfg(target_os = "linux")]
    {
        "so"
    }
    #[cfg(target_os = "windows")]
    {
        "dll"
    }
    #[cfg(target_arch = "wasm32")]
    {
        "wasm"
    }
    #[cfg(not(any(
        target_os = "linux",
        target_os = "windows",
        target_os = "macos",
        target_arch = "wasm32"
    )))]
    {
        panic!("Unknown OS")
    }
}

pub fn get_accelerator() -> &'static str {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        "metal"
    }
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    {
        "vulkan"
    }
    #[cfg(target_arch = "wasm32")]
    {
        "webgpu"
    }
    #[cfg(not(any(
        target_os = "linux",
        target_os = "windows",
        all(target_arch = "aarch64", target_os = "macos"),
        target_arch = "wasm32",
    )))]
    {
        panic!("accelerator not found")
    }
}

pub fn claim_files(
    cache: Cache,
    key: impl AsRef<str>,
) -> BoxFuture<'static, anyhow::Result<CacheClaim>> {
    let dirname = vec![key.as_ref().replace("/", "--")].join("--");
    let elem = CacheEntry::new(&dirname, "ndarray-cache.json");
    Box::pin(async move {
        let ndarray_cache_bytes = cache.get(&elem).await?;
        let ndarray_cache_str =
            std::str::from_utf8(&ndarray_cache_bytes).context("Internal error")?;
        let ndarray_cache: serde_json::Value =
            serde_json::from_str(ndarray_cache_str).context("JSON deserialization failed")?;
        let mut rv = ndarray_cache
            .as_object()
            .unwrap()
            .get("records")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                CacheEntry::new(
                    &dirname,
                    v.as_object()
                        .unwrap()
                        .get("dataPath")
                        .unwrap()
                        .as_str()
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        rv.push(CacheEntry::new(&dirname, "ndarray-cache.json"));
        rv.push(CacheEntry::new(
            format!(
                "{}--{}--{}",
                dirname,
                env!("BUILD_TARGET_TRIPLE"),
                get_accelerator()
            ),
            format!("rt.{}", get_lib_extension()),
        ));
        Ok(CacheClaim::new(rv))
    })
}

#[cfg(any(target_family = "unix", target_family = "windows"))]
mod tvm_runtime {
    use cxx::UniquePtr;

    use super::*;
    use crate::{
        cache::{Cache, CacheContents, TryFromCache},
        ffi::cxx_bridge::{
            DLPackTensor, TVMEmbeddingModel, TVMLanguageModel, create_dldevice,
            create_tvm_embedding_model, create_tvm_language_model,
        },
        utils::BoxFuture,
    };

    pub fn get_device_type(accelerator: &str) -> i32 {
        if accelerator == "metal" {
            8
        } else if accelerator == "vulkan" {
            7
        } else {
            0
        }
    }

    pub fn get_device_id(_: &str) -> i32 {
        0
    }

    #[derive(Debug)]
    pub struct EmbeddingModelInferencer {
        inner: UniquePtr<TVMEmbeddingModel>,
    }

    impl EmbeddingModelInferencer {
        pub fn infer(&mut self, tokens: &[u32]) -> DLPackTensor {
            self.inner.pin_mut().infer(tokens)
        }
    }

    impl TryFromCache for EmbeddingModelInferencer {
        fn claim_files(
            cache: Cache,
            key: impl AsRef<str>,
        ) -> BoxFuture<'static, anyhow::Result<CacheClaim>> {
            claim_files(cache, key)
        }

        fn try_from_contents(
            mut contents: CacheContents,
        ) -> BoxFuture<'static, anyhow::Result<Self>> {
            Box::pin(async move {
                let device = create_dldevice(
                    get_device_type(get_accelerator()),
                    get_device_id(get_accelerator()),
                );
                let inner = create_tvm_embedding_model(&mut contents, device);

                Ok(EmbeddingModelInferencer { inner })
            })
        }
    }

    #[derive(Debug)]
    pub struct LanguageModelInferencer {
        inner: UniquePtr<TVMLanguageModel>,
    }

    impl LanguageModelInferencer {
        pub fn prefill(&mut self, tokens: &[u32]) -> () {
            self.inner.pin_mut().prefill(tokens)
        }

        pub fn decode(&mut self, last_token: u32) -> u32 {
            let logits = self.inner.pin_mut().decode(last_token);
            self.inner.pin_mut().sample(logits)
        }
    }

    impl TryFromCache for LanguageModelInferencer {
        fn claim_files(
            cache: Cache,
            key: impl AsRef<str>,
        ) -> BoxFuture<'static, anyhow::Result<CacheClaim>> {
            claim_files(cache, key)
        }

        fn try_from_contents(
            mut contents: CacheContents,
        ) -> BoxFuture<'static, anyhow::Result<Self>> {
            Box::pin(async move {
                let device = create_dldevice(
                    get_device_type(get_accelerator()),
                    get_device_id(get_accelerator()),
                );
                let inner = create_tvm_language_model(&mut contents, device);

                Ok(LanguageModelInferencer { inner })
            })
        }
    }
}

#[cfg(any(target_family = "wasm"))]
mod tvmjs_runtime {
    use std::fmt;

    use js_sys::{Float32Array, Object, Reflect, Uint8Array, Uint32Array};
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::JsFuture;

    use super::*;
    use crate::{
        cache::{Cache, CacheContents, TryFromCache},
        ffi::js_bridge::{
            JSEmbeddingModel, JSLanguageModel, init_embedding_model_js, init_language_model_js,
        },
        utils::{BoxFuture, float16},
    };

    pub struct LanguageModelInferencer {
        inner: JSLanguageModel,
    }

    impl fmt::Debug for LanguageModelInferencer {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("LanguageModelInferencer")
                // .field("init", &self.init)
                // .field("inner", &self.inner)
                .finish()
        }
    }

    impl LanguageModelInferencer {
        pub async fn prefill(&mut self, tokens: &[u32]) -> () {
            let arr = unsafe { Uint32Array::view(tokens) };
            JsFuture::from(self.inner.prefill(arr)).await.unwrap();
        }

        pub async fn decode(&mut self, last_token: u32) -> u32 {
            let logits: Float32Array = JsFuture::from(self.inner.decode(last_token))
                .await
                .unwrap()
                .into();
            self.inner.sample(logits)
        }
    }

    impl TryFromCache for LanguageModelInferencer {
        fn claim_files(
            cache: Cache,
            key: impl AsRef<str>,
        ) -> BoxFuture<'static, anyhow::Result<CacheClaim>> {
            claim_files(cache, key)
        }

        fn try_from_contents(
            mut contents: CacheContents,
        ) -> BoxFuture<'static, anyhow::Result<Self>> {
            Box::pin(async move {
                let cache_contents = {
                    let obj = Object::new();
                    for (entry, buf) in contents.drain() {
                        let filename = entry.filename();
                        let u8arr = Uint8Array::new_with_length(buf.len() as u32);
                        u8arr.copy_from(&buf[..]);
                        Reflect::set(&obj, &JsValue::from_str(filename), &u8arr.buffer().into())
                            .unwrap();
                    }
                    obj
                };

                let prom = init_language_model_js(&cache_contents);
                let js_lm = match JsFuture::from(prom).await {
                    Ok(out) => {
                        let lm: JSLanguageModel =
                            out.dyn_into().context("Conversion failed: {:?}", e)?;
                        lm
                    }
                    Err(err) => {
                        bail!("JS inferencer init failed: {:?}", err);
                    }
                };

                Ok(LanguageModelInferencer { inner: js_lm })
            })
        }
    }

    pub struct EmbeddingModelInferencer {
        inner: JSEmbeddingModel,
    }

    impl fmt::Debug for EmbeddingModelInferencer {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("EmbeddingModelInferencer")
                // .field("init", &self.init)
                // .field("inner", &self.inner)
                .finish()
        }
    }

    impl EmbeddingModelInferencer {
        pub async fn infer(&mut self, tokens: &[u32]) -> Vec<f32> {
            let arr = unsafe { js_sys::Uint32Array::view(tokens) };
            let res = self.inner.infer(arr);
            let result_vector = JsFuture::from(res).await.unwrap();

            let f32_vec = if let Some(f32_array) = result_vector.dyn_ref::<js_sys::Float32Array>() {
                f32_array.to_vec()
            } else if let Some(u16_array) = result_vector.dyn_ref::<js_sys::Uint16Array>() {
                u16_array
                    .to_vec()
                    .into_iter()
                    .map(|val| float16::f16_to_f32(val))
                    .collect()
            } else {
                vec![]
            };
            f32_vec
        }
    }

    impl TryFromCache for EmbeddingModelInferencer {
        fn claim_files(
            cache: Cache,
            key: impl AsRef<str>,
        ) -> BoxFuture<'static, anyhow::Result<CacheClaim>> {
            claim_files(cache, key)
        }

        fn try_from_contents(
            mut contents: CacheContents,
        ) -> BoxFuture<'static, anyhow::Result<Self>> {
            Box::pin(async move {
                let cache_contents = {
                    let obj = Object::new();
                    for (entry, buf) in contents.drain() {
                        let filename = entry.filename();
                        let u8arr = Uint8Array::new_with_length(buf.len() as u32);
                        u8arr.copy_from(&buf[..]);
                        Reflect::set(&obj, &JsValue::from_str(filename), &u8arr.buffer().into())
                            .unwrap();
                    }
                    obj
                };

                let prom = init_embedding_model_js(&cache_contents);
                let js_em: JSEmbeddingModel = match JsFuture::from(prom).await {
                    Ok(out) => {
                        let em: JSEmbeddingModel =
                            out.dyn_into().context("Conversion failed: {:?}", e)?;
                        em
                    }
                    Err(err) => {
                        bail!("JS inferencer init failed: {:?}", err);
                    }
                };

                Ok(EmbeddingModelInferencer { inner: js_em })
            })
        }
    }
}
