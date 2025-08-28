#[cfg(any(target_family = "unix", target_family = "windows"))]
mod tvm_runtime {
    use cxx::UniquePtr;

    use crate::{
        cache::{Cache, CacheContents, CacheEntry, TryFromCache},
        ffi::{
            DLPackTensor, TVMEmbeddingModel, TVMLanguageModel, create_dldevice,
            create_tvm_embedding_model, create_tvm_language_model,
        },
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
        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
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
        #[cfg(not(any(
            target_os = "linux",
            target_os = "windows",
            all(target_arch = "aarch64", target_os = "macos"),
        )))]
        {
            panic!("accelerator not found")
        }
    }

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

    fn inferencer_claim_files(
        cache: Cache,
        key: impl AsRef<str>,
    ) -> BoxFuture<'static, Result<Vec<CacheEntry>, String>> {
        let dirname = vec![key.as_ref().replace("/", "--")].join("--");
        let elem = CacheEntry::new(&dirname, "ndarray-cache.json");
        Box::pin(async move {
            let ndarray_cache_bytes = cache.get(&elem).await?;
            let ndarray_cache_str =
                std::str::from_utf8(&ndarray_cache_bytes).map_err(|_| format!("Internal error"))?;
            let ndarray_cache: serde_json::Value = serde_json::from_str(ndarray_cache_str)
                .map_err(|e| format!("JSON deserialization failed: {}", e.to_string()))?;
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
            Ok(rv)
        })
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
        ) -> BoxFuture<'static, Result<Vec<CacheEntry>, String>> {
            inferencer_claim_files(cache, key)
        }

        fn try_from_contents(contents: &mut CacheContents) -> Result<Self, String> {
            let device = create_dldevice(
                get_device_type(get_accelerator()),
                get_device_id(get_accelerator()),
            );
            let inner = create_tvm_embedding_model(contents, device);

            Ok(EmbeddingModelInferencer { inner })
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
        ) -> BoxFuture<'static, Result<Vec<CacheEntry>, String>> {
            inferencer_claim_files(cache, key)
        }

        fn try_from_contents(contents: &mut CacheContents) -> Result<Self, String> {
            let device = create_dldevice(
                get_device_type(get_accelerator()),
                get_device_id(get_accelerator()),
            );
            let inner = create_tvm_language_model(contents, device);

            Ok(LanguageModelInferencer { inner })
        }
    }
}

#[cfg(any(target_family = "wasm"))]
mod tvmjs_runtime {
    use crate::{
        cache::{Cache, CacheContents, CacheEntry, TryFromCache},
        utils::BoxFuture,
    };

    #[derive(Debug)]
    pub struct LanguageModelInferencer {}

    impl LanguageModelInferencer {
        pub fn prefill(&mut self, _: &[u32]) -> () {
            todo!()
        }

        pub fn decode(&mut self, _: u32) -> u32 {
            todo!()
        }
    }

    impl TryFromCache for LanguageModelInferencer {
        fn claim_files(
            cache: Cache,
            key: impl AsRef<str>,
        ) -> BoxFuture<'static, Result<Vec<CacheEntry>, String>> {
            Box::pin(async move { Ok(vec![]) })
        }

        fn try_from_contents(contents: &mut CacheContents) -> Result<Self, String> {
            Ok(LanguageModelInferencer {})
        }
    }

    #[derive(Debug)]
    pub struct EmbeddingModelInferencer {}

    impl EmbeddingModelInferencer {
        pub fn infer(&mut self, tokens: &[i32]) -> DLPackTensor {
            todo!()
        }
    }

    impl TryFromCache for EmbeddingModelInferencer {
        fn claim_files(
            cache: Cache,
            key: impl AsRef<str>,
        ) -> BoxFuture<'static, Result<Vec<CacheEntry>, String>> {
            Box::pin(async move { Ok(vec![]) })
        }

        fn try_from_contents(contents: &mut CacheContents) -> Result<Self, String> {
            Ok(EmbeddingModelInferencer {})
        }
    }
}

#[cfg(any(target_family = "unix", target_family = "windows"))]
pub use tvm_runtime::EmbeddingModelInferencer;
#[cfg(any(target_family = "unix", target_family = "windows"))]
pub use tvm_runtime::LanguageModelInferencer;

#[cfg(any(target_family = "wasm"))]
pub use tvmjs_runtime::EmbeddingModelInferencer;
#[cfg(any(target_family = "wasm"))]
pub use tvmjs_runtime::LanguageModelInferencer;
