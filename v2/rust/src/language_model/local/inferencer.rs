#[cfg(any(target_family = "unix", target_family = "windows"))]
mod tvm_runtime {
    use std::{path::PathBuf, pin::Pin};

    use cxx::UniquePtr;

    use crate::{
        cache::{Cache, CacheElement, TryFromCache},
        ffi::{TVMLanguageModel, create_dldevice, create_tvm_language_model},
    };

    pub fn get_accelerator() -> &'static str {
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        {
            "metal"
        }
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            "vulkan"
        }
        #[cfg(target_family = "wasm")]
        {
            "webgpu"
        }
        #[cfg(not(any(
            target_os = "linux",
            target_os = "windows",
            all(target_arch = "aarch64", target_os = "macos"),
            target_family = "wasm"
        )))]
        {
            "unknown"
        }
    }

    #[derive(Debug)]
    pub struct Inferencer {
        inner: UniquePtr<TVMLanguageModel>,
    }

    impl Inferencer {
        pub fn prefill(&mut self, tokens: &Vec<u32>) -> () {
            self.inner.pin_mut().prefill(tokens)
        }

        pub fn decode(&mut self, last_token: u32) -> u32 {
            let logits = self.inner.pin_mut().decode(last_token);
            self.inner.pin_mut().sample(logits)
        }
    }

    impl TryFromCache for Inferencer {
        fn claim_files(
            cache: Cache,
            key: impl AsRef<str>,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<CacheElement>, String>>>> {
            let dirname = vec![key.as_ref().replace("/", "--")].join("--");
            let elem = CacheElement::new(&dirname, "ndarray-cache.json");
            Box::pin(async move {
                let ndarray_cache_bytes = cache.get(&elem).await?;
                let ndarray_cache_str = std::str::from_utf8(&ndarray_cache_bytes)
                    .map_err(|_| format!("Internal error"))?;
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
                        CacheElement::new(
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
                rv.push(CacheElement::new(&dirname, "ndarray-cache.json"));
                rv.push(CacheElement::new(
                    format!(
                        "{}--{}--{}",
                        dirname,
                        env!("BUILD_TARGET_TRIPLE"),
                        get_accelerator()
                    ),
                    "rt.dylib",
                ));
                Ok(rv)
            })
        }

        fn try_from_files(
            cache: &Cache,
            files: Vec<(CacheElement, Vec<u8>)>,
        ) -> Result<Self, String> {
            let cache_root = cache.get_root();
            let mut cache_contents = crate::ffi::create_cache();
            let mut lib_path: Option<PathBuf> = None;
            for (elem, data) in files {
                if elem.filename().starts_with("rt.") {
                    lib_path = Some(cache_root.join(elem.dirname()).join(elem.filename()));
                    continue;
                }
                cache_contents
                    .pin_mut()
                    .write_binary(elem.filename().to_owned(), data);
            }

            let lib_path = match lib_path {
                Some(v) => {
                    if !v.exists() {
                        return Err("Runtime not exists".to_owned());
                    };
                    v
                }
                None => return Err("No rt found".to_owned()),
            };
            let device = create_dldevice(8, 0);
            let inner = create_tvm_language_model(
                lib_path.to_string_lossy().to_string(),
                cache_contents,
                device,
            );

            Ok(Inferencer { inner })
        }
    }
}

#[cfg(any(target_family = "wasm"))]
mod tvmjs_runtime {
    #[derive(Debug)]
    pub struct Inferencer {}
}

#[cfg(any(target_family = "unix", target_family = "windows"))]
pub use tvm_runtime::Inferencer;

#[cfg(any(target_family = "wasm"))]
pub use tvmjs_runtime::Inferencer;
