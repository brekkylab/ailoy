#[cfg(any(target_family = "unix", target_family = "windows"))]
mod tvm_runtime {
    use std::{path::PathBuf, pin::Pin};

    use cxx::UniquePtr;

    use crate::{
        cache::{Cache, CacheEntry, TryFromCache},
        ffi::{TVMLanguageModel, create_dldevice, create_tvm_language_model},
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
        ) -> Pin<Box<dyn Future<Output = Result<Vec<CacheEntry>, String>>>> {
            let dirname = vec![key.as_ref().replace("/", "--")].join("--");
            let elem = CacheEntry::new(&dirname, "ndarray-cache.json");
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

        fn try_from_files(
            cache: &Cache,
            files: Vec<(CacheEntry, Vec<u8>)>,
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
            let device = create_dldevice(
                get_device_type(get_accelerator()),
                get_device_id(get_accelerator()),
            );
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

    impl Inferencer {
        pub fn prefill(&mut self, _: &Vec<u32>) -> () {
            todo!()
        }

        pub fn decode(&mut self, _: u32) -> u32 {
            todo!()
        }
    }
}

#[cfg(any(target_family = "unix", target_family = "windows"))]
pub use tvm_runtime::Inferencer;

#[cfg(any(target_family = "wasm"))]
pub use tvmjs_runtime::Inferencer;
