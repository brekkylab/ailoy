#[cfg(any(target_family = "unix", target_family = "windows"))]
mod tvm_runtime {
    use std::pin::Pin;

    use crate::cache::{Cache, CacheElement, TryFromCache};

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
        inner: *mut ffi::TvmModel,
    }

    impl Inferencer {
        pub fn new(lib_filename: &str) -> Self {
            // let mut inner: *mut ffi::TvmModel = std::ptr::null_mut();
            // let lib_full_path = get_cache_root().join(lib_filename);
            todo!()
            // let ret = unsafe {
            //     ffi::ailoy_tvm_model_create(
            //         lib_full_path.as_os_str().to_string_lossy().as_ptr() as *const _,
            //         contents,
            //         &mut tvm_model,
            //     )
            // };
            // if ret != 0 {
            //     unsafe { ffi::ailoy_file_contents_destroy(contents) };
            //     return Err(format!("ailoy_tvm_model_create failed"));
            // }

            // Inferencer { inner }
        }

        pub fn embed(&self, input: impl AsRef<[u8]>) -> () {
            todo!()
        }

        pub fn prefill(&self, input: impl AsRef<[u8]>) -> u64 {
            todo!()
        }

        pub fn decode(&self, input: impl AsRef<[u8]>) -> u64 {
            todo!()
        }
    }

    impl Drop for Inferencer {
        fn drop(&mut self) {
            unsafe { ffi::ailoy_tvm_model_destroy(self.inner) };
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
                    "lib",
                ));
                todo!()

                // Ok(rv)
            })
        }

        fn try_from_files(files: Vec<(CacheElement, Vec<u8>)>) -> Result<Self, String> {
            todo!()
            // let mut contents: *mut ffi::FileContents = std::ptr::null_mut();

            // if unsafe { ffi::ailoy_file_contents_create(&mut contents) } != 0 {
            //     return Err("ailoy_file_contents_create failed".to_owned());
            // }

            // let mut lib_filename: Option<String> = None;
            // for (path, data) in files {
            //     let path_str = path.as_os_str().to_string_lossy();
            //     if path_str.ends_with("lib.dylib")
            //         || path_str.ends_with("lib.so")
            //         || path_str.ends_with("lib.dll")
            //         || path_str.ends_with("lib.wasm")
            //     {
            //         lib_filename = Some(path.to_str().unwrap().to_owned());
            //         continue;
            //     }
            //     let ret = unsafe {
            //         ffi::ailoy_file_contents_insert(
            //             contents,
            //             path_str.as_ptr() as *const _,
            //             data.len(),
            //             data.as_ptr() as *const _,
            //         )
            //     };
            //     if ret != 0 {
            //         unsafe { ffi::ailoy_file_contents_destroy(contents) };
            //         return Err(format!("Failed to insert file: {}", path.to_string_lossy()));
            //     }
            // }

            // unsafe { ffi::ailoy_file_contents_destroy(contents) };
            // Ok(Inferencer::new())
        }
    }

    mod ffi {
        #[repr(C)]
        pub struct FileContents;

        #[repr(C)]
        pub struct TvmModel;

        unsafe extern "C" {
            pub fn ailoy_file_contents_create(out: *mut *mut FileContents) -> i32;

            pub fn ailoy_file_contents_destroy(contents: *mut FileContents) -> i32;

            pub fn ailoy_file_contents_insert(
                contents: *mut FileContents,
                filename: *const std::os::raw::c_char,
                len: usize,
                content: *const std::os::raw::c_char,
            ) -> i32;

            pub fn ailoy_tvm_model_create(
                lib_filename: *const std::os::raw::c_char,
                contents: *const FileContents,
                out: *mut *mut TvmModel,
            ) -> i32;

            pub fn ailoy_tvm_model_destroy(model: *mut TvmModel) -> i32;
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(test)]
    mod tests {

        use super::*;

        #[tokio::test]
        async fn test_tvm_model() {
            let cache = crate::cache::Cache::new();
            println!(
                "{:?}",
                Inferencer::claim_files(cache, "Qwen/Qwen3-0.6B")
                    .await
                    .unwrap()
            );
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
