#[cfg(any(target_family = "unix", target_family = "windows"))]
mod tvm_runtime {
    use std::{path::PathBuf, pin::Pin};

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
        inner: *mut ffi::TVMRuntime,
    }

    impl Inferencer {
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
            // unsafe { ffi::ailoy_tvm_runtime_destroy(self.inner) };
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
            // let v = dlpackrs::ManagedTensor::new(tensor, manager_ctx)
            // let cache_root = cache.get_root();
            // let mut lib_path: Option<PathBuf> = None;
            // let mut files_c: *mut ffi::FileContents = std::ptr::null_mut();
            // unsafe { ffi::ailoy_file_contents_create(&mut files_c) };
            // for (elem, data) in files {
            //     if elem.filename().starts_with("rt.") {
            //         lib_path = Some(cache_root.join(elem.dirname()).join(elem.filename()));
            //         continue;
            //     }
            //     unsafe {
            //         ffi::ailoy_file_contents_insert(
            //             files_c,
            //             elem.filename().as_ptr() as *const _,
            //             data.len(),
            //             data.as_ptr() as *const _,
            //         );
            //     }
            // }

            // let lib_path = match lib_path {
            //     Some(v) => {
            //         if !v.exists() {
            //             return Err("Runtime not exists".to_owned());
            //         };
            //         v
            //     }
            //     None => return Err("No rt found".to_owned()),
            // };
            // let mut inferencer_c: *mut ffi::TVMRuntime = std::ptr::null_mut();
            // unsafe {
            //     let ret = ffi::ailoy_tvm_runtime_create(
            //         lib_path.as_os_str().to_string_lossy().as_ptr() as *const _,
            //         files_c,
            //         &mut inferencer_c,
            //     );
            //     if ret != 0 {
            //         return Err(format!("ailoy_tvm_runtime_create failed: {}", ret));
            //     }
            // }

            // unsafe { ffi::ailoy_file_contents_destroy(files_c) };

            // Ok(Inferencer {
            //     inner: inferencer_c,
            // })
            todo!()
        }
    }

    mod ffi {
        #[repr(C)]
        pub struct FileContents {
            _private: (),
        }

        #[repr(C)]
        pub struct TVMRuntime {
            _private: (),
        }

        unsafe extern "C" {
            pub fn ailoy_tvm_language_model_prefill(
                len: usize,
                v: *const dlpackrs::ffi::DLManagedTensor,
            );
            // pub fn ailoy_file_contents_create(out: *mut *mut FileContents) -> i32;

            // pub fn ailoy_file_contents_destroy(contents: *mut FileContents) -> i32;

            // pub fn ailoy_file_contents_insert(
            //     contents: *mut FileContents,
            //     filename: *const std::os::raw::c_char,
            //     len: usize,
            //     content: *const std::os::raw::c_char,
            // ) -> i32;

            // pub fn ailoy_tvm_runtime_create(
            //     lib_path: *const std::os::raw::c_char,
            //     contents: *const FileContents,
            //     out: *mut *mut TVMRuntime,
            // ) -> i32;

            // pub fn ailoy_tvm_runtime_destroy(model: *mut TVMRuntime) -> i32;
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
