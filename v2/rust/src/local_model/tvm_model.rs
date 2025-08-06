use std::{path::PathBuf, pin::Pin, str::FromStr};

use crate::cache::{Cache, TryFromCache, get_cache_root};

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
        target_arch = "wasm32"
    )))]
    {
        "unknown"
    }
}

#[derive(Debug)]
pub struct TVMModel {
    inner: *mut ffi::TvmModel,
}

impl TVMModel {
    pub fn new(inner: *mut ffi::TvmModel) -> Self {
        TVMModel { inner }
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

impl Drop for TVMModel {
    fn drop(&mut self) {
        unsafe { ffi::ailoy_tvm_model_destroy(self.inner) };
    }
}

impl TryFromCache for TVMModel {
    fn claim_files(
        cache: Cache,
        key: impl AsRef<str>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<PathBuf>, String>>>> {
        let dir = vec![key.as_ref().replace("/", "--")].join("--");
        Box::pin(async move {
            let ndarray_cache = std::str::from_utf8(&cache.get(&dir, "ndarray-cache.json").await?)
                .map_err(|_| format!("Internal error"))?
                .to_owned();
            let ndarray_cache: serde_json::Value = serde_json::from_str(&ndarray_cache)
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
                    PathBuf::from_str(&dir).unwrap().join(
                        v.as_object()
                            .unwrap()
                            .get("dataPath")
                            .unwrap()
                            .as_str()
                            .unwrap()
                            .to_owned(),
                    )
                })
                .collect::<Vec<_>>();
            rv.push(
                PathBuf::from_str(&dir)
                    .unwrap()
                    .join("ndarray-cache.json".to_owned()),
            );
            rv.push(
                PathBuf::from_str(&format!(
                    "{}--{}--{}",
                    &dir,
                    env!("BUILD_TARGET_TRIPLE"),
                    get_accelerator()
                ))
                .unwrap()
                .join("lib.dylib".to_owned()),
            );

            Ok(rv)
        })
    }

    fn try_from_files(files: Vec<(PathBuf, Vec<u8>)>) -> Result<Self, String> {
        let mut contents: *mut ffi::FileContents = std::ptr::null_mut();

        if unsafe { ffi::ailoy_file_contents_create(&mut contents) } != 0 {
            return Err("ailoy_file_contents_create failed".to_owned());
        }

        let mut lib_filename: Option<String> = None;
        for (path, data) in files {
            let path_str = path.as_os_str().to_string_lossy();
            if path_str.ends_with("lib.dylib")
                || path_str.ends_with("lib.so")
                || path_str.ends_with("lib.dll")
                || path_str.ends_with("lib.wasm")
            {
                lib_filename = Some(path.to_str().unwrap().to_owned());
                continue;
            }
            let ret = unsafe {
                ffi::ailoy_file_contents_insert(
                    contents,
                    path_str.as_ptr() as *const _,
                    data.len(),
                    data.as_ptr() as *const _,
                )
            };
            if ret != 0 {
                unsafe { ffi::ailoy_file_contents_destroy(contents) };
                return Err(format!("Failed to insert file: {}", path.to_string_lossy()));
            }
        }
        let mut tvm_model: *mut ffi::TvmModel = std::ptr::null_mut();
        let lib_full_path = get_cache_root().join(lib_filename.unwrap().to_owned());
        let ret = unsafe {
            ffi::ailoy_tvm_model_create(
                lib_full_path.as_os_str().to_string_lossy().as_ptr() as *const _,
                contents,
                &mut tvm_model,
            )
        };
        if ret != 0 {
            unsafe { ffi::ailoy_file_contents_destroy(contents) };
            return Err(format!("ailoy_tvm_model_create failed"));
        }
        unsafe { ffi::ailoy_file_contents_destroy(contents) };
        Ok(TVMModel { inner: tvm_model })
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
            TVMModel::claim_files(cache, "Qwen/Qwen3-0.6B")
                .await
                .unwrap()
        );
    }
}
