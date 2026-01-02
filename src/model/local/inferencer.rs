use anyhow::Context;
#[cfg(any(target_family = "unix", target_family = "windows"))]
pub use native::EmbeddingModelInferencer;
#[cfg(any(target_family = "unix", target_family = "windows"))]
pub use native::LanguageModelInferencer;
#[cfg(any(target_family = "wasm"))]
pub use wasm::EmbeddingModelInferencer;
#[cfg(any(target_family = "wasm"))]
pub use wasm::LanguageModelInferencer;

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
    Box::pin(async move {
        let mut tensor_cache_json_filename = "tensor-cache.json";
        let tensor_cache_bytes = match cache
            .get(&CacheEntry::new(&dirname, tensor_cache_json_filename), None)
            .await
        {
            Ok(data) => data,
            Err(_) => {
                tensor_cache_json_filename = "ndarray-cache.json";
                match cache
                    .get(&CacheEntry::new(&dirname, tensor_cache_json_filename), None)
                    .await
                {
                    Ok(data) => data,
                    Err(_) => {
                        anyhow::bail!("Cannot find either tensor-cache.json or ndarray-cache.json");
                    }
                }
            }
        };

        let tensor_cache_str =
            std::str::from_utf8(&tensor_cache_bytes).context("Internal error")?;
        let tensor_cache: serde_json::Value =
            serde_json::from_str(tensor_cache_str).context("JSON deserialization failed")?;
        let mut rv = tensor_cache
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
        rv.push(CacheEntry::new(&dirname, tensor_cache_json_filename));
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
mod native {
    use std::path::PathBuf;

    use anyhow::anyhow;
    use tvm_ffi::{
        AnyView, Array, DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, Function, Module,
    };
    use tvm_runtime::{Tensor, TensorCache};

    use super::*;
    use crate::{
        cache::{Cache, CacheContents, TryFromCache},
        model::{KVCache, KVCacheConfig, KVCacheOps},
        utils::BoxFuture,
    };

    pub fn get_device_type(accelerator: &str) -> DLDeviceType {
        if accelerator == "metal" {
            DLDeviceType::kDLMetal
        } else if accelerator == "vulkan" {
            DLDeviceType::kDLVulkan
        } else {
            DLDeviceType::kDLCPU
        }
    }

    #[allow(dead_code)]
    pub struct EmbeddingModelInferencer {
        device: DLDevice,
        vm: Module,
        params: Array<Tensor>,
        fprefill: Function,
    }

    impl std::fmt::Debug for EmbeddingModelInferencer {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("EmbeddingModelInferencer").finish()
        }
    }

    unsafe impl crate::utils::MaybeSend for EmbeddingModelInferencer {}

    impl EmbeddingModelInferencer {
        pub fn new(
            runtime_path: &PathBuf,
            tensor_cache_path: &PathBuf,
            device: DLDevice,
        ) -> anyhow::Result<Self> {
            let exec = Module::load_from_file(runtime_path.to_string_lossy())
                .map_err(|e| anyhow!("Failed to load TVM module: {:?}", e))?;
            let vm: Module = exec
                .get_function("vm_load_executable")
                .map_err(|e| anyhow!("Failed to get `vm_load_executable` function: {:?}", e))?
                .call_tuple(())
                .map_err(|e| anyhow!("Failed to call `vm_load_executable`: {:?}", e))?
                .try_into()
                .map_err(|e| anyhow!("Failed to convert to Module: {:?}", e))?;
            vm.get_function("vm_initialization")
                .map_err(|e| anyhow!("Failed to get `vm_initialization` function: {:?}", e))?
                .call_tuple((
                    device.device_type as i32,            // device_type
                    device.device_id as i32,              // device_id
                    2i32,                                 // vm_allocator_type
                    tvm_ffi::DLDeviceType::kDLCPU as i32, // host_device_type
                    0i32,                                 // host_device_id
                    2i32,                                 // host_vm_allocator_type
                ))
                .map_err(|e| anyhow!("Failed to call `vm_initialization`: {:?}", e))?;

            let metadata: tvm_ffi::String = vm
                .get_function("_metadata")
                .map_err(|e| anyhow!("Failed to get `_metadata` function: {:?}", e))?
                .call_tuple(())
                .map_err(|e| anyhow!("Failed to call `_metadata`: {:?}", e))?
                .try_into()
                .map_err(|e| anyhow!("Failed to convert to String: {:?}", e))?;
            let metadata: serde_json::Value = serde_json::from_str(&metadata)
                .map_err(|e| anyhow!("Failed to parse metadata json: {:?}", e))?;

            let tensor_cache = TensorCache::from(tensor_cache_path, device)
                .map_err(|e| anyhow!("Failed to initialize tensor cache: {:?}", e))?;
            let param_names = metadata
                .get("params")
                .ok_or(anyhow!("Failed to get `params` attribute"))?
                .as_array()
                .ok_or(anyhow!("Failed to convert `params` to array"))?
                .iter()
                .map(|v| {
                    v.get("name")
                        .ok_or(anyhow!("Failed to get `name` attribute"))?
                        .as_str()
                        .ok_or(anyhow!("Failed to convert `name` to str"))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            let params = tensor_cache.get_params(param_names);

            let fprefill = vm
                .get_function("prefill")
                .map_err(|e| anyhow!("Failed to get `prefill` function: {:?}", e))?;

            Ok(Self {
                device,
                vm,
                params,
                fprefill,
            })
        }

        pub fn infer(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
            let dtype_i32 = DLDataType {
                code: DLDataTypeCode::kDLInt as u8,
                bits: 32,
                lanes: 1,
            };

            let mut input = Tensor::empty(&[1, tokens.len() as i64], dtype_i32, self.device);
            let tokens_i32: Vec<i32> = tokens.to_vec().into_iter().map(|v| v as i32).collect();
            // SAFETY: `input` has the same amount of buffer with tokens
            unsafe {
                let tokens_slice = std::slice::from_raw_parts(
                    tokens_i32.as_ptr() as *const u8,
                    tokens.len() * std::mem::size_of::<i32>(),
                );
                input
                    .copy_from_slice(tokens_slice)
                    .map_err(|e| anyhow!("Failed to copy tokens from host to device: {:?}", e))?;
            }

            let mut mask = Tensor::empty(&[1, tokens.len() as i64], dtype_i32, self.device);
            // SAFETY: `mask` has the same amount of buffer with tokens
            unsafe {
                let mask_i32 = std::slice::from_raw_parts(
                    vec![1i32; tokens.len()].as_ptr() as *const u8,
                    tokens.len() * std::mem::size_of::<i32>(),
                );
                mask.copy_from_slice(mask_i32)
                    .map_err(|e| anyhow!("Failed to copy mask from host to device: {:?}", e))?;
            }

            let logits: Tensor = self
                .fprefill
                .call_packed(&[
                    AnyView::from(&input),
                    AnyView::from(&mask),
                    AnyView::from(&self.params),
                ])
                .map_err(|e| anyhow!("Failed to call `prefill`: {:?}", e))?
                .try_into()
                .map_err(|e| anyhow!("Failed to convert to Tensor: {:?}", e))?;

            let mut logits_cpu = Tensor::empty_like(
                &logits,
                DLDevice {
                    device_type: DLDeviceType::kDLCPU,
                    device_id: 0,
                },
            );
            logits_cpu
                .copy_from(&logits)
                .map_err(|e| anyhow!("Failed to copy from device to host: {:?}", e))?;

            // Copy the dense vector only
            let last_dim = logits_cpu
                .shape()
                .last()
                .ok_or(anyhow!("last dim should be exist"))?
                .clone() as usize;
            let dense_vec = if logits_cpu.dtype().bits == 16 {
                // Copy FP16
                let mut buffer_u16: Vec<u16> = vec![0u16; last_dim];
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        logits_cpu.data_ptr() as *const u16,
                        buffer_u16.as_mut_ptr(),
                        last_dim,
                    );
                }
                let buffer_f32: Vec<f32> = buffer_u16
                    .into_iter()
                    .map(|v| crate::utils::float16::f16_to_f32(v))
                    .collect();
                buffer_f32
            } else {
                // Copy FP32
                let mut buffer: Vec<f32> = vec![0f32; last_dim];
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        logits_cpu.data_ptr() as *const f32,
                        buffer.as_mut_ptr(),
                        last_dim,
                    );
                }
                buffer
            };

            Ok(dense_vec)
        }
    }

    impl<'this> TryFromCache<'this> for EmbeddingModelInferencer {
        fn claim_files<'a: 'this>(
            cache: Cache,
            key: impl AsRef<str>,
            _: &'a mut std::collections::HashMap<String, crate::value::Value>,
        ) -> BoxFuture<'a, anyhow::Result<CacheClaim>> {
            claim_files(cache, key)
        }

        fn try_from_contents<'a: 'this>(
            contents: &'a mut CacheContents,
            ctx: &'a std::collections::HashMap<String, crate::value::Value>,
        ) -> BoxFuture<'a, anyhow::Result<Self>> {
            Box::pin(async move {
                let device_id = if let Some(devid) = ctx.get("device_id") {
                    devid.as_integer().unwrap_or(0) as i32
                } else {
                    0i32
                };
                let device = DLDevice {
                    device_type: get_device_type(get_accelerator()),
                    device_id,
                };

                let runtime_filename = format!("rt.{}", get_lib_extension());
                let runtime_path = if let Some((entry, source)) =
                    contents.remove_with_filename(&runtime_filename)
                {
                    // For native, prefer using the lazy path directly to avoid loading into memory
                    if let Some(path) = source.as_path() {
                        path.to_path_buf()
                    } else {
                        // Fallback: if it's eager, use the entry path
                        contents.root.join(entry.path())
                    }
                } else {
                    anyhow::bail!("{} does not exist", runtime_filename)
                };

                let tensor_cache_path = if let Some((entry, source)) =
                    contents.remove_with_filename("tensor-cache.json")
                {
                    if let Some(path) = source.as_path() {
                        path.to_path_buf()
                    } else {
                        contents.root.join(entry.path())
                    }
                } else if let Some((entry, source)) =
                    contents.remove_with_filename("ndarray-cache.json")
                {
                    if let Some(path) = source.as_path() {
                        path.to_path_buf()
                    } else {
                        contents.root.join(entry.path())
                    }
                } else {
                    anyhow::bail!("tensor cache json does not exist")
                };

                // TensorCache::from() already loads params incrementally from disk
                let inferencer =
                    EmbeddingModelInferencer::new(&runtime_path, &tensor_cache_path, device)?;

                Ok(inferencer)
            })
        }
    }

    #[allow(dead_code)]
    pub struct LanguageModelInferencer {
        device: DLDevice,
        vm: Module,
        params: Array<Tensor>,
        kv_cache: KVCache,
        history: Vec<u32>,

        fembed: Function,
        fprefill: Function,
        fdecode: Function,
        fapply_bitmask_inplace: Function,
        fsample_top_p_from_logits: Function,
    }

    impl std::fmt::Debug for LanguageModelInferencer {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("LanguageModelInferencer").finish()
        }
    }

    unsafe impl crate::utils::MaybeSend for LanguageModelInferencer {}

    impl LanguageModelInferencer {
        pub fn new(
            runtime_path: &PathBuf,
            tensor_cache_path: &PathBuf,
            device: DLDevice,
            kv_cache_config: KVCacheConfig,
        ) -> anyhow::Result<Self> {
            let exec = Module::load_from_file(runtime_path.to_string_lossy())
                .map_err(|e| anyhow!("Failed to load TVM module: {:?}", e))?;
            let vm: Module = exec
                .get_function("vm_load_executable")
                .map_err(|e| anyhow!("Failed to get `vm_load_executable` function: {:?}", e))?
                .call_tuple(())
                .map_err(|e| anyhow!("Failed to call `vm_load_executable`: {:?}", e))?
                .try_into()
                .map_err(|e| anyhow!("Failed to convert to Module: {:?}", e))?;
            vm.get_function("vm_initialization")
                .map_err(|e| anyhow!("Failed to get `vm_initialization` function: {:?}", e))?
                .call_tuple((
                    device.device_type as i32,            // device_type
                    device.device_id as i32,              // device_id
                    2i32,                                 // vm_allocator_type
                    tvm_ffi::DLDeviceType::kDLCPU as i32, // host_device_type
                    0i32,                                 // host_device_id
                    2i32,                                 // host_vm_allocator_type
                ))
                .map_err(|e| anyhow!("Failed to call `vm_initialization`: {:?}", e))?;

            let metadata: tvm_ffi::String = vm
                .get_function("_metadata")
                .map_err(|e| anyhow!("Failed to get `_metadata` function: {:?}", e))?
                .call_tuple(())
                .map_err(|e| anyhow!("Failed to call `_metadata`: {:?}", e))?
                .try_into()
                .map_err(|e| anyhow!("Failed to convert to String: {:?}", e))?;
            let metadata: serde_json::Value = serde_json::from_str(&metadata)
                .map_err(|e| anyhow!("Failed to parse metadata json: {:?}", e))?;

            let tensor_cache = TensorCache::from(tensor_cache_path, device)
                .map_err(|e| anyhow!("Failed to initialize tensor cache: {:?}", e))?;
            let param_names = metadata
                .get("params")
                .ok_or(anyhow!("Failed to get `params` attribute"))?
                .as_array()
                .ok_or(anyhow!("Failed to convert `params` to array"))?
                .iter()
                .map(|v| {
                    v.get("name")
                        .ok_or(anyhow!("Failed to get `name` attribute"))?
                        .as_str()
                        .ok_or(anyhow!("Failed to convert `name` to str"))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            let params = tensor_cache.get_params(param_names);

            let kv_cache = KVCache::new(&vm, kv_cache_config)?;

            let fembed = vm
                .get_function("embed")
                .map_err(|e| anyhow!("Failed to get `embed` function: {:?}", e))?;
            let fprefill = vm
                .get_function("prefill")
                .map_err(|e| anyhow!("Failed to get `prefill` function: {:?}", e))?;
            let fdecode = vm
                .get_function("decode")
                .map_err(|e| anyhow!("Failed to get `decode` function: {:?}", e))?;
            let fapply_bitmask_inplace = vm
                .get_function("apply_bitmask_inplace")
                .map_err(|e| anyhow!("Failed to get `apply_bitmask_inplace` function: {:?}", e))?;
            let fsample_top_p_from_logits =
                Function::get_global("vm.builtin.sample_top_p_from_logits").map_err(|e| {
                    anyhow!(
                        "Failed to get global function `vm.builtin.sample_top_p_from_logits`: {:?}",
                        e
                    )
                })?;

            Ok(Self {
                device,
                vm,
                params,
                kv_cache,
                history: Vec::new(),

                fembed,
                fprefill,
                fdecode,
                fapply_bitmask_inplace,
                fsample_top_p_from_logits,
            })
        }

        pub fn embed(&self, tokens: &[i32]) -> anyhow::Result<Tensor> {
            let mut input = Tensor::empty(
                &[tokens.len() as i64],
                DLDataType {
                    code: DLDataTypeCode::kDLInt as u8,
                    bits: 32,
                    lanes: 1,
                },
                self.device,
            );
            // SAFETY: `input` has the same amount of buffer with tokens
            unsafe {
                let tokens_slice = std::slice::from_raw_parts(
                    tokens.as_ptr() as *const u8,
                    tokens.len() * std::mem::size_of::<i32>(),
                );
                input
                    .copy_from_slice(tokens_slice)
                    .map_err(|e| anyhow!("Failed to copy tokens from host to device: {:?}", e))?;
            }

            let embedding: Tensor = self
                .fembed
                .call_packed(&[AnyView::from(&input), AnyView::from(&self.params)])
                .map_err(|e| anyhow!("Failed to call `embed`: {:?}", e))?
                .try_into()
                .map_err(|e| anyhow!("Failed to convert to Tensor: {:?}", e))?;
            let embedding_reshaped = embedding
                .reshape(&[1, embedding.shape()[0], embedding.shape()[1]])
                .map_err(|e| anyhow!("Failed to reshape embedding: {:?}", e))?;

            Ok(embedding_reshaped.into())
        }

        pub fn clear(&mut self) -> anyhow::Result<()> {
            self.kv_cache.clear().map_err(|e| anyhow!("{e:?}"))?;
            self.history.clear();
            Ok(())
        }

        pub fn prefill(&mut self, tokens: &[u32]) -> anyhow::Result<()> {
            if tokens.is_empty() {
                anyhow::bail!("Token must not be empty");
            }

            // Make sure that kv-cache and history is sync
            if self.kv_cache.get_total_sequence_length()? != self.history.len() as i64 {
                self.clear()?;
            }

            // The longest common prefix (LCP) between inputs & previous conversations
            let lcp_index = self
                .history
                .iter()
                .zip(tokens.iter())
                .take_while(|(h, t)| h == t)
                .count();

            // Rewind the head of kv-cache to the LCP
            if lcp_index < self.history.len() {
                self.kv_cache
                    .popn(0, (self.history.len() - lcp_index) as i64)
                    .map_err(|e| anyhow!("{e:?}"))?;
            }

            // Tokens to be added (without common prefixes)
            let new_tokens: Vec<i32> = tokens[lcp_index..].iter().map(|t| *t as i32).collect();

            if new_tokens.is_empty() {
                self.history = tokens.to_vec();
                return Ok(());
            }

            // Calculate remaining space in KV cache
            if new_tokens.len() as i64
                >= self.kv_cache.get_num_available_pages()? * self.kv_cache.page_size
            {
                anyhow::bail!("Context length limit exceed");
            }

            let prefill_chunk_size = self.kv_cache.prefill_chunk_size as usize;
            for i in (0..new_tokens.len()).step_by(prefill_chunk_size) {
                let j = std::cmp::min(i + prefill_chunk_size, new_tokens.len());
                let length = j - i;
                let tokens_sliced = &new_tokens[i..j];
                let embedding = self.embed(tokens_sliced)?;

                self.kv_cache
                    .begin_forward(0, length as i64)
                    .map_err(|e| anyhow!("{e:?}"))?;
                self.fprefill
                    .call_packed(&[
                        AnyView::from(&embedding),
                        AnyView::from(self.kv_cache.get_state()),
                        AnyView::from(&self.params),
                    ])
                    .map_err(|e| anyhow!("{e:?}"))?;
                self.kv_cache.end_forward().map_err(|e| anyhow!("{e:?}"))?;
            }

            // Update history
            self.history = tokens.to_vec();

            Ok(())
        }

        pub fn decode(&mut self, last_token: u32) -> anyhow::Result<Tensor> {
            let embedding = self.embed(&[last_token as i32])?;

            self.kv_cache
                .begin_forward(0, 1)
                .map_err(|e| anyhow!("Failed to begin forward: {:?}", e))?;
            let output = self
                .fdecode
                .call_packed(&[
                    AnyView::from(&embedding),
                    AnyView::from(self.kv_cache.get_state()),
                    AnyView::from(&self.params),
                ])
                .map_err(|e| anyhow!("Failed to call `decode`: {:?}", e))?;
            self.kv_cache
                .end_forward()
                .map_err(|e| anyhow!("Failed to end forward: {:?}", e))?;

            // The output of decode is an Array of 2 items: logits(Tensor) and kv cache.
            let logits = unsafe {
                tvm_ffi::collections::array::get_from_any_array(output, 0)
                    .map_err(|e| anyhow!("Failed to get logits from output array: {:?}", e))?
            };

            Ok(logits)
        }

        pub fn sample(
            &mut self,
            logits: Tensor,
            temperature: f64,
            top_p: f64,
        ) -> anyhow::Result<u32> {
            let uniform_dist_threshold: f64 = crate::utils::get_random_f64();
            let sampled_token: i32 = self
                .fsample_top_p_from_logits
                .call_tuple((logits, &temperature, &top_p, &uniform_dist_threshold))
                .map_err(|e| anyhow!("Failed to call `sample_top_p_from_logits`: {:?}", e))?
                .try_into()
                .map_err(|e| anyhow!("Failed to convert sampled token to i32: {:?}", e))?;
            let sampled_token = sampled_token as u32;
            self.history.push(sampled_token);
            Ok(sampled_token)
        }
    }

    impl<'this> TryFromCache<'this> for LanguageModelInferencer {
        fn claim_files<'a: 'this>(
            cache: Cache,
            key: impl AsRef<str>,
            _: &'a mut std::collections::HashMap<String, crate::value::Value>,
        ) -> BoxFuture<'a, anyhow::Result<CacheClaim>> {
            claim_files(cache, key)
        }

        fn try_from_contents<'a: 'this>(
            contents: &'a mut CacheContents,
            ctx: &'a std::collections::HashMap<String, crate::value::Value>,
        ) -> BoxFuture<'a, anyhow::Result<Self>> {
            Box::pin(async move {
                let device_id = if let Some(devid) = ctx.get("device_id") {
                    devid.as_integer().unwrap_or(0) as i32
                } else {
                    0i32
                };
                let device = DLDevice {
                    device_type: get_device_type(get_accelerator()),
                    device_id,
                };

                let kv_cache_config = if let Some(kv_cache) = ctx.get("kv_cache") {
                    serde_json::from_value(kv_cache.clone().into())
                        .map_err(|e| anyhow!("Failed to parse kv_cache config: {:?}", e))?
                } else {
                    KVCacheConfig::default()
                };

                let runtime_filename = format!("rt.{}", get_lib_extension());
                let runtime_path = if let Some((entry, source)) =
                    contents.remove_with_filename(&runtime_filename)
                {
                    // For native, prefer using the lazy path directly to avoid loading into memory
                    if let Some(path) = source.as_path() {
                        path.to_path_buf()
                    } else {
                        // Fallback: if it's eager, use the entry path
                        contents.root.join(entry.path())
                    }
                } else {
                    anyhow::bail!("{} does not exist", runtime_filename)
                };

                let tensor_cache_path = if let Some((entry, source)) =
                    contents.remove_with_filename("tensor-cache.json")
                {
                    if let Some(path) = source.as_path() {
                        path.to_path_buf()
                    } else {
                        contents.root.join(entry.path())
                    }
                } else if let Some((entry, source)) =
                    contents.remove_with_filename("ndarray-cache.json")
                {
                    if let Some(path) = source.as_path() {
                        path.to_path_buf()
                    } else {
                        contents.root.join(entry.path())
                    }
                } else {
                    anyhow::bail!("tensor cache json does not exist")
                };

                // TensorCache::from() already loads params incrementally from disk
                let inferencer = LanguageModelInferencer::new(
                    &runtime_path,
                    &tensor_cache_path,
                    device,
                    kv_cache_config,
                )?;

                Ok(inferencer)
            })
        }
    }
}

#[cfg(any(target_family = "wasm"))]
mod wasm {
    use std::fmt;

    use anyhow::{Result, anyhow};
    use js_sys::Uint8Array;
    use wasm_bindgen::prelude::*;

    use super::*;
    use crate::{
        cache::{Cache, CacheContents, TryFromCache},
        ffi::web::{
            conversion::u32_slice_to_js,
            tvmjs_bridge::{self as tvmjs},
        },
        model::{KVCache, KVCacheConfig, KVCacheOps},
        utils::BoxFuture,
    };

    async fn instantiate_tvm<'a>(contents: &'a mut CacheContents) -> Result<tvmjs::Instance> {
        let runtime_filename = "rt.wasm";
        let runtime_bytes =
            if let Some((_, source)) = contents.remove_with_filename(&runtime_filename) {
                let bytes = source.read_all().await?;
                Uint8Array::new_from_slice(bytes.as_slice())
            } else {
                anyhow::bail!("{} does not exist", runtime_filename)
            };

        let tvm = tvmjs::instantiate(runtime_bytes.buffer()).await;

        // initialize webgpu
        let gpu = tvmjs::get_gpu_device().await;
        tvm.init_webgpu(gpu);

        Ok(tvm)
    }

    fn initialize_vm<'a>(tvm: &tvmjs::Instance, device: &tvmjs::DLDevice) -> Result<tvmjs::Module> {
        let fload_exec: tvmjs::PackedFunc =
            tvm.system_lib().get_function("vm_load_executable").into();
        let vm: tvmjs::Module = tvm.detach(fload_exec.call0()?);

        let fvm_init: tvmjs::PackedFunc = vm.get_function("vm_initialization").into();
        fvm_init.call6(
            &tvmjs::Scalar::new(device.device_type() as f64, "int"), // webgpu device type
            &tvmjs::Scalar::new(device.device_id() as f64, "int"),   // webgpu device id
            &tvmjs::Scalar::new(2., "int"),                          // pooled allocator
            &tvmjs::Scalar::new(1., "int"),                          // host device type (cpu: 1)
            &tvmjs::Scalar::new(0., "int"),                          // host device id
            &tvmjs::Scalar::new(2., "int"),                          // pooled allocator
        )?;

        Ok(vm)
    }

    fn get_metadata(vm: &tvmjs::Module) -> Result<serde_json::Value> {
        let fmetadata: tvmjs::PackedFunc = vm.get_function("_metadata").into();
        let metadata_str = fmetadata
            .call0()?
            .as_string()
            .ok_or(anyhow!("_metadata result should be string"))?;
        let metadata: serde_json::Value = serde_json::from_str(metadata_str.as_str())
            .with_context(|| "Failed to parse metadata")?;
        Ok(metadata)
    }

    async fn initialize_params<'a>(
        tvm: &tvmjs::Instance,
        device: &tvmjs::DLDevice,
        metadata: &serde_json::Value,
        contents: &'a mut CacheContents,
    ) -> Result<tvmjs::TVMObject> {
        let tensor_cache_bytes =
            if let Some((_, source)) = contents.remove_with_filename("tensor-cache.json") {
                source.read_all().await?
            } else if let Some((_, source)) = contents.remove_with_filename("ndarray-cache.json") {
                source.read_all().await?
            } else {
                anyhow::bail!("tensor cache json does not exist")
            };

        let tensor_cache: tvmjs::TensorCache =
            serde_json::from_slice(tensor_cache_bytes.as_slice())?;

        for shard_entry in tensor_cache.records {
            let buffer =
                if let Some((_, source)) = contents.remove_with_filename(&shard_entry.data_path) {
                    source.read_all().await?
                } else {
                    anyhow::bail!(
                        "Tensor Cache shard {} does not exist",
                        shard_entry.data_path
                    );
                };

            for param_record in shard_entry.records {
                let buffer_part = &buffer
                    [param_record.byte_offset..(param_record.byte_offset + param_record.nbytes)];

                tvm.tensor_cache_update_buffer(
                    device.clone(),
                    param_record,
                    Uint8Array::new_from_slice(buffer_part).buffer(),
                )
                .await;
            }
        }
        let param_names = metadata
            .get("params")
            .ok_or(anyhow!("Failed to get `params` attribute"))?
            .as_array()
            .ok_or(anyhow!("Failed to convert `params` to array"))?
            .iter()
            .map(|v| {
                v.get("name")
                    .ok_or(anyhow!("Failed to get `name` attribute"))?
                    .as_str()
                    .ok_or(anyhow!("Failed to convert `name` to str"))
                    .map(|s| s.to_string())
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let params: tvmjs::TVMObject = tvm.detach(tvm.get_params_from_cache_by_name(param_names));

        Ok(params)
    }

    pub struct LanguageModelInferencer {
        tvm: tvmjs::Instance,
        device: tvmjs::DLDevice,
        kv_cache: KVCache,
        params: tvmjs::TVMObject,
        history: Vec<u32>,

        fembed: tvmjs::PackedFunc,
        fprefill: tvmjs::PackedFunc,
        fdecode: tvmjs::PackedFunc,
        fsample_top_p_from_logits: tvmjs::PackedFunc,
    }

    impl fmt::Debug for LanguageModelInferencer {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("LanguageModelInferencer").finish()
        }
    }

    impl LanguageModelInferencer {
        fn clear(&mut self) -> anyhow::Result<()> {
            self.kv_cache.clear()?;
            self.history.clear();
            Ok(())
        }

        async fn embed(&self, tokens: &[i32]) -> anyhow::Result<tvmjs::Tensor> {
            let input = self.tvm.empty(
                u32_slice_to_js(&[tokens.len() as u32]),
                "int32",
                self.device.clone().into(),
            );
            input.copy_from_i32array(tokens);
            self.device.sync().await;

            let embedding: tvmjs::Tensor = self.fembed.call2(&input, &self.params).unwrap().into();
            let embedding_reshaped = embedding.view(
                u32_slice_to_js(&[1, embedding.shape()[0], embedding.shape()[1]]),
                None,
                None,
            );

            Ok(embedding_reshaped)
        }

        pub async fn prefill(&mut self, tokens: &[u32]) -> anyhow::Result<()> {
            if tokens.is_empty() {
                anyhow::bail!("Token must not be empty");
            }

            self.tvm.begin_scope();

            // Make sure that kv-cache and history is sync
            if self.kv_cache.get_total_sequence_length()? != self.history.len() as i64 {
                self.clear()?;
            }

            // The longest common prefix (LCP) between inputs & previous conversations
            let lcp_index = self
                .history
                .iter()
                .zip(tokens.iter())
                .take_while(|(h, t)| h == t)
                .count();

            // Rewind the head of kv-cache to the LCP
            if lcp_index < self.history.len() {
                self.kv_cache
                    .popn(0, (self.history.len() - lcp_index) as i64)
                    .map_err(|e| anyhow!("{e:?}"))?;
            }

            // Tokens to be added (without common prefixes)
            let new_tokens: Vec<i32> = tokens[lcp_index..].iter().map(|t| *t as i32).collect();

            if new_tokens.is_empty() {
                self.history = tokens.to_vec();
                return Ok(());
            }

            // Calculate remaining space in KV cache
            if new_tokens.len() as i64
                >= self.kv_cache.get_num_available_pages()? * self.kv_cache.page_size
            {
                anyhow::bail!("Context length limit exceed");
            }

            let prefill_chunk_size = self.kv_cache.prefill_chunk_size as usize;
            for i in (0..new_tokens.len()).step_by(prefill_chunk_size) {
                let j = std::cmp::min(i + prefill_chunk_size, new_tokens.len());
                let length = j - i;
                let tokens_sliced = &new_tokens[i..j];
                let embedding = self.embed(tokens_sliced).await?;

                self.kv_cache
                    .begin_forward(0, length as i64)
                    .map_err(|e| anyhow!("{e:?}"))?;
                self.fprefill
                    .call3(&embedding, self.kv_cache.get_state(), &self.params)
                    .map_err(|e| anyhow!("{e:?}"))?;
                self.kv_cache.end_forward().map_err(|e| anyhow!("{e:?}"))?;
            }

            // Update history
            self.history = tokens.to_vec();

            self.tvm.end_scope();

            Ok(())
        }

        pub async fn decode(
            &mut self,
            last_token: u32,
            temperature: f64,
            top_p: f64,
        ) -> anyhow::Result<u32> {
            self.tvm.begin_scope();

            let embedding = self.embed(&[last_token as i32]).await?;

            self.kv_cache
                .begin_forward(0, 1)
                .map_err(|e| anyhow!("Failed to begin forward: {:?}", e))?;
            let output: tvmjs::TVMArray = self
                .fdecode
                .call3(&embedding, self.kv_cache.get_state(), &self.params)
                .map_err(|e| anyhow!("Failed to call `decode`: {:?}", e))?
                .into();
            self.kv_cache
                .end_forward()
                .map_err(|e| anyhow!("Failed to end forward: {:?}", e))?;

            // The output of decode is an Array of 2 items: logits(Tensor) and kv cache.
            let logits: tvmjs::Tensor = self.tvm.detach(output.get(0));
            let logits_cpu: tvmjs::Tensor = self.tvm.detach(self.tvm.empty(
                u32_slice_to_js(logits.shape().as_slice()),
                &logits.dtype(),
                self.tvm.cpu(),
            ));
            logits_cpu.copy_from_tensor(&logits);
            self.device.sync().await;
            logits.dispose();

            let sampled_token = self
                .fsample_top_p_from_logits
                .call4(
                    &logits_cpu,
                    &JsValue::from_f64(temperature),
                    &JsValue::from_f64(top_p),
                    &JsValue::from_f64(crate::utils::get_random_f64()),
                )
                .unwrap()
                .as_f64()
                .unwrap();
            logits_cpu.dispose();

            self.tvm.end_scope();

            Ok(sampled_token as u32)
        }
    }

    impl<'this> TryFromCache<'this> for LanguageModelInferencer {
        fn claim_files<'a: 'this>(
            cache: Cache,
            key: impl AsRef<str>,
            _: &'a mut std::collections::HashMap<String, crate::value::Value>,
        ) -> BoxFuture<'a, anyhow::Result<CacheClaim>> {
            claim_files(cache, key)
        }

        fn try_from_contents<'a: 'this>(
            contents: &'a mut CacheContents,
            ctx: &'a std::collections::HashMap<String, crate::value::Value>,
        ) -> BoxFuture<'a, anyhow::Result<Self>> {
            Box::pin(async move {
                let tvm = instantiate_tvm(contents).await?;
                tvm.begin_scope();

                let device = tvm.webgpu(0);
                let vm = initialize_vm(&tvm, &device)?;
                let metadata = get_metadata(&vm)?;
                let params = initialize_params(&tvm, &device, &metadata, contents).await?;

                let fembed: tvmjs::PackedFunc = tvm.detach(vm.get_function("embed"));
                let fprefill: tvmjs::PackedFunc = tvm.detach(vm.get_function("prefill"));
                let fdecode: tvmjs::PackedFunc = tvm.detach(vm.get_function("decode"));
                let fsample_top_p_from_logits: tvmjs::PackedFunc =
                    tvm.detach(tvm.get_global_func("vm.builtin.sample_top_p_from_logits"));

                let kv_cache_config = if let Some(kv_cache) = ctx.get("kv_cache") {
                    serde_json::from_value::<KVCacheConfig>(kv_cache.clone().into())
                        .unwrap_or_default()
                } else {
                    KVCacheConfig::default()
                };
                let kv_cache =
                    KVCache::new(tvm.clone().into(), &vm, &metadata, kv_cache_config).unwrap();

                tvm.end_scope();

                Ok(LanguageModelInferencer {
                    tvm,
                    device,
                    kv_cache,
                    params,
                    history: Vec::new(),
                    fembed,
                    fprefill,
                    fdecode,
                    fsample_top_p_from_logits,
                })
            })
        }
    }

    pub struct EmbeddingModelInferencer {
        tvm: tvmjs::Instance,
        device: tvmjs::DLDevice,
        params: tvmjs::TVMObject,
        fprefill: tvmjs::PackedFunc,
    }

    impl fmt::Debug for EmbeddingModelInferencer {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("EmbeddingModelInferencer").finish()
        }
    }

    impl EmbeddingModelInferencer {
        pub async fn infer(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
            self.tvm.begin_scope();

            let input: tvmjs::Tensor = self.tvm.detach(self.tvm.empty(
                u32_slice_to_js(&[1, tokens.len() as u32]),
                "int32",
                self.device.clone().into(),
            ));
            let tokens_i32: Vec<i32> = tokens.to_vec().into_iter().map(|v| v as i32).collect();
            input.copy_from_i32array(tokens_i32.as_slice());
            self.device.sync().await;

            let mask: tvmjs::Tensor = self.tvm.detach(self.tvm.empty(
                u32_slice_to_js(&[1, tokens.len() as u32]),
                "int32",
                self.device.clone().into(),
            ));
            let mask_i32 = vec![1i32; tokens.len()];
            mask.copy_from_i32array(mask_i32.as_slice());
            self.device.sync().await;

            let logits: tvmjs::Tensor = self
                .fprefill
                .call3(&input, &mask, &self.params)
                .map_err(|e| anyhow!("Failed to call `prefill`: {:?}", e))?
                .into();
            input.dispose();
            mask.dispose();

            let logits_cpu = self.tvm.empty(
                u32_slice_to_js(logits.shape().as_slice()),
                &logits.dtype(),
                self.tvm.cpu(),
            );
            logits_cpu.copy_from_tensor(&logits);
            self.device.sync().await;
            logits.dispose();

            let logits_shape = logits_cpu.shape();
            let hidden_size = logits_shape
                .last()
                .ok_or(anyhow!("last dim should be exist"))?
                .clone();
            let mut dense_shape = vec![1; logits_shape.len()];
            dense_shape[logits_shape.len() - 1] = hidden_size;

            // Copy the dense vector only
            let logits_cpu = logits_cpu.view(u32_slice_to_js(&dense_shape), None, Some(0));
            let dense_vec = if logits_cpu.dtype() == "float16" {
                // Copy FP16
                let buffer_u16: Vec<u16> = logits_cpu.to_u16array();
                let buffer_f32: Vec<f32> = buffer_u16
                    .into_iter()
                    .map(|v| crate::utils::float16::f16_to_f32(v))
                    .collect();
                buffer_f32
            } else {
                // Copy FP32
                let buffer: Vec<f32> = logits_cpu.to_f32array();
                buffer
            };
            logits_cpu.dispose();

            self.tvm.end_scope();

            Ok(dense_vec)
        }
    }

    impl<'this> TryFromCache<'this> for EmbeddingModelInferencer {
        fn claim_files<'a: 'this>(
            cache: Cache,
            key: impl AsRef<str>,
            _: &'a mut std::collections::HashMap<String, crate::value::Value>,
        ) -> BoxFuture<'a, anyhow::Result<CacheClaim>> {
            claim_files(cache, key)
        }

        fn try_from_contents<'a: 'this>(
            contents: &'a mut CacheContents,
            _: &'a std::collections::HashMap<String, crate::value::Value>,
        ) -> BoxFuture<'a, anyhow::Result<Self>> {
            Box::pin(async move {
                let tvm = instantiate_tvm(contents).await?;
                tvm.begin_scope();

                let device = tvm.webgpu(0);
                let vm = initialize_vm(&tvm, &device)?;
                let metadata = get_metadata(&vm)?;
                let params = initialize_params(&tvm, &device, &metadata, contents).await?;
                let fprefill: tvmjs::PackedFunc = tvm.detach(vm.get_function("prefill"));

                tvm.end_scope();

                Ok(EmbeddingModelInferencer {
                    tvm,
                    device,
                    params,
                    fprefill,
                })
            })
        }
    }
}
