use anyhow::Context;
#[cfg(any(target_family = "unix", target_family = "windows"))]
pub use tvm::EmbeddingModelInferencer;
#[cfg(any(target_family = "unix", target_family = "windows"))]
pub use tvm::LanguageModelInferencer;
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
mod tvm {
    use std::path::PathBuf;

    use tvm_ffi::{
        AnyView, Array, DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, Function, Module,
        Shape, Tensor as TVMFFITensor,
    };
    use tvm_runtime::{Tensor, TensorCache};

    use super::*;
    use crate::{
        cache::{Cache, CacheContents, TryFromCache},
        model::KvCacheConfig,
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
        params: Array<TVMFFITensor>,
        fprefill: Function,
    }

    impl std::fmt::Debug for EmbeddingModelInferencer {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("EmbeddingModelInferencer").finish()
        }
    }

    unsafe impl crate::utils::MaybeSend for EmbeddingModelInferencer {}

    impl EmbeddingModelInferencer {
        pub fn new(runtime_path: &PathBuf, tensor_cache_path: &PathBuf, device: DLDevice) -> Self {
            let exec = Module::load_from_file(runtime_path.to_string_lossy()).unwrap();
            let vm: Module = exec
                .get_function("vm_load_executable")
                .unwrap()
                .call_tuple(())
                .unwrap()
                .try_into()
                .unwrap();
            vm.get_function("vm_initialization")
                .unwrap()
                .call_tuple((
                    device.device_type as i32,            // device_type
                    device.device_id as i32,              // device_id
                    2i32,                                 // vm_allocator_type
                    tvm_ffi::DLDeviceType::kDLCPU as i32, // host_device_type
                    0i32,                                 // host_device_id
                    2i32,                                 // host_vm_allocator_type
                ))
                .unwrap();

            let metadata: tvm_ffi::String = vm
                .get_function("_metadata")
                .unwrap()
                .call_tuple(())
                .unwrap()
                .try_into()
                .unwrap();
            let metadata: serde_json::Value = serde_json::from_str(&metadata).unwrap();

            let tensor_cache = TensorCache::from(tensor_cache_path, device).unwrap();
            let param_names = metadata
                .get("params")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.get("name").unwrap().as_str().unwrap())
                .collect::<Vec<_>>();
            let params = tensor_cache.get_params(param_names);

            let fprefill = vm.get_function("prefill").unwrap();

            Self {
                device,
                vm,
                params,
                fprefill,
            }
        }

        pub fn infer(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
            let dtype_i32 = DLDataType {
                code: DLDataTypeCode::kDLInt as u8,
                bits: 32,
                lanes: 1,
            };
            let mut input = Tensor::empty(&[1, tokens.len() as i64], dtype_i32, self.device);
            let mut mask = Tensor::empty(&[1, tokens.len() as i64], dtype_i32, self.device);
            unsafe {
                let tokens_i32: Vec<i32> = tokens.to_vec().into_iter().map(|v| v as i32).collect();
                let tokens_slice = std::slice::from_raw_parts(
                    tokens_i32.as_ptr() as *const u8,
                    tokens.len() * std::mem::size_of::<i32>(),
                );
                input.copy_from_slice(tokens_slice).unwrap();

                let mask_i32 = std::slice::from_raw_parts(
                    vec![1i32; tokens.len()].as_ptr() as *const u8,
                    tokens.len() * std::mem::size_of::<i32>(),
                );
                mask.copy_from_slice(mask_i32).unwrap();
            }

            let logits: tvm_ffi::Tensor = self
                .fprefill
                .call_packed(&[
                    tvm_ffi::AnyView::from(&<tvm_ffi::Tensor as From<Tensor>>::from(input)),
                    tvm_ffi::AnyView::from(&<tvm_ffi::Tensor as From<Tensor>>::from(mask)),
                    tvm_ffi::AnyView::from(&self.params),
                ])
                .unwrap()
                .try_into()
                .unwrap();
            let logits: Tensor = logits.into();

            let mut logits_cpu = Tensor::empty_like(
                &logits,
                DLDevice {
                    device_type: DLDeviceType::kDLCPU,
                    device_id: 0,
                },
            );
            logits_cpu.copy_from(&logits).unwrap();

            // Copy the dense vector only
            let last_dim = logits_cpu.shape().last().unwrap().clone() as usize;
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

    impl TryFromCache for EmbeddingModelInferencer {
        fn claim_files<'a>(
            cache: Cache,
            key: impl AsRef<str>,
            _: &'a mut std::collections::HashMap<String, crate::value::Value>,
        ) -> BoxFuture<'a, anyhow::Result<CacheClaim>> {
            claim_files(cache, key)
        }

        fn try_from_contents<'a>(
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
                let runtime_path =
                    if let Some((entry, _)) = contents.remove_with_filename(&runtime_filename) {
                        entry.path()
                    } else {
                        anyhow::bail!("{} does not exist", runtime_filename)
                    };

                let tensor_cache_path = if let Some((entry, _)) =
                    contents.remove_with_filename("tensor-cache.json")
                {
                    entry.path()
                } else if let Some((entry, _)) = contents.remove_with_filename("ndarray-cache.json")
                {
                    entry.path()
                } else {
                    anyhow::bail!("tensor cache json does not exist")
                };

                let inferencer = EmbeddingModelInferencer::new(
                    &contents.root.join(runtime_path),
                    &contents.root.join(tensor_cache_path),
                    device,
                );

                Ok(inferencer)
            })
        }
    }

    #[allow(dead_code)]
    pub struct KVCache {
        state: tvm_ffi::Any,

        pub context_window_size: i64,
        pub prefill_chunk_size: i64,
        pub sliding_window_size: i64,
        pub page_size: i64,

        pub fkv_state_clear: Function,
        pub fkv_state_add_sequence: Function,
        pub fkv_state_remove_sequence: Function,
        pub fkv_state_fork_sequence: Function,
        pub fkv_state_begin_forward: Function,
        pub fkv_state_end_forward: Function,
        pub fkv_state_popn: Function,
        pub fkv_cache_get_num_available_pages: Function,
        pub fkv_cache_get_total_sequence_length: Function,
    }

    impl KVCache {
        pub fn new(
            vm: &Module,
            context_window_size: Option<i64>,
            prefill_chunk_size: Option<i64>,
            sliding_window_size: Option<i64>,
        ) -> Self {
            let metadata_str: tvm_ffi::String = vm
                .get_function("_metadata")
                .unwrap()
                .call_tuple(())
                .unwrap()
                .try_into()
                .unwrap();
            let metadata: serde_json::Value = serde_json::from_str(&metadata_str).unwrap();

            let context_window_size = context_window_size.unwrap_or(
                metadata
                    .get("context_window_size")
                    .unwrap()
                    .as_i64()
                    .unwrap(),
            );
            let prefill_chunk_size = prefill_chunk_size.unwrap_or(
                metadata
                    .get("prefill_chunk_size")
                    .unwrap()
                    .as_i64()
                    .unwrap(),
            );
            let sliding_window_size = sliding_window_size.unwrap_or(
                metadata
                    .get("sliding_window_size")
                    .unwrap()
                    .as_i64()
                    .unwrap(),
            );

            const PAGE_SIZE: i64 = 16;
            let state = vm
                .get_function("create_tir_paged_kv_cache")
                .unwrap()
                .call_tuple((
                    Shape::from([1]),                                  // max_batch_size
                    Shape::from([context_window_size]),                // max_total_seq_len
                    Shape::from([prefill_chunk_size]),                 // prefill_chunk_size
                    Shape::from([PAGE_SIZE]),                          // page_size
                    Shape::from([(sliding_window_size != -1) as i64]), // support_sliding_window
                ))
                .unwrap();

            let fkv_state_clear = Function::get_global("vm.builtin.kv_state_clear").unwrap();
            let fkv_state_add_sequence =
                Function::get_global("vm.builtin.kv_state_add_sequence").unwrap();
            let fkv_state_remove_sequence =
                Function::get_global("vm.builtin.kv_state_remove_sequence").unwrap();
            let fkv_state_fork_sequence =
                Function::get_global("vm.builtin.kv_state_fork_sequence").unwrap();
            let fkv_state_begin_forward =
                Function::get_global("vm.builtin.kv_state_begin_forward").unwrap();
            let fkv_state_end_forward =
                Function::get_global("vm.builtin.kv_state_end_forward").unwrap();
            let fkv_state_popn = Function::get_global("vm.builtin.kv_state_popn").unwrap();
            let fkv_cache_get_num_available_pages =
                Function::get_global("vm.builtin.attention_kv_cache_get_num_available_pages")
                    .unwrap();
            let fkv_cache_get_total_sequence_length =
                Function::get_global("vm.builtin.attention_kv_cache_get_total_sequence_length")
                    .unwrap();

            let mut self_ = Self {
                state,
                context_window_size,
                prefill_chunk_size,
                sliding_window_size,
                page_size: PAGE_SIZE,
                fkv_state_clear,
                fkv_state_add_sequence,
                fkv_state_remove_sequence,
                fkv_state_fork_sequence,
                fkv_state_begin_forward,
                fkv_state_end_forward,
                fkv_state_popn,
                fkv_cache_get_num_available_pages,
                fkv_cache_get_total_sequence_length,
            };
            self_.clear().unwrap();
            self_
        }

        pub fn get_state(&self) -> &tvm_ffi::Any {
            &self.state
        }

        pub fn add_sequence(&mut self) -> tvm_ffi::Result<()> {
            self.fkv_state_add_sequence
                .call_packed(&[
                    AnyView::from(&self.state),
                    AnyView::from(&tvm_ffi::Any::from(0)), // sequence id
                ])
                .unwrap();
            Ok(())
        }

        pub fn remove_sequence(&mut self) -> tvm_ffi::Result<()> {
            self.fkv_state_remove_sequence
                .call_packed(&[
                    AnyView::from(&self.state),
                    AnyView::from(&tvm_ffi::Any::from(0)), // sequence id
                ])
                .unwrap();
            Ok(())
        }

        pub fn begin_forward(&mut self, length: impl Into<i64>) -> tvm_ffi::Result<tvm_ffi::Any> {
            self.fkv_state_begin_forward.call_packed(&[
                AnyView::from(&self.state),
                AnyView::from(&Shape::from(vec![0])),
                AnyView::from(&Shape::from(vec![length.into()])),
            ])
        }

        pub fn end_forward(&mut self) -> tvm_ffi::Result<tvm_ffi::Any> {
            self.fkv_state_end_forward
                .call_packed(&[(&self.state).into()])
        }

        pub fn clear(&mut self) -> tvm_ffi::Result<()> {
            self.fkv_state_clear
                .call_packed(&[AnyView::from(&self.state)])
                .unwrap();
            self.add_sequence().unwrap();
            Ok(())
        }

        pub fn popn(&mut self, num_tokens: i64) -> tvm_ffi::Result<()> {
            self.fkv_state_popn
                .call_packed(&[
                    AnyView::from(&self.state),
                    AnyView::from(&tvm_ffi::Any::from(0)), // sequence id
                    AnyView::from(&tvm_ffi::Any::from(num_tokens)),
                ])
                .unwrap();
            Ok(())
        }

        pub fn get_num_available_pages(&self) -> i64 {
            let res = self
                .fkv_cache_get_num_available_pages
                .call_packed(&[AnyView::from(&self.state)])
                .unwrap();
            res.try_into().unwrap()
        }

        pub fn get_total_sequence_length(&self) -> i64 {
            let res = self
                .fkv_cache_get_total_sequence_length
                .call_packed(&[AnyView::from(&self.state)])
                .unwrap();
            res.try_into().unwrap()
        }
    }

    impl Drop for KVCache {
        fn drop(&mut self) {
            self.remove_sequence().unwrap();
        }
    }

    #[allow(dead_code)]
    pub struct LanguageModelInferencer {
        device: DLDevice,
        vm: Module,
        params: Array<TVMFFITensor>,
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
            kv_cache_config: KvCacheConfig,
        ) -> Self {
            let exec = Module::load_from_file(runtime_path.to_string_lossy()).unwrap();
            let vm: Module = exec
                .get_function("vm_load_executable")
                .unwrap()
                .call_tuple(())
                .unwrap()
                .try_into()
                .unwrap();
            vm.get_function("vm_initialization")
                .unwrap()
                .call_tuple((
                    device.device_type as i32,            // device_type
                    device.device_id as i32,              // device_id
                    2i32,                                 // vm_allocator_type
                    tvm_ffi::DLDeviceType::kDLCPU as i32, // host_device_type
                    0i32,                                 // host_device_id
                    2i32,                                 // host_vm_allocator_type
                ))
                .unwrap();

            let metadata: tvm_ffi::String = vm
                .get_function("_metadata")
                .unwrap()
                .call_tuple(())
                .unwrap()
                .try_into()
                .unwrap();
            let metadata: serde_json::Value = serde_json::from_str(&metadata).unwrap();

            let tensor_cache = TensorCache::from(tensor_cache_path, device).unwrap();
            let param_names = metadata
                .get("params")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.get("name").unwrap().as_str().unwrap())
                .collect::<Vec<_>>();
            let params = tensor_cache.get_params(param_names);

            let kv_cache = KVCache::new(
                &vm,
                kv_cache_config.context_window_size.map(|v| v as i64),
                None,
                None,
            );

            let fembed = vm.get_function("embed").unwrap();
            let fprefill = vm.get_function("prefill").unwrap();
            let fdecode = vm.get_function("decode").unwrap();
            let fapply_bitmask_inplace = vm.get_function("apply_bitmask_inplace").unwrap();
            let fsample_top_p_from_logits =
                Function::get_global("vm.builtin.sample_top_p_from_logits").unwrap();

            Self {
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
            }
        }

        pub fn embed(&self, tokens: &[i32]) -> anyhow::Result<TVMFFITensor> {
            let mut input = Tensor::empty(
                &[tokens.len() as i64],
                DLDataType {
                    code: DLDataTypeCode::kDLInt as u8,
                    bits: 32,
                    lanes: 1,
                },
                self.device,
            );
            unsafe {
                let tokens_slice = std::slice::from_raw_parts(
                    tokens.as_ptr() as *const u8,
                    tokens.len() * std::mem::size_of::<i32>(),
                );
                input.copy_from_slice(tokens_slice).unwrap();
            }

            let embedding: TVMFFITensor = self
                .fembed
                .call_packed(&[
                    AnyView::from(&<TVMFFITensor as From<Tensor>>::from(input)),
                    AnyView::from(&self.params),
                ])
                .unwrap()
                .try_into()
                .unwrap();
            let mut embedding: Tensor = embedding.into();
            let embedding_reshaped = embedding
                .reshape(&[1, embedding.shape()[0], embedding.shape()[1]])
                .unwrap();

            Ok(embedding_reshaped.into())
        }

        pub fn clear(&mut self) -> anyhow::Result<()> {
            self.kv_cache
                .clear()
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            self.history.clear();
            Ok(())
        }

        pub fn prefill(&mut self, tokens: &[u32]) -> anyhow::Result<()> {
            if tokens.is_empty() {
                anyhow::bail!("Token must not be empty");
            }

            // Make sure that kv-cache and history is sync
            if self.kv_cache.get_total_sequence_length() != self.history.len() as i64 {
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
                    .popn((self.history.len() - lcp_index) as i64)
                    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            }

            // Tokens to be added (without common prefixes)
            let new_tokens: Vec<i32> = tokens[lcp_index..].iter().map(|t| *t as i32).collect();

            if new_tokens.is_empty() {
                self.history = tokens.to_vec();
                return Ok(());
            }

            // Calculate remaining space in KV cache
            if new_tokens.len() as i64
                >= self.kv_cache.get_num_available_pages() * self.kv_cache.page_size
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
                    .begin_forward(length as i64)
                    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
                self.fprefill
                    .call_packed(&[
                        AnyView::from(&embedding),
                        AnyView::from(self.kv_cache.get_state()),
                        AnyView::from(&self.params),
                    ])
                    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
                self.kv_cache
                    .end_forward()
                    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            }

            // Update history
            self.history = tokens.to_vec();

            Ok(())
        }

        pub fn decode(&mut self, last_token: u32) -> anyhow::Result<TVMFFITensor> {
            let embedding = self.embed(&[last_token as i32]).unwrap();

            self.kv_cache.begin_forward(1).unwrap();
            let output = self
                .fdecode
                .call_packed(&[
                    AnyView::from(&embedding),
                    AnyView::from(self.kv_cache.get_state()),
                    AnyView::from(&self.params),
                ])
                .unwrap();
            self.kv_cache.end_forward().unwrap();

            // The output of decode is an Array of 2 items: logits(Tensor) and kv cache.
            let logits =
                unsafe { tvm_ffi::collections::array::get_from_any_array(output, 0).unwrap() };

            Ok(logits)
        }

        pub fn sample(
            &mut self,
            logits: &TVMFFITensor,
            temperature: f64,
            top_p: f64,
        ) -> anyhow::Result<u32> {
            let uniform_dist_threshold: f64 = crate::utils::get_random_f64();
            let sampled_token: i32 = self
                .fsample_top_p_from_logits
                .call_tuple((logits, &temperature, &top_p, &uniform_dist_threshold))
                .unwrap()
                .try_into()
                .unwrap();
            let sampled_token = sampled_token as u32;
            self.history.push(sampled_token);
            Ok(sampled_token)
        }
    }

    impl TryFromCache for LanguageModelInferencer {
        fn claim_files<'a>(
            cache: Cache,
            key: impl AsRef<str>,
            _: &'a mut std::collections::HashMap<String, crate::value::Value>,
        ) -> BoxFuture<'a, anyhow::Result<CacheClaim>> {
            claim_files(cache, key)
        }

        fn try_from_contents<'a>(
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
                    serde_json::from_value(kv_cache.clone().into()).unwrap()
                } else {
                    KvCacheConfig::default()
                };

                let runtime_filename = format!("rt.{}", get_lib_extension());
                let runtime_path =
                    if let Some((entry, _)) = contents.remove_with_filename(&runtime_filename) {
                        entry.path()
                    } else {
                        anyhow::bail!("{} does not exist", runtime_filename)
                    };

                let tensor_cache_path = if let Some((entry, _)) =
                    contents.remove_with_filename("tensor-cache.json")
                {
                    entry.path()
                } else if let Some((entry, _)) = contents.remove_with_filename("ndarray-cache.json")
                {
                    entry.path()
                } else {
                    anyhow::bail!("tensor cache json does not exist")
                };

                let inferencer = LanguageModelInferencer::new(
                    &contents.root.join(runtime_path),
                    &contents.root.join(tensor_cache_path),
                    device,
                    kv_cache_config,
                );

                Ok(inferencer)
            })
        }
    }
}

#[cfg(any(target_family = "wasm"))]
mod tvmjs_runtime {
    use std::fmt;

    use anyhow::{anyhow, bail};
    use js_sys::{Float32Array, Object, Reflect, Uint8Array, Uint32Array};
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::JsFuture;

    use super::*;
    use crate::{
        cache::{Cache, CacheContents, TryFromCache},
        ffi::js_bridge::{
            JSTVMEmbeddingModel, JSTVMLanguageModel, init_tvm_embedding_model_js,
            init_tvm_language_model_js,
        },
        model::KvCacheConfig,
        utils::{BoxFuture, float16},
    };

    pub struct LanguageModelInferencer {
        inner: JSTVMLanguageModel,
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

        pub async fn decode(&mut self, last_token: u32, temperature: f64, top_p: f64) -> u32 {
            let logits: Float32Array = JsFuture::from(self.inner.decode(last_token))
                .await
                .unwrap()
                .into();
            self.inner.sample(logits, temperature, top_p)
        }
    }

    impl TryFromCache for LanguageModelInferencer {
        fn claim_files<'a>(
            cache: Cache,
            key: impl AsRef<str>,
            _: &'a mut std::collections::HashMap<String, crate::value::Value>,
        ) -> BoxFuture<'a, anyhow::Result<CacheClaim>> {
            claim_files(cache, key)
        }

        fn try_from_contents<'a>(
            contents: &'a mut CacheContents,
            ctx: &'a std::collections::HashMap<String, crate::value::Value>,
        ) -> BoxFuture<'a, anyhow::Result<Self>> {
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
                let config = {
                    let config = Object::new();
                    if let Some(kv_cache) = ctx.get("kv_cache") {
                        let kv_cache_config =
                            serde_json::from_value::<KvCacheConfig>(kv_cache.clone().into())
                                .unwrap_or_default();
                        Reflect::set(&config, &"kvCache".into(), &kv_cache_config.into()).unwrap();
                    }
                    config
                };

                let prom = init_tvm_language_model_js(&cache_contents, Some(config));
                let js_lm = match JsFuture::from(prom).await {
                    Ok(out) => {
                        let lm: JSTVMLanguageModel = out
                            .dyn_into()
                            .map_err(|e| anyhow!("Conversion failed: {:?}", e))?;
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
        inner: JSTVMEmbeddingModel,
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
        fn claim_files<'a>(
            cache: Cache,
            key: impl AsRef<str>,
            _: &'a mut std::collections::HashMap<String, crate::value::Value>,
        ) -> BoxFuture<'a, anyhow::Result<CacheClaim>> {
            claim_files(cache, key)
        }

        fn try_from_contents<'a>(
            contents: &'a mut CacheContents,
            _: &'a std::collections::HashMap<String, crate::value::Value>,
        ) -> BoxFuture<'a, anyhow::Result<Self>> {
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

                let prom = init_tvm_embedding_model_js(&cache_contents);
                let js_em: JSTVMEmbeddingModel = match JsFuture::from(prom).await {
                    Ok(out) => {
                        let em: JSTVMEmbeddingModel = out
                            .dyn_into()
                            .map_err(|e| anyhow!("Conversion failed: {:?}", e))?;
                        em
                    }
                    Err(err) => {
                        bail!("JS inferencer init failed: {:?}", err)
                    }
                };

                Ok(EmbeddingModelInferencer { inner: js_em })
            })
        }
    }
}
