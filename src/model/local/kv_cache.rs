///! Unified KV Cache implementation for both native and web platforms
///!
///! This module provides a platform-agnostic KV cache interface that works with
///! both native TVM runtime (via tvm-ffi) and web platform (via JavaScript PackedFuncs).
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "python", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(module = "ailoy._core", get_all, set_all)
)]
#[cfg_attr(
    feature = "nodejs",
    napi_derive::napi(object, js_name = "KVCacheConfig")
)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KVCacheConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window_size: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_chunk_size: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sliding_window_size: Option<u32>,
}

#[cfg(feature = "python")]
mod py {
    use pyo3::prelude::*;
    use pyo3_stub_gen_derive::*;

    use super::*;

    #[gen_stub_pymethods]
    #[pymethods]
    impl KVCacheConfig {
        #[new]
        fn __new__(
            context_window_size: Option<u32>,
            prefill_chunk_size: Option<u32>,
            sliding_window_size: Option<u32>,
        ) -> Self {
            Self {
                context_window_size,
                prefill_chunk_size,
                sliding_window_size,
            }
        }
    }
}

pub trait KVCacheOps {
    /// Clear the KV cache
    fn clear(&mut self) -> Result<()>;

    /// Add a sequence to the cache
    fn add_sequence(&mut self, seq_id: i64) -> Result<()>;

    /// Remove a sequence from the cache
    fn remove_sequence(&mut self, seq_id: i64) -> Result<()>;

    /// Begin forward pass
    fn begin_forward(&mut self, seq_id: i64, length: i64) -> Result<()>;

    /// End forward pass
    fn end_forward(&mut self) -> Result<()>;

    /// Pop N tokens from the cache
    fn popn(&mut self, seq_id: i64, num_tokens: i64) -> Result<()>;

    /// Get number of available pages
    fn get_num_available_pages(&self) -> Result<i64>;

    /// Get total sequence length
    fn get_total_sequence_length(&self) -> Result<i64>;
}

#[cfg(any(target_family = "unix", target_family = "windows"))]
mod native {
    use anyhow::{Result, anyhow};

    use super::{KVCacheConfig, KVCacheOps};

    const PAGE_SIZE: i64 = 16;

    #[allow(dead_code)]
    pub struct KVCache {
        state: tvm_ffi::Any,

        pub context_window_size: i64,
        pub prefill_chunk_size: i64,
        pub sliding_window_size: i64,
        pub page_size: i64,

        fkv_state_clear: tvm_ffi::Function,
        fkv_state_add_sequence: tvm_ffi::Function,
        fkv_state_remove_sequence: tvm_ffi::Function,
        fkv_state_begin_forward: tvm_ffi::Function,
        fkv_state_end_forward: tvm_ffi::Function,
        fkv_state_popn: tvm_ffi::Function,
        fkv_cache_get_num_available_pages: tvm_ffi::Function,
        fkv_cache_get_total_sequence_length: tvm_ffi::Function,
    }

    impl KVCache {
        pub fn new(vm: &tvm_ffi::Module, config: KVCacheConfig) -> Result<Self> {
            use tvm_ffi::Shape;

            let metadata_str: tvm_ffi::String = vm
                .get_function("_metadata")
                .map_err(|e| anyhow!("Failed to get `_metadata` function: {:?}", e))?
                .call_tuple(())
                .map_err(|e| anyhow!("Failed to call `_metadata`: {:?}", e))?
                .try_into()
                .map_err(|e| anyhow!("Failed to convert to String: {:?}", e))?;
            let metadata: serde_json::Value = serde_json::from_str(&metadata_str)
                .map_err(|e| anyhow!("Failed to parse metadata json: {:?}", e))?;

            let context_window_size = config.context_window_size.map(|v| v as i64).unwrap_or(
                metadata
                    .get("context_window_size")
                    .ok_or(anyhow!("Failed to get `context_window_size` attribute"))?
                    .as_i64()
                    .ok_or(anyhow!("Failed to convert `context_window_size` to i64"))?,
            );
            let prefill_chunk_size = config.prefill_chunk_size.map(|v| v as i64).unwrap_or(
                metadata
                    .get("prefill_chunk_size")
                    .ok_or(anyhow!("Failed to get `prefill_chunk_size` attribute"))?
                    .as_i64()
                    .ok_or(anyhow!("Failed to convert `prefill_chunk_size` to i64"))?,
            );
            let sliding_window_size = config.sliding_window_size.map(|v| v as i64).unwrap_or(
                metadata
                    .get("sliding_window_size")
                    .ok_or(anyhow!("Failed to get `sliding_window_size` attribute"))?
                    .as_i64()
                    .ok_or(anyhow!("Failed to convert `sliding_window_size` to i64"))?,
            );

            let state = vm
                .get_function("create_tir_paged_kv_cache")
                .map_err(|e| {
                    anyhow!(
                        "Failed to get `create_tir_paged_kv_cache` function: {:?}",
                        e
                    )
                })?
                .call_tuple((
                    Shape::from([1]),
                    Shape::from([context_window_size]),
                    Shape::from([prefill_chunk_size]),
                    Shape::from([PAGE_SIZE]),
                    Shape::from([(sliding_window_size != -1) as i64]),
                ))
                .map_err(|e| anyhow!("Failed to call `create_tir_paged_kv_cache`: {:?}", e))?;

            let fkv_state_clear = tvm_ffi::Function::get_global("vm.builtin.kv_state_clear")
                .map_err(|e| {
                    anyhow!(
                        "Failed to get global function `vm.builtin.kv_state_clear`: {:?}",
                        e
                    )
                })?;
            let fkv_state_add_sequence =
                tvm_ffi::Function::get_global("vm.builtin.kv_state_add_sequence").map_err(|e| {
                    anyhow!(
                        "Failed to get global function `vm.builtin.kv_state_add_sequence`: {:?}",
                        e
                    )
                })?;
            let fkv_state_remove_sequence = tvm_ffi::Function::get_global(
                "vm.builtin.kv_state_remove_sequence",
            )
            .map_err(|e| {
                anyhow!(
                    "Failed to get global function `vm.builtin.kv_state_remove_sequence`: {:?}",
                    e
                )
            })?;
            let fkv_state_begin_forward = tvm_ffi::Function::get_global(
                "vm.builtin.kv_state_begin_forward",
            )
            .map_err(|e| {
                anyhow!(
                    "Failed to get global function `vm.builtin.kv_state_begin_forward`: {:?}",
                    e
                )
            })?;
            let fkv_state_end_forward =
                tvm_ffi::Function::get_global("vm.builtin.kv_state_end_forward").map_err(|e| {
                    anyhow!(
                        "Failed to get global function `vm.builtin.kv_state_end_forward`: {:?}",
                        e
                    )
                })?;
            let fkv_state_popn = tvm_ffi::Function::get_global("vm.builtin.kv_state_popn")
                .map_err(|e| {
                    anyhow!(
                        "Failed to get global function `vm.builtin.kv_state_popn`: {:?}",
                        e
                    )
                })?;
            let fkv_cache_get_num_available_pages = tvm_ffi::Function::get_global("vm.builtin.attention_kv_cache_get_num_available_pages")
            .map_err(|e| anyhow!("Failed to get global function `vm.builtin.attention_kv_cache_get_num_available_pages`: {:?}", e))?;
            let fkv_cache_get_total_sequence_length = tvm_ffi::Function::get_global("vm.builtin.attention_kv_cache_get_total_sequence_length")
            .map_err(|e| anyhow!("Failed to get global function `vm.builtin.attention_kv_cache_get_total_sequence_length`: {:?}", e))?;

            let mut inner = Self {
                state,
                context_window_size,
                prefill_chunk_size,
                sliding_window_size,
                page_size: PAGE_SIZE,
                fkv_state_clear,
                fkv_state_add_sequence,
                fkv_state_remove_sequence,
                fkv_state_begin_forward,
                fkv_state_end_forward,
                fkv_state_popn,
                fkv_cache_get_num_available_pages,
                fkv_cache_get_total_sequence_length,
            };
            inner.clear()?;
            Ok(inner)
        }

        /// Get the native state object (only available on native platforms)
        #[cfg(any(target_family = "unix", target_family = "windows"))]
        pub fn get_state(&self) -> &tvm_ffi::Any {
            &self.state
        }
    }

    impl KVCacheOps for KVCache {
        fn clear(&mut self) -> Result<()> {
            use tvm_ffi::AnyView;
            self.fkv_state_clear
                .call_packed(&[AnyView::from(&self.state)])
                .map_err(|e| anyhow!("{e:?}"))?;
            self.add_sequence(0)?;
            Ok(())
        }

        fn add_sequence(&mut self, seq_id: i64) -> Result<()> {
            use tvm_ffi::AnyView;
            self.fkv_state_add_sequence
                .call_packed(&[AnyView::from(&self.state), AnyView::from(&seq_id)])
                .map_err(|e| anyhow!("{e:?}"))?;
            Ok(())
        }

        fn remove_sequence(&mut self, seq_id: i64) -> Result<()> {
            use tvm_ffi::AnyView;
            self.fkv_state_remove_sequence
                .call_packed(&[AnyView::from(&self.state), AnyView::from(&seq_id)])
                .map_err(|e| anyhow!("{e:?}"))?;
            Ok(())
        }

        fn begin_forward(&mut self, seq_id: i64, length: i64) -> Result<()> {
            use tvm_ffi::{AnyView, Shape};
            let seq_ids_shape = Shape::from(vec![seq_id]);
            let lengths_shape = Shape::from(vec![length]);
            self.fkv_state_begin_forward
                .call_packed(&[
                    AnyView::from(&self.state),
                    AnyView::from(&seq_ids_shape),
                    AnyView::from(&lengths_shape),
                ])
                .map_err(|e| anyhow!("{e:?}"))?;
            Ok(())
        }

        fn end_forward(&mut self) -> Result<()> {
            use tvm_ffi::AnyView;
            self.fkv_state_end_forward
                .call_packed(&[AnyView::from(&self.state)])
                .map_err(|e| anyhow!("{e:?}"))?;
            Ok(())
        }

        fn popn(&mut self, seq_id: i64, num_tokens: i64) -> Result<()> {
            use tvm_ffi::AnyView;
            self.fkv_state_popn
                .call_packed(&[
                    AnyView::from(&self.state),
                    AnyView::from(&seq_id),
                    AnyView::from(&num_tokens),
                ])
                .map_err(|e| anyhow!("{e:?}"))?;
            Ok(())
        }

        fn get_num_available_pages(&self) -> Result<i64> {
            use tvm_ffi::AnyView;
            let res = self
                .fkv_cache_get_num_available_pages
                .call_packed(&[AnyView::from(&self.state)])
                .map_err(|e| anyhow!("Failed to get num available pages: {:?}", e))?;
            res.try_into()
                .map_err(|e| anyhow!("Failed to convert to i64: {:?}", e))
        }

        fn get_total_sequence_length(&self) -> Result<i64> {
            use tvm_ffi::AnyView;
            let res = self
                .fkv_cache_get_total_sequence_length
                .call_packed(&[AnyView::from(&self.state)])
                .map_err(|e| anyhow!("Failed to get total sequence length: {:?}", e))?;
            res.try_into()
                .map_err(|e| anyhow!("Failed to convert to i64: {:?}", e))
        }
    }

    impl Drop for KVCache {
        fn drop(&mut self) {
            let _ = self.remove_sequence(0);
        }
    }
}

#[cfg(any(target_family = "unix", target_family = "windows"))]
pub use native::*;

#[cfg(target_family = "wasm")]
mod wasm {
    use anyhow::{Result, anyhow};

    use super::{KVCacheConfig, KVCacheOps};
    use crate::ffi::web::{
        conversion::*,
        tvmjs_bridge::{
            PackedFunc, {self as tvmjs},
        },
    };

    const PAGE_SIZE: i64 = 16;

    #[allow(dead_code)]
    pub struct KVCache {
        tvm: tvmjs::Instance,
        state: tvmjs::TVMObject,

        pub context_window_size: i64,
        pub prefill_chunk_size: i64,
        pub sliding_window_size: i64,
        pub page_size: i64,

        fkv_state_clear: PackedFunc,
        fkv_state_add_sequence: PackedFunc,
        fkv_state_remove_sequence: PackedFunc,
        fkv_state_begin_forward: PackedFunc,
        fkv_state_end_forward: PackedFunc,
        fkv_state_popn: PackedFunc,
        fkv_cache_get_num_available_pages: PackedFunc,
        fkv_cache_get_total_sequence_length: PackedFunc,
    }

    impl KVCache {
        pub fn new(
            tvm: tvmjs::Instance,
            vm: &tvmjs::Module,
            metadata: &serde_json::Value,
            config: KVCacheConfig,
        ) -> Result<Self> {
            tvm.begin_scope();

            let context_window_size = config.context_window_size.map(|v| v as i64).unwrap_or(
                metadata
                    .get("context_window_size")
                    .ok_or(anyhow!("Failed to get `context_window_size` attribute"))?
                    .as_i64()
                    .ok_or(anyhow!("Failed to convert `context_window_size` to i64"))?,
            );
            let prefill_chunk_size = config.prefill_chunk_size.map(|v| v as i64).unwrap_or(
                metadata
                    .get("prefill_chunk_size")
                    .ok_or(anyhow!("Failed to get `prefill_chunk_size` attribute"))?
                    .as_i64()
                    .ok_or(anyhow!("Failed to convert `prefill_chunk_size` to i64"))?,
            );
            let sliding_window_size = config.sliding_window_size.map(|v| v as i64).unwrap_or(
                metadata
                    .get("sliding_window_size")
                    .ok_or(anyhow!("Failed to get `sliding_window_size` attribute"))?
                    .as_i64()
                    .ok_or(anyhow!("Failed to convert `sliding_window_size` to i64"))?,
            );

            let fcreate_kv_cache: PackedFunc = vm.get_function("create_tir_paged_kv_cache").into();
            let state: tvmjs::TVMObject = tvm.detach(fcreate_kv_cache.call5(
                &tvm.make_shape_tuple(i64_slice_to_js(&[1])),
                &tvm.make_shape_tuple(i64_slice_to_js(&[context_window_size])),
                &tvm.make_shape_tuple(i64_slice_to_js(&[prefill_chunk_size])),
                &tvm.make_shape_tuple(i64_slice_to_js(&[PAGE_SIZE])),
                &tvm.make_shape_tuple(i64_slice_to_js(&[(sliding_window_size != -1) as i64])),
            )?);

            let fkv_state_clear: PackedFunc =
                tvm.detach(tvm.get_global_func("vm.builtin.kv_state_clear"));
            let fkv_state_add_sequence: PackedFunc =
                tvm.detach(tvm.get_global_func("vm.builtin.kv_state_add_sequence"));
            let fkv_state_remove_sequence: PackedFunc =
                tvm.detach(tvm.get_global_func("vm.builtin.kv_state_remove_sequence"));
            let fkv_state_begin_forward: PackedFunc =
                tvm.detach(tvm.get_global_func("vm.builtin.kv_state_begin_forward"));
            let fkv_state_end_forward: PackedFunc =
                tvm.detach(tvm.get_global_func("vm.builtin.kv_state_end_forward"));
            let fkv_state_popn: PackedFunc =
                tvm.detach(tvm.get_global_func("vm.builtin.kv_state_popn"));
            let fkv_cache_get_num_available_pages: PackedFunc = tvm.detach(
                tvm.get_global_func("vm.builtin.attention_kv_cache_get_num_available_pages"),
            );
            let fkv_cache_get_total_sequence_length: PackedFunc = tvm.detach(
                tvm.get_global_func("vm.builtin.attention_kv_cache_get_total_sequence_length"),
            );

            let mut inner = Self {
                tvm,
                state,
                context_window_size,
                prefill_chunk_size,
                sliding_window_size,
                page_size: PAGE_SIZE,
                fkv_state_clear,
                fkv_state_add_sequence,
                fkv_state_remove_sequence,
                fkv_state_begin_forward,
                fkv_state_end_forward,
                fkv_state_popn,
                fkv_cache_get_num_available_pages,
                fkv_cache_get_total_sequence_length,
            };
            inner.clear()?;
            Ok(inner)
        }

        /// Get the web state object (only available on web platform)
        pub fn get_state(&self) -> &wasm_bindgen::JsValue {
            &self.state
        }
    }

    impl KVCacheOps for KVCache {
        fn clear(&mut self) -> Result<()> {
            self.fkv_state_clear.call1(&self.state)?;
            self.add_sequence(0)?;
            Ok(())
        }

        fn add_sequence(&mut self, seq_id: i64) -> Result<()> {
            self.fkv_state_add_sequence
                .call2(&self.state, &tvmjs::Scalar::new(seq_id as f64, "int"))?;
            Ok(())
        }

        fn remove_sequence(&mut self, seq_id: i64) -> Result<()> {
            self.fkv_state_remove_sequence
                .call2(&self.state, &tvmjs::Scalar::new(seq_id as f64, "int"))?;
            Ok(())
        }

        fn begin_forward(&mut self, seq_id: i64, length: i64) -> Result<()> {
            self.fkv_state_begin_forward.call3(
                &self.state,
                &self.tvm.make_shape_tuple(i64_slice_to_js(&[seq_id])),
                &self.tvm.make_shape_tuple(i64_slice_to_js(&[length])),
            )?;
            Ok(())
        }

        fn end_forward(&mut self) -> Result<()> {
            self.fkv_state_end_forward.call1(&self.state)?;
            Ok(())
        }

        fn popn(&mut self, seq_id: i64, num_tokens: i64) -> Result<()> {
            self.fkv_state_popn.call3(
                &self.state,
                &tvmjs::Scalar::new(seq_id as f64, "int"),
                &tvmjs::Scalar::new(num_tokens as f64, "int"),
            )?;
            Ok(())
        }

        fn get_num_available_pages(&self) -> Result<i64> {
            let result = self.fkv_cache_get_num_available_pages.call1(&self.state)?;
            js_to_i64(&result)
        }

        fn get_total_sequence_length(&self) -> Result<i64> {
            let result = self
                .fkv_cache_get_total_sequence_length
                .call1(&self.state)?;
            js_to_i64(&result)
        }
    }

    impl Drop for KVCache {
        fn drop(&mut self) {
            let _ = self.remove_sequence(0);

            let _ = self.fkv_cache_get_num_available_pages.dispose();
            let _ = self.fkv_cache_get_total_sequence_length.dispose();
            let _ = self.fkv_state_add_sequence.dispose();
            let _ = self.fkv_state_begin_forward.dispose();
            let _ = self.fkv_state_clear.dispose();
            let _ = self.fkv_state_end_forward.dispose();
            let _ = self.fkv_state_popn.dispose();
            let _ = self.fkv_state_remove_sequence.dispose();
            let _ = self.state.dispose();
        }
    }
}

#[cfg(target_family = "wasm")]
pub use wasm::*;
