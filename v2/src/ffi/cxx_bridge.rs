pub use ffi::*;

use crate::cache::{CacheContents, CacheEntry};
use crate::ffi::util::*;
use anyhow::{Result, bail};

fn cache_entry_new(dirname: &str, filename: &str) -> Box<CacheEntry> {
    Box::new(CacheEntry::new(dirname, filename))
}

fn cache_contents_get_root(self_: &CacheContents) -> String {
    self_.get_root().to_string_lossy().to_string()
}

fn cache_contents_remove_with_filename_out(
    self_: &mut CacheContents,
    filename: &str,
    out_dirname: &mut String,
    out_filename: &mut String,
    out_bytes: &mut Vec<u8>,
) -> bool {
    if let Some((entry, bytes)) = self_.remove_with_filename(filename) {
        out_dirname.clear();
        out_filename.clear();
        out_bytes.clear();

        out_dirname.push_str(entry.dirname());
        out_filename.push_str(entry.filename());
        out_bytes.extend_from_slice(&bytes);
        true
    } else {
        false
    }
}

#[cxx::bridge]
mod ffi {
    // Rust wrapper of DLManagedTensorVersioned
    pub struct DLPackTensor {
        inner: UniquePtr<ManagedTensor>,
    }

    extern "C++" {
        include!("dlpack/dlpack.h");

        type DLDevice;
        type DLManagedTensorVersioned;
    }

    #[namespace = "ailoy"]
    unsafe extern "C++" {
        include!("language_model.hpp");

        #[cxx_name = "tvm_language_model_t"]
        type TVMLanguageModel;

        pub fn create_tvm_language_model(
            cache: &mut CacheContents,
            device: UniquePtr<DLDevice>,
        ) -> UniquePtr<TVMLanguageModel>;

        #[cxx_name = "prefill_from_rs"]
        pub fn prefill(self: Pin<&mut TVMLanguageModel>, tokens: &Vec<u32>) -> ();

        #[cxx_name = "decode_from_rs"]
        pub fn decode(self: Pin<&mut TVMLanguageModel>, last_token: u32) -> DLPackTensor;

        #[cxx_name = "sample_from_rs"]
        pub fn sample(self: Pin<&mut TVMLanguageModel>, logits: DLPackTensor) -> u32;
    }

    #[namespace = "dlpack_bridge"]
    unsafe extern "C++" {
        include!("dlpack_bridge.hpp");

        pub fn create_dldevice(device_type: i32, device_id: i32) -> UniquePtr<DLDevice>;

        type ManagedTensor;

        unsafe fn create_managed_tensor(
            tensor: *mut DLManagedTensorVersioned,
        ) -> Result<UniquePtr<ManagedTensor>>;

        fn get_ndim(self: &ManagedTensor) -> i32;
        fn get_dimension(self: &ManagedTensor) -> i64;
        fn is_cpu_tensor(self: &ManagedTensor) -> bool;
        fn has_int_dtype(self: &ManagedTensor, bits: u8) -> bool;
        fn has_float_dtype(self: &ManagedTensor, bits: u8) -> bool;
        fn get_data_ptr_f32(self: &ManagedTensor) -> *const f32;
        fn get_data_ptr_u16(self: &ManagedTensor) -> *const u16;
    }
    #[namespace = "ailoy"]
    extern "Rust" {
        type CacheEntry;

        fn cache_entry_new(dirname: &str, filename: &str) -> Box<CacheEntry>;

        pub fn path(self: &CacheEntry) -> String;

        pub fn dirname(self: &CacheEntry) -> &str;

        pub fn filename(self: &CacheEntry) -> &str;

        type CacheContents;

        fn cache_contents_get_root(cache: &CacheContents) -> String;

        /// Remove by filename and return results via out-parameters.
        ///
        /// Why out-params? cxx does not support returning `Option<(A, B)>` or tuple types directly.
        /// Returns:
        ///   - true  : entry found and removed; out_* are filled
        ///   - false : not found; out_* remain unchanged
        fn cache_contents_remove_with_filename_out(
            cache: &mut CacheContents,
            filename: &str,
            out_dirname: &mut String,
            out_filename: &mut String,
            out_bytes: &mut Vec<u8>,
        ) -> bool;
    }
}

unsafe impl Send for ffi::DLDevice {}

unsafe impl Send for ffi::DLPackTensor {}

unsafe impl Send for ffi::TVMLanguageModel {}

unsafe impl Sync for ffi::TVMLanguageModel {}

impl std::fmt::Debug for ffi::TVMLanguageModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TVMLanguageModel").finish()
    }
}

impl ffi::DLPackTensor {
    // /// from raw DLManagedTensorVersioned pointer
    // pub unsafe fn from_raw(ptr: *mut DLManagedTensorVersioned) -> Result<Self> {
    //     let managed = unsafe { ffi::create_managed_tensor(ptr) }?;
    //     Ok(Self { inner: managed })
    // }

    /// 1-dimensional float32 Tensor to Vec<f32>
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        let dimension = self.inner.get_dimension();
        if dimension == -1 {
            bail!("Tensor is not 1D.");
        }
        if !self.inner.is_cpu_tensor() {
            bail!("GPU tensors not yet supported. CPU tensors only");
        }

        if self.inner.has_float_dtype(32) {
            let data_ptr = self.inner.get_data_ptr_f32();
            if data_ptr.is_null() {
                bail!("Tensor data pointer is null");
            }
            let vec = unsafe { std::slice::from_raw_parts(data_ptr, dimension as usize).to_vec() };
            Ok(vec)
        } else if self.inner.has_float_dtype(16) {
            let data_ptr = self.inner.get_data_ptr_u16();
            if data_ptr.is_null() {
                bail!("Tensor data pointer is null");
            }
            let vec = unsafe { std::slice::from_raw_parts(data_ptr, dimension as usize).to_vec() }
                .into_iter()
                .map(|val| util::f16_to_f32(val))
                .collect();
            Ok(vec)
        } else {
            bail!("Tensor has unsupported dtype.");
        }
    }

    // /// Rust type to C++ UniquePtr
    // fn into_inner(self) -> cxx::UniquePtr<ffi::ManagedTensor> {
    //     self.inner
    // }
}
