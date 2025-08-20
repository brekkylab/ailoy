pub use ffi::*;

use crate::cache::{CacheContents, CacheEntry};
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
    // DLManagedTensorVersioned Rust 래퍼 타입
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

        // #[namespace = "ailoy"]
        pub fn create_tvm_language_model(
            cache: &mut CacheContents,
            device: UniquePtr<DLDevice>,
        ) -> UniquePtr<TVMLanguageModel>;

        // #[namespace = "ailoy"]
        #[cxx_name = "prefill_from_rs"]
        pub fn prefill(self: Pin<&mut TVMLanguageModel>, tokens: &Vec<u32>) -> ();

        // #[namespace = "ailoy"]
        #[cxx_name = "decode_from_rs"]
        pub fn decode(self: Pin<&mut TVMLanguageModel>, last_token: u32) -> DLPackTensor;

        // #[namespace = "ailoy"]
        #[cxx_name = "sample_from_rs"]
        pub fn sample(self: Pin<&mut TVMLanguageModel>, logits: DLPackTensor) -> u32;
    }

    #[namespace = "dlpack_bridge"]
    unsafe extern "C++" {
        include!("dlpack_bridge.hpp");

        pub fn create_dldevice(device_type: i32, device_id: i32) -> UniquePtr<DLDevice>;

        // C++의 ManagedTensor 타입
        // #[namespace = "dlpack_bridge"]
        type ManagedTensor;

        // #[namespace = "dlpack_bridge"]
        unsafe fn create_managed_tensor(
            tensor: *mut DLManagedTensorVersioned,
        ) -> Result<UniquePtr<ManagedTensor>>;

        // 메서드들
        // #[namespace = "dlpack_bridge"]
        fn is_1d_float32(self: &ManagedTensor) -> bool;
        // #[namespace = "dlpack_bridge"]
        fn get_size(self: &ManagedTensor) -> i64;
        // #[namespace = "dlpack_bridge"]
        fn is_cpu_tensor(self: &ManagedTensor) -> bool;
        // #[namespace = "dlpack_bridge"]
        fn get_data_ptr(self: &ManagedTensor) -> *const f32;
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
    /// 외부에서 받은 DLManagedTensorVersioned 포인터로부터 생성
    ///
    /// # Safety
    /// - `ptr`은 유효한 DLManagedTensorVersioned 포인터여야 함
    /// - deleter 함수가 올바르게 설정되어 있어야 함
    pub unsafe fn from_raw(ptr: *mut DLManagedTensorVersioned) -> Result<Self> {
        // Rust 타입인 `*mut c_void`를 cxx가 이해하는 `*mut u8`로 캐스팅합니다.
        // 이 캐스팅은 성능 저하가 없는 제로-코스트 추상화입니다.
        let managed = unsafe { ffi::create_managed_tensor(ptr) }?;
        Ok(Self { inner: managed })
    }

    /// 1차원 float32 텐서를 Vec<f32>로 변환
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        if !self.inner.is_1d_float32() {
            bail!("Tensor is not 1D float32. Other types not yet implemented");
        }
        if !self.inner.is_cpu_tensor() {
            bail!("GPU tensors not yet supported. CPU tensors only");
        }

        let size = self.inner.get_size() as usize;
        if size <= 0 {
            return Ok(Vec::new()); // 빈 벡터 반환
        }
        let data_ptr = self.inner.get_data_ptr();
        if data_ptr.is_null() {
            bail!("Tensor data pointer is null");
        }

        let vec = unsafe { std::slice::from_raw_parts(data_ptr, size).to_vec() };
        Ok(vec)
    }

    // // Rust 타입을 C++ UniquePtr로 변환
    // fn into_inner(self) -> cxx::UniquePtr<ffi::ManagedTensor> {
    //     self.inner
    // }
}
