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
    // Rust wrapper of DLManagedTensorVersioned
    pub struct DLPackTensor {
        inner: UniquePtr<ManagedTensor>,
    }

    extern "C++" {
        include!("dlpack/dlpack.h");

        type DLDevice;
        type DLManagedTensorVersioned;
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

    /// Rust wrapper of faiss::Index
    pub struct FaissIndex {
        inner: UniquePtr<FaissIndexWrapper>,
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum FaissMetricType {
        InnerProduct = 0, // Inner product (for cosine similarity with normalized vectors)
        L2 = 1,           // Euclidean distance
    }

    #[namespace = "faiss_bridge"]
    unsafe extern "C++" {
        include!("faiss_bridge.hpp");

        type FaissIndexWrapper;

        // FaissIndex 생성
        unsafe fn create_index(
            dimension: i32,
            description: &str,
            metric: FaissMetricType,
        ) -> Result<UniquePtr<FaissIndexWrapper>>;

        // 벡터 추가 (add_with_ids 사용)
        unsafe fn add_vectors_with_ids(
            self: Pin<&mut FaissIndexWrapper>,
            vectors: &[f32],
            num_vectors: usize,
            ids: &[i64],
        ) -> Result<()>;

        // 검색
        unsafe fn search_vectors(
            self: &FaissIndexWrapper,
            query_vectors: &[f32],
            num_queries: usize,
            k: usize,
            distances: &mut [f32],
            indices: &mut [i64],
        ) -> Result<()>;

        // 인덱스 정보 확인
        fn is_trained(self: &FaissIndexWrapper) -> bool;
        fn get_ntotal(self: &FaissIndexWrapper) -> i64;
        fn get_dimension(self: &FaissIndexWrapper) -> i32;

        // 학습 (필요한 경우)
        unsafe fn train_index(
            self: Pin<&mut FaissIndexWrapper>,
            training_vectors: &[f32],
            num_training_vectors: usize,
        ) -> Result<()>;
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

        // #[cxx_name = "faiss_vector_store_t"]
        // type FAISSVectorStore;

        // pub fn create_faiss_vector_store(dimension: u32) -> UniquePtr<FAISSVectorStore>;
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

unsafe impl Send for ffi::TVMLanguageModel {}

unsafe impl Sync for ffi::TVMLanguageModel {}

impl std::fmt::Debug for ffi::TVMLanguageModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TVMLanguageModel").finish()
    }
}
