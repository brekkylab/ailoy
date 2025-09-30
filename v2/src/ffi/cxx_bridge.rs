pub use ffi::*;

use crate::cache::{CacheContents, CacheEntry};

fn cache_entry_new(dirname: &str, filename: &str) -> Box<CacheEntry> {
    Box::new(CacheEntry::new(dirname, filename))
}

fn cache_contents_get_root(self_: &CacheContents) -> String {
    self_.root.to_string_lossy().to_string()
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
    extern "C++" {
        include!("dlpack/dlpack.h");

        type DLDevice;
        type DLManagedTensorVersioned;
    }

    // Rust wrapper of DLManagedTensorVersioned
    pub struct DLPackTensor {
        inner: UniquePtr<ManagedTensor>,
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

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum FaissMetricType {
        /// basic metrics
        InnerProduct = 0,
        L2 = 1,
        L1,
        Linf,
        Lp,

        /// some additional metrics defined in scipy.spatial.distance
        Canberra = 20,
        BrayCurtis,
        JensenShannon,

        /// sum_i(min(a_i, b_i)) / sum_i(max(a_i, b_i)) where a_i, b_i > 0
        Jaccard,
        /// Squared Eucliden distance, ignoring NaNs
        NaNEuclidean,
        /// Gower's distance - numeric dimensions are in [0,1] and categorical
        /// dimensions are negative integers
        Gower,
    }

    #[derive(Debug, Clone)]
    struct FaissIndexSearchResult {
        pub distances: Vec<f32>,
        pub indexes: Vec<i64>,
    }

    #[namespace = "faiss_bridge"]
    unsafe extern "C++" {
        include!("faiss_bridge.hpp");

        type FaissIndexInner;

        // Creator functions
        unsafe fn create_index(
            dimension: i32,
            description: &str,
            metric: FaissMetricType,
        ) -> Result<UniquePtr<FaissIndexInner>>;

        unsafe fn read_index(filename: &str) -> Result<UniquePtr<FaissIndexInner>>;

        // Methods
        fn is_trained(self: &FaissIndexInner) -> bool;
        fn get_ntotal(self: &FaissIndexInner) -> i64;
        fn get_dimension(self: &FaissIndexInner) -> i32;
        fn get_metric_type(self: &FaissIndexInner) -> FaissMetricType;

        unsafe fn train_index(
            self: Pin<&mut FaissIndexInner>,
            training_vectors: &[f32],
            num_training_vectors: usize,
        ) -> Result<()>;

        unsafe fn add_vectors_with_ids(
            self: Pin<&mut FaissIndexInner>,
            vectors: &[f32],
            num_vectors: usize,
            ids: &[i64],
        ) -> Result<()>;

        unsafe fn search_vectors(
            self: &FaissIndexInner,
            query_vectors: &[f32],
            k: usize,
        ) -> Result<FaissIndexSearchResult>;

        unsafe fn get_by_ids(self: &FaissIndexInner, ids: &[i64]) -> Result<Vec<f32>>;

        unsafe fn remove_vectors(self: Pin<&mut FaissIndexInner>, ids: &[i64]) -> Result<usize>;

        unsafe fn clear(self: Pin<&mut FaissIndexInner>) -> Result<()>;

        unsafe fn write_index(self: &FaissIndexInner, filename: &str) -> Result<()>;

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
        pub fn prefill(self: Pin<&mut TVMLanguageModel>, tokens: &[u32]) -> ();

        #[cxx_name = "decode_from_rs"]
        pub fn decode(self: Pin<&mut TVMLanguageModel>, last_token: u32) -> DLPackTensor;

        #[cxx_name = "sample_from_rs"]
        pub fn sample(self: Pin<&mut TVMLanguageModel>, logits: DLPackTensor) -> u32;
    }

    #[namespace = "ailoy"]
    unsafe extern "C++" {
        include!("embedding_model.hpp");

        #[cxx_name = "tvm_embedding_model_t"]
        type TVMEmbeddingModel;

        pub fn create_tvm_embedding_model(
            cache: &mut CacheContents,
            device: UniquePtr<DLDevice>,
        ) -> UniquePtr<TVMEmbeddingModel>;

        #[cxx_name = "infer_from_rs"]
        pub fn infer(self: Pin<&mut TVMEmbeddingModel>, tokens: &[u32]) -> DLPackTensor;
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

unsafe impl Send for ffi::FaissIndexInner {}

unsafe impl Sync for ffi::FaissIndexInner {}

unsafe impl Send for ffi::TVMEmbeddingModel {}

unsafe impl Sync for ffi::TVMEmbeddingModel {}

impl std::fmt::Debug for ffi::TVMEmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TVMEmbeddingModel").finish()
    }
}

unsafe impl Send for ffi::TVMLanguageModel {}

impl std::fmt::Debug for ffi::TVMLanguageModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TVMLanguageModel").finish()
    }
}
