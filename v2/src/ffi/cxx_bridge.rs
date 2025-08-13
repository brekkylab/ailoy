pub use ffi::*;

use crate::cache::{CacheContents, CacheEntry};

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
    unsafe extern "C++" {
        include!("language_model.hpp");

        type DLDevice;

        #[namespace = "ailoy"]
        pub fn create_dldevice(device_type: i32, device_id: i32) -> UniquePtr<DLDevice>;

        type DLManagedTensorVersioned;

        #[namespace = "ailoy"]
        #[cxx_name = "tvm_language_model_t"]
        type TVMLanguageModel;

        #[namespace = "ailoy"]
        pub fn create_tvm_language_model(
            cache: &mut CacheContents,
            device: UniquePtr<DLDevice>,
        ) -> UniquePtr<TVMLanguageModel>;

        #[namespace = "ailoy"]
        #[cxx_name = "prefill_from_rs"]
        pub fn prefill(self: Pin<&mut TVMLanguageModel>, tokens: &Vec<u32>) -> ();

        #[namespace = "ailoy"]
        #[cxx_name = "decode_from_rs"]
        pub fn decode(
            self: Pin<&mut TVMLanguageModel>,
            last_token: u32,
        ) -> UniquePtr<DLManagedTensorVersioned>;

        #[namespace = "ailoy"]
        #[cxx_name = "sample_from_rs"]
        pub fn sample(
            self: Pin<&mut TVMLanguageModel>,
            logits: UniquePtr<DLManagedTensorVersioned>,
        ) -> u32;
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

unsafe impl Send for ffi::DLManagedTensorVersioned {}

unsafe impl Send for ffi::TVMLanguageModel {}

unsafe impl Sync for ffi::TVMLanguageModel {}

impl std::fmt::Debug for ffi::TVMLanguageModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TVMLanguageModel").finish()
    }
}
