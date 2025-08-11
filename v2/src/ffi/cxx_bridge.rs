pub use ffi::*;

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("cache.hpp");

        #[namespace = "ailoy"]
        #[cxx_name = "cache_t"]
        type CacheContents;

        #[namespace = "ailoy"]
        pub fn create_cache() -> UniquePtr<CacheContents>;

        #[namespace = "ailoy"]
        #[cxx_name = "write_from_rs"]
        pub fn write(self: Pin<&mut CacheContents>, key: String, value: String) -> ();

        #[namespace = "ailoy"]
        #[cxx_name = "write_binary_from_rs"]
        pub fn write_binary(self: Pin<&mut CacheContents>, key: String, value: Vec<u8>) -> ();

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
            lib_filename: String,
            cache_contents: UniquePtr<CacheContents>,
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
}

unsafe impl Send for ffi::CacheContents {}

unsafe impl Send for ffi::DLDevice {}

unsafe impl Send for ffi::DLManagedTensorVersioned {}

unsafe impl Send for ffi::TVMLanguageModel {}

impl std::fmt::Debug for ffi::TVMLanguageModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TVMLanguageModel").finish()
    }
}
