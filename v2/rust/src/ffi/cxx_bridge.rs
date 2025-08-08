use cxx::{CxxString, UniquePtr};

#[cxx::bridge]
pub mod ffi {

    unsafe extern "C++" {
        include!("cache.hpp");

        #[namespace = "ailoy"]
        #[cxx_name = "cache_t"]
        type Cache;

        #[namespace = "ailoy"]
        pub fn create_cache() -> UniquePtr<Cache>;

        #[namespace = "ailoy"]
        #[cxx_name = "write_from_rs"]
        pub fn write(self: Pin<&mut Cache>, key: String, value: String) -> ();

        #[namespace = "ailoy"]
        #[cxx_name = "write_binary_from_rs"]
        pub fn write_binary(self: Pin<&mut Cache>, key: String, value: Vec<u8>) -> ();

        include!("language_model.hpp");

        type DLDevice;

        #[namespace = "ailoy"]
        pub fn create_dldevice(device_type: i32, device_id: i32) -> UniquePtr<DLDevice>;

        #[namespace = "ailoy"]
        #[cxx_name = "tvm_language_model_t"]
        type TVMLanguageModel;

        #[namespace = "ailoy"]
        pub fn create_tvm_language_model(
            lib_filename: String,
            cache: UniquePtr<Cache>,
            device: UniquePtr<DLDevice>,
        ) -> UniquePtr<TVMLanguageModel>;
    }
}

pub use ffi::*;

impl std::fmt::Debug for TVMLanguageModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TVMLanguageModel").finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test1() {
        let mut cache = ffi::create_cache();
        cache.pin_mut().write("key".to_owned(), "value".to_owned());
        cache.pin_mut().write_binary("key".to_owned(), Vec::new());
        let device = ffi::create_dldevice(8, 0);
        let mut lang_model = ffi::create_tvm_language_model(
            "/Users/ijaehwan/.cache/ailoy/Qwen--Qwen3-0.6B--aarch64-apple-darwin--metal/rt.dylib"
                .to_owned(),
            cache,
            device,
        );
        // let_cxx_string!(lib_filename = "lib.dylib");
        // let device = create_dldevice(15, 0);
        // let cpp_class = super::ffi::create_tvm_language_model(&lib_filename, device);
        // println!("{}", cpp_class.get_result());
    }
}
