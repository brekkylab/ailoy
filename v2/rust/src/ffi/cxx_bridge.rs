use cxx::{CxxString, UniquePtr};

#[cxx::bridge]
mod ffi {

    unsafe extern "C++" {
        include!("cache.hpp");

        #[namespace = "ailoy"]
        #[cxx_name = "cache_t"]
        type Cache;

        #[namespace = "ailoy"]
        fn create_cache() -> UniquePtr<Cache>;

        #[namespace = "ailoy"]
        #[cxx_name = "write_from_rs"]
        fn write(self: Pin<&mut Cache>, key: String, value: String) -> ();

        #[namespace = "ailoy"]
        #[cxx_name = "write_binary_from_rs"]
        fn write_binary(self: Pin<&mut Cache>, key: String, value: Vec<u8>) -> ();
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
        // let_cxx_string!(lib_filename = "lib.dylib");
        // let device = create_dldevice(15, 0);
        // let cpp_class = super::ffi::create_tvm_language_model(&lib_filename, device);
        // println!("{}", cpp_class.get_result());
    }
}
