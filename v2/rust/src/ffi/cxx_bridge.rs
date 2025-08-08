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
        fn write(self: Pin<&mut Cache>, key: &CxxString, value: String) -> ();

        #[namespace = "ailoy"]
        #[cxx_name = "write_binary_from_rs"]
        fn write_binary(self: Pin<&mut Cache>, key: &CxxString, value: Vec<u8>) -> ();
    }
}

pub struct Cache {
    inner: UniquePtr<ffi::Cache>,
}

impl Cache {
    pub fn new() -> Self {
        Cache {
            inner: ffi::create_cache(),
        }
    }

    pub fn write(&mut self, key: &str, value: String) {
        // let value = ffi::make_cxx_string(String::from_utf8(value).unwrap().as_str());
        // self.inner.pin_mut().write(key, value);
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use cxx::let_cxx_string;

    use super::*;

    #[test]
    fn test1() {
        let_cxx_string!(lib_filename = "lib.dylib");
        // let device = create_dldevice(15, 0);
        // let cpp_class = super::ffi::create_tvm_language_model(&lib_filename, device);
        // println!("{}", cpp_class.get_result());
    }
}
