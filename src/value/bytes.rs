use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;

use crate::utils::Ellipsis;

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(from_wasm_abi, into_wasm_abi))]
pub struct Bytes(pub ByteBuf);

impl std::fmt::Debug for Bytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const MAX_LEN: usize = 64;
        let bytes_str = format!("{:?}", self.0);
        write!(
            f,
            "Bytes({} ({} bytes total))",
            bytes_str.truncate_ellipsis(MAX_LEN),
            self.0.len()
        )
    }
}

#[cfg(feature = "python")]
mod py {
    use pyo3::{
        Borrowed, Bound, FromPyObject, IntoPyObject, PyAny, PyErr, PyResult,
        exceptions::PyTypeError,
        types::{PyBytes, PyBytesMethods as _},
    };
    use pyo3_stub_gen::{PyStubType, TypeInfo};

    use super::*;

    impl<'a, 'py> FromPyObject<'a, 'py> for Bytes {
        type Error = PyErr;

        fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            if let Ok(pybytes) = &ob.cast::<PyBytes>() {
                Ok(Bytes(pybytes.as_bytes().to_vec().into()))
            } else {
                Err(PyTypeError::new_err("Expected a bytes object"))
            }
        }
    }

    impl<'py> IntoPyObject<'py> for Bytes {
        type Target = PyBytes;
        type Output = Bound<'py, PyBytes>;
        type Error = PyErr;

        fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<Self::Output, Self::Error> {
            Ok(PyBytes::new(py, &self.0))
        }
    }

    impl PyStubType for Bytes {
        fn type_output() -> TypeInfo {
            // @jhlee: Add proper stub
            TypeInfo::any()
        }
    }
}

#[cfg(feature = "nodejs")]
mod node {
    use napi::{Env, ValueType, bindgen_prelude::*};
    use napi_derive::napi;

    use super::Bytes;

    #[allow(unused)]
    #[napi(js_name = "Bytes")]
    pub type JsBytes = Buffer; // dummy type to generate type alias in d.ts

    unsafe fn read_buffer(env: sys::napi_env, val: sys::napi_value) -> Result<Vec<u8>> {
        let mut data_ptr: *mut core::ffi::c_void = std::ptr::null_mut();
        let mut len: usize = 0;
        let status = unsafe { sys::napi_get_buffer_info(env, val, &mut data_ptr, &mut len) };
        if status == sys::Status::napi_ok {
            let slice = unsafe { std::slice::from_raw_parts(data_ptr as *const u8, len) };
            return Ok(slice.to_vec());
        }
        Err(Error::new(Status::InvalidArg, "Not a Buffer"))
    }

    impl FromNapiValue for Bytes {
        unsafe fn from_napi_value(env: sys::napi_env, val: sys::napi_value) -> Result<Self> {
            let env = Env::from_raw(env);
            if let Ok(data) = unsafe { read_buffer(env.raw(), val) } {
                return Ok(Bytes(data.into()));
            }
            Err(Error::new(
                Status::InvalidArg,
                "Expected Buffer".to_string(),
            ))
        }
    }

    impl ToNapiValue for Bytes {
        unsafe fn to_napi_value(env: sys::napi_env, this: Self) -> Result<sys::napi_value> {
            use std::{mem::ManuallyDrop, os::raw::c_void, ptr};

            // Zero-copy: use napi_create_external_buffer
            extern "C" fn finalize_vec(
                _env: sys::napi_env,
                _finalize_data: *mut c_void, // = data ptr (not used)
                finalize_hint: *mut c_void,  // our (ptr, len, cap) tuple
            ) {
                unsafe {
                    // Recover the Box -> get (ptr,len,cap) -> rebuild Vec -> drop
                    let tuple = Box::from_raw(finalize_hint as *mut (*mut u8, usize, usize));
                    let (ptr, len, cap) = *tuple;
                    let _ = Vec::from_raw_parts(ptr, len, cap);
                }
            }

            // Transfer ownership of Vec<u8> to JS
            let mut vec = this.0;
            let ptr_u8 = vec.as_mut_ptr();
            let len = vec.len();
            let cap = vec.capacity();

            // Save (ptr,len,cap) as hint, to reconstruct Vec in finalizer
            let hint = Box::into_raw(Box::new((ptr_u8, len, cap))) as *mut c_void;

            // Prevent Rust from dropping vec
            let _leak = ManuallyDrop::new(vec);

            let mut out: sys::napi_value = ptr::null_mut();
            let status = unsafe {
                sys::napi_create_external_buffer(
                    env,
                    len,                   // size in bytes
                    ptr_u8 as *mut c_void, // data pointer
                    Some(finalize_vec),    // finalizer
                    hint,                  // hint (tuple)
                    &mut out,
                )
            };

            if status == sys::Status::napi_ok {
                Ok(out)
            } else {
                Err(Error::new(
                    Status::GenericFailure,
                    "Failed to create external Buffer",
                ))
            }
        }
    }

    impl TypeName for Bytes {
        fn type_name() -> &'static str {
            "Buffer"
        }

        fn value_type() -> ValueType {
            ValueType::Object
        }
    }

    impl ValidateNapiValue for Bytes {
        unsafe fn validate(
            env: sys::napi_env,
            napi_val: sys::napi_value,
        ) -> Result<sys::napi_value> {
            unsafe { <Buffer as ValidateNapiValue>::validate(env, napi_val) }
        }
    }
}
