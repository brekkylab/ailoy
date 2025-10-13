use base64::Engine as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bytes(pub Vec<u8>);

impl Serialize for Bytes {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if serializer.is_human_readable() {
            let b64 = base64::engine::general_purpose::STANDARD.encode(&self.0);
            serializer.serialize_str(&b64)
        } else {
            serializer.serialize_bytes(&self.0)
        }
    }
}

impl<'de> Deserialize<'de> for Bytes {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            let s = String::deserialize(deserializer)?;
            let data = base64::engine::general_purpose::STANDARD
                .decode(s.as_bytes())
                .map_err(D::Error::custom)?;
            Ok(Bytes(data))
        } else {
            let b = <Vec<u8>>::deserialize(deserializer)?;
            Ok(Bytes(b))
        }
    }
}

#[cfg(feature = "python")]
mod py {
    use pyo3::{
        Bound, FromPyObject, PyAny, PyResult,
        exceptions::PyTypeError,
        types::{PyAnyMethods, PyBytes, PyBytesMethods as _},
    };
    #[cfg(feature = "python")]
    use pyo3::{IntoPyObject, PyErr};
    use pyo3_stub_gen::{PyStubType, TypeInfo};

    use super::*;

    impl<'py> FromPyObject<'py> for Bytes {
        fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
            if let Ok(pybytes) = ob.downcast::<PyBytes>() {
                Ok(Bytes(pybytes.as_bytes().to_vec()))
            } else {
                Err(PyTypeError::new_err("Expected a bytes object"))
            }
        }
    }

    #[cfg(feature = "python")]
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

#[cfg(feature = "wasm")]
mod wasm {
    use js_sys::{ArrayBuffer, Uint8Array};
    use wasm_bindgen::{
        convert::{FromWasmAbi, IntoWasmAbi, OptionFromWasmAbi, OptionIntoWasmAbi},
        describe::WasmDescribe,
        prelude::*,
    };

    use super::Bytes;

    impl WasmDescribe for Bytes {
        fn describe() {
            JsValue::describe()
        }
    }

    impl FromWasmAbi for Bytes {
        type Abi = <JsValue as FromWasmAbi>::Abi;

        #[inline]
        unsafe fn from_abi(js: Self::Abi) -> Self {
            let bytes = unsafe { JsValue::from_abi(js) };
            Bytes::try_from(bytes).unwrap()
        }
    }

    impl IntoWasmAbi for Bytes {
        type Abi = <JsValue as IntoWasmAbi>::Abi;

        #[inline]
        fn into_abi(self) -> Self::Abi {
            let js_value = JsValue::from(self);
            js_value.into_abi()
        }
    }

    impl OptionFromWasmAbi for Bytes {
        #[inline]
        fn is_none(js: &Self::Abi) -> bool {
            let js_value = unsafe { JsValue::from_abi(*js) };
            let is_none = js_value.is_null() || js_value.is_undefined();
            std::mem::forget(js_value);
            is_none
        }
    }

    impl OptionIntoWasmAbi for Bytes {
        #[inline]
        fn none() -> Self::Abi {
            JsValue::NULL.into_abi()
        }
    }

    impl TryFrom<JsValue> for Bytes {
        type Error = js_sys::Error;

        fn try_from(js_val: JsValue) -> Result<Self, Self::Error> {
            // Try to handle Uint8Array
            if let Some(uint8_array) = js_val.dyn_ref::<Uint8Array>() {
                let len = uint8_array.length() as usize;
                let mut vec = vec![0u8; len];
                uint8_array.copy_to(&mut vec);
                return Ok(Bytes(vec));
            }

            // Try to handle ArrayBuffer
            if let Some(array_buffer) = js_val.dyn_ref::<ArrayBuffer>() {
                let uint8_array = Uint8Array::new(array_buffer);
                let len = uint8_array.length() as usize;
                let mut vec = vec![0u8; len];
                uint8_array.copy_to(&mut vec);
                return Ok(Bytes(vec));
            }

            // Try to handle Array of numbers
            if js_sys::Array::is_array(&js_val) {
                let arr = js_sys::Array::from(&js_val);
                let len = arr.length() as usize;
                let mut vec = Vec::with_capacity(len);

                for i in 0..len {
                    if let Some(num) = arr.get(i as u32).as_f64() {
                        vec.push(num as u8);
                    }
                }

                return Ok(Bytes(vec));
            }

            Err(js_sys::Error::new("Cannot convert to Bytes"))
        }
    }

    impl From<Bytes> for JsValue {
        fn from(bytes: Bytes) -> Self {
            let uint8_array = Uint8Array::new_with_length(bytes.0.len() as u32);
            uint8_array.copy_from(&bytes.0);
            uint8_array.into()
        }
    }
}
