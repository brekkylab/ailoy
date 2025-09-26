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
            todo!()
        }
    }

    impl PyStubType for Bytes {
        fn type_output() -> TypeInfo {
            TypeInfo::any()
        }
    }
}
