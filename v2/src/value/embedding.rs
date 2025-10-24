use serde::{Deserialize, Serialize};

use crate::utils::Normalize;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Embedding(Vec<f32>);

impl From<Vec<f32>> for Embedding {
    fn from(value: Vec<f32>) -> Self {
        Self(value)
    }
}

impl Into<Vec<f32>> for Embedding {
    fn into(self) -> Vec<f32> {
        self.0
    }
}

impl std::ops::Mul for &Embedding {
    type Output = f32;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.len() != rhs.len() {
            panic!("Cannot dot-product two embeddings of different lengths");
        }

        self.0.iter().zip(rhs.0.iter()).map(|(x, y)| x * y).sum()
    }
}

impl std::ops::Mul for Embedding {
    type Output = f32;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl Normalize for Embedding {
    fn normalized(&self) -> Self {
        Self(self.0.normalized())
    }
}

impl Embedding {
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

#[cfg(feature = "python")]
mod py {
    use pyo3::{
        prelude::*,
        types::{PyFloat, PyList},
    };
    use pyo3_stub_gen::{PyStubType, TypeInfo};

    use super::*;

    impl<'py> IntoPyObject<'py> for Embedding {
        type Target = PyList;
        type Output = Bound<'py, PyList>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            PyList::new(py, self.0).into()
        }
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for Embedding {
        type Error = PyErr;

        fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            let vec: Vec<f32> = ob.extract()?;
            Ok(Embedding(vec))
        }
    }

    impl PyStubType for Embedding {
        fn type_output() -> pyo3_stub_gen::TypeInfo {
            TypeInfo::list_of::<PyFloat>()
        }
    }
}

#[cfg(feature = "nodejs")]
mod node {
    use napi::bindgen_prelude::*;
    use napi_derive::*;

    use super::*;

    #[allow(unused)]
    #[napi(js_name = "Embedding")]
    pub type JsEmbedding = Float32Array; // dummy type to generate type alias in d.ts

    impl FromNapiValue for Embedding {
        unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> Result<Self> {
            let array = unsafe { Float32Array::from_napi_value(env, napi_val) }?;
            Ok(Self(array.to_vec()))
        }
    }

    impl ToNapiValue for Embedding {
        unsafe fn to_napi_value(env: sys::napi_env, val: Self) -> Result<sys::napi_value> {
            let env = Env::from_raw(env);
            let array = Float32Array::new(val.0.clone());
            let unknown = array.into_unknown(&env)?;
            Ok(unknown.raw())
        }
    }

    impl TypeName for Embedding {
        fn type_name() -> &'static str {
            "Float32Array"
        }

        fn value_type() -> ValueType {
            ValueType::Object
        }
    }

    impl ValidateNapiValue for Embedding {}
}

#[cfg(feature = "wasm")]
mod wasm {
    use js_sys::Float32Array;
    use wasm_bindgen::{
        convert::{FromWasmAbi, IntoWasmAbi},
        describe::WasmDescribe,
        prelude::*,
    };

    use super::*;

    #[wasm_bindgen(typescript_custom_section)]
    const TS_APPEND_CONTENT: &'static str = dedent::dedent!(
        r#"
        type Embedding = Float32Array;
        "#
    );

    impl WasmDescribe for Embedding {
        fn describe() {
            Float32Array::describe()
        }
    }

    impl IntoWasmAbi for Embedding {
        type Abi = <Float32Array as IntoWasmAbi>::Abi;

        fn into_abi(self) -> Self::Abi {
            let array = Float32Array::new_with_length(self.0.len() as u32);
            array.copy_from(&self.0);
            array.into_abi()
        }
    }

    impl FromWasmAbi for Embedding {
        type Abi = <Float32Array as IntoWasmAbi>::Abi;

        unsafe fn from_abi(js: Self::Abi) -> Self {
            let array = unsafe { Float32Array::from_abi(js) };
            let vec = array.to_vec();
            Embedding(vec)
        }
    }

    impl From<JsValue> for Embedding {
        fn from(value: JsValue) -> Self {
            let array = Float32Array::from(value);
            let vec = array.to_vec();
            Self(vec)
        }
    }

    impl Into<JsValue> for Embedding {
        fn into(self) -> JsValue {
            let array = Float32Array::new_with_length(self.0.len() as u32);
            array.copy_from(&self.0);
            array.into()
        }
    }
}
