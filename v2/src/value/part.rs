use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};

use crate::value::{Value, delta::Delta};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "nodejs", napi_derive::napi(object))]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub struct PartFunction {
    pub name: String,
    pub arguments: Value,
}

#[derive(
    Clone, Debug, PartialEq, Eq, Serialize, Deserialize, strum::EnumString, strum::Display,
)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
#[cfg_attr(feature = "nodejs", napi_derive::napi(string_enum = "lowercase"))]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub enum PartImageColorspace {
    Grayscale,
    RGB,
    RGBA,
}

impl PartImageColorspace {
    pub fn channel(&self) -> u32 {
        match self {
            PartImageColorspace::Grayscale => 1,
            PartImageColorspace::RGB => 3,
            PartImageColorspace::RGBA => 4,
        }
    }
}

impl TryFrom<String> for PartImageColorspace {
    type Error = strum::ParseError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
#[cfg_attr(
    feature = "python",
    pyo3_stub_gen_derive::gen_stub_pyclass_complex_enum
)]
#[cfg_attr(feature = "python", pyo3::pyclass(eq))]
#[cfg_attr(feature = "nodejs", napi_derive::napi(discriminant_case = "lowercase"))]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub enum PartImage {
    Binary {
        height: u32,
        width: u32,
        colorspace: PartImageColorspace,
        #[cfg_attr(feature = "nodejs", napi_derive::napi(ts_type = "Buffer"))]
        data: super::bytes::Bytes,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
#[cfg_attr(
    feature = "python",
    pyo3_stub_gen_derive::gen_stub_pyclass_complex_enum
)]
#[cfg_attr(feature = "python", pyo3::pyclass(eq))]
#[cfg_attr(feature = "nodejs", napi_derive::napi(discriminant_case = "lowercase"))]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub enum Part {
    Text {
        text: String,
    },
    Function {
        id: Option<String>,
        function: PartFunction,
    },
    Value {
        value: Value,
    },
    Image {
        image: PartImage,
    },
}

impl Part {
    pub fn text(v: impl Into<String>) -> Self {
        Self::Text { text: v.into() }
    }

    pub fn image_binary(
        height: u32,
        width: u32,
        colorspace: impl TryInto<PartImageColorspace>,
        data: impl IntoIterator<Item = u8>,
    ) -> anyhow::Result<Self> {
        let colorspace: PartImageColorspace = colorspace
            .try_into()
            .ok()
            .context("Colorspace parsing failed")?;
        let data = data.into_iter().collect::<Vec<_>>();
        let nbytes = data.len() as u32 / height / width / colorspace.channel();
        if !(nbytes == 1 || nbytes == 2 || nbytes == 3 || nbytes == 4) {
            panic!("Invalid data length");
        }
        Ok(Self::Image {
            image: PartImage::Binary {
                height: height as u32,
                width: width as u32,
                colorspace,
                data: super::bytes::Bytes(data.into()),
            },
        })
    }

    pub fn function(name: impl Into<String>, args: impl Into<Value>) -> Self {
        Self::Function {
            id: None,
            function: PartFunction {
                name: name.into(),
                arguments: args.into(),
            },
        }
    }

    pub fn function_with_id(
        id: impl Into<String>,
        name: impl Into<String>,
        args: impl Into<Value>,
    ) -> Self {
        Self::Function {
            id: Some(id.into()),
            function: PartFunction {
                name: name.into(),
                arguments: args.into(),
            },
        }
    }

    pub fn is_text(&self) -> bool {
        match self {
            Self::Text { .. } => true,
            _ => false,
        }
    }

    pub fn is_function(&self) -> bool {
        match self {
            Self::Function { .. } => true,
            _ => false,
        }
    }

    pub fn is_value(&self) -> bool {
        match self {
            Self::Value { .. } => true,
            _ => false,
        }
    }

    pub fn is_image(&self) -> bool {
        match self {
            Self::Image { .. } => true,
            _ => false,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text } => Some(text.as_str()),
            _ => None,
        }
    }

    pub fn as_text_mut(&mut self) -> Option<&mut String> {
        match self {
            Self::Text { text } => Some(text),
            _ => None,
        }
    }

    pub fn as_function(&self) -> Option<(Option<&str>, &str, &Value)> {
        match self {
            Self::Function {
                id,
                function: PartFunction { name, arguments },
            } => Some((id.as_deref(), name.as_str(), arguments)),
            _ => None,
        }
    }

    pub fn as_function_mut(&mut self) -> Option<(Option<&mut String>, &mut String, &mut Value)> {
        match self {
            Self::Function {
                id,
                function: PartFunction { name, arguments },
            } => Some((id.as_mut(), name, arguments)),
            _ => None,
        }
    }

    pub fn as_value(&self) -> Option<&Value> {
        match self {
            Self::Value { value } => Some(value),
            _ => None,
        }
    }

    pub fn as_value_mut(&mut self) -> Option<&mut Value> {
        match self {
            Self::Value { value } => Some(value),
            _ => None,
        }
    }

    pub fn as_image(&self) -> Option<image::DynamicImage> {
        fn bytes_to_u16_ne(b: &[u8]) -> Option<Vec<u16>> {
            if b.len() % 2 != 0 {
                return None;
            }
            let mut v = Vec::with_capacity(b.len() / 2);
            for ch in b.chunks_exact(2) {
                v.push(u16::from_ne_bytes([ch[0], ch[1]]));
            }
            Some(v)
        }

        match self {
            Self::Image {
                image:
                    PartImage::Binary {
                        height,
                        width,
                        colorspace,
                        data: super::bytes::Bytes(buf),
                    },
            } => {
                let (h, w) = (*height as u32, *width as u32);
                let nbytes = buf.len() as u32 / h / w / colorspace.channel();
                match (colorspace, nbytes) {
                    // Grayscale 8-bit
                    (&PartImageColorspace::Grayscale, 1) => {
                        let buf = image::GrayImage::from_raw(w, h, buf.to_vec())?;
                        Some(image::DynamicImage::ImageLuma8(buf))
                    }
                    // Grayscale 16-bit
                    (&PartImageColorspace::Grayscale, 2) => {
                        let buf = image::ImageBuffer::<image::Luma<u16>, _>::from_raw(
                            w,
                            h,
                            bytes_to_u16_ne(buf)?,
                        )?;
                        Some(image::DynamicImage::ImageLuma16(buf))
                    }
                    // RGB 8-bit
                    (&PartImageColorspace::RGB, 1) => {
                        let buf = image::RgbImage::from_raw(w, h, buf.to_vec())?;
                        Some(image::DynamicImage::ImageRgb8(buf))
                    }
                    // RGBA 8-bit
                    (&PartImageColorspace::RGBA, 1) => {
                        let buf = image::RgbaImage::from_raw(w, h, buf.to_vec())?;
                        Some(image::DynamicImage::ImageRgba8(buf))
                    }
                    // RGB 16-bit
                    (&PartImageColorspace::RGB, 2) => {
                        let buf = image::ImageBuffer::<image::Rgb<u16>, _>::from_raw(
                            w,
                            h,
                            bytes_to_u16_ne(buf)?,
                        )?;
                        Some(image::DynamicImage::ImageRgb16(buf))
                    }
                    // RGBA 16-bit
                    (&PartImageColorspace::RGBA, 2) => {
                        let buf = image::ImageBuffer::<image::Rgba<u16>, _>::from_raw(
                            w,
                            h,
                            bytes_to_u16_ne(buf)?,
                        )?;
                        Some(image::DynamicImage::ImageRgba16(buf))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(
    feature = "python",
    pyo3_stub_gen_derive::gen_stub_pyclass_complex_enum
)]
#[cfg_attr(feature = "python", pyo3::pyclass(eq))]
#[cfg_attr(
    feature = "nodejs",
    napi_derive::napi(discriminant_case = "snake_case")
)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub enum PartDeltaFunction {
    Verbatim { text: String },
    WithStringArgs { name: String, arguments: String },
    WithParsedArgs { name: String, arguments: Value },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(
    feature = "python",
    pyo3_stub_gen_derive::gen_stub_pyclass_complex_enum
)]
#[cfg_attr(feature = "python", pyo3::pyclass(eq))]
#[cfg_attr(
    feature = "nodejs",
    napi_derive::napi(discriminant_case = "snake_case")
)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub enum PartDelta {
    Text {
        text: String,
    },
    Function {
        id: Option<String>,
        function: PartDeltaFunction,
    },
    Value {
        value: Value,
    },
    Null {},
}

impl PartDelta {
    pub fn is_text(&self) -> bool {
        match self {
            Self::Text { .. } => true,
            _ => false,
        }
    }
    pub fn is_verbatim_function(&self) -> bool {
        match self {
            Self::Function {
                function: PartDeltaFunction::Verbatim { .. },
                ..
            } => true,
            _ => false,
        }
    }

    pub fn is_function(&self) -> bool {
        match self {
            Self::Function {
                function: PartDeltaFunction::WithStringArgs { .. },
                ..
            } => true,
            _ => false,
        }
    }

    pub fn is_parsed_function(&self) -> bool {
        match self {
            Self::Function {
                function: PartDeltaFunction::WithParsedArgs { .. },
                ..
            } => true,
            _ => false,
        }
    }

    pub fn is_value(&self) -> bool {
        match self {
            Self::Value { .. } => true,
            _ => false,
        }
    }

    pub fn to_text(self) -> Option<String> {
        match self {
            Self::Text { text } => Some(text),
            Self::Function {
                function: PartDeltaFunction::Verbatim { text },
                ..
            } => Some(text),
            _ => None,
        }
    }

    pub fn to_function(self) -> Option<(Option<String>, String, String)> {
        match self {
            Self::Function {
                id,
                function: PartDeltaFunction::WithStringArgs { name, arguments },
            } => Some((id, name, arguments)),
            _ => None,
        }
    }

    pub fn to_parsed_function(self) -> Option<(Option<String>, String, Value)> {
        match self {
            Self::Function {
                id,
                function: PartDeltaFunction::WithParsedArgs { name, arguments },
            } => Some((id, name, arguments)),
            _ => None,
        }
    }

    pub fn to_value(self) -> Option<Value> {
        match self {
            Self::Value { value } => Some(value),
            _ => None,
        }
    }
}

impl Default for PartDelta {
    fn default() -> Self {
        Self::Null {}
    }
}

impl Delta for PartDelta {
    type Item = Part;
    type Err = anyhow::Error; // TODO: Define custom error for this.

    fn aggregate(self, other: Self) -> anyhow::Result<Self> {
        match (self, other) {
            (PartDelta::Null {}, other) => Ok(other),
            (PartDelta::Text { text: mut t1 }, PartDelta::Text { text: t2 }) => {
                t1.push_str(&t2);
                Ok(PartDelta::Text { text: t1 })
            }
            (
                PartDelta::Function {
                    id: id1,
                    function: f1,
                },
                PartDelta::Function {
                    id: id2,
                    function: f2,
                },
            ) => {
                let id = match (id1, id2) {
                    (Some(id1), Some(id2)) => {
                        if id1 != id2 {
                            bail!(
                                "Cannot aggregate two functions with different ids. ({} != {}).",
                                id1,
                                id2
                            )
                        }
                        Some(id1)
                    }
                    (None, Some(id2)) => Some(id2),
                    (Some(id1), None) => Some(id1),
                    (None, None) => None,
                };
                let f = match (f1, f2) {
                    (
                        PartDeltaFunction::Verbatim { text: mut t1 },
                        PartDeltaFunction::Verbatim { text: t2 },
                    ) => {
                        t1.push_str(&t2);
                        PartDeltaFunction::Verbatim { text: t1 }
                    }
                    (
                        PartDeltaFunction::WithStringArgs {
                            name: mut n1,
                            arguments: mut a1,
                        },
                        PartDeltaFunction::WithStringArgs {
                            name: n2,
                            arguments: a2,
                        },
                    ) => {
                        n1.push_str(&n2);
                        a1.push_str(&a2);
                        PartDeltaFunction::WithStringArgs {
                            name: n1,
                            arguments: a1,
                        }
                    }
                    (
                        PartDeltaFunction::WithParsedArgs {
                            name: mut n1,
                            arguments: _,
                        },
                        PartDeltaFunction::WithParsedArgs {
                            name: n2,
                            arguments: a2,
                        },
                    ) => {
                        // @jhlee: Rather than just replacing, merge logic could be helpful
                        n1.push_str(&n2);
                        PartDeltaFunction::WithParsedArgs {
                            name: n1,
                            arguments: a2,
                        }
                    }
                    (f1, f2) => bail!(
                        "Aggregation between those two function delta {:?}, {:?} is not defined.",
                        f1,
                        f2
                    ),
                };
                Ok(PartDelta::Function { id, function: f })
            }
            (pd1, pd2) => {
                bail!(
                    "Aggregation between those two part delta {:?}, {:?} is not defined.",
                    pd1,
                    pd2
                )
            }
        }
    }

    fn finish(self) -> anyhow::Result<Self::Item> {
        match self {
            PartDelta::Null {} => Ok(Part::Text {
                text: String::new(),
            }),
            PartDelta::Text { text } => Ok(Part::Text { text }),
            PartDelta::Function { id, function } => {
                let function = match function {
                    // Try json deserialization if verbatim
                    PartDeltaFunction::Verbatim { text } => {
                        match serde_json::from_str::<Value>(&text) {
                            Ok(root) => {
                                match (root.pointer_as::<str>("/name"), root.pointer("/arguments"))
                                {
                                    (Some(name), Some(args)) => PartFunction {
                                        name: name.to_owned(),
                                        arguments: args.to_owned(),
                                    },
                                    _ => bail!("Invalid function JSON"),
                                }
                            }
                            Err(_) => bail!("Invalid JSON"),
                        }
                    }
                    // Try json deserialization for args
                    PartDeltaFunction::WithStringArgs { name, arguments } => {
                        let arguments =
                            serde_json::from_str::<Value>(&arguments).context("Invalid JSON")?;
                        PartFunction { name, arguments }
                    }
                    // As-is
                    PartDeltaFunction::WithParsedArgs { name, arguments } => {
                        PartFunction { name, arguments }
                    }
                };
                Ok(Part::Function { id, function })
            }
            PartDelta::Value { value } => Ok(Part::Value { value }),
        }
    }
}

#[cfg(feature = "python")]
mod py {
    use pyo3::{
        Bound, FromPyObject, IntoPyObject, PyAny, PyErr, PyResult, Python,
        exceptions::{PyTypeError, PyValueError},
        types::{PyAnyMethods, PyDict, PyString},
    };
    use pyo3_stub_gen::{PyStubType, TypeInfo};

    use super::*;

    impl<'py> FromPyObject<'py> for PartFunction {
        fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
            if let Ok(pydict) = ob.downcast::<PyDict>() {
                let name_any = pydict.get_item("name")?;
                let name: String = name_any.extract()?;
                let args_any = pydict.get_item("args")?;
                let arguments: Value = args_any.extract()?;
                Ok(Self { name, arguments })
            } else {
                Err(PyTypeError::new_err(
                    "PartFunction must be a dict with keys 'name' and 'args'",
                ))
            }
        }
    }

    impl<'py> IntoPyObject<'py> for PartFunction {
        type Target = PyDict;
        type Output = Bound<'py, PyDict>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            let d = PyDict::new(py);
            d.set_item("name", self.name)?;
            let py_args = self.arguments.into_pyobject(py)?;
            d.set_item("args", py_args)?;
            Ok(d)
        }
    }

    impl PyStubType for PartFunction {
        fn type_output() -> TypeInfo {
            let TypeInfo {
                name: value_name,
                import: mut imports,
            } = Value::type_output();
            imports.insert("builtins".into());
            imports.insert("typing".into());

            TypeInfo {
                name: format!(
                    "dict[typing.Literal[\"name\", \"args\"], typing.Union[str, {}]]",
                    value_name
                ),
                import: imports,
            }
        }
    }

    impl<'py> IntoPyObject<'py> for PartImageColorspace {
        type Target = PyString;
        type Output = Bound<'py, PyString>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            Ok(PyString::new(py, &self.to_string()))
        }
    }

    impl<'py> FromPyObject<'py> for PartImageColorspace {
        fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
            let s: &str = ob.extract()?;
            s.parse::<PartImageColorspace>()
                .map_err(|_| PyValueError::new_err(format!("Invalid colorspace: {s}")))
        }
    }

    impl PyStubType for PartImageColorspace {
        fn type_output() -> TypeInfo {
            let mut import = std::collections::HashSet::new();
            import.insert("typing".into());

            TypeInfo {
                name: r#"typing.Literal["grayscale", "rgb", "rgba"]"#.into(),
                import,
            }
        }
    }
}
