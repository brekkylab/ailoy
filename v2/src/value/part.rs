use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};

use crate::value::{Value, delta::Delta};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "nodejs", napi_derive::napi(object))]
#[cfg_attr(
    feature = "wasm",
    wasm_bindgen::prelude::wasm_bindgen(getter_with_clone)
)]
pub struct PartFunction {
    pub name: String,
    #[serde(rename = "arguments")]
    pub args: Value,
}

#[derive(
    Clone, Debug, PartialEq, Eq, Serialize, Deserialize, strum::EnumString, strum::Display,
)]
#[serde(untagged)]
#[cfg_attr(feature = "nodejs", napi_derive::napi(string_enum))]
#[cfg_attr(feature = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
pub enum PartImageColorspace {
    #[strum(serialize = "grayscale")]
    #[serde(rename = "grayscale")]
    Grayscale,
    #[strum(serialize = "rgb")]
    #[serde(rename = "rgb")]
    RGB,
    #[strum(serialize = "rgba")]
    #[serde(rename = "rgba")]
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
#[serde(tag = "media-type")]
#[cfg_attr(
    feature = "python",
    pyo3_stub_gen_derive::gen_stub_pyclass_complex_enum
)]
#[cfg_attr(feature = "python", pyo3::pyclass(eq))]
#[cfg_attr(feature = "nodejs", napi_derive::napi)]
pub enum PartImage {
    #[serde(rename = "image/x-binary")]
    Binary {
        #[serde(rename = "height")]
        h: u32,
        #[serde(rename = "width")]
        w: u32,
        #[serde(rename = "colorspace")]
        c: PartImageColorspace,
        #[cfg_attr(feature = "nodejs", napi_derive::napi(ts_type = "Buffer"))]
        data: super::bytes::Bytes,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
#[cfg_attr(
    feature = "python",
    pyo3_stub_gen_derive::gen_stub_pyclass_complex_enum
)]
#[cfg_attr(feature = "python", pyo3::pyclass(eq))]
#[cfg_attr(feature = "nodejs", napi_derive::napi)]
pub enum Part {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "function")]
    Function {
        id: Option<String>,
        #[serde(rename = "function")]
        f: PartFunction,
    },
    #[serde(rename = "value")]
    Value { value: Value },
    #[serde(rename = "image")]
    Image { image: PartImage },
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
                h: height as u32,
                w: width as u32,
                c: colorspace,
                data: super::bytes::Bytes(data),
            },
        })
    }

    pub fn function(name: impl Into<String>, args: impl Into<Value>) -> Self {
        Self::Function {
            id: None,
            f: PartFunction {
                name: name.into(),
                args: args.into(),
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
            f: PartFunction {
                name: name.into(),
                args: args.into(),
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
                f: PartFunction { name, args },
            } => Some((id.as_deref(), name.as_str(), args)),
            _ => None,
        }
    }

    pub fn as_function_mut(&mut self) -> Option<(Option<&mut String>, &mut String, &mut Value)> {
        match self {
            Self::Function {
                id,
                f: PartFunction { name, args },
            } => Some((id.as_mut(), name, args)),
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
                        h,
                        w,
                        c,
                        data: super::bytes::Bytes(buf),
                    },
            } => {
                let (h, w) = (*h as u32, *w as u32);
                let nbytes = buf.len() as u32 / h / w / c.channel();
                match (c, nbytes) {
                    // Grayscale 8-bit
                    (&PartImageColorspace::Grayscale, 1) => {
                        let buf = image::GrayImage::from_raw(w, h, buf.clone())?;
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
                        let buf = image::RgbImage::from_raw(w, h, buf.clone())?;
                        Some(image::DynamicImage::ImageRgb8(buf))
                    }
                    // RGBA 8-bit
                    (&PartImageColorspace::RGBA, 1) => {
                        let buf = image::RgbaImage::from_raw(w, h, buf.clone())?;
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
#[serde(untagged)]
#[cfg_attr(
    feature = "python",
    pyo3_stub_gen_derive::gen_stub_pyclass_complex_enum
)]
#[cfg_attr(feature = "python", pyo3::pyclass(eq))]
#[cfg_attr(feature = "nodejs", napi_derive::napi)]
pub enum PartDeltaFunction {
    Verbatim(String),
    WithStringArgs {
        name: String,
        #[serde(rename = "arguments")]
        args: String,
    },
    WithParsedArgs {
        name: String,
        #[serde(rename = "arguments")]
        args: Value,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
#[cfg_attr(
    feature = "python",
    pyo3_stub_gen_derive::gen_stub_pyclass_complex_enum
)]
#[cfg_attr(feature = "python", pyo3::pyclass(eq))]
#[cfg_attr(feature = "nodejs", napi_derive::napi)]
pub enum PartDelta {
    Text {
        text: String,
    },
    Function {
        id: Option<String>,
        #[serde(rename = "function")]
        f: PartDeltaFunction,
    },
    Value {
        value: Value,
    },
    Null(),
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
                f: PartDeltaFunction::Verbatim(..),
                ..
            } => true,
            _ => false,
        }
    }

    pub fn is_function(&self) -> bool {
        match self {
            Self::Function {
                f: PartDeltaFunction::WithStringArgs { .. },
                ..
            } => true,
            _ => false,
        }
    }

    pub fn is_parsed_function(&self) -> bool {
        match self {
            Self::Function {
                f: PartDeltaFunction::WithParsedArgs { .. },
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
                f: PartDeltaFunction::Verbatim(text),
                ..
            } => Some(text),
            _ => None,
        }
    }

    pub fn to_function(self) -> Option<(Option<String>, String, String)> {
        match self {
            Self::Function {
                id,
                f: PartDeltaFunction::WithStringArgs { name, args },
            } => Some((id, name, args)),
            _ => None,
        }
    }

    pub fn to_parsed_function(self) -> Option<(Option<String>, String, Value)> {
        match self {
            Self::Function {
                id,
                f: PartDeltaFunction::WithParsedArgs { name, args },
            } => Some((id, name, args)),
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
        Self::Null()
    }
}

impl Delta for PartDelta {
    type Item = Part;
    type Err = anyhow::Error; // TODO: Define custom error for this.

    fn aggregate(self, other: Self) -> anyhow::Result<Self> {
        match (self, other) {
            (PartDelta::Null(), other) => Ok(other),
            (PartDelta::Text { text: mut t1 }, PartDelta::Text { text: t2 }) => {
                t1.push_str(&t2);
                Ok(PartDelta::Text { text: t1 })
            }
            (PartDelta::Function { id: id1, f: f1 }, PartDelta::Function { id: id2, f: f2 }) => {
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
                    (PartDeltaFunction::Verbatim(mut t1), PartDeltaFunction::Verbatim(t2)) => {
                        t1.push_str(&t2);
                        PartDeltaFunction::Verbatim(t1)
                    }
                    (
                        PartDeltaFunction::WithStringArgs {
                            name: mut n1,
                            args: mut a1,
                        },
                        PartDeltaFunction::WithStringArgs { name: n2, args: a2 },
                    ) => {
                        n1.push_str(&n2);
                        a1.push_str(&a2);
                        PartDeltaFunction::WithStringArgs { name: n1, args: a1 }
                    }
                    (
                        PartDeltaFunction::WithParsedArgs {
                            name: mut n1,
                            args: _,
                        },
                        PartDeltaFunction::WithParsedArgs { name: n2, args: a2 },
                    ) => {
                        // @jhlee: Rather than just replacing, merge logic could be helpful
                        n1.push_str(&n2);
                        PartDeltaFunction::WithParsedArgs { name: n1, args: a2 }
                    }
                    (f1, f2) => bail!(
                        "Aggregation between those two function delta {:?}, {:?} is not defined.",
                        f1,
                        f2
                    ),
                };
                Ok(PartDelta::Function { id, f })
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
            PartDelta::Null() => Ok(Part::Text {
                text: String::new(),
            }),
            PartDelta::Text { text } => Ok(Part::Text { text }),
            PartDelta::Function { id, f } => {
                let f = match f {
                    // Try json deserialization if verbatim
                    PartDeltaFunction::Verbatim(text) => match serde_json::from_str::<Value>(&text)
                    {
                        Ok(root) => {
                            match (root.pointer_as::<str>("/name"), root.pointer("/arguments")) {
                                (Some(name), Some(args)) => PartFunction {
                                    name: name.to_owned(),
                                    args: args.to_owned(),
                                },
                                _ => bail!("Invalid function JSON"),
                            }
                        }
                        Err(_) => bail!("Invalid JSON"),
                    },
                    // Try json deserialization for args
                    PartDeltaFunction::WithStringArgs { name, args } => {
                        let args = serde_json::from_str::<Value>(&args).context("Invalid JSON")?;
                        PartFunction { name, args }
                    }
                    // As-is
                    PartDeltaFunction::WithParsedArgs { name, args } => PartFunction { name, args },
                };
                Ok(Part::Function { id, f })
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
        fn extract_bound(ob: &Bound<'py, PyAny>) -> anyhow::Result<Self> {
            if let Ok(pydict) = ob.downcast::<PyDict>() {
                let name_any = pydict.get_item("name")?;
                let name: String = name_any.extract()?;
                let args_any = pydict.get_item("args")?;
                let args: Value = args_any.extract()?;
                Ok(Self { name, args })
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
            let py_args = self.args.into_pyobject(py)?;
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
        fn extract_bound(ob: &Bound<'py, PyAny>) -> anyhow::Result<Self> {
            let s: &str = ob.extract()?;
            s.parse::<PartImageColorspace>()
                .map_err(|_| PyValueError::new_bail!("Invalid colorspace: {s}"))
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

#[cfg(feature = "wasm")]
mod wasm {
    use std::convert::TryFrom;

    use js_sys::{Object, Reflect};
    use wasm_bindgen::{
        convert::{
            FromWasmAbi, IntoWasmAbi, OptionFromWasmAbi, OptionIntoWasmAbi, TryFromJsValue,
            VectorFromWasmAbi, VectorIntoWasmAbi,
        },
        describe::{WasmDescribe, WasmDescribeVector},
        prelude::*,
    };

    use super::*;

    #[wasm_bindgen]
    pub struct PartImageBinary {
        height: usize,
        width: usize,
        colorspace: PartImageColorspace,
        data: crate::value::bytes::Bytes,
    }

    impl WasmDescribe for Part {
        fn describe() {
            JsValue::describe()
        }
    }

    impl WasmDescribeVector for Part {
        fn describe_vector() {
            <JsValue as WasmDescribeVector>::describe_vector()
        }
    }

    impl FromWasmAbi for Part {
        type Abi = <JsValue as FromWasmAbi>::Abi;

        #[inline]
        unsafe fn from_abi(js: Self::Abi) -> Self {
            let js_value = unsafe { JsValue::from_abi(js) };
            Part::try_from(js_value).unwrap()
        }
    }

    impl IntoWasmAbi for Part {
        type Abi = <JsValue as IntoWasmAbi>::Abi;

        #[inline]
        fn into_abi(self) -> Self::Abi {
            let js_value = JsValue::from(self);
            js_value.into_abi()
        }
    }

    impl OptionFromWasmAbi for Part {
        #[inline]
        fn is_none(js: &Self::Abi) -> bool {
            let js_value = unsafe { JsValue::from_abi(*js) };
            let is_none = js_value.is_null() || js_value.is_undefined();
            std::mem::forget(js_value);
            is_none
        }
    }

    impl OptionIntoWasmAbi for Part {
        #[inline]
        fn none() -> Self::Abi {
            JsValue::NULL.into_abi()
        }
    }

    impl VectorFromWasmAbi for Part {
        type Abi = <Box<[JsValue]> as FromWasmAbi>::Abi;

        unsafe fn vector_from_abi(js: Self::Abi) -> Box<[Self]> {
            let js_values = unsafe { <Box<[JsValue]> as FromWasmAbi>::from_abi(js) };
            let vec: Vec<Part> = js_values
                .iter()
                .map(|js_val| {
                    Part::try_from(js_val.clone())
                        .expect("Failed to convert JsValue to Part in array")
                })
                .collect();

            vec.into_boxed_slice()
        }
    }

    impl VectorIntoWasmAbi for Part {
        type Abi = <Box<[JsValue]> as IntoWasmAbi>::Abi;

        fn vector_into_abi(vector: Box<[Self]>) -> Self::Abi {
            let js_values: Vec<JsValue> = vector.iter().map(|part| part.clone().into()).collect();
            js_values.into_boxed_slice().into_abi()
        }
    }

    impl TryFrom<JsValue> for Part {
        type Error = js_sys::Error;

        fn try_from(js_val: JsValue) -> Result<Self, Self::Error> {
            if !js_val.is_object() {
                return Err(js_sys::Error::new("The value is not an object"));
            }

            let obj = Object::from(js_val.clone());
            let part_type = Reflect::get(&obj, &JsValue::from_str("partType"))
                .map_err(|_| js_sys::Error::new("partType field does not exist"))?
                .as_string()
                .ok_or(js_sys::Error::new("partType should be a string"))?;
            match part_type.as_str() {
                "text" => {
                    let text = Reflect::get(&obj, &JsValue::from_str("text"))
                        .map_err(|_| js_sys::Error::new("text field does not exist"))?
                        .as_string()
                        .ok_or(js_sys::Error::new("text field should be a string"))?;
                    return Ok(Part::Text { text });
                }
                "function" => {
                    let id = Reflect::get(&obj, &JsValue::from_str("id"))
                        .map_err(|_| js_sys::Error::new("'id' field does not exist"))?
                        .as_string();
                    let f = PartFunction::try_from_js_value(
                        Reflect::get(&obj, &JsValue::from_str("function"))
                            .map_err(|_| js_sys::Error::new("'function' field does not exist"))?,
                    )?;
                    return Ok(Part::Function { id, f });
                }
                "value" => {
                    let js_val = Reflect::get(&obj, &JsValue::from_str("value"))
                        .map_err(|_| js_sys::Error::new("'value' field does not exist"))?;
                    let value = Value::try_from(js_val)?;
                    return Ok(Part::Value { value });
                }
                "image" => {
                    if let Ok(binary) = Reflect::get(&obj, &JsValue::from_str("binary")) {
                        let img = PartImageBinary::try_from_js_value(binary)?;
                        return Ok(Part::Image {
                            image: PartImage::Binary {
                                h: img.height,
                                w: img.width,
                                c: img.colorspace,
                                data: img.data,
                            },
                        });
                    }
                    return Err(js_sys::Error::new("Invalid image object"));
                }
                _ => {
                    return Err(js_sys::Error::new(
                        format!("Unknown partType: {}", part_type).as_str(),
                    ));
                }
            }
        }
    }

    impl From<Part> for JsValue {
        fn from(value: Part) -> Self {
            let obj = Object::new();
            match value {
                Part::Text { text } => {
                    Reflect::set(
                        &obj,
                        &JsValue::from_str("partType"),
                        &JsValue::from_str("text"),
                    )
                    .unwrap();
                    Reflect::set(&obj, &JsValue::from_str("text"), &JsValue::from_str(&text))
                        .unwrap();
                }
                Part::Function { id, f } => {
                    Reflect::set(
                        &obj,
                        &JsValue::from_str("partType"),
                        &JsValue::from_str("function"),
                    )
                    .unwrap();
                    let id = if let Some(id) = id {
                        &JsValue::from_str(&id)
                    } else {
                        &JsValue::NULL
                    };
                    Reflect::set(&obj, &JsValue::from_str("id"), id).unwrap();
                    Reflect::set(&obj, &JsValue::from_str("function"), &JsValue::from(f)).unwrap();
                }
                Part::Value { value } => {
                    Reflect::set(
                        &obj,
                        &JsValue::from_str("partType"),
                        &JsValue::from_str("value"),
                    )
                    .unwrap();
                    Reflect::set(&obj, &JsValue::from_str("value"), &JsValue::from(value)).unwrap();
                }
                Part::Image { image } => {
                    Reflect::set(
                        &obj,
                        &JsValue::from_str("partType"),
                        &JsValue::from_str("image"),
                    )
                    .unwrap();
                    match image {
                        PartImage::Binary { h, w, c, data } => {
                            let img = PartImageBinary {
                                height: h,
                                width: w,
                                colorspace: c,
                                data: data,
                            };
                            Reflect::set(&obj, &JsValue::from_str("binary"), &JsValue::from(img))
                                .unwrap();
                        }
                    }
                }
            }
            obj.into()
        }
    }
}
