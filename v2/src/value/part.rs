use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};

use crate::value::{Value, delta::Delta};

/// Represents a function call contained within a message part.
///
/// Many language models (LLMs) use a **function calling** mechanism to extend their capabilities.
/// When an LLM decides to use external *tools*, it produces a structured output called a `function`.
/// A function conventionally consists of two fields: a `name`, and an `arguments` field formatted as JSON.
/// This is conceptually similar to making an HTTP POST request, where the request body carries a single JSON object.
///
/// This struct models that convention, representing a function invocation request
/// from an LLM to an external tool or API.
///
/// # Examples
/// ```rust
/// let f = PartFunction {
///     name: "translate".to_string(),
///     args: Value::from_json(r#"{"text": "hello", "lang": "cn"}"#).unwrap(),
/// };
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "nodejs", napi_derive::napi(object))]
pub struct PartFunction {
    /// The name of the function
    pub name: String,

    /// The arguments of the function, usually represented as a JSON object.
    #[serde(rename = "arguments")]
    pub args: Value,
}

/// Represents the color space of an image part.
///
/// This enum defines the supported pixel formats of image data. It determines
/// how many channels each pixel has and how the image should be interpreted.
///
/// # Examples
/// ```rust
/// let c = PartImageColorspace::RGB;
/// assert_eq!(c.channel(), 3);
/// ```
#[derive(
    Clone, Debug, PartialEq, Eq, Serialize, Deserialize, strum::EnumString, strum::Display,
)]
#[serde(untagged)]
#[cfg_attr(feature = "nodejs", napi_derive::napi(string_enum))]
pub enum PartImageColorspace {
    /// Single-channel grayscale image
    #[strum(serialize = "grayscale")]
    #[serde(rename = "grayscale")]
    Grayscale,

    /// Three-channel color image without alpha    
    #[strum(serialize = "rgb")]
    #[serde(rename = "rgb")]
    RGB,

    /// Four-channel color image with alpha
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

/// Represents the image data contained in a [`Part`].
///
/// `PartImage` provides structured access to image data.
/// Currently, it only implments "binary" types.
///
/// # Example
/// ```rust
/// let part = Part::image_binary(640, 480, "rgb", (0..640*480*3).map(|i| (i % 255) as u8)).unwrap();
///
/// if let Some(img) = part.as_image() {
///     assert_eq!(img.height(), 640);
///     assert_eq!(img.width(), 480);
/// }
/// ```
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

/// Represents a semantically meaningful content unit exchanged between the model and the user.
///
/// Conceptually, each `Part` encapsulates a piece of **data** that contributes
/// to a chat message — such as text, a function invocation, or an image.  
///
/// For example, a single message consisting of a sequence like  
/// `(text..., image, text...)` is represented as a `Message` containing
/// an array of three `Part` elements.
///
/// Note that a `Part` does **not** carry "intent", such as "reasoning" or "tool call".
/// These higher-level semantics are determined by the context of a [`Message`].
///
/// # Example
///
/// ## Rust
/// ```rust
/// let part = Part::text("Hello, world!");
/// assert!(part.is_text());
/// ```
///
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
#[cfg_attr(
    feature = "python",
    pyo3_stub_gen_derive::gen_stub_pyclass_complex_enum
)]
#[cfg_attr(feature = "python", pyo3::pyclass(eq))]
#[cfg_attr(feature = "nodejs", napi_derive::napi)]
pub enum Part {
    /// Plain text content.
    #[serde(rename = "text")]
    Text { text: String },

    /// Represents a structured function call to an external tool.
    #[serde(rename = "function")]
    Function {
        id: Option<String>,
        #[serde(rename = "function")]
        f: PartFunction,
    },

    /// Holds a structured data value, typically a JSON object.
    #[serde(rename = "value")]
    Value { value: Value },

    /// Contains an image payload or reference used within a message part.
    /// The image may be provided as raw binary data or an encoded format (e.g., PNG, JPEG),
    /// or as a reference via a URL. Optional metadata can be included alongside the image.
    #[serde(rename = "image")]
    Image { image: PartImage },
}

impl Part {
    pub fn text(v: impl Into<String>) -> Self {
        Self::Text { text: v.into() }
    }

    /// Create a new image pixel map.
    ///
    /// # Encoding Notes
    /// * The `data` field is expected to contain pixel data in **row-major order**.
    /// * The bytes per channel depend on the color depth (e.g., 1 byte for 8-bit, 2 bytes for 16-bit).
    /// * The total size of `data` must be equal to: `height × width × colorspace.channel() × bytes_per_channel`.
    ///
    /// # Errors
    /// Image construction fails if:
    /// * The colorspace cannot be parsed from the provided input.
    /// * The data length does not match the expected size given dimensions and channels.
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

/// Represents an incremental update (delta) of a function part.
///
/// This type is used during streaming or partial message generation, when function calls are being streamed as text chunks or partial JSON fragments.
///
/// # Variants
/// * `Verbatim(String)` — Raw text content, typically a partial JSON fragment.
/// * `WithStringArgs { name, args }` — Function name and its serialized arguments as strings.
/// * `WithParsedArgs { name, args }` — Function name and parsed arguments as a `Value`.
///
/// # Use Case
/// When the model streams out a function call response (e.g., `"function_call":{"name":...}`),
/// the incremental deltas can be aggregated until the full function payload is formed.
///
/// # Example
/// ```rust
/// let delta = PartDeltaFunction::WithStringArgs {
///     name: "translate".into(),
///     args: r#"{"text":"hi"}"#.into(),
/// };
/// `
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

/// Represents a partial or incremental update (delta) of a [`Part`].
///
/// This type enables composable, streaming updates to message parts.
/// For example, text may be produced token-by-token, or a function call
/// may be emitted gradually as its arguments stream in.
///
/// # Example
///
/// ## Rust
/// ```rust
/// let d1 = PartDelta::Text { text: "Hel".into() };
/// let d2 = PartDelta::Text { text: "lo".into() };
/// let merged = d1.aggregate(d2).unwrap();
/// assert_eq!(merged.to_text().unwrap(), "Hello");
/// ```
///
/// # Error Handling
/// Aggregation or finalization may return an error if incompatible deltas
/// (e.g. mismatched function IDs) are combined or invalid JSON arguments are given.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
#[cfg_attr(
    feature = "python",
    pyo3_stub_gen_derive::gen_stub_pyclass_complex_enum
)]
#[cfg_attr(feature = "python", pyo3::pyclass(eq))]
#[cfg_attr(feature = "nodejs", napi_derive::napi)]
pub enum PartDelta {
    /// Incremental text fragment.
    Text { text: String },

    /// Incremental function call fragment.
    Function {
        id: Option<String>,
        #[serde(rename = "function")]
        f: PartDeltaFunction,
    },

    /// JSON-like value update.
    Value { value: Value },

    /// Placeholder representing no data yet.
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
