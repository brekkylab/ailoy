use std::fmt;

use anyhow::{Context, bail};
use base64::Engine;
use serde::{Deserialize, Serialize};
use url::Url;

use crate::value::{Value, bytes::Bytes, delta::Delta};

/// Represents a function call contained within a message part.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3_stub_gen_derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "ailoy._core", eq))]
#[cfg_attr(feature = "nodejs", napi_derive::napi(object))]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub struct PartFunction {
    /// The name of the function
    pub name: String,

    /// The arguments of the function, usually represented as a JSON object.
    pub arguments: Value,
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
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
#[cfg_attr(feature = "python", derive(ailoy_macros::PyStringEnum))]
#[cfg_attr(feature = "nodejs", napi_derive::napi(string_enum = "lowercase"))]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub enum PartImageColorspace {
    /// Single-channel grayscale image
    Grayscale,

    /// Three-channel color image without alpha    
    RGB,

    /// Four-channel color image with alpha
    RGBA,
}

impl From<image::ColorType> for PartImageColorspace {
    fn from(value: image::ColorType) -> Self {
        match value {
            image::ColorType::L8
            | image::ColorType::La8
            | image::ColorType::L16
            | image::ColorType::La16 => PartImageColorspace::Grayscale,
            image::ColorType::Rgb8 | image::ColorType::Rgb16 | image::ColorType::Rgb32F => {
                PartImageColorspace::RGB
            }
            image::ColorType::Rgba8 | image::ColorType::Rgba16 | image::ColorType::Rgba32F => {
                PartImageColorspace::RGBA
            }
            _ => panic!("invalid color type"),
        }
    }
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
#[serde(tag = "type", rename_all = "lowercase")]
#[cfg_attr(
    feature = "python",
    pyo3_stub_gen_derive::gen_stub_pyclass_complex_enum
)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "ailoy._core", eq))]
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
    Url {
        url: String,
    },
}

impl TryInto<image::DynamicImage> for &PartImage {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<image::DynamicImage, Self::Error> {
        match self {
            PartImage::Binary {
                height,
                width,
                colorspace,
                data: Bytes(buf),
            } => {
                fn bytes_to_u16_ne(b: &[u8]) -> anyhow::Result<Vec<u16>> {
                    if b.len() % 2 != 0 {
                        return Err(anyhow::anyhow!("bytes should be even"));
                    }
                    let mut v = Vec::with_capacity(b.len() / 2);
                    for ch in b.chunks_exact(2) {
                        v.push(u16::from_ne_bytes([ch[0], ch[1]]));
                    }
                    Ok(v)
                }

                let height = *height as u32;
                let width = *width as u32;
                let nbytes = buf.len() as u32 / height / width / colorspace.channel();
                match (colorspace, nbytes) {
                    // Grayscale 8-bit
                    (PartImageColorspace::Grayscale, 1) => {
                        let buf = image::GrayImage::from_raw(width, height, buf.to_vec())
                            .ok_or(anyhow::anyhow!("Failed to read image buffer"))?;
                        Ok(image::DynamicImage::ImageLuma8(buf))
                    }
                    // Grayscale 16-bit
                    (PartImageColorspace::Grayscale, 2) => {
                        let buf = image::ImageBuffer::<image::Luma<u16>, _>::from_raw(
                            width,
                            height,
                            bytes_to_u16_ne(&buf)?,
                        )
                        .ok_or(anyhow::anyhow!("Failed to read image buffer"))?;
                        Ok(image::DynamicImage::ImageLuma16(buf))
                    }
                    // RGB 8-bit
                    (PartImageColorspace::RGB, 1) => {
                        let buf = image::RgbImage::from_raw(width, height, buf.to_vec())
                            .ok_or(anyhow::anyhow!("Failed to read image buffer"))?;
                        Ok(image::DynamicImage::ImageRgb8(buf))
                    }
                    // RGBA 8-bit
                    (PartImageColorspace::RGBA, 1) => {
                        let buf = image::RgbaImage::from_raw(width, height, buf.to_vec())
                            .ok_or(anyhow::anyhow!("Failed to read image buffer"))?;
                        Ok(image::DynamicImage::ImageRgba8(buf))
                    }
                    // RGB 16-bit
                    (PartImageColorspace::RGB, 2) => {
                        let buf = image::ImageBuffer::<image::Rgb<u16>, _>::from_raw(
                            width,
                            height,
                            bytes_to_u16_ne(&buf)?,
                        )
                        .ok_or(anyhow::anyhow!("Failed to read image buffer"))?;
                        Ok(image::DynamicImage::ImageRgb16(buf))
                    }
                    // RGBA 16-bit
                    (PartImageColorspace::RGBA, 2) => {
                        let buf = image::ImageBuffer::<image::Rgba<u16>, _>::from_raw(
                            width,
                            height,
                            bytes_to_u16_ne(&buf)?,
                        )
                        .ok_or(anyhow::anyhow!("Failed to read image buffer"))?;
                        Ok(image::DynamicImage::ImageRgba16(buf))
                    }
                    _ => Err(anyhow::anyhow!("Invalid colorspace or channel")),
                }
            }
            PartImage::Url { .. } => {
                todo!("Request to url and get the data, and load to DynamicImage")
            }
        }
    }
}

impl PartImage {
    /// Returns base64 encoded string with PNG format
    pub fn base64(&self) -> anyhow::Result<String> {
        match self {
            PartImage::Binary { .. } => {
                let img: image::DynamicImage = self.try_into()?;

                // dump as PNG
                let mut png_bytes: Vec<u8> = Vec::new();
                img.write_to(
                    &mut std::io::Cursor::new(&mut png_bytes),
                    image::ImageFormat::Png,
                )?;

                // base64 encoding
                let encoded =
                    base64::engine::general_purpose::STANDARD.encode(png_bytes.as_slice());
                Ok(encoded)
            }
            _ => Err(anyhow::anyhow!(
                "base64 is available for PartImage::Binary only"
            )),
        }
    }
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
#[serde(tag = "type", rename_all = "lowercase")]
#[cfg_attr(
    feature = "python",
    pyo3_stub_gen_derive::gen_stub_pyclass_complex_enum
)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "ailoy._core", eq))]
#[cfg_attr(feature = "nodejs", napi_derive::napi(discriminant_case = "lowercase"))]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub enum Part {
    /// Plain utf-8 encoded text.
    Text { text: String },

    /// Represents a structured function call to an external tool.
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
    ///     arguments: Value::from_json(r#"{"source": "hello", "lang": "cn"}"#).unwrap(),
    /// };
    /// ```
    Function {
        id: Option<String>,
        function: PartFunction,
    },

    /// Holds a structured data value, typically considered as a JSON structure.
    Value { value: Value },

    /// Contains an image payload or reference used within a message part.
    /// The image may be provided as raw binary data or an encoded format (e.g., PNG, JPEG),
    /// or as a reference via a URL. Optional metadata can be included alongside the image.
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
            bail!("Invalid data length");
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

    pub fn image_binary_from_bytes(data: &[u8]) -> anyhow::Result<Self> {
        let img = image::load_from_memory(data).expect("Failed to load image from base64 data");
        Ok(Self::Image {
            image: PartImage::Binary {
                height: img.height(),
                width: img.width(),
                colorspace: img.color().into(),
                data: super::bytes::Bytes(img.into_bytes().into()),
            },
        })
    }

    pub fn image_binary_from_base64(data: impl Into<String>) -> anyhow::Result<Self> {
        let data = base64::engine::general_purpose::STANDARD
            .decode(data.into().as_bytes())
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Self::image_binary_from_bytes(data.as_slice())
    }

    pub fn image_url(url: String) -> anyhow::Result<Self> {
        let url = Url::parse(&url).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Ok(Part::Image {
            image: PartImage::Url { url: url.into() },
        })
    }

    pub fn function(name: impl Into<String>, arguments: impl Into<Value>) -> Self {
        Self::Function {
            id: None,
            function: PartFunction {
                name: name.into(),
                arguments: arguments.into(),
            },
        }
    }

    pub fn function_with_id(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<Value>,
    ) -> Self {
        Self::Function {
            id: Some(id.into()),
            function: PartFunction {
                name: name.into(),
                arguments: arguments.into(),
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
        match self {
            Self::Image { image } => image.try_into().ok(),
            _ => None,
        }
    }
}

impl fmt::Display for Part {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = serde_json::to_string(self).map_err(|_| fmt::Error)?;
        write!(f, "Part {}", s)
    }
}

/// Represents an incremental update (delta) of a function part.
///
/// This type is used during streaming or partial message generation, when function calls are being streamed as text chunks or partial JSON fragments.
///
/// # Variants
/// * `Verbatim(String)` — Raw text content, typically a partial JSON fragment.
/// * `WithStringArgs { name, arguments }` — Function name and its serialized arguments as strings.
/// * `WithParsedArgs { name, arguments }` — Function name and parsed arguments as a `Value`.
///
/// # Use Case
/// When the model streams out a function call response (e.g., `"function_call":{"name":...}`),
/// the incremental deltas can be accumulated until the full function payload is formed.
///
/// # Example
/// ```rust
/// let delta = PartDeltaFunction::WithStringArgs {
///     name: "translate".into(),
///     arguments: r#"{"text":"hi"}"#.into(),
/// };
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(
    feature = "python",
    pyo3_stub_gen_derive::gen_stub_pyclass_complex_enum
)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "ailoy._core", eq))]
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
/// let merged = d1.accumulate(d2).unwrap();
/// assert_eq!(merged.to_text().unwrap(), "Hello");
/// ```
///
/// # Error Handling
/// Accumulation or finalization may return an error if incompatible deltas
/// (e.g. mismatched function IDs) are combined or invalid JSON arguments are given.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(
    feature = "python",
    pyo3_stub_gen_derive::gen_stub_pyclass_complex_enum
)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "ailoy._core", eq))]
#[cfg_attr(
    feature = "nodejs",
    napi_derive::napi(discriminant_case = "snake_case")
)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub enum PartDelta {
    /// Incremental text fragment.
    Text { text: String },

    /// Incremental function call fragment.
    Function {
        id: Option<String>,
        function: PartDeltaFunction,
    },

    /// JSON-like value update.
    Value { value: Value },

    /// Placeholder representing no data yet.
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

    fn accumulate(self, other: Self) -> anyhow::Result<Self> {
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
                                "Cannot accumulate two functions with different ids. ({} != {}).",
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
                        "Accumulation between those two function delta {:?}, {:?} is not defined.",
                        f1,
                        f2
                    ),
                };
                Ok(PartDelta::Function { id, function: f })
            }
            (pd1, pd2) => {
                bail!(
                    "Accumulation between those two part delta {:?}, {:?} is not defined.",
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

impl fmt::Display for PartDelta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = serde_json::to_string(self).map_err(|_| fmt::Error)?;
        write!(f, "PartDelta {}", s)
    }
}

#[cfg(feature = "python")]
mod py {
    use pyo3::{
        Bound, IntoPyObject, PyAny, PyResult, Python, exceptions::PyValueError, pymethods,
        types::PyType,
    };
    use pyo3_stub_gen::derive::*;

    use super::*;

    #[gen_stub_pymethods]
    #[pymethods]
    impl PartFunction {
        #[getter]
        fn name(&self) -> String {
            self.name.clone()
        }

        #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any]"))]
        #[getter]
        fn arguments<'a>(&'a self, py: Python<'a>) -> Bound<'a, PyAny> {
            self.arguments.clone().into_pyobject(py).unwrap()
        }
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl Part {
        pub fn __repr__(&self) -> String {
            let s = match &self {
                Part::Text { text } => format!("Text(\"{}\")", text.replace('\n', "\\n")),
                Part::Function { .. } => {
                    format!(
                        "Function({})",
                        serde_json::to_string(self).unwrap_or("".to_owned())
                    )
                }
                Part::Value { value } => format!(
                    "Value({})",
                    serde_json::to_string(value).unwrap_or("{...}".to_owned())
                ),
                Part::Image { image } => {
                    format!(
                        "Image(\"{}\")",
                        serde_json::to_string(image).unwrap_or("".to_owned())
                    )
                }
            };
            format!("Part.{}", s)
        }

        #[getter]
        fn part_type(&self) -> &'static str {
            match &self {
                Part::Text { .. } => "text",
                Part::Function { .. } => "function",
                Part::Value { .. } => "value",
                Part::Image { .. } => "image",
            }
        }

        #[classmethod]
        #[pyo3(name = "image_from_bytes")]
        pub fn image_from_bytes_py<'a>(
            _cls: &Bound<'a, PyType>,
            #[gen_stub(override_type(type_repr = "bytes"))] data: &[u8],
        ) -> PyResult<Part> {
            Part::image_binary_from_bytes(data).map_err(|e| PyValueError::new_err(e.to_string()))
        }

        #[classmethod]
        #[pyo3(name = "image_from_base64")]
        pub fn image_from_base64_py<'a>(_cls: &Bound<'a, PyType>, data: String) -> PyResult<Part> {
            Part::image_binary_from_base64(data).map_err(|e| PyValueError::new_err(e.to_string()))
        }

        #[classmethod]
        #[pyo3(name = "image_from_url")]
        pub fn image_from_url_py<'a>(_cls: &Bound<'a, PyType>, url: String) -> PyResult<Part> {
            Part::image_url(url).map_err(|e| PyValueError::new_err(e.to_string()))
        }
    }
}

#[cfg(feature = "nodejs")]
mod node {
    use napi::bindgen_prelude::*;
    use napi_derive::napi;

    use super::*;

    #[allow(unused)]
    #[napi]
    pub fn image_from_bytes(data: Uint8Array) -> napi::Result<Part> {
        Part::image_binary_from_bytes(data.to_vec().as_slice())
            .map_err(|e| napi::Error::new(Status::InvalidArg, e.to_string()))
    }

    #[allow(unused)]
    #[napi]
    pub fn image_from_base64(data: String) -> napi::Result<Part> {
        Part::image_binary_from_base64(data)
            .map_err(|e| napi::Error::new(Status::InvalidArg, e.to_string()))
    }

    #[allow(unused)]
    #[napi]
    pub fn image_from_url(url: String) -> napi::Result<Part> {
        Part::image_url(url).map_err(|e| napi::Error::new(Status::InvalidArg, e.to_string()))
    }
}

#[cfg(feature = "wasm")]
mod wasm {
    use js_sys::Uint8Array;
    use wasm_bindgen::prelude::*;

    use super::*;

    #[wasm_bindgen(js_name = "imageFromBytes")]
    pub fn image_from_bytes(data: Uint8Array) -> Result<Part, js_sys::Error> {
        Part::image_binary_from_bytes(data.to_vec().as_slice())
            .map_err(|e| js_sys::Error::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = "imageFromBase64")]
    pub fn image_from_base64(data: String) -> Result<Part, js_sys::Error> {
        Part::image_binary_from_base64(data).map_err(|e| js_sys::Error::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = "imageFromUrl")]
    pub fn image_from_url(url: String) -> Result<Part, js_sys::Error> {
        Part::image_url(url).map_err(|e| js_sys::Error::new(&e.to_string()))
    }
}
