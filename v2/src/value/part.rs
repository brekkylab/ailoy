use serde::{Deserialize, Serialize};

use crate::value::{Value, delta::Delta};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3_stub_gen_derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass(eq, get_all, set_all))]
pub struct PartFunction {
    pub name: String,
    #[serde(rename = "arguments")]
    pub args: Value,
}

#[derive(
    Clone, Debug, PartialEq, Eq, Serialize, Deserialize, strum::EnumString, strum::Display,
)]
#[serde(untagged)]
#[cfg_attr(feature = "python", pyo3_stub_gen_derive::gen_stub_pyclass_enum)]
#[cfg_attr(feature = "python", pyo3::pyclass(eq))]
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
    pub fn channel(&self) -> usize {
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
pub enum PartImage {
    #[serde(rename = "image/x-binary")]
    Binary {
        #[serde(rename = "height")]
        h: usize,
        #[serde(rename = "width")]
        w: usize,
        #[serde(rename = "colorspace")]
        c: PartImageColorspace,
        nbytes: usize,
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
        height: usize,
        width: usize,
        colorspace: impl TryInto<PartImageColorspace>,
        data: impl IntoIterator<Item = u8>,
    ) -> Result<Self, String> {
        let colorspace: PartImageColorspace = colorspace
            .try_into()
            .map_err(|_| String::from("Colorspace parsing failed"))?;
        let data = data.into_iter().collect::<Vec<_>>();
        let nbytes = data.len() / height / width / colorspace.channel();
        if !(nbytes == 1 || nbytes == 2 || nbytes == 3 || nbytes == 4) {
            panic!("Invalid data length");
        }
        Ok(Self::Image {
            image: PartImage::Binary {
                h: height,
                w: width,
                c: colorspace,
                nbytes,
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
                        nbytes,
                        data: super::bytes::Bytes(buf),
                    },
            } => {
                let (h, w) = (*h as u32, *w as u32);
                match (c, *nbytes) {
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
pub enum PartDelta {
    Text {
        text: String,
    },
    Function {
        id: Option<String>,
        #[serde(rename = "function")]
        f: PartDeltaFunction,
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
}

impl Default for PartDelta {
    fn default() -> Self {
        Self::Null()
    }
}

impl Delta for PartDelta {
    type Item = Part;

    fn aggregate(self, other: Self) -> Result<Self, ()> {
        match (self, other) {
            (PartDelta::Null(), other) => Ok(other),
            (PartDelta::Text { text: mut t1 }, PartDelta::Text { text: t2 }) => {
                t1.push_str(&t2);
                Ok(PartDelta::Text { text: t1 })
            }
            (PartDelta::Function { id: id1, f: f1 }, PartDelta::Function { id: id2, f: f2 }) => {
                let id = match (id1, id2) {
                    (Some(id1), Some(id2)) if id1 == id2 => Some(id1),
                    (Some(_), Some(_)) => return Err(()),
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
                    _ => return Err(()),
                };
                Ok(PartDelta::Function { id, f })
            }
            _ => Err(()),
        }
    }

    fn finish(self) -> Result<Self::Item, String> {
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
                                (Some(name), Some(args)) => Ok(PartFunction {
                                    name: name.to_owned(),
                                    args: args.to_owned(),
                                }),
                                _ => Err(String::from("Invalid function JSON")),
                            }
                        }
                        Err(_) => Err(String::from("Invalid JSON")),
                    },
                    // Try json deserialization for args
                    PartDeltaFunction::WithStringArgs { name, args } => {
                        let args = serde_json::from_str::<Value>(&args)
                            .map_err(|_| String::from("Invalid JSON"))?;
                        Ok(PartFunction { name, args })
                    }
                    // As-is
                    PartDeltaFunction::WithParsedArgs { name, args } => {
                        Ok(PartFunction { name, args })
                    }
                }?;
                Ok(Part::Function { id, f })
            }
        }
    }
}
