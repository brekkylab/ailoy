use base64::Engine;
use serde::{Deserialize, Serialize, Serializer, ser::SerializeMap as _};

use crate::value::{Value, delta::Delta};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, strum::EnumString, strum::Display)]
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

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "python",
    pyo3_stub_gen_derive::gen_stub_pyclass_complex_enum
)]
#[cfg_attr(feature = "python", pyo3::pyclass(eq))]
pub enum Part {
    Text(String),
    Function {
        id: Option<String>,
        name: String,
        arguments: Value,
    },
    Value(Value),
    Image {
        h: usize,
        w: usize,
        c: PartImageColorspace,
        nbytes: usize,
        buf: Vec<u8>,
    },
}

impl Part {
    pub fn text(v: impl Into<String>) -> Self {
        Self::Text(v.into())
    }

    pub fn image(
        height: usize,
        width: usize,
        colorspace: impl TryInto<PartImageColorspace>,
        buf: impl IntoIterator<Item = u8>,
    ) -> Result<Self, String> {
        let colorspace: PartImageColorspace = colorspace
            .try_into()
            .map_err(|_| String::from("Colorspace parsing failed"))?;
        let buf = buf.into_iter().collect::<Vec<_>>();
        let nbytes = buf.len() / height / width / colorspace.channel();
        if !(nbytes == 1 || nbytes == 2 || nbytes == 3 || nbytes == 4) {
            panic!("Invalid buffer length");
        }
        Ok(Self::Image {
            h: height,
            w: width,
            c: colorspace,
            nbytes,
            buf,
        })
    }

    pub fn function(name: impl Into<String>, args: impl Into<Value>) -> Self {
        Self::Function {
            id: None,
            name: name.into(),
            arguments: args.into(),
        }
    }

    pub fn function_with_id(
        id: impl Into<String>,
        name: impl Into<String>,
        args: impl Into<Value>,
    ) -> Self {
        Self::Function {
            id: Some(id.into()),
            name: name.into(),
            arguments: args.into(),
        }
    }

    pub fn is_text(&self) -> bool {
        match self {
            Self::Text(..) => true,
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
            Self::Value(..) => true,
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
            Self::Text(text) => Some(text.as_str()),
            _ => None,
        }
    }

    pub fn as_text_mut(&mut self) -> Option<&mut String> {
        match self {
            Self::Text(text) => Some(text),
            _ => None,
        }
    }

    pub fn as_function(&self) -> Option<(Option<&str>, &str, &Value)> {
        match self {
            Self::Function {
                id,
                name,
                arguments,
            } => Some((id.as_deref(), name.as_str(), arguments)),
            _ => None,
        }
    }

    pub fn as_function_mut(&mut self) -> Option<(Option<&mut String>, &mut String, &mut Value)> {
        match self {
            Self::Function {
                id,
                name,
                arguments,
            } => Some((id.as_mut(), name, arguments)),
            _ => None,
        }
    }

    pub fn as_value(&self) -> Option<&Value> {
        match self {
            Self::Value(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_value_mut(&mut self) -> Option<&mut Value> {
        match self {
            Self::Value(value) => Some(value),
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
                h,
                w,
                c,
                nbytes,
                buf,
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

impl Serialize for Part {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Text(text) => {
                let mut s = serializer.serialize_map(Some(2))?;
                s.serialize_entry("type", "text")?;
                s.serialize_entry("text", text)?;
                s
            }
            Self::Function {
                id,
                name,
                arguments,
            } => {
                let mut s = serializer.serialize_map(Some(3))?;
                s.serialize_entry("type", "function")?;
                if let Some(id) = id {
                    s.serialize_entry("id", id)?;
                }
                s.serialize_entry("name", name)?;
                s.serialize_entry("arguments", arguments)?;
                s
            }
            Self::Image {
                h,
                w,
                c,
                nbytes: _,
                buf,
            } => {
                let human_readable = serializer.is_human_readable();
                let mut s = serializer.serialize_map(Some(6))?;
                s.serialize_entry("type", "image")?;
                s.serialize_entry("media-type", "image/x-binary")?;
                s.serialize_entry("height", h)?;
                s.serialize_entry("width", w)?;
                s.serialize_entry("colorspace", c)?;
                if human_readable {
                    s.serialize_entry(
                        "data",
                        &base64::engine::general_purpose::STANDARD.encode(buf),
                    )?;
                } else {
                    s.serialize_entry("data", &buf[..])?;
                }
                s
            }
            Self::Value(value) => {
                let mut s = serializer.serialize_map(Some(2))?;
                s.serialize_entry("type", "value")?;
                s.serialize_entry("value", value)?;
                s
            }
        }
        .end()
    }
}

impl<'de> Deserialize<'de> for Part {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PartDelta {
    Null,
    Text(String),
    VerbatimFunction(String),
    Function {
        id: Option<String>,
        name: String,
        arguments: String,
    },
    ParsedFunction {
        id: Option<String>,
        name: String,
        arguments: Value,
    },
}

impl PartDelta {
    pub fn is_text(&self) -> bool {
        match self {
            Self::Text(..) => true,
            _ => false,
        }
    }
    pub fn is_verbatim_function(&self) -> bool {
        match self {
            Self::VerbatimFunction(..) => true,
            _ => false,
        }
    }

    pub fn is_function(&self) -> bool {
        match self {
            Self::Function { .. } => true,
            _ => false,
        }
    }

    pub fn is_parsed_function(&self) -> bool {
        match self {
            Self::ParsedFunction { .. } => true,
            _ => false,
        }
    }

    pub fn to_text(self) -> Option<String> {
        match self {
            Self::Text(text) | Self::VerbatimFunction(text) => Some(text),
            _ => None,
        }
    }

    pub fn to_function(self) -> Option<(Option<String>, String, String)> {
        match self {
            Self::Function {
                id,
                name,
                arguments,
            } => Some((id, name, arguments)),
            _ => None,
        }
    }

    pub fn to_parsed_function(self) -> Option<(Option<String>, String, Value)> {
        match self {
            Self::ParsedFunction {
                id,
                name,
                arguments,
            } => Some((id, name, arguments)),
            _ => None,
        }
    }
}

impl Default for PartDelta {
    fn default() -> Self {
        Self::Null
    }
}

impl Delta for PartDelta {
    type Item = Part;

    fn aggregate(self, other: Self) -> Result<Self, ()> {
        match (self, other) {
            (PartDelta::Null, other) => Ok(other),
            (PartDelta::Text(mut lhs), PartDelta::Text(rhs)) => {
                lhs.push_str(&rhs);
                Ok(PartDelta::Text(lhs))
            }
            (PartDelta::VerbatimFunction(mut lhs), PartDelta::VerbatimFunction(rhs)) => {
                lhs.push_str(&rhs);
                Ok(PartDelta::VerbatimFunction(lhs))
            }
            (
                PartDelta::Function {
                    id: lid,
                    name: mut lname,
                    arguments: mut largs,
                },
                PartDelta::Function {
                    id: rid,
                    name: rname,
                    arguments: rargs,
                },
            ) => {
                let id = match (lid, rid) {
                    (None, None) => None,
                    (None, Some(rid)) => Some(rid),
                    (Some(lid), None) => Some(lid),
                    (Some(lid), Some(rid)) if lid == rid => Some(lid),
                    (Some(_), Some(_)) => return Err(()),
                };
                lname.push_str(&rname);
                largs.push_str(&rargs);
                Ok(PartDelta::Function {
                    id,
                    name: lname,
                    arguments: largs,
                })
            }
            (
                PartDelta::ParsedFunction {
                    id: lid,
                    name: mut lname,
                    arguments: mut largs,
                },
                PartDelta::ParsedFunction {
                    id: rid,
                    name: rname,
                    arguments: rargs,
                },
            ) => {
                todo!()
            }
            _ => Err(()),
        }
    }

    fn finish(self) -> Result<Self::Item, String> {
        match self {
            PartDelta::Null => Ok(Part::Text(String::new())),
            PartDelta::Text(s) => Ok(Part::Text(s)),
            PartDelta::VerbatimFunction(s) => match serde_json::from_str::<Value>(&s) {
                Ok(root) => match (root.pointer_as::<str>("/name"), root.pointer("/arguments")) {
                    (Some(name), Some(args)) => Ok(Part::Function {
                        id: None,
                        name: name.into(),
                        arguments: args.clone(),
                    }),
                    _ => Err(String::from("Invalid function JSON")),
                },
                Err(_) => Err(String::from("Invalid JSON")),
            },
            PartDelta::Function {
                id,
                name,
                arguments,
            } => Ok(Part::Function {
                id,
                name,
                arguments: serde_json::from_str::<Value>(&arguments)
                    .map_err(|_| String::from("Invalid JSON"))?,
            }),
            PartDelta::ParsedFunction {
                id,
                name,
                arguments,
            } => Ok(Part::Function {
                id,
                name,
                arguments,
            }),
        }
    }
}
