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
    TextReasoning {
        text: String,
        signature: Option<String>,
    },
    TextContent(String),
    ImageContent {
        h: usize,
        w: usize,
        c: PartImageColorspace,
        nbytes: usize,
        buf: Vec<u8>,
    },
    FunctionToolCall {
        id: Option<String>,
        name: String,
        arguments: Value,
    },
    TextRefusal(String),
}

impl Part {
    pub fn text_reasoning(v: impl Into<String>) -> Self {
        Self::TextReasoning {
            text: v.into(),
            signature: None,
        }
    }

    pub fn text_reasoning_with_signature(v: impl Into<String>, sig: impl Into<String>) -> Self {
        Self::TextReasoning {
            text: v.into(),
            signature: Some(sig.into()),
        }
    }

    pub fn text_content(v: impl Into<String>) -> Self {
        Self::TextContent(v.into())
    }

    pub fn image_content(
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
        Ok(Self::ImageContent {
            h: height,
            w: width,
            c: colorspace,
            nbytes,
            buf,
        })
    }

    pub fn function_tool_call(name: impl Into<String>, args: impl Into<Value>) -> Self {
        Self::FunctionToolCall {
            id: None,
            name: name.into(),
            arguments: args.into(),
        }
    }

    pub fn function_tool_call_with_id(
        name: impl Into<String>,
        args: impl Into<Value>,
        id: impl Into<String>,
    ) -> Self {
        Self::FunctionToolCall {
            id: Some(id.into()),
            name: name.into(),
            arguments: args.into(),
        }
    }

    pub fn text_refusal(v: impl Into<String>) -> Self {
        Self::TextRefusal(v.into())
    }

    pub fn is_reasoning(&self) -> bool {
        match self {
            Self::TextReasoning { .. } => true,
            _ => false,
        }
    }

    pub fn is_content(&self) -> bool {
        match self {
            Self::TextContent { .. } => true,
            _ => false,
        }
    }

    pub fn is_tool_call(&self) -> bool {
        match self {
            Self::FunctionToolCall { .. } => true,
            _ => false,
        }
    }

    pub fn is_text(&self) -> bool {
        match self {
            Self::TextReasoning { .. } | Self::TextContent(..) => true,
            _ => false,
        }
    }

    pub fn is_function(&self) -> bool {
        match self {
            Self::FunctionToolCall { .. } => true,
            _ => false,
        }
    }

    pub fn is_image(&self) -> bool {
        match self {
            Self::ImageContent { .. } => true,
            _ => false,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::TextReasoning { text, .. } => Some(text.as_str()),
            Self::TextContent(text) => Some(text.as_str()),
            _ => None,
        }
    }

    pub fn as_text_mut(&mut self) -> Option<&mut String> {
        match self {
            Self::TextReasoning { text, .. } => Some(text),
            Self::TextContent(text) => Some(text),
            _ => None,
        }
    }

    pub fn as_function(&self) -> Option<(Option<&str>, &str, &Value)> {
        match self {
            Self::FunctionToolCall {
                id,
                name,
                arguments,
            } => Some((id.as_ref().map(|x| x.as_str()), name.as_str(), arguments)),
            _ => None,
        }
    }

    pub fn as_function_mut(&mut self) -> Option<(Option<&mut String>, &mut String, &mut Value)> {
        match self {
            Self::FunctionToolCall {
                id,
                name,
                arguments,
            } => Some((id.as_mut(), name, arguments)),
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
            Self::ImageContent {
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
            Part::TextReasoning { text, signature } => {
                let n = if signature.is_some() { 4 } else { 3 };
                let mut s = serializer.serialize_map(Some(n))?;
                s.serialize_entry("type", "text")?;
                s.serialize_entry("mode", "reasoning")?;
                s.serialize_entry("text", text)?;
                if let Some(sig) = signature {
                    s.serialize_entry("signature", sig)?;
                }
                s
            }
            Part::TextContent(text) => {
                let mut s = serializer.serialize_map(Some(2))?;
                s.serialize_entry("type", "text")?;
                s.serialize_entry("text", text)?;
                s
            }
            Part::ImageContent {
                h,
                w,
                c,
                nbytes: _,
                buf,
            } => {
                #[derive(Serialize)]
                struct ImgBase64<'a> {
                    height: &'a usize,
                    width: &'a usize,
                    colorspace: &'a PartImageColorspace,
                    data: &'a str, // base64
                }

                #[derive(Serialize)]
                struct ImgBin<'a> {
                    height: &'a usize,
                    width: &'a usize,
                    colorspace: &'a PartImageColorspace,
                    #[serde(with = "serde_bytes")]
                    data: &'a [u8], // raw bytes
                }

                let human_readable = serializer.is_human_readable();
                let mut s = serializer.serialize_map(Some(2))?;
                s.serialize_entry("type", "image")?;

                if human_readable {
                    s.serialize_entry(
                        "image",
                        &ImgBase64 {
                            height: h,
                            width: w,
                            colorspace: c,
                            data: &base64::engine::general_purpose::STANDARD.encode(buf),
                        },
                    )?;
                } else {
                    s.serialize_entry(
                        "image",
                        &ImgBin {
                            height: h,
                            width: w,
                            colorspace: c,
                            data: &buf[..],
                        },
                    )?;
                }
                s
            }
            Part::FunctionToolCall {
                id,
                name,
                arguments,
            } => {
                #[derive(Serialize)]
                struct Inner<'a> {
                    name: &'a String,
                    arguments: &'a Value,
                }

                let n = if id.is_some() { 3 } else { 2 };
                let mut s = serializer.serialize_map(Some(n))?;
                s.serialize_entry("type", "function")?;
                s.serialize_entry("function", &Inner { name, arguments })?;
                if let Some(id) = id {
                    s.serialize_entry("id", id)?;
                }
                s
            }
            Part::TextRefusal(_) => todo!(),
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
    TextReasoning {
        text: String,
        signature: Option<String>,
    },
    TextContent(String),
    TextToolCall(String),
    FunctionToolCall {
        id: Option<String>,
        name: String,
        arguments: String,
    },
    TextRefusal(String),
}

impl PartDelta {
    pub fn is_reasoning(&self) -> bool {
        match self {
            Self::TextReasoning { .. } => true,
            _ => false,
        }
    }

    pub fn is_content(&self) -> bool {
        match self {
            Self::TextContent { .. } => true,
            _ => false,
        }
    }

    pub fn is_tool_call(&self) -> bool {
        match self {
            Self::TextToolCall { .. } | Self::FunctionToolCall { .. } => true,
            _ => false,
        }
    }

    pub fn is_text(&self) -> bool {
        match self {
            Self::TextReasoning { .. } | Self::TextContent(..) | Self::TextToolCall(..) => true,
            _ => false,
        }
    }

    pub fn is_function(&self) -> bool {
        match self {
            Self::FunctionToolCall { .. } => true,
            _ => false,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::TextReasoning { text, .. } => Some(text.as_str()),
            Self::TextContent(text) => Some(text.as_str()),
            Self::TextToolCall(text) => Some(text.as_str()),
            _ => None,
        }
    }

    pub fn as_text_mut(&mut self) -> Option<&mut String> {
        match self {
            Self::TextReasoning { text, .. } => Some(text),
            Self::TextContent(text) => Some(text),
            Self::TextToolCall(text) => Some(text),
            _ => None,
        }
    }

    pub fn as_function(&self) -> Option<(Option<&str>, &str, &str)> {
        match self {
            Self::FunctionToolCall {
                id,
                name,
                arguments,
            } => Some((
                id.as_ref().map(|x| x.as_str()),
                name.as_str(),
                arguments.as_str(),
            )),
            _ => None,
        }
    }

    pub fn as_function_mut(&mut self) -> Option<(&mut String, &mut String)> {
        match self {
            Self::FunctionToolCall {
                name, arguments, ..
            } => Some((name, arguments)),
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
            (
                PartDelta::TextReasoning {
                    text: mut ltext,
                    signature: lsig,
                },
                PartDelta::TextReasoning {
                    text: rtext,
                    signature: rsig,
                },
            ) => {
                ltext.push_str(&rtext);
                let sig = match (lsig, rsig) {
                    (None, None) => None,
                    (None, Some(rsig)) => Some(rsig),
                    (Some(lsig), None) => Some(lsig),
                    (Some(_), Some(rsig)) => Some(rsig),
                };
                Ok(PartDelta::TextReasoning {
                    text: ltext,
                    signature: sig,
                })
            }
            (PartDelta::TextContent(mut lhs), PartDelta::TextContent(rhs)) => {
                lhs.push_str(&rhs);
                Ok(PartDelta::TextContent(lhs))
            }
            (PartDelta::TextToolCall(mut lhs), PartDelta::TextToolCall(rhs)) => {
                lhs.push_str(&rhs);
                Ok(PartDelta::TextToolCall(lhs))
            }
            (
                PartDelta::FunctionToolCall {
                    id: lid,
                    name: mut lname,
                    arguments: mut largs,
                },
                PartDelta::FunctionToolCall {
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
                Ok(PartDelta::FunctionToolCall {
                    id,
                    name: lname,
                    arguments: largs,
                })
            }
            _ => Err(()),
        }
    }

    fn finish(self) -> Result<Self::Item, String> {
        match self {
            PartDelta::Null => Ok(Part::text_content(String::new())),
            PartDelta::TextReasoning { text, signature } => {
                Ok(Part::TextReasoning { text, signature })
            }
            PartDelta::TextContent(s) => Ok(Part::text_content(s)),
            PartDelta::TextToolCall(s) => match serde_json::from_str::<Value>(&s) {
                Ok(root) => {
                    match (
                        root.pointer_as::<str>("/name"),
                        root.pointer_as::<str>("/arguments"),
                    ) {
                        (Some(name), Some(args)) => Ok(Part::function_tool_call(
                            name,
                            serde_json::from_str::<Value>(&args)
                                .map_err(|_| String::from("Invalid JSON"))?,
                        )),
                        _ => Err(String::from("Invalid function JSON")),
                    }
                }
                Err(_) => Err(String::from("Invalid JSON")),
            },
            PartDelta::FunctionToolCall {
                id,
                name,
                arguments,
            } => Ok(Part::FunctionToolCall {
                id,
                name,
                arguments: serde_json::from_str::<Value>(&arguments)
                    .map_err(|_| String::from("Invalid JSON"))?,
            }),
            PartDelta::TextRefusal(text) => Ok(Part::text_refusal(text)),
        }
    }
}
