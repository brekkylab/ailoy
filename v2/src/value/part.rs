use crate::value::{Value, delta::Delta};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Part {
    TextReasoning {
        text: String,
        signature: Option<String>,
    },
    TextContent(String),
    ImageContent {
        h: usize,
        w: usize,
        // c == 1 (grayscale), c == 2 (grayscale + alpha), c == 3 (RGB), c == 4 (RGBA)
        c: usize,
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
        channel: usize,
        buf: impl IntoIterator<Item = u8>,
    ) -> Self {
        let buf = buf.into_iter().collect::<Vec<_>>();
        let nbytes = buf.len() / height / width / channel;
        if !(nbytes == 1 || nbytes == 2 || nbytes == 3 || nbytes == 4) {
            panic!("Invalid buffer length");
        }
        Self::ImageContent {
            h: height,
            w: width,
            c: channel,
            nbytes,
            buf,
        }
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
                match (*c, *nbytes) {
                    // Grayscale 8-bit
                    (1, 1) => {
                        let buf = image::GrayImage::from_raw(w, h, buf.clone())?;
                        Some(image::DynamicImage::ImageLuma8(buf))
                    }
                    // Grayscale 16-bit
                    (1, 2) => {
                        let buf = image::ImageBuffer::<image::Luma<u16>, _>::from_raw(
                            w,
                            h,
                            bytes_to_u16_ne(buf)?,
                        )?;
                        Some(image::DynamicImage::ImageLuma16(buf))
                    }
                    // Grayscale + Alpha (8-bit each)
                    (2, 1) => {
                        let buf = image::GrayAlphaImage::from_raw(w, h, buf.clone())?;
                        Some(image::DynamicImage::ImageLumaA8(buf))
                    }
                    // RGB 8-bit
                    (3, 1) => {
                        let buf = image::RgbImage::from_raw(w, h, buf.clone())?;
                        Some(image::DynamicImage::ImageRgb8(buf))
                    }
                    // RGBA 8-bit
                    (4, 1) => {
                        let buf = image::RgbaImage::from_raw(w, h, buf.clone())?;
                        Some(image::DynamicImage::ImageRgba8(buf))
                    }
                    // RGB 16-bit
                    (3, 2) => {
                        let buf = image::ImageBuffer::<image::Rgb<u16>, _>::from_raw(
                            w,
                            h,
                            bytes_to_u16_ne(buf)?,
                        )?;
                        Some(image::DynamicImage::ImageRgb16(buf))
                    }
                    // RGBA 16-bit
                    (4, 2) => {
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
