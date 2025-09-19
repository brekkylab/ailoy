use serde::Deserialize;
use serde_json::from_str;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FunctionData {
    /// The **verbatim string** of a tool/function payload as it was streamed/received
    /// (often a JSON string). This may be incomplete or invalid while streaming. It is
    /// intended for *as-is accumulation* and later parsing by the caller.
    String(String),

    ///  A parsed function.
    ///  - `name`: function/tool name. May be assembled from streaming chunks.
    ///  - `args`: raw arguments **string** (typically JSON), preserved verbatim.
    Parsed { name: String, args: String },
}

impl FunctionData {
    /// Parse raw function string to parsed one
    pub fn parse(self) -> Result<(String, String), String> {
        match self {
            Self::String(s) => {
                #[derive(Deserialize)]
                struct Inner {
                    name: String,
                    args: String,
                }
                let parsed: Inner = from_str(&s).unwrap();
                Ok((parsed.name, parsed.args))
            }
            Self::Parsed { name, args } => Ok((name, args)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaData {
    /// A web-addressable image URL (no fetching/validation is performed).
    URL(String),

    /// Inline base64-encoded image bytes with IANA media type
    Base64 { media_type: String, data: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Data {
    /// Plain UTF-8 text
    Text(String),

    /// Function call
    Function(FunctionData),

    /// Image
    Image(MediaData),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mode {
    Think { signature: Option<String> },
    Content,
    ToolCall { id: Option<String> },
    Refusal,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Part {
    pub data: Data,
    pub mode: Mode,
}

#[derive(Clone, Debug)]
pub struct PartBuilder {
    data: Option<Data>,
    mode: Mode,
}

impl PartBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn data(self, data: Data) -> Self {
        Self {
            data: Some(data),
            mode: self.mode,
        }
    }

    pub fn text(self, data: impl Into<String>) -> Self {
        Self {
            data: Some(Data::Text(data.into())),
            mode: self.mode,
        }
    }

    pub fn function_string(self, data: impl Into<String>) -> Self {
        Self {
            data: Some(Data::Function(FunctionData::String(data.into()))),
            mode: self.mode,
        }
    }

    pub fn function(self, name: impl Into<String>, args: impl Into<String>) -> Self {
        Self {
            data: Some(Data::Function(FunctionData::Parsed {
                name: name.into(),
                args: args.into(),
            })),
            mode: self.mode,
        }
    }

    pub fn image_url(self, url: impl Into<String>) -> Self {
        Self {
            data: Some(Data::Image(MediaData::URL(url.into()))),
            mode: self.mode,
        }
    }

    pub fn image_base64(self, media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self {
            data: Some(Data::Image(MediaData::Base64 {
                media_type: media_type.into(),
                data: data.into(),
            })),
            mode: self.mode,
        }
    }

    pub fn mode(self, mode: Mode) -> Self {
        Self {
            data: self.data,
            mode,
        }
    }

    pub fn think(self) -> Self {
        Self {
            data: self.data,
            mode: Mode::Think { signature: None },
        }
    }

    pub fn think_with_signature(self, sig: impl Into<String>) -> Self {
        Self {
            data: self.data,
            mode: Mode::Think {
                signature: Some(sig.into()),
            },
        }
    }

    pub fn content(self) -> Self {
        Self {
            data: self.data,
            mode: Mode::Content,
        }
    }

    pub fn tool_call(self) -> Self {
        Self {
            data: self.data,
            mode: Mode::ToolCall { id: None },
        }
    }

    pub fn tool_call_with_id(self, id: impl Into<String>) -> Self {
        Self {
            data: self.data,
            mode: Mode::ToolCall {
                id: Some(id.into()),
            },
        }
    }

    pub fn build(self) -> Result<Part, ()> {
        if let Some(data) = self.data {
            Ok(Part {
                data,
                mode: self.mode,
            })
        } else {
            Err(())
        }
    }
}

impl Default for PartBuilder {
    fn default() -> Self {
        Self {
            data: None,
            mode: Mode::Content,
        }
    }
}
