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
    pub fn parse(self) -> Self {
        match self {
            Self::String(_) => todo!(),
            _ => self,
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
        Self {
            data: None,
            mode: Mode::Content,
        }
    }

    pub fn data(&mut self, data: Data) -> &mut Self {
        self.data = Some(data);
        self
    }

    pub fn text(&mut self, data: impl Into<String>) -> &mut Self {
        self.data = Some(Data::Text(data.into()));
        self
    }

    pub fn function_string(&mut self, data: impl Into<String>) -> &mut Self {
        self.data = Some(Data::Function(FunctionData::String(data.into())));
        self
    }

    pub fn function(&mut self, name: impl Into<String>, args: impl Into<String>) -> &mut Self {
        self.data = Some(Data::Function(FunctionData::Parsed {
            name: name.into(),
            args: args.into(),
        }));
        self
    }

    pub fn mode(&mut self, mode: Mode) -> &mut Self {
        self.mode = mode;
        self
    }

    pub fn think(&mut self) -> &mut Self {
        self.mode = Mode::Think { signature: None };
        self
    }

    pub fn think_with_signature(&mut self, sig: impl Into<String>) -> &mut Self {
        self.mode = Mode::Think {
            signature: Some(sig.into()),
        };
        self
    }

    pub fn content(&mut self) -> &mut Self {
        self.mode = Mode::Content;
        self
    }

    pub fn tool_call(&mut self) -> &mut Self {
        self.mode = Mode::ToolCall { id: None };
        self
    }

    pub fn tool_call_with_id(&mut self, id: impl Into<String>) -> &mut Self {
        self.mode = Mode::ToolCall {
            id: Some(id.into()),
        };
        self
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
