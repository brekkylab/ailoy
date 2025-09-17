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
    Think,
    Content,
    ToolCall,
    Refusal,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Part {
    data: Data,
    mode: Mode,
}

impl Part {
    // pub fn text(text: impl Into<String>) -> Self {
    //     Self::Text(text.into())
    // }

    // pub fn function_string(function: impl Into<String>) -> Self {
    //     Self::Function(PartFunction::String(function.into()))
    // }

    // pub fn function(name: impl Into<String>, args: impl Into<String>) -> Self {
    //     Self::Function(PartFunction::Parsed {
    //         name: name.into(),
    //         args: args.into(),
    //     })
    // }

    // pub fn image_url(url: impl Into<String>) -> Self {
    //     Self::Image(PartMedia::URL(url.into()))
    // }

    // pub fn image_base64(data: impl Into<String>, media_type: impl Into<String>) -> Self {
    //     Self::Image(PartMedia::Base64 {
    //         data: data.into(),
    //         media_type: media_type.into(),
    //     })
    // }

    // pub fn is_text(&self) -> bool {
    //     match self {
    //         Part::Text(_) => true,
    //         _ => false,
    //     }
    // }

    // pub fn is_function(&self) -> bool {
    //     match self {
    //         Part::Function(_) => true,
    //         _ => false,
    //     }
    // }

    // pub fn is_image(&self) -> bool {
    //     match self {
    //         Part::Image(_) => true,
    //         _ => false,
    //     }
    // }

    // /// Merges adjacent parts of the **same variant** in place:
    // ///
    // /// # Returns
    // /// `None`` if successfully merged
    // /// `Some(Value)` if something cannot be merged
    // ///
    // /// # Concatenation semantics
    // /// - `Text` + `Text`: appends right to left.
    // /// - `FunctionString` + `FunctionString`: appends right to left (for streaming).
    // /// - `Function` + `Function`:
    // ///   - If both IDs are non-empty and **different**, denies merging (returns `Some(other)`).
    // ///   - Otherwise, empty `id` on the left is filled from the right; `name` and `arguments`
    // ///     are appended, then merge **succeeds** (`None`).
    // /// - Any other pair: not mergeable; returns `Some(other)`.
    // pub fn concatenate(&mut self, other: Self) -> Option<Self> {
    //     match (self, other) {
    //         (Part::Text(lhs), Part::Text(rhs)) => {
    //             lhs.push_str(&rhs);
    //             None
    //         }
    //         (Part::FunctionString(lhs), Part::FunctionString(rhs)) => {
    //             lhs.push_str(&rhs);
    //             None
    //         }
    //         (
    //             Part::Function {
    //                 id: id1,
    //                 name: name1,
    //                 arguments: arguments1,
    //             },
    //             Part::Function {
    //                 id: id2,
    //                 name: name2,
    //                 arguments: arguments2,
    //             },
    //         ) => {
    //             // Function ID changed: Treat as a different function call
    //             if !id1.is_empty() && !id2.is_empty() && id1 != &id2 {
    //                 Some(Part::Function {
    //                     id: id2,
    //                     name: name2,
    //                     arguments: arguments2,
    //                 })
    //             } else {
    //                 if id1.is_empty() {
    //                     *id1 = id2;
    //                 }
    //                 name1.push_str(&name2);
    //                 arguments1.push_str(&arguments2);
    //                 None
    //             }
    //         }
    //         (_, other) => Some(other),
    //     }
    // }

    // pub fn to_string(&self) -> Option<String> {
    //     match self {
    //         Part::Text(str) => Some(str.into()),
    //         Part::FunctionString(str) => Some(str.into()),
    //         Part::ImageURL(str) => Some(str.into()),
    //         Part::ImageData(data, mime_type) => {
    //             Some(format!("data:{};base64,{}", mime_type, data).to_owned())
    //         }
    //         _ => None,
    //     }
    // }
}
