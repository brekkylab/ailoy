use std::fmt;

use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self, MapAccess},
    ser::SerializeMap as _,
};

/// Represents one typed unit of message content.
///
/// A `Part` is a single element inside a message’s `content` (and, for tools, sometimes
/// under `tool_calls`). The enum itself is **transport-agnostic**; any OpenAI-style JSON
/// shape is produced/consumed by higher-level (de)serializers.
///
/// # Notes
/// - No validation is performed. It just store the value as-is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Part {
    /// Plain UTF-8 text.
    ///
    /// Without style, it can be serialized as:
    /// ```json
    /// { "type": "text", "text": "hello" }
    /// ```
    Text(String),

    /// The **verbatim string** of a tool/function payload as it was streamed/received
    /// (often a JSON string). This may be incomplete or invalid while streaming. It is
    /// intended for *as-is accumulation* and later parsing by the caller.
    ///
    /// ```json
    /// "{\"type\": \"function\", \"function\": \"...\""}
    /// ```
    FunctionString(String),

    ///  A **partially parsed** function.
    ///  - `id`: the `tool_call_id` to correlate results. Use an empty string if undefined.
    ///  - `name`: function/tool name. May be assembled from streaming chunks.
    ///  - `arguments`: raw arguments **string** (typically JSON), preserved verbatim.
    ///
    ///  Can be mapped to wire formats (e.g., OpenAI `tool_calls[].function`).
    ///
    /// ```json
    /// {
    ///   "id": "call_abc",
    ///   "type": "function",
    ///   "function": { "name": "weather", "arguments": "{ \"city\": \"Paris\" }" }
    /// }
    /// ```
    Function {
        id: String,
        name: String,
        arguments: String,
    },

    /// A web-addressable image URL (no fetching/validation is performed).
    /// ```json
    /// { "type": "image", "url": "https://example.com/cat.png" }
    /// ```
    ImageURL(String),

    /// Inline base64-encoded image bytes with MIME type (no decoding/validation is performed).
    /// ```json
    /// { "type": "image", "data": "<base64>" }
    /// ```
    ImageData(String, String),
}

impl Part {
    pub fn new_text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    pub fn new_function_string(fnstr: impl Into<String>) -> Self {
        Self::FunctionString(fnstr.into())
    }

    pub fn new_function(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self::Function {
            id: id.into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }

    pub fn new_image_url(url: impl Into<String>) -> Self {
        Self::ImageURL(url.into())
    }

    pub fn new_image_data(data: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self::ImageData(data.into(), mime_type.into())
    }

    /// Merges adjacent parts of the **same variant** in place:
    ///
    /// # Returns
    /// `None`` if successfully merged
    /// `Some(Value)` if something cannot be merged
    ///
    /// # Concatenation semantics
    /// - `Text` + `Text`: appends right to left.
    /// - `FunctionString` + `FunctionString`: appends right to left (for streaming).
    /// - `Function` + `Function`:  
    ///   - If both IDs are non-empty and **different**, denies merging (returns `Some(other)`).
    ///   - Otherwise, empty `id` on the left is filled from the right; `name` and `arguments`
    ///     are appended, then merge **succeeds** (`None`).
    /// - Any other pair: not mergeable; returns `Some(other)`.
    pub fn concatenate(&mut self, other: Self) -> Option<Self> {
        match (self, other) {
            (Part::Text(lhs), Part::Text(rhs)) => {
                lhs.push_str(&rhs);
                None
            }
            (Part::FunctionString(lhs), Part::FunctionString(rhs)) => {
                lhs.push_str(&rhs);
                None
            }
            (
                Part::Function {
                    id: id1,
                    name: name1,
                    arguments: arguments1,
                },
                Part::Function {
                    id: id2,
                    name: name2,
                    arguments: arguments2,
                },
            ) => {
                // Function ID changed: Treat as a different function call
                if !id1.is_empty() && !id2.is_empty() && id1 != &id2 {
                    Some(Part::Function {
                        id: id2,
                        name: name2,
                        arguments: arguments2,
                    })
                } else {
                    if id1.is_empty() {
                        *id1 = id2;
                    }
                    name1.push_str(&name2);
                    arguments1.push_str(&arguments2);
                    None
                }
            }
            (_, other) => Some(other),
        }
    }

    pub fn to_string(&self) -> Option<String> {
        match self {
            Part::Text(str) => Some(str.into()),
            Part::FunctionString(str) => Some(str.into()),
            Part::ImageURL(str) => Some(str.into()),
            Part::ImageData(data, mime_type) => {
                Some(format!("data:{};base64,{}", mime_type, data).to_owned())
            }
            _ => None,
        }
    }

    pub fn is_text(&self) -> bool {
        match self {
            Part::Text(_) => true,
            _ => false,
        }
    }
}

/// Describes how a [`Part`] should be mapped to a wire-format (key names & type tags).
///
/// `PartStyle` is a **pure naming schema**: it tells the serializer/deserializer which
/// field names to use for each logical piece of a [`Part`]. This allows you to stay
/// transport-agnostic while targeting different provider shapes (OpenAI, “parameters”
/// instead of “arguments”, custom `type` tags, etc.).
///
/// The defaults correspond to a common OpenAI-style mapping:
/// - Text      → `{ "type": "text",  "text":  "..." }`
/// - Function  → `{ "type": "function", "id": "...", "function": { "name": "...", "arguments": "..." } }`
/// - Image URL → `{ "type": "image", "url":   "..." }`
/// - Image B64 → `{ "type": "image", "data":  "..." }`
///
/// You can override individual fields to adapt to other APIs. For example, if an API uses
/// `"parameters"` instead of `"arguments"`, set `function_arguments_field = "parameters"`.
///
/// # Examples
/// Switching function arguments key:
/// ```rust
/// # use crate::value::PartStyle;
/// let mut style = PartStyle::default();
/// style.function_arguments_field = "parameters".into();
/// ```
///
/// Custom image mapping:
/// ```rust
/// # use crate::value::PartStyle;
/// let mut style = PartStyle::default();
/// style.image_url_type = "image_url".into(); // { "type": "image_url", "url": "..." }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartStyle {
    /// `{"type": "||HERE||", "text": "..."}`
    /// Default: `"text"`
    pub text_type: String,

    /// `{"type": "text", "||HERE||": "..."}`
    /// Default: `"text"`
    pub text_field: String,

    /// `{"type": "||HERE||", "id": "...", "function": { "name": "...", "arguments": "..." }}`
    /// Default: `"function"`
    pub function_type: String,

    /// `{"type": "function", "id": "...", "||HERE||": { "name": "...", "arguments": "..." }}`
    /// Default: `"function"`
    pub function_field: String,

    /// `{"type": "function", "||HERE||": "...", "function": { ... }}`
    /// Default: `"id"`
    pub function_id_field: String,

    /// Inside the `function` object: `{"||HERE||": "...", "arguments": "..." }`
    /// Default: `"name"`
    pub function_name_field: String,

    /// Inside the `function` object: `{"name": "...", "||HERE||": "..." }`
    /// Default: `"arguments"`
    pub function_arguments_field: String,

    /// `{"type": "||HERE||", "url": "http://..."}`
    /// Default: `"image"`
    pub image_url_type: String,

    /// `{"type": "image", "||HERE||": "http://..."}`
    /// Default: `"url"`
    pub image_url_field: String,

    /// `{"type": "||HERE||", "data": "<base64>" }`
    /// Default: `"image"`
    pub image_data_type: String,

    /// `{"type": "image", "||HERE||": "<base64>" }`
    /// Default: `"data"`
    pub image_data_field: String,
}

impl PartStyle {
    pub fn new() -> Self {
        Self::default()
    }

    /// Merge `other` into `self`, respecting defaults and rejecting conflicting overrides.
    ///
    /// A field from `other` is applied only if it differs from the default. If `self`
    /// already holds a non-default value for that field and it disagrees with `other`,
    /// this returns `Err("Conflicting style ...")`.
    pub fn update(&mut self, other: Self) -> Result<(), String> {
        let default = Self::default();
        let update_string_field = |key: &str,
                                   val_self: &mut String,
                                   val_other: String,
                                   val_default: &str|
         -> Result<(), String> {
            if val_other != val_default {
                if val_self != val_default && val_self != &val_other {
                    return Err(format!(
                        "Conflicting style in {} ({} vs. {})",
                        key, val_self, val_other
                    ));
                }
                *val_self = val_other;
            }
            Ok(())
        };
        update_string_field(
            "text_type",
            &mut self.text_type,
            other.text_type,
            &default.text_type,
        )?;
        update_string_field(
            "text_field",
            &mut self.text_field,
            other.text_field,
            &default.text_field,
        )?;
        update_string_field(
            "function_type",
            &mut self.function_type,
            other.function_type,
            &default.function_type,
        )?;
        update_string_field(
            "function_field",
            &mut self.function_field,
            other.function_field,
            &default.function_field,
        )?;
        update_string_field(
            "function_id_field",
            &mut self.function_id_field,
            other.function_id_field,
            &default.function_id_field,
        )?;
        update_string_field(
            "function_name_field",
            &mut self.function_name_field,
            other.function_name_field,
            &default.function_name_field,
        )?;
        update_string_field(
            "function_arguments_field",
            &mut self.function_arguments_field,
            other.function_arguments_field,
            &default.function_arguments_field,
        )?;
        update_string_field(
            "image_url_type",
            &mut self.image_url_type,
            other.image_url_type,
            &default.image_url_type,
        )?;
        update_string_field(
            "image_url_field",
            &mut self.image_url_field,
            other.image_url_field,
            &default.image_url_field,
        )?;
        update_string_field(
            "image_data_type",
            &mut self.image_data_type,
            other.image_data_type,
            &default.image_data_type,
        )?;
        update_string_field(
            "image_data_field",
            &mut self.image_data_field,
            other.image_data_field,
            &default.image_data_field,
        )?;
        Ok(())
    }
}

impl Default for PartStyle {
    fn default() -> Self {
        Self {
            text_type: String::from("text"),
            text_field: String::from("text"),
            function_type: String::from("function"),
            function_field: String::from("function"),
            function_id_field: String::from("id"),
            function_name_field: String::from("name"),
            function_arguments_field: String::from("arguments"),
            image_url_type: String::from("image"),
            image_url_field: String::from("url"),
            image_data_type: String::from("image"),
            image_data_field: String::from("data"),
        }
    }
}

/// A [`Part`] bundled with a concrete [`PartStyle`] to drive (de)serialization.
///
/// `StyledPart` keeps the **data** (`Part`) separate from the **mapping** (`PartStyle`).
/// All `new_*` constructors attach the default style; call [`StyledPart::with_style`]
/// to swap in a custom mapping before serialization.
///
/// # Serialization
/// - Emits a `type` tag and the style-specific keys for the chosen variant.
/// - `Function` emits an object under `style.function_field`, with per-key names taken
///   from `function_name_field` / `function_arguments_field`.
/// - `FunctionString` emits the raw payload under `style.function_field`.
///
/// # Deserialization
/// The visitor reads `"type"` and dispatches according to `PartStyle`. It accepts both
/// `"arguments"` and `"parameters"` (when present) and captures which was used so it can
/// round-trip the original key names through `PartStyle`.
///
/// # Example
/// Serialize a function call using `"parameters"` instead of `"arguments"`:
/// ```rust
/// # use ailoy::value::{Part, PartStyle, StyledPart};
/// let part = Part::new_function("", "foo", r#"{ "x": 1 }"#);
/// let mut style = PartStyle::default();
/// style.function_arguments_field = "parameters".into();
/// let styled = StyledPart { data: part, style };
/// let json = serde_json::to_string(&styled).unwrap();
/// assert!(json.contains("\"parameters\""));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StyledPart {
    pub data: Part,
    pub style: PartStyle,
}

impl StyledPart {
    pub fn new_text(text: impl Into<String>) -> Self {
        Self {
            data: Part::new_text(text),
            style: PartStyle::default(),
        }
    }

    pub fn new_function_string(fnstr: impl Into<String>) -> Self {
        Self {
            data: Part::new_function_string(fnstr),
            style: PartStyle::default(),
        }
    }

    pub fn new_function(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            data: Part::new_function(id, name, arguments),
            style: PartStyle::default(),
        }
    }

    pub fn new_image_url(url: impl Into<String>) -> Self {
        Self {
            data: Part::new_image_url(url),
            style: PartStyle::default(),
        }
    }

    pub fn new_image_data(data: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self {
            data: Part::new_image_data(data, mime_type),
            style: PartStyle::default(),
        }
    }

    pub fn with_style(self, style: PartStyle) -> Self {
        Self {
            data: self.data,
            style,
        }
    }

    pub fn is_text(&self) -> bool {
        match self.data {
            Part::Text(_) => true,
            _ => false,
        }
    }
}

impl fmt::Display for StyledPart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.data {
            Part::Text(text) => {
                f.write_fmt(format_args!("Part {{\"type\": \"{}\", \"{}\"=\"{}\"}}", self.style.text_type, self.style.text_field, text))?
            }
            Part::FunctionString(text) => f.write_fmt(format_args!(
                "Part {{\"type\": \"{}\", {}: \"{}\"}}",
                self.style.function_type, self.style.function_field, text
            ))?,
            Part::Function {
                id,
                name,
                arguments,
            } => f.write_fmt(format_args!(
                "Part {{\"type\": \"{}\", \"{}\"=\"{}\", \"{}\": {{\"{}\": \"{}\", \"{}\": \"{}\"}}}}",
                self.style.function_type,
                self.style.function_id_field,
                id,
                self.style.function_field,
                self.style.function_name_field,
                name,
                self.style.function_arguments_field,
                arguments
            ))?,
            Part::ImageURL(url) => f.write_fmt(format_args!(
                "Part {{\"type\": \"{}\", \"{}\"=\"{}\"}}",
                self.style.image_url_type,
                self.style.image_url_field,
                url
            ))?,
            Part::ImageData(data, mime_type) => f.write_fmt(format_args!(
                "Part {{\"type\": \"{}\", \"{}\"=({}, {} bytes)}}",
                self.style.image_data_type,
                self.style.image_data_field,
                mime_type,
                data.len()
            ))?,
        };
        Ok(())
    }
}

struct PartFunctionRef<'a> {
    name: &'a str,
    arguments: &'a str,
    style: &'a PartStyle,
}

impl<'a> Serialize for PartFunctionRef<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(None)?;
        if !self.name.is_empty() {
            map.serialize_entry(&self.style.function_name_field, self.name)?;
        }
        if !self.arguments.is_empty() {
            map.serialize_entry(&self.style.function_arguments_field, self.arguments)?;
        }
        map.end()
    }
}

struct PartFunctionOwned {
    name: String,
    arguments: String,
    style: PartStyle,
}

impl<'de> Deserialize<'de> for PartFunctionOwned {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(PartFunctionVisitor)
    }
}

struct PartFunctionVisitor;

impl<'de> de::Visitor<'de> for PartFunctionVisitor {
    type Value = PartFunctionOwned;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str(r#"a map with "type" field and one other content key"#)
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut name: Option<(String, String)> = None;
        let mut arguments: Option<(String, String)> = None;
        while let Some(k) = map.next_key::<String>()? {
            if k == "name" {
                name = Some((String::from("name"), map.next_value()?));
            } else if k == "arguments" {
                arguments = Some((String::from("arguments"), map.next_value()?));
            } else if k == "parameters" {
                arguments = Some((String::from("parameters"), map.next_value()?));
            }
        }
        let mut rv = PartFunctionOwned {
            name: String::new(),
            arguments: String::new(),
            style: PartStyle::new(),
        };
        if let Some((name_key, name)) = name {
            rv.style.function_name_field = name_key;
            rv.name = name;
        }
        if let Some((arguments_key, arguments)) = arguments {
            rv.style.function_arguments_field = arguments_key;
            rv.arguments = arguments;
        }
        Ok(rv)
    }
}

impl Serialize for StyledPart {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(2))?;
        match &self.data {
            Part::Text(text) => {
                map.serialize_entry("type", &self.style.text_type)?;
                map.serialize_entry(&self.style.text_field, text)?;
            }
            Part::FunctionString(function_string) => {
                map.serialize_entry("type", &self.style.function_type)?;
                map.serialize_entry(&self.style.function_field, function_string)?;
            }
            Part::Function {
                id,
                name,
                arguments,
            } => {
                map.serialize_entry("type", &self.style.function_type)?;
                if !id.is_empty() {
                    map.serialize_entry(&self.style.function_id_field, &id)?;
                }
                map.serialize_entry(
                    &self.style.function_field,
                    &PartFunctionRef {
                        name,
                        arguments,
                        style: &self.style,
                    },
                )?;
            }
            Part::ImageURL(url) => {
                map.serialize_entry("type", &self.style.image_url_type)?;
                map.serialize_entry(&self.style.image_url_field, url.as_str())?;
            }
            Part::ImageData(..) => {
                map.serialize_entry("type", &self.style.image_data_type)?;
                map.serialize_entry(&self.style.image_data_field, &self.data.to_string())?;
            }
        };
        map.end()
    }
}

impl<'de> Deserialize<'de> for StyledPart {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(PartVisitor)
    }
}

struct PartVisitor;

impl PartVisitor {
    fn parse_base64_data_url(&self, data_url: &str) -> Option<(String, String)> {
        let parts: Vec<&str> = data_url.splitn(2, ',').collect();
        if parts.len() != 2 {
            return None;
        }

        let header = parts[0];
        let data = parts[1];

        // Check if header starts with "data:" and ends with ";base64"
        if !header.starts_with("data:") || !header.ends_with(";base64") {
            return None;
        }

        // Extract mime type (between "data:" and ";base64")
        let mime_type = &header[5..header.len() - 7]; // Remove "data:" and ";base64"

        Some((mime_type.to_string(), data.to_string()))
    }
}

impl<'de> de::Visitor<'de> for PartVisitor {
    type Value = StyledPart;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str(r#"a map with "type" field and one other content key"#)
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut ty: Option<String> = None;
        let mut id: Option<(String, String)> = None;
        let mut function_field: Option<(String, PartFunctionOwned)> = None;
        let mut field: Option<(String, String)> = None;

        while let Some(k) = map.next_key::<String>()? {
            if k == "type" {
                if ty.is_some() {
                    return Err(de::Error::duplicate_field("type"));
                }
                ty = Some(map.next_value()?);
            } else if k == "id" {
                if id.is_some() {
                    return Err(de::Error::custom(format!(
                        "multiple id fields found ({} & {})",
                        id.unwrap().0,
                        k
                    )));
                }
                id = Some((String::from("id"), map.next_value()?));
            } else if k == "function_id" {
                if id.is_some() {
                    return Err(de::Error::custom(format!(
                        "multiple id fields found ({} & {})",
                        id.unwrap().0,
                        k
                    )));
                }
                id = Some((String::from("function_id"), map.next_value()?));
            } else if k == "function" {
                #[derive(Deserialize)]
                #[serde(untagged)]
                enum FunctionEither {
                    String(String),
                    Object(PartFunctionOwned),
                }
                if function_field.is_some() {
                    return Err(de::Error::custom(format!(
                        "multiple part fields found ({} & {})",
                        function_field.unwrap().0,
                        k
                    )));
                }
                if field.is_some() {
                    return Err(de::Error::custom(format!(
                        "multiple part fields found ({} & {})",
                        field.unwrap().0,
                        k
                    )));
                }
                match map.next_value::<FunctionEither>()? {
                    FunctionEither::String(v) => {
                        field = Some((k, v));
                    }
                    FunctionEither::Object(v) => {
                        function_field = Some((k, v));
                    }
                };
            } else if k == "text"
                || k == "url"
                || k == "data"
                || k == "image_url"
                || k == "image_data"
                || k == "audio_url"
                || k == "audio_data"
            {
                if field.is_some() {
                    return Err(de::Error::custom(format!(
                        "multiple part fields found ({} & {})",
                        field.unwrap().0,
                        k
                    )));
                }
                field = Some((k, map.next_value()?));
            }
        }

        let mut style = PartStyle::new();
        match (ty, field, id, function_field) {
            (_, None, _, None) => Err(de::Error::custom("Missing part field")),
            (ty, Some((k, v)), _, _) if k == "text" => {
                if let Some(ty) = ty {
                    style.text_type = ty;
                }
                style.text_field = k;
                Ok(StyledPart::new_text(v).with_style(style))
            }
            (ty, Some((k, v)), _, _) if k == "function" => {
                if let Some(ty) = ty {
                    style.text_type = ty;
                }
                style.function_field = k;
                Ok(StyledPart::new_function_string(v).with_style(style))
            }
            (ty, _, id, Some((k, v))) if k == "function" => {
                if let Some(ty) = ty {
                    style.text_type = ty;
                }
                style.function_field = k;
                if let Some((idk, idv)) = id {
                    style.function_id_field = idk;
                    style.function_name_field = v.style.function_name_field;
                    style.function_arguments_field = v.style.function_arguments_field;
                    Ok(StyledPart::new_function(idv, v.name, v.arguments).with_style(style))
                } else {
                    style.function_name_field = v.style.function_name_field;
                    style.function_arguments_field = v.style.function_arguments_field;
                    Ok(StyledPart::new_function(String::new(), v.name, v.arguments)
                        .with_style(style))
                }
            }
            (Some(ty), Some((k, v)), _, _)
                if (ty == "image" || ty == "image_url") && (k == "url" || k == "image_url") =>
            {
                style.image_url_type = ty;
                style.image_url_field = k;
                Ok(StyledPart::new_image_url(v).with_style(style))
            }
            (Some(ty), Some((k, v)), _, _)
                if (ty == "image" || ty == "image_data")
                    && (k == "data"
                        || k == "image_data"
                        || k == "base64"
                        || k == "image_base64") =>
            {
                style.image_url_type = ty;
                style.image_url_field = k;
                if let Some((mime_type, base64)) = self.parse_base64_data_url(&v) {
                    Ok(StyledPart::new_image_data(base64, mime_type).with_style(style))
                } else {
                    Err(de::Error::custom("Invalid base64 data url"))
                }
            }
            (Some(ty), Some((k, v)), _, _) => Err(de::Error::custom(format!(
                "Invalid Part format(type: {}, {}: {})",
                ty, k, v
            ))),
            _ => Err(de::Error::custom("Invalid Part format")),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn part_serde_default() {
        {
            let part = StyledPart::new_text("This is a text");
            let s = serde_json::to_string(&part).unwrap();
            assert_eq!(s, r#"{"type":"text","text":"This is a text"}"#);
            let roundtrip = serde_json::from_str::<StyledPart>(&s).unwrap();
            assert_eq!(roundtrip, part);
        }
        {
            let part = StyledPart::new_function("asdf1234", "fn", "false");
            let s = serde_json::to_string(&part).unwrap();
            assert_eq!(
                s,
                r#"{"type":"function","id":"asdf1234","function":{"name":"fn","arguments":false}}"#
            );
            let roundtrip = serde_json::from_str::<StyledPart>(&s).unwrap();
            assert_eq!(roundtrip, part);
        }
    }

    #[test]
    fn part_serde_with_format() {
        let mut style = PartStyle::new();
        style.text_type = String::from("text1");
        style.text_field = String::from("text2");
        {
            let part = StyledPart::new_text("This is a text").with_style(style);
            let s = serde_json::to_string(&part).unwrap();
            assert_eq!(s, r#"{"type":"text1","text2":"This is a text"}"#);
            let roundtrip = serde_json::from_str::<StyledPart>(&s).unwrap();
            assert_eq!(roundtrip, part);
        }
    }
}
