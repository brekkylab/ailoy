use std::fmt;

use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self, MapAccess},
    ser::SerializeMap as _,
};

/// Represents a single, typed unit of message content (multi-modal).
///
/// A `Part` models one element inside a message’s `content` (and related fields like
/// `reasoning` / `tool_calls`). It is designed to be **OpenAI-compatible** when serialized,
/// using the modern “array-of-parts” shape such as:
/// `{ "type": "text", "text": "..." }` or `{ "type": "image", "url": "..." }`.
///
/// # Variants
/// - [`Part::Text`]: Plain UTF-8 text.
/// - [`Part::Function`]: Raw JSON string for a tool/function call payload, with an optional
///   `tool_call_id` to correlate results.
/// - [`Part::ImageURL`]: A web-accessible image (HTTP(S) URL).
/// - [`Part::ImageData`]: An inline, base64-encoded image payload.
///
/// # OpenAI-compatible mapping (typical)
/// These are the common wire shapes produced/consumed when targeting OpenAI-style APIs:
///
/// - `Text(s)`
///   ```json
///   { "type": "text", "text": "hello" }
///   ```
///
/// - `ImageURL(url)`
///   ```json
///   { "type": "image", "url": "https://example.com/cat.png" }
///   ```
///
/// - `ImageBase64(data)`
///   ```json
///   { "type": "image", "data": "<base64-bytes>" }
///   ```
///
/// - `Function { id, function }`
///   The `function` field holds the **raw JSON** for the tool/function call (often the
///   `arguments` string in OpenAI tool calls). The `id` corresponds to `tool_call_id`
///   and is used to link the tool’s eventual result back to this call. Serialization of
///   this variant is handled by higher-level message (de)serializers that place it under
///   `tool_calls` as appropriate.
///
/// # Notes on `Function`
/// - **As-is storage:** While streaming, `function` may be incomplete or invalid JSON.
///   This is expected. Parse only after the stream has finalized/aggregated.
/// - **No validation:** The variant does not validate or mutate the JSON string.
///   Converting into a strongly-typed [`crate::value::ToolCall`] will fail if the JSON
///   is malformed or incomplete.
/// - **Correlation:** When present, `id` should be echoed back as `tool_call_id` when
///   returning the tool’s output, matching OpenAI’s linking behavior.
///
/// # Invariants & behavior
/// - Insertion order is preserved by the container (e.g., `Message.content`).
/// - Image parts do not fetch or validate URLs/bytes at construction time.
/// - The enum is transport-agnostic; OpenAI compatibility is achieved by the
///   surrounding (de)serializer layer.
///
/// # Examples
/// Building a multi-modal user message:
/// ```rust
/// # use url::Url;
/// # use crate::value::{Message, Part, Role};
/// let mut msg = Message::new(Role::User);
/// msg.push_content(Part::Text("What does this sign say?".into()));
/// msg.push_content(Part::ImageURL(Url::parse("https://example.com/sign.jpg").unwrap()));
/// ```
///
/// Collecting tool-call arguments during streaming:
/// ```rust
/// # use crate::value::Part;
/// let mut buf = String::new();
/// // Append chunks as they arrive:
/// buf.push_str("{\"location\":\"Paris\"");
/// buf.push_str(",\"unit\":\"Celsius\"}");
/// let part = Part::Function { id: None, function: buf }; // parse after finalization
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Part {
    /// Plain UTF-8 text.
    Text(String),

    /// Raw function, holding JSON call as a string. holds the unmodified JSON; it may be incomplete while streaming.
    /// serialized as a string: `"\{\"name\": \"function_name\", \"arguments\": \"...\""}`
    FunctionString(String),

    /// Tool/function call payload captured.
    ///
    /// `id` corresponds to `tool_call_id` (when available) for correlating tool results.
    /// Empty string if it has no value
    /// Note that argument is raw string, (it is actually JSON)
    /// serialized as an object: `{\"name\": \"function_name\", \"arguments\": \"...\""}``
    Function {
        /// Optional `tool_call_id` used to correlate tool results.
        id: String,
        name: String,
        arguments: String,
    },

    /// Web-addressable image (HTTP(S) URL).
    ///
    /// Typically serialized as:
    /// `{ "type": "image", "url": "<...>" }`.
    ImageURL(String),

    /// Inline, base64-encoded image bitmap bytes.
    ///
    /// Typically serialized as:
    /// `{ "type": "image", "data": "<base64>" }`.
    ImageData(String),
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

    pub fn new_image_data(data: impl Into<String>) -> Self {
        Self::ImageData(data.into())
    }

    /// Returns none if successfully merged
    /// Some(Value) if it cannot be merged (the portion of cannot be merged)
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
            (Part::ImageURL(lhs), Part::ImageURL(rhs)) => {
                lhs.push_str(&rhs);
                None
            }
            (Part::ImageData(lhs), Part::ImageData(rhs)) => {
                lhs.push_str(&rhs);
                None
            }
            (_, other) => Some(other),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Part::Text(str) => Some(str.as_str()),
            Part::FunctionString(str) => Some(str.as_str()),
            Part::ImageURL(str) => Some(str.as_str()),
            Part::ImageData(str) => Some(str.as_str()),
            _ => None,
        }
    }

    pub fn as_mut_string(&mut self) -> Option<&mut String> {
        match self {
            Part::Text(str) => Some(str),
            Part::FunctionString(str) => Some(str),
            Part::ImageURL(str) => Some(str),
            Part::ImageData(str) => Some(str),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartStyle {
    /// {"type": "||HERE||", "text": "It's a text..."}
    /// default: "text"
    pub text_type: String,

    /// {"type": "text", "||HERE||": "It's a text..."}
    /// default: "text"
    pub text_field: String,

    /// {"type": "||HERE||", "id": "1234asdf", "function": {"name": "function name", "arguments": "function args"}}
    /// default: "function"
    pub function_type: String,

    /// {"type": "function", "id": "1234asdf", "||HERE||": {"name": "function name", "arguments": "function args"}}
    /// default: "function"
    pub function_field: String,

    /// {"type": "function", "||HERE||": "1234asdf", "function": {"name": "function name", "arguments": "function args"}}
    /// default: "id"
    pub function_id_field: String,

    /// {"type": "function", "id": "1234asdf", "function": {"||HERE||": "function name", "arguments": "function args"}}
    /// default: "name"
    pub function_name_field: String,

    /// {"type": "function", "id": "1234asdf", "function": {"name": "function name", "||HERE||": "function args"}}
    /// default: "arguments"
    pub function_arguments_field: String,

    /// {"type": "||HERE||", "url": "http://..."}
    /// default: "image"
    pub image_url_type: String,

    /// {"type": "image", "||HERE||": "http://..."}
    /// default: "url"
    pub image_url_field: String,

    /// {"type": "||HERE||", "data": "base64 encoded bytes..."}
    /// default: "image"
    pub image_data_type: String,

    /// {"type": "image", "||HERE||": "base64 encoded bytes..."}
    /// default: "data"
    pub image_data_field: String,
}

impl PartStyle {
    pub fn new() -> Self {
        Self::default()
    }

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

    pub fn new_image_data(data: impl Into<String>) -> Self {
        Self {
            data: Part::new_image_data(data),
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
            Part::ImageData(data) => f.write_fmt(format_args!(
                "Part {{\"type\": \"{}\", \"{}\"=({} bytes)}}",
                self.style.image_data_type,
                self.style.image_data_field,
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
            Part::ImageData(encoded) => {
                map.serialize_entry("type", &self.style.image_data_type)?;
                map.serialize_entry(&self.style.image_data_field, encoded.as_str())?;
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
                Ok(StyledPart::new_image_data(v).with_style(style))
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
