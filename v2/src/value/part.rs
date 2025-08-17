use std::fmt;

use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self, MapAccess},
    ser::SerializeMap as _,
};
use url::Url;

use crate::value::{ToolCall, ToolCallFmt, ToolCallWithFmt};

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
/// buf.push_str("{\"location\":\"Dubai\"");
/// buf.push_str(",\"unit\":\"Celsius\"}");
/// let part = Part::Function { id: None, function: buf }; // parse after finalization
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Part {
    /// Plain UTF-8 text.
    Text(String),

    /// Tool/function call payload captured as a raw JSON string.
    ///
    /// `id` corresponds to `tool_call_id` (when available) for correlating tool results.
    /// `function` holds the unmodified JSON; it may be incomplete while streaming.
    Function {
        /// Optional `tool_call_id` used to correlate tool results.
        id: Option<String>,
        /// Raw JSON for the function call payload (often the `arguments` string).
        function: ToolCall,
    },

    /// Web-addressable image (HTTP(S) URL).
    ///
    /// Typically serialized as:
    /// `{ "type": "image", "url": "<...>" }`.
    ImageURL(Url),

    /// Inline, base64-encoded image bitmap bytes.
    ///
    /// Typically serialized as:
    /// `{ "type": "image", "data": "<base64>" }`.
    ImageData(String),

    /// Web-addressable audio (HTTP(S) URL).
    ///
    /// Typically serialized as:
    /// `{ "type": "audio", "url": "<...>" }`.
    AudioURL(Url),

    /// Inline, base64-encoded audio.
    ///
    /// Typically serialized as:
    /// `{ "type": "audio", "data": "<base64>" }`.
    AudioData(String),
}

impl Part {
    pub fn new_text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    pub fn is_text(&self) -> bool {
        match self {
            Part::Text(_) => true,
            _ => false,
        }
    }

    pub fn get_text(&self) -> Option<&str> {
        match self {
            Part::Text(str) => Some(str.as_str()),
            _ => None,
        }
    }

    pub fn get_text_mut(&mut self) -> Option<&mut String> {
        match self {
            Part::Text(str) => Some(str),
            _ => None,
        }
    }

    pub fn new_function(id: Option<String>, function: impl Into<ToolCall>) -> Self {
        Self::Function {
            id,
            function: function.into(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Part::Text(v) => v.is_empty(),
            Part::Function { .. } => false,
            Part::ImageURL(_) => false,
            Part::ImageData(v) => v.is_empty(),
            Part::AudioURL(_) => false,
            Part::AudioData(v) => v.is_empty(),
        }
    }
}

impl fmt::Display for Part {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Part::Text(text) => {
                f.write_fmt(format_args!("Part {{ type=\"text\", text=\"{}\" }}", text))?
            }
            Part::Function { id, function } => f.write_fmt(format_args!(
                "Part {{ type=\"function\", id=\"{}\", function=\"{}\" }}",
                if let Some(id) = id { id } else { "null" },
                function
            ))?,
            Part::ImageURL(url) => f.write_fmt(format_args!(
                "Part {{ type=\"image\", url=\"{}\" }}",
                url.to_string()
            ))?,
            Part::ImageData(data) => f.write_fmt(format_args!(
                "Part {{ type=\"image\", data=({} bytes) }}",
                data.len()
            ))?,
            Part::AudioURL(url) => f.write_fmt(format_args!(
                "Part {{ type=\"audio\", url=\"{}\" }}",
                url.to_string()
            ))?,
            Part::AudioData(data) => f.write_fmt(format_args!(
                "Part {{ type=\"audio\", data=({} bytes) }}",
                data.len()
            ))?,
        };
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct PartFmt {
    pub tool_call_fmt: ToolCallFmt,

    /// {"type": "<HERE>", "text": "It's a text..."}
    /// default: "text"
    pub text_type: String,

    /// {"type": "text", "<HERE>": "It's a text..."}
    /// default: "text"
    pub text_field: String,

    /// {"type": "<HERE>", "id": "1234asdf", "function": {"name": "function name", "arguments": "function args"}}
    /// default: "function"
    pub function_type: String,

    /// {"type": "function", "id": "1234asdf", "<HERE>": {"name": "function name", "arguments": "function args"}}
    /// default: "function"
    pub function_field: String,

    /// {"type": "function", "<HERE>": "1234asdf", "function": {"name": "function name", "arguments": "function args"}}
    /// default: "function"
    pub function_id_field: String,

    /// {"type": "function", "id": null, "function": (tool call)}
    /// it the value is true, put `"id": null` markers
    /// if the value is false, no field.
    /// default: false
    pub function_id_null_marker: bool,

    /// {"type": "<HERE>", "url": "http://..."}
    /// default: "image"
    pub image_url_type: String,

    /// {"type": "image", "<HERE>": "http://..."}
    /// default: "url"
    pub image_url_field: String,

    /// {"type": "<HERE>", "data": "base64 encoded bytes..."}
    /// default: "image"
    pub image_data_type: String,

    /// {"type": "image", "<HERE>": "base64 encoded bytes..."}
    /// default: "data"
    pub image_data_field: String,

    /// {"type": "<HERE>", "url": "http://..."}
    /// default: "audio"
    pub audio_url_type: String,

    /// {"type": "audio", "<HERE>": "http://..."}
    /// default: "url"
    pub audio_url_field: String,

    /// {"type": "<HERE>", "data": "base64 encoded bytes..."}
    /// default: "audio"
    pub audio_data_type: String,

    /// {"type": "audio", "<HERE>": "base64 encoded bytes..."}
    /// default: "data"
    pub audio_data_field: String,
}

impl PartFmt {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_tool_call_fmt(mut self, tc_fmt: ToolCallFmt) -> Self {
        self.tool_call_fmt = tc_fmt;
        self
    }
}

impl Default for PartFmt {
    fn default() -> Self {
        Self {
            tool_call_fmt: ToolCallFmt::default(),
            text_type: String::from("text"),
            text_field: String::from("text"),
            function_type: String::from("function"),
            function_field: String::from("function"),
            function_id_field: String::from("id"),
            function_id_null_marker: false,
            image_url_type: String::from("image"),
            image_url_field: String::from("url"),
            image_data_type: String::from("image"),
            image_data_field: String::from("data"),
            audio_url_type: String::from("audio"),
            audio_url_field: String::from("url"),
            audio_data_type: String::from("audio"),
            audio_data_field: String::from("data"),
        }
    }
}

impl Serialize for Part {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let formatted = PartWithFmt::new(self, PartFmt::new());
        formatted.serialize(serializer)
    }
}

#[derive(Debug, Clone)]
pub struct PartWithFmt<'a>(&'a Part, PartFmt);

impl<'a> PartWithFmt<'a> {
    pub fn new(inner: &'a Part, fmt: PartFmt) -> Self {
        Self(inner, fmt)
    }
}

impl<'a> Serialize for PartWithFmt<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(2))?;
        match &self.0 {
            Part::Text(text) => {
                map.serialize_entry("type", &self.1.text_type)?;
                map.serialize_entry(&self.1.text_field, text)?;
            }
            Part::Function { id, function } => {
                map.serialize_entry("type", &self.1.function_type)?;
                if id.is_some() {
                    map.serialize_entry(&self.1.function_id_field, id.as_ref().unwrap())?;
                } else if self.1.function_id_null_marker {
                    map.serialize_entry(&self.1.function_id_field, &())?;
                }
                map.serialize_entry(
                    &self.1.function_field,
                    &ToolCallWithFmt::new(function, self.1.tool_call_fmt.clone()),
                )?;
            }
            Part::ImageURL(url) => {
                map.serialize_entry("type", &self.1.image_url_type)?;
                map.serialize_entry(&self.1.image_url_field, url.as_str())?;
            }
            Part::ImageData(encoded) => {
                map.serialize_entry("type", &self.1.image_data_type)?;
                map.serialize_entry(&self.1.image_data_field, encoded.as_str())?;
            }
            Part::AudioURL(url) => {
                map.serialize_entry("type", &self.1.audio_url_type)?;
                map.serialize_entry(&self.1.audio_url_field, url.as_str())?;
            }
            Part::AudioData(encoded) => {
                map.serialize_entry("type", &self.1.audio_data_type)?;
                map.serialize_entry(&self.1.audio_data_field, encoded.as_str())?;
            }
        };
        map.end()
    }
}

impl<'de> Deserialize<'de> for Part {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(PartVisitor)
    }
}

struct PartVisitor;

impl<'de> de::Visitor<'de> for PartVisitor {
    type Value = Part;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str(r#"a map with "type" field and one other content key"#)
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut ty: Option<String> = None;
        let mut field: Option<(String, String)> = None;
        let mut func_field: Option<(String, ToolCall)> = None;
        let mut id: Option<String> = None;

        while let Some(k) = map.next_key::<String>()? {
            if k == "type" {
                if ty.is_some() {
                    return Err(de::Error::duplicate_field("type"));
                }
                ty = Some(map.next_value()?);
            } else if k == "id" || k == "function_id" {
                if id.is_some() {
                    return Err(de::Error::duplicate_field("id"));
                }
                id = Some(map.next_value()?);
            } else if k == "function" {
                if field.is_some() {
                    return Err(de::Error::custom("multiple part fields found"));
                }
                func_field = Some((k, map.next_value::<ToolCall>()?));
            } else if k == "text"
                || k == "url"
                || k == "data"
                || k == "image_url"
                || k == "image_data"
                || k == "audio_url"
                || k == "audio_data"
            {
                if field.is_some() {
                    return Err(de::Error::custom("multiple part fields found"));
                }
                field = Some((k, map.next_value()?));
            }
        }

        match (ty, field, func_field) {
            (None, _, _) => Err(de::Error::missing_field("type")),
            (_, None, None) => Err(de::Error::custom("Missing part field")),
            (Some(ty), Some((k, v)), _) if ty == "text" && k == "text" => Ok(Part::Text(v)),
            (Some(ty), _, Some((k, v))) if ty == "function" && k == "function" => {
                Ok(Part::Function { id, function: v })
            }
            (Some(ty), Some((k, v)), _)
                if (ty == "image" || ty == "image_url") && (k == "url" || k == "image_url") =>
            {
                let url = url::Url::parse(&v)
                    .map_err(|_| de::Error::invalid_value(de::Unexpected::Str(&v), &"Valid URL"))?;
                Ok(Part::ImageURL(url))
            }
            (Some(ty), Some((k, v)), _)
                if (ty == "image" || ty == "image_data") && (k == "data" || k == "image_data") =>
            {
                Ok(Part::ImageData(v))
            }
            (Some(ty), Some((k, v)), _)
                if (ty == "audio" || ty == "audio_url") && (k == "url" || k == "audio_url") =>
            {
                let url = url::Url::parse(&v)
                    .map_err(|_| de::Error::invalid_value(de::Unexpected::Str(&v), &"Valid URL"))?;
                Ok(Part::AudioURL(url))
            }
            (Some(ty), Some((k, v)), _)
                if (ty == "audio" || ty == "audio_data") && (k == "data" || k == "audio_data") =>
            {
                Ok(Part::AudioData(v))
            }
            (Some(_), _, _) => Err(de::Error::custom("Invalid Part format")),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::value::ToolCallArg;

    use super::*;

    #[test]
    fn part_serde_default() {
        {
            let part = Part::Text("This is a text".to_owned());
            let s = serde_json::to_string(&part).unwrap();
            assert_eq!(s, r#"{"type":"text","text":"This is a text"}"#);
            let roundtrip = serde_json::from_str::<Part>(&s).unwrap();
            assert_eq!(roundtrip, part);
        }
        {
            let part = Part::Function {
                id: Some(String::from("asdf1234")),
                function: ToolCall {
                    name: String::from("fn"),
                    arguments: ToolCallArg::Boolean(false),
                },
            };
            let s = serde_json::to_string(&part).unwrap();
            assert_eq!(
                s,
                r#"{"type":"function","id":"asdf1234","function":{"name":"fn","arguments":false}}"#
            );
            let roundtrip = serde_json::from_str::<Part>(&s).unwrap();
            assert_eq!(roundtrip, part);
        }
    }

    #[test]
    fn part_serde_with_format() {
        let mut fmt2 = PartFmt::new();
        fmt2.text_type = String::from("text1");
        fmt2.text_field = String::from("text2");
        {
            let part = Part::Text("This is a text".to_owned());
            let s = serde_json::to_string(&PartWithFmt::new(&part, fmt2.clone())).unwrap();
            assert_eq!(s, r#"{"type":"text1","text2":"This is a text"}"#);
            let roundtrip = serde_json::from_str::<Part>(&s).unwrap();
            assert_eq!(roundtrip, part);
        }
    }
}
