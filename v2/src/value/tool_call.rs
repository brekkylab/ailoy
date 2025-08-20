use std::fmt;

use indexmap::IndexMap;
use ordered_float::NotNan;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de, ser::SerializeMap as _};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolCallArg {
    String(String),
    Number(NotNan<f64>),
    Boolean(bool),
    Object(IndexMap<String, Box<ToolCallArg>>),
    Array(Vec<Box<ToolCallArg>>),
    Null,
}

impl ToolCallArg {
    pub fn try_from_string(s: impl AsRef<str>) -> Result<Self, String> {
        match serde_json::from_str::<ToolCallArg>(s.as_ref()) {
            Ok(tc) => Ok(tc),
            Err(e) => Err(format!("Invalid tool call arg: {}", e.to_string())),
        }
    }

    pub fn new_string(s: impl Into<String>) -> Self {
        Self::String(s.into())
    }

    pub fn new_number(n: impl Into<f64>) -> Self {
        Self::Number(NotNan::new(n.into()).unwrap())
    }

    pub fn new_boolean(b: impl Into<bool>) -> Self {
        Self::Boolean(b.into())
    }

    pub fn new_object(o: impl IntoIterator<Item = (impl Into<String>, ToolCallArg)>) -> Self {
        let o = o.into_iter().map(|(k, v)| (k.into(), Box::new(v)));
        Self::Object(o.collect())
    }

    pub fn new_array(a: impl IntoIterator<Item = ToolCallArg>) -> Self {
        let a = a.into_iter().map(|v| Box::new(v));
        Self::Array(a.collect())
    }

    pub fn new_null() -> Self {
        Self::Null
    }

    /// Returns the inner `&str` if this is `String`, otherwise `None`.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            ToolCallArg::String(v) => Some(v),
            _ => None,
        }
    }

    /// Returns the inner as `i64` if this is `Number` and the value is a finite, in-range integer.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            ToolCallArg::Number(v) => Some(v.round() as i64),
            _ => None,
        }
    }

    /// Returns the inner as `u64` if this is `Number` and the value is a finite, in-range integer.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            ToolCallArg::Number(v) => Some(v.round() as u64),
            _ => None,
        }
    }

    /// Returns the inner `f64` if this is `Number`, otherwise `None`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ToolCallArg::Number(v) => Some(f64::from(v.to_owned())),
            _ => None,
        }
    }

    /// Returns the inner `bool` if this is `Boolean`, otherwise `None`.
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            ToolCallArg::Boolean(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the inner object map if this is `Object`, otherwise `None`.
    pub fn as_object(&self) -> Option<&IndexMap<String, Box<ToolCallArg>>> {
        match self {
            ToolCallArg::Object(map) => Some(map),
            _ => None,
        }
    }

    /// Returns the inner array as a slice if this is `Array`, otherwise `None`.
    pub fn as_array(&self) -> Option<&[Box<ToolCallArg>]> {
        match self {
            ToolCallArg::Array(vec) => Some(vec.as_slice()),
            _ => None,
        }
    }

    /// Returns `true` if this is `Null`.
    pub fn as_null(&self) -> bool {
        matches!(self, ToolCallArg::Null)
    }
}

impl fmt::Display for ToolCallArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolCallArg::String(s) => format!("\"{}\"", s).fmt(f),
            ToolCallArg::Number(not_nan) => not_nan.fmt(f),
            ToolCallArg::Boolean(b) => b.fmt(f),
            ToolCallArg::Object(hash_map) => {
                let elems = hash_map
                    .iter()
                    .map(|(k, v)| format!("{}:{}", k, v.to_string()))
                    .collect::<Vec<_>>();
                f.write_fmt(format_args!("{{{}}}", elems.join(", ")))
            }
            ToolCallArg::Array(arr) => {
                let elems = arr.iter().map(|v| v.to_string()).collect::<Vec<_>>();
                f.write_fmt(format_args!("[{}]", elems.join(", ")))
            }
            ToolCallArg::Null => fmt::Debug::fmt(&(), f),
        }
    }
}

/// Represents a single tool invocation requested by an LLM.
///
/// This struct models one entry in an assistant message’s `tool_calls` array
/// (e.g., OpenAI-style tool/function calling). It pairs the canonical tool
/// identifier with the raw JSON arguments intended for that tool.
///
/// # Fields
/// - `name`: Canonical identifier of the tool to invoke (e.g., `"search"`,
///   `"fetch_weather"`). The string should match a tool the client can resolve.
/// - `arguments`: The arguments payload as a flexible JSON value modeled by
///   [`ToolCallArg`]. When [`ToolCallArg`] is defined as an
///   **untagged** enum (recommended), each variant serializes to its natural
///   JSON form (string/number/bool/object/array/null).
///
/// See also: [`ToolCallArg`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: ToolCallArg,
}

impl ToolCall {
    pub fn new(name: impl Into<String>, arguments: ToolCallArg) -> Self {
        Self {
            name: name.into(),
            arguments,
        }
    }

    pub fn try_from_string(s: impl AsRef<str>) -> Result<Self, String> {
        serde_json::from_str::<ToolCall>(s.as_ref())
            .map_err(|e| format!("Invalid JSON: {}", e.to_string()))
    }
}

impl fmt::Display for ToolCall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!(
            "ToolCall {{ name: \"{}\", arguments: {} }}",
            self.name, self.arguments
        ))
    }
}

impl Serialize for ToolCall {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let formatted = ToolCallWithFmt(self, ToolCallFmt::default());
        formatted.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ToolCall {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(ToolCallVisitor)
    }
}

impl TryFrom<crate::value::Part> for ToolCall {
    type Error = String;

    fn try_from(value: crate::value::Part) -> Result<Self, Self::Error> {
        match value {
            super::Part::FunctionString(s) => ToolCall::try_from_string(s),
            super::Part::Function {
                id: _,
                name,
                arguments,
            } => Ok(ToolCall::new(
                name,
                ToolCallArg::try_from_string(arguments)?,
            )),
            _ => Err(String::from("Unsupported part type")),
        }
    }
}

struct ToolCallVisitor;

impl<'de> de::Visitor<'de> for ToolCallVisitor {
    type Value = ToolCall;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter
            .write_str(r#"a map with "name" field and another field "arguments" or parameters"#)
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: de::MapAccess<'de>,
    {
        let mut name: Option<String> = None;
        let mut arguments: Option<ToolCallArg> = None;

        while let Some(k) = map.next_key::<String>()? {
            if k == "name" {
                if name.is_some() {
                    return Err(de::Error::duplicate_field("name"));
                }
                name = Some(map.next_value()?);
            } else if k == "arguments" || k == "parameters" {
                if arguments.is_some() {
                    return Err(de::Error::duplicate_field("arguments"));
                }
                arguments = Some(map.next_value()?);
            }
        }

        match (name, arguments) {
            (None, _) => Err(de::Error::missing_field("name")),
            (Some(_), None) => Err(de::Error::missing_field("arguments")),
            (Some(name), Some(args)) => Ok(ToolCall::new(name, args)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToolCallFmt {
    /// {"type": "function", "id": "1234asdf", "function": {"name": "function name", "||HERE||": "function args"}}
    /// default: "arguments"
    pub arguments_field: String,
}

impl ToolCallFmt {
    pub fn new() -> Self {
        ToolCallFmt::default()
    }
}

impl Default for ToolCallFmt {
    fn default() -> Self {
        ToolCallFmt {
            arguments_field: String::from("arguments"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToolCallWithFmt<'a>(&'a ToolCall, ToolCallFmt);

impl<'a> ToolCallWithFmt<'a> {
    pub fn new(inner: &'a ToolCall, fmt: ToolCallFmt) -> Self {
        Self(inner, fmt)
    }
}

impl<'a> Serialize for ToolCallWithFmt<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("name", &self.0.name)?;
        map.serialize_entry(&self.1.arguments_field, &self.0.arguments)?;
        map.end()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn tool_call_serde_default() {
        let mut obj: IndexMap<String, Box<ToolCallArg>> = IndexMap::new();
        obj.insert("msg".into(), Box::new(ToolCallArg::String("Hi".into())));
        obj.insert(
            "count".into(),
            Box::new(ToolCallArg::Number(NotNan::new(3.0).unwrap())),
        );
        obj.insert(
            "flags".into(),
            Box::new(ToolCallArg::Array(vec![
                Box::new(ToolCallArg::Boolean(true)),
                Box::new(ToolCallArg::Null),
            ])),
        );
        let tc = ToolCall::new("echo", ToolCallArg::Object(obj));
        let s = serde_json::to_string(&tc).unwrap();
        assert_eq!(
            s,
            r#"{"name":"echo","arguments":{"msg":"Hi","count":3.0,"flags":[true,null]}}"#
        );
        let roundtrip = serde_json::from_str::<ToolCall>(&s).unwrap();
        assert_eq!(roundtrip, tc);
    }

    #[test]
    fn tool_call_serde_with_fmt() {
        let mut tc_fmt = ToolCallFmt::default();
        tc_fmt.arguments_field = String::from("parameters");

        let mut obj: IndexMap<String, Box<ToolCallArg>> = IndexMap::new();
        obj.insert("msg".into(), Box::new(ToolCallArg::String("Hi".into())));
        obj.insert(
            "count".into(),
            Box::new(ToolCallArg::Number(NotNan::new(3.0).unwrap())),
        );
        obj.insert(
            "flags".into(),
            Box::new(ToolCallArg::Array(vec![
                Box::new(ToolCallArg::Boolean(true)),
                Box::new(ToolCallArg::Null),
            ])),
        );
        let tc = ToolCall::new("echo", ToolCallArg::Object(obj));
        let s = serde_json::to_string(&ToolCallWithFmt::new(&tc, tc_fmt.clone())).unwrap();
        assert_eq!(
            s,
            r#"{"name":"echo","parameters":{"msg":"Hi","count":3.0,"flags":[true,null]}}"#
        );
        let roundtrip: ToolCall = serde_json::from_str(&s).unwrap();
        assert_eq!(roundtrip, tc);
    }
}
