use indexmap::IndexMap;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

/// RFC 6901: "~1" => "/", "~0" => "~"
fn decode_json_pointer_token(token: &str) -> String {
    let mut out = String::with_capacity(token.len());
    let mut chars = token.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '~' {
            match chars.next() {
                Some('0') => out.push('~'),
                Some('1') => out.push('/'),
                Some(other) => {
                    // For malformed sequences, conservatively preserve the original ("~" + other)
                    out.push('~');
                    out.push(other);
                }
                None => {
                    // If the string ends with a lone '~', preserve it as-is
                    out.push('~');
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    Null,
    Bool(bool),
    Unsigned(u64),
    Integer(i64),
    Float(OrderedFloat<f64>),
    String(String),
    Object(IndexMap<String, Value>),
    Array(Vec<Value>),
}

impl Value {
    pub fn null() -> Self {
        Self::Null
    }

    pub fn bool(v: bool) -> Self {
        Self::Bool(v)
    }

    pub fn unsigned(v: u64) -> Self {
        Self::Unsigned(v)
    }

    pub fn integer(v: i64) -> Self {
        Self::Integer(v)
    }

    pub fn float(v: f64) -> Self {
        Self::Float(OrderedFloat(v))
    }

    pub fn string(v: impl Into<String>) -> Self {
        Self::String(v.into())
    }

    pub fn object(v: impl IntoIterator<Item = (impl Into<String>, impl Into<Value>)>) -> Self {
        Self::Object(v.into_iter().map(|(k, v)| (k.into(), v.into())).collect())
    }

    pub fn object_empty() -> Self {
        Self::Object(IndexMap::new())
    }

    pub fn object_with_capacity(capacity: usize) -> Self {
        Self::Object(IndexMap::with_capacity(capacity))
    }

    pub fn array(v: impl IntoIterator<Item = impl Into<Value>>) -> Self {
        Self::Array(v.into_iter().map(|v| v.into()).collect())
    }

    pub fn array_empty() -> Self {
        Self::Array(Vec::new())
    }

    pub fn array_with_capacity(capacity: usize) -> Self {
        Self::Array(Vec::with_capacity(capacity))
    }

    pub fn ty(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::Unsigned(_) => "number",
            Value::Integer(_) => "number",
            Value::Float(_) => "number",
            Value::String(_) => "string",
            Value::Object(_) => "object",
            Value::Array(_) => "array",
        }
    }

    pub fn is_null(&self) -> bool {
        match self {
            Value::Null => true,
            _ => false,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn is_number(&self) -> bool {
        match self {
            Value::Unsigned(_) => true,
            Value::Integer(_) => true,
            Value::Float(_) => true,
            _ => false,
        }
    }

    pub fn as_unsigned(&self) -> Option<u64> {
        match self {
            Value::Unsigned(u) => Some(*u),
            Value::Integer(i) if *i >= 0 => Some(*i as u64),
            _ => None,
        }
    }

    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Value::Unsigned(u) => Some(*u as i64),
            Value::Integer(i) => Some(*i),
            _ => None,
        }
    }

    pub fn is_string(&self) -> bool {
        match self {
            Value::String(_) => true,
            _ => false,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_string_mut(&mut self) -> Option<&mut String> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn is_array(&self) -> bool {
        match self {
            Value::Array(_) => true,
            _ => false,
        }
    }

    pub fn as_array(&self) -> Option<&Vec<Value>> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_array_mut(&mut self) -> Option<&mut Vec<Value>> {
        match self {
            Value::Array(m) => Some(m),
            _ => None,
        }
    }

    pub fn is_object(&self) -> bool {
        match self {
            Value::Object(_) => true,
            _ => false,
        }
    }

    pub fn as_object(&self) -> Option<&IndexMap<String, Value>> {
        match self {
            Value::Object(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_object_mut(&mut self) -> Option<&mut IndexMap<String, Value>> {
        match self {
            Value::Object(m) => Some(m),
            _ => None,
        }
    }

    /// Get a value by JSON Pointer (immutable).
    ///
    /// Rules:
    /// - `""` (empty string): returns `Some(self)`
    /// - Must start with `/`; otherwise returns `None`
    /// - Objects are accessed by key; arrays are accessed by decimal index
    /// - Unescape tokens: `~1` → `/`, `~0` → `~`
    /// - The JSON Patch `-` token is not supported
    pub fn pointer(&self, pointer: &str) -> Option<&Value> {
        if pointer.is_empty() {
            return Some(self);
        }
        if !pointer.starts_with('/') {
            return None;
        }

        let mut cur = self;
        for raw in pointer.split('/').skip(1) {
            let token = decode_json_pointer_token(raw);
            match cur {
                Value::Object(map) => {
                    cur = map.get(&token)?;
                }
                Value::Array(vec) => {
                    // JSON Pointer encodes indices as strings; negative or non-integer -> None
                    if token == "-" {
                        return None; // JSON Patch only
                    }
                    let idx: usize = token.parse().ok()?;
                    cur = vec.get(idx)?;
                }
                _ => return None, // Cannot descend into a scalar
            }
        }
        Some(cur)
    }

    pub fn pointer_as<T>(&self, pointer: &str) -> Option<&T>
    where
        T: ?Sized,
        for<'a> &'a T: core::convert::TryFrom<&'a Value>,
    {
        let prim = self.pointer(pointer)?;
        let value = prim.try_into();
        match value {
            Ok(v) => Some(v),
            Err(_) => None,
        }
    }

    /// Get a value by JSON Pointer (mutable).
    ///
    /// Usage/constraints are identical to [`pointer`].
    pub fn pointer_mut(&mut self, pointer: &str) -> Option<&mut Value> {
        if pointer.is_empty() {
            return Some(self);
        }
        if !pointer.starts_with('/') {
            return None;
        }

        let mut cur = self;
        for raw in pointer.split('/').skip(1) {
            let token = decode_json_pointer_token(raw);
            match cur {
                Value::Object(map) => {
                    cur = map.get_mut(&token)?;
                }
                Value::Array(vec) => {
                    if token == "-" {
                        return None; // JSON Patch only
                    }
                    let idx: usize = token.parse().ok()?;
                    cur = vec.get_mut(idx)?;
                }
                _ => return None, // Cannot descend into a scalar
            }
        }
        Some(cur)
    }

    pub fn pointer_as_mut<T>(&mut self, pointer: &str) -> Option<&mut T>
    where
        T: ?Sized,
        for<'a> &'a mut T: core::convert::TryFrom<&'a mut Value>,
    {
        let prim = self.pointer_mut(pointer)?;
        let value: Result<&mut T, _> = prim.try_into();
        value.ok()
    }
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Value::Bool(value)
    }
}

impl TryFrom<Value> for bool {
    type Error = ValueError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Bool(v) => Ok(v),
            _ => Err(ValueError::InvalidType),
        }
    }
}

impl<'a> TryFrom<&'a Value> for &'a bool {
    type Error = ValueError;
    fn try_from(value: &'a Value) -> Result<Self, Self::Error> {
        match value {
            Value::Bool(v) => Ok(v),
            _ => Err(ValueError::InvalidType),
        }
    }
}

impl<'a> TryFrom<&'a mut Value> for &'a mut bool {
    type Error = ValueError;

    fn try_from(value: &'a mut Value) -> Result<Self, Self::Error> {
        match value {
            Value::Bool(v) => Ok(v),
            _ => Err(ValueError::InvalidType),
        }
    }
}

impl From<u64> for Value {
    fn from(value: u64) -> Self {
        Value::Unsigned(value)
    }
}

impl TryFrom<Value> for u64 {
    type Error = ValueError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Unsigned(v) => Ok(v),
            _ => Err(ValueError::InvalidType),
        }
    }
}

impl<'a> TryFrom<&'a Value> for &'a u64 {
    type Error = ValueError;

    fn try_from(value: &'a Value) -> Result<Self, Self::Error> {
        match value {
            Value::Unsigned(v) => Ok(v),
            _ => Err(ValueError::InvalidType),
        }
    }
}

impl<'a> TryFrom<&'a mut Value> for &'a mut u64 {
    type Error = ValueError;

    fn try_from(value: &'a mut Value) -> Result<Self, Self::Error> {
        match value {
            Value::Unsigned(v) => Ok(v),
            _ => Err(ValueError::InvalidType),
        }
    }
}

impl From<usize> for Value {
    fn from(value: usize) -> Self {
        Value::Unsigned(value as u64)
    }
}

impl TryFrom<Value> for usize {
    type Error = ValueError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Unsigned(v) => Ok(v as usize),
            _ => Err(ValueError::InvalidType),
        }
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Value::Integer(value)
    }
}

impl TryFrom<Value> for i64 {
    type Error = ValueError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Integer(v) => Ok(v),
            _ => Err(ValueError::InvalidType),
        }
    }
}

impl<'a> TryFrom<&'a Value> for &'a i64 {
    type Error = ValueError;

    fn try_from(value: &'a Value) -> Result<Self, Self::Error> {
        match value {
            Value::Integer(v) => Ok(v),
            _ => Err(ValueError::InvalidType),
        }
    }
}

impl<'a> TryFrom<&'a mut Value> for &'a mut i64 {
    type Error = ValueError;

    fn try_from(value: &'a mut Value) -> Result<Self, Self::Error> {
        match value {
            Value::Integer(v) => Ok(v),
            _ => Err(ValueError::InvalidType),
        }
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Value::Float(ordered_float::OrderedFloat(value))
    }
}

impl TryFrom<Value> for f64 {
    type Error = ValueError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Float(v) => Ok(*v),
            _ => Err(ValueError::InvalidType),
        }
    }
}

impl<'a> TryFrom<&'a Value> for &'a f64 {
    type Error = ValueError;

    fn try_from(value: &'a Value) -> Result<Self, Self::Error> {
        match value {
            Value::Float(v) => Ok(v),
            _ => Err(ValueError::InvalidType),
        }
    }
}

impl<'a> TryFrom<&'a mut Value> for &'a mut f64 {
    type Error = ValueError;

    fn try_from(value: &'a mut Value) -> Result<Self, Self::Error> {
        match value {
            Value::Float(v) => Ok(v),
            _ => Err(ValueError::InvalidType),
        }
    }
}

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Value::String(value.to_owned())
    }
}

impl<'a> TryFrom<&'a Value> for &'a str {
    type Error = ValueError;

    fn try_from(value: &'a Value) -> Result<Self, Self::Error> {
        match value {
            Value::String(v) => Ok(v.as_str()),
            _ => Err(ValueError::InvalidType),
        }
    }
}

impl<'a> TryFrom<&'a Value> for &'a String {
    type Error = ValueError;

    fn try_from(value: &'a Value) -> Result<Self, Self::Error> {
        match value {
            Value::String(v) => Ok(v),
            _ => Err(ValueError::InvalidType),
        }
    }
}

impl From<String> for Value {
    fn from(value: String) -> Self {
        Value::String(value)
    }
}

impl TryFrom<Value> for String {
    type Error = ValueError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::String(v) => Ok(v),
            _ => Err(ValueError::InvalidType),
        }
    }
}

impl From<&String> for Value {
    fn from(value: &String) -> Self {
        Value::String(value.clone())
    }
}

impl<'a> TryFrom<&'a mut Value> for &'a mut String {
    type Error = ValueError;

    fn try_from(value: &'a mut Value) -> Result<Self, Self::Error> {
        match value {
            Value::String(v) => Ok(v),
            _ => Err(ValueError::InvalidType),
        }
    }
}

impl<T> FromIterator<T> for Value
where
    T: Into<Value>,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Value::Array(iter.into_iter().map(Into::into).collect())
    }
}

impl TryFrom<Value> for Vec<Value> {
    type Error = ValueError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Array(v) => Ok(v),
            _ => Err(ValueError::InvalidType),
        }
    }
}

impl<'a> TryFrom<&'a Value> for &'a Vec<Value> {
    type Error = ValueError;

    fn try_from(value: &'a Value) -> Result<Self, Self::Error> {
        match value {
            Value::Array(v) => Ok(v),
            _ => Err(ValueError::InvalidType),
        }
    }
}

impl<'a> TryFrom<&'a mut Value> for &'a mut Vec<Value> {
    type Error = ValueError;

    fn try_from(value: &'a mut Value) -> Result<Self, Self::Error> {
        match value {
            Value::Array(v) => Ok(v),
            _ => Err(ValueError::InvalidType),
        }
    }
}

impl<K, V> FromIterator<(K, V)> for Value
where
    K: AsRef<str>,
    V: Into<Value>,
{
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let m: IndexMap<String, Value> = iter
            .into_iter()
            .map(|(k, v)| (k.as_ref().to_owned(), v.into()))
            .collect();
        Value::Object(m)
    }
}

impl TryFrom<Value> for IndexMap<String, Value> {
    type Error = ValueError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Object(v) => Ok(v),
            _ => Err(ValueError::InvalidType),
        }
    }
}

impl<'a> TryFrom<&'a Value> for &'a IndexMap<String, Value> {
    type Error = ValueError;

    fn try_from(value: &'a Value) -> Result<Self, Self::Error> {
        match value {
            Value::Object(v) => Ok(v),
            _ => Err(ValueError::InvalidType),
        }
    }
}

impl<'a> TryFrom<&'a mut Value> for &'a mut IndexMap<String, Value> {
    type Error = ValueError;

    fn try_from(value: &'a mut Value) -> Result<Self, Self::Error> {
        match value {
            Value::Object(v) => Ok(v),
            _ => Err(ValueError::InvalidType),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ValueError {
    InvalidType,
    InvalidValue,
    MissingField,
}

pub trait EncodeValue<T>: Default {
    fn encode(&mut self, item: &T) -> Value;
}

pub trait DecodeValue<T>: Default {
    fn decode(&mut self, prim: &Value) -> Result<T, ValueError>;
}

pub struct Encodable<'data, Data, Encoder: EncodeValue<Data>> {
    data: &'data Data,
    encoder: std::marker::PhantomData<Encoder>,
}

impl<'data, Data, Encoder: EncodeValue<Data>> Encodable<'data, Data, Encoder> {
    pub fn new(data: &'data Data) -> Self {
        Self {
            data,
            encoder: std::marker::PhantomData::default(),
        }
    }
}

impl<'data, Data, Encoder: EncodeValue<Data>> Serialize for Encodable<'data, Data, Encoder> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut encoder = Encoder::default();
        let prim = encoder.encode(self.data);
        prim.serialize(serializer)
    }
}

pub struct Decodable<Data, Decoder: DecodeValue<Data>> {
    data: Data,
    decoder: std::marker::PhantomData<Decoder>,
}

impl<Data, Decoder: DecodeValue<Data>> Decodable<Data, Decoder> {
    pub fn to_data(self) -> Data {
        self.data
    }
}

impl<'de, Data, Decoder: DecodeValue<Data>> Deserialize<'de> for Decodable<Data, Decoder> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let prim = Value::deserialize(deserializer)?;
        let mut decoder = Decoder::default();
        let data = decoder
            .decode(&prim)
            .map_err(|_| serde::de::Error::custom("Unable to decode"))?;
        Ok(Decodable {
            data,
            decoder: std::marker::PhantomData::default(),
        })
    }
}
