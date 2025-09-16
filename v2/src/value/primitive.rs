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
pub enum Primitive {
    Null,
    Bool(bool),
    Unsigned(u64),
    Integer(i64),
    Float(OrderedFloat<f64>),
    String(String),
    Object(IndexMap<String, Primitive>),
    Array(Vec<Primitive>),
}

impl Primitive {
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

    pub fn object(v: impl IntoIterator<Item = (impl Into<String>, impl Into<Primitive>)>) -> Self {
        Self::Object(v.into_iter().map(|(k, v)| (k.into(), v.into())).collect())
    }

    pub fn object_empty() -> Self {
        Self::Object(IndexMap::new())
    }

    pub fn object_with_capacity(capacity: usize) -> Self {
        Self::Object(IndexMap::with_capacity(capacity))
    }

    pub fn array(v: impl IntoIterator<Item = impl Into<Primitive>>) -> Self {
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
            Primitive::Null => "null",
            Primitive::Bool(_) => "bool",
            Primitive::Unsigned(_) => "number",
            Primitive::Integer(_) => "number",
            Primitive::Float(_) => "number",
            Primitive::String(_) => "string",
            Primitive::Object(_) => "object",
            Primitive::Array(_) => "array",
        }
    }

    pub fn is_null(&self) -> bool {
        match self {
            Primitive::Null => true,
            _ => false,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Primitive::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn is_number(&self) -> bool {
        match self {
            Primitive::Unsigned(_) => true,
            Primitive::Integer(_) => true,
            Primitive::Float(_) => true,
            _ => false,
        }
    }

    pub fn as_unsigned(&self) -> Option<u64> {
        match self {
            Primitive::Unsigned(u) => Some(*u),
            Primitive::Integer(i) if *i >= 0 => Some(*i as u64),
            _ => None,
        }
    }

    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Primitive::Unsigned(u) => Some(*u as i64),
            Primitive::Integer(i) => Some(*i),
            _ => None,
        }
    }

    pub fn is_string(&self) -> bool {
        match self {
            Primitive::String(_) => true,
            _ => false,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Primitive::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_string_mut(&mut self) -> Option<&mut String> {
        match self {
            Primitive::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn is_array(&self) -> bool {
        match self {
            Primitive::Array(_) => true,
            _ => false,
        }
    }

    pub fn as_array(&self) -> Option<&Vec<Primitive>> {
        match self {
            Primitive::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_array_mut(&mut self) -> Option<&mut Vec<Primitive>> {
        match self {
            Primitive::Array(m) => Some(m),
            _ => None,
        }
    }

    pub fn is_object(&self) -> bool {
        match self {
            Primitive::Object(_) => true,
            _ => false,
        }
    }

    pub fn as_object(&self) -> Option<&IndexMap<String, Primitive>> {
        match self {
            Primitive::Object(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_object_mut(&mut self) -> Option<&mut IndexMap<String, Primitive>> {
        match self {
            Primitive::Object(m) => Some(m),
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
    pub fn pointer(&self, pointer: &str) -> Option<&Primitive> {
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
                Primitive::Object(map) => {
                    cur = map.get(&token)?;
                }
                Primitive::Array(vec) => {
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
        for<'a> &'a T: core::convert::TryFrom<&'a Primitive>,
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
    pub fn pointer_mut(&mut self, pointer: &str) -> Option<&mut Primitive> {
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
                Primitive::Object(map) => {
                    cur = map.get_mut(&token)?;
                }
                Primitive::Array(vec) => {
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
        for<'a> &'a mut T: core::convert::TryFrom<&'a mut Primitive>,
    {
        let prim = self.pointer_mut(pointer)?;
        let value: Result<&mut T, _> = prim.try_into();
        value.ok()
    }
}

impl From<bool> for Primitive {
    fn from(value: bool) -> Self {
        Primitive::Bool(value)
    }
}

impl TryFrom<Primitive> for bool {
    type Error = PrimitiveError;

    fn try_from(value: Primitive) -> Result<Self, Self::Error> {
        match value {
            Primitive::Bool(v) => Ok(v),
            _ => Err(PrimitiveError::InvalidType),
        }
    }
}

impl<'a> TryFrom<&'a Primitive> for &'a bool {
    type Error = PrimitiveError;
    fn try_from(value: &'a Primitive) -> Result<Self, Self::Error> {
        match value {
            Primitive::Bool(v) => Ok(v),
            _ => Err(PrimitiveError::InvalidType),
        }
    }
}

impl<'a> TryFrom<&'a mut Primitive> for &'a mut bool {
    type Error = PrimitiveError;

    fn try_from(value: &'a mut Primitive) -> Result<Self, Self::Error> {
        match value {
            Primitive::Bool(v) => Ok(v),
            _ => Err(PrimitiveError::InvalidType),
        }
    }
}

impl From<u64> for Primitive {
    fn from(value: u64) -> Self {
        Primitive::Unsigned(value)
    }
}

impl TryFrom<Primitive> for u64 {
    type Error = PrimitiveError;

    fn try_from(value: Primitive) -> Result<Self, Self::Error> {
        match value {
            Primitive::Unsigned(v) => Ok(v),
            _ => Err(PrimitiveError::InvalidType),
        }
    }
}

impl<'a> TryFrom<&'a Primitive> for &'a u64 {
    type Error = PrimitiveError;

    fn try_from(value: &'a Primitive) -> Result<Self, Self::Error> {
        match value {
            Primitive::Unsigned(v) => Ok(v),
            _ => Err(PrimitiveError::InvalidType),
        }
    }
}

impl<'a> TryFrom<&'a mut Primitive> for &'a mut u64 {
    type Error = PrimitiveError;

    fn try_from(value: &'a mut Primitive) -> Result<Self, Self::Error> {
        match value {
            Primitive::Unsigned(v) => Ok(v),
            _ => Err(PrimitiveError::InvalidType),
        }
    }
}

impl From<usize> for Primitive {
    fn from(value: usize) -> Self {
        Primitive::Unsigned(value as u64)
    }
}

impl TryFrom<Primitive> for usize {
    type Error = PrimitiveError;

    fn try_from(value: Primitive) -> Result<Self, Self::Error> {
        match value {
            Primitive::Unsigned(v) => Ok(v as usize),
            _ => Err(PrimitiveError::InvalidType),
        }
    }
}

impl From<i64> for Primitive {
    fn from(value: i64) -> Self {
        Primitive::Integer(value)
    }
}

impl TryFrom<Primitive> for i64 {
    type Error = PrimitiveError;

    fn try_from(value: Primitive) -> Result<Self, Self::Error> {
        match value {
            Primitive::Integer(v) => Ok(v),
            _ => Err(PrimitiveError::InvalidType),
        }
    }
}

impl<'a> TryFrom<&'a Primitive> for &'a i64 {
    type Error = PrimitiveError;

    fn try_from(value: &'a Primitive) -> Result<Self, Self::Error> {
        match value {
            Primitive::Integer(v) => Ok(v),
            _ => Err(PrimitiveError::InvalidType),
        }
    }
}

impl<'a> TryFrom<&'a mut Primitive> for &'a mut i64 {
    type Error = PrimitiveError;

    fn try_from(value: &'a mut Primitive) -> Result<Self, Self::Error> {
        match value {
            Primitive::Integer(v) => Ok(v),
            _ => Err(PrimitiveError::InvalidType),
        }
    }
}

impl From<f64> for Primitive {
    fn from(value: f64) -> Self {
        Primitive::Float(ordered_float::OrderedFloat(value))
    }
}

impl TryFrom<Primitive> for f64 {
    type Error = PrimitiveError;

    fn try_from(value: Primitive) -> Result<Self, Self::Error> {
        match value {
            Primitive::Float(v) => Ok(*v),
            _ => Err(PrimitiveError::InvalidType),
        }
    }
}

impl<'a> TryFrom<&'a Primitive> for &'a f64 {
    type Error = PrimitiveError;

    fn try_from(value: &'a Primitive) -> Result<Self, Self::Error> {
        match value {
            Primitive::Float(v) => Ok(v),
            _ => Err(PrimitiveError::InvalidType),
        }
    }
}

impl<'a> TryFrom<&'a mut Primitive> for &'a mut f64 {
    type Error = PrimitiveError;

    fn try_from(value: &'a mut Primitive) -> Result<Self, Self::Error> {
        match value {
            Primitive::Float(v) => Ok(v),
            _ => Err(PrimitiveError::InvalidType),
        }
    }
}

impl From<&str> for Primitive {
    fn from(value: &str) -> Self {
        Primitive::String(value.to_owned())
    }
}

impl<'a> TryFrom<&'a Primitive> for &'a str {
    type Error = PrimitiveError;

    fn try_from(value: &'a Primitive) -> Result<Self, Self::Error> {
        match value {
            Primitive::String(v) => Ok(v.as_str()),
            _ => Err(PrimitiveError::InvalidType),
        }
    }
}

impl<'a> TryFrom<&'a Primitive> for &'a String {
    type Error = PrimitiveError;

    fn try_from(value: &'a Primitive) -> Result<Self, Self::Error> {
        match value {
            Primitive::String(v) => Ok(v),
            _ => Err(PrimitiveError::InvalidType),
        }
    }
}

impl From<String> for Primitive {
    fn from(value: String) -> Self {
        Primitive::String(value)
    }
}

impl TryFrom<Primitive> for String {
    type Error = PrimitiveError;

    fn try_from(value: Primitive) -> Result<Self, Self::Error> {
        match value {
            Primitive::String(v) => Ok(v),
            _ => Err(PrimitiveError::InvalidType),
        }
    }
}

impl From<&String> for Primitive {
    fn from(value: &String) -> Self {
        Primitive::String(value.clone())
    }
}

impl<'a> TryFrom<&'a mut Primitive> for &'a mut String {
    type Error = PrimitiveError;

    fn try_from(value: &'a mut Primitive) -> Result<Self, Self::Error> {
        match value {
            Primitive::String(v) => Ok(v),
            _ => Err(PrimitiveError::InvalidType),
        }
    }
}

impl<T> FromIterator<T> for Primitive
where
    T: Into<Primitive>,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Primitive::Array(iter.into_iter().map(Into::into).collect())
    }
}

impl TryFrom<Primitive> for Vec<Primitive> {
    type Error = PrimitiveError;

    fn try_from(value: Primitive) -> Result<Self, Self::Error> {
        match value {
            Primitive::Array(v) => Ok(v),
            _ => Err(PrimitiveError::InvalidType),
        }
    }
}

impl<'a> TryFrom<&'a Primitive> for &'a Vec<Primitive> {
    type Error = PrimitiveError;

    fn try_from(value: &'a Primitive) -> Result<Self, Self::Error> {
        match value {
            Primitive::Array(v) => Ok(v),
            _ => Err(PrimitiveError::InvalidType),
        }
    }
}

impl<'a> TryFrom<&'a mut Primitive> for &'a mut Vec<Primitive> {
    type Error = PrimitiveError;

    fn try_from(value: &'a mut Primitive) -> Result<Self, Self::Error> {
        match value {
            Primitive::Array(v) => Ok(v),
            _ => Err(PrimitiveError::InvalidType),
        }
    }
}

impl<K, V> FromIterator<(K, V)> for Primitive
where
    K: AsRef<str>,
    V: Into<Primitive>,
{
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let m: IndexMap<String, Primitive> = iter
            .into_iter()
            .map(|(k, v)| (k.as_ref().to_owned(), v.into()))
            .collect();
        Primitive::Object(m)
    }
}

impl TryFrom<Primitive> for IndexMap<String, Primitive> {
    type Error = PrimitiveError;

    fn try_from(value: Primitive) -> Result<Self, Self::Error> {
        match value {
            Primitive::Object(v) => Ok(v),
            _ => Err(PrimitiveError::InvalidType),
        }
    }
}

impl<'a> TryFrom<&'a Primitive> for &'a IndexMap<String, Primitive> {
    type Error = PrimitiveError;

    fn try_from(value: &'a Primitive) -> Result<Self, Self::Error> {
        match value {
            Primitive::Object(v) => Ok(v),
            _ => Err(PrimitiveError::InvalidType),
        }
    }
}

impl<'a> TryFrom<&'a mut Primitive> for &'a mut IndexMap<String, Primitive> {
    type Error = PrimitiveError;

    fn try_from(value: &'a mut Primitive) -> Result<Self, Self::Error> {
        match value {
            Primitive::Object(v) => Ok(v),
            _ => Err(PrimitiveError::InvalidType),
        }
    }
}

#[derive(Debug, Clone)]
pub enum PrimitiveError {
    InvalidType,
    InvalidValue,
    MissingField,
}

pub trait EncodePrimitive<T>: Default {
    fn encode(&mut self, item: &T) -> Primitive;
}

pub trait DecodePrimitive<T>: Default {
    fn decode(&mut self, prim: &Primitive) -> Result<T, PrimitiveError>;
}

pub struct Encodable<'data, Data, Encoder: EncodePrimitive<Data>> {
    data: &'data Data,
    encoder: std::marker::PhantomData<Encoder>,
}

impl<'data, Data, Encoder: EncodePrimitive<Data>> Encodable<'data, Data, Encoder> {
    pub fn new(data: &'data Data) -> Self {
        Self {
            data,
            encoder: std::marker::PhantomData::default(),
        }
    }
}

impl<'data, Data, Encoder: EncodePrimitive<Data>> Serialize for Encodable<'data, Data, Encoder> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut encoder = Encoder::default();
        let prim = encoder.encode(self.data);
        prim.serialize(serializer)
    }
}

pub struct Decodable<Data, Decoder: DecodePrimitive<Data>> {
    data: Data,
    decoder: std::marker::PhantomData<Decoder>,
}

impl<Data, Decoder: DecodePrimitive<Data>> Decodable<Data, Decoder> {
    pub fn to_data(self) -> Data {
        self.data
    }
}

impl<'de, Data, Decoder: DecodePrimitive<Data>> Deserialize<'de> for Decodable<Data, Decoder> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let prim = Primitive::deserialize(deserializer)?;
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
