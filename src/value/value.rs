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

    pub fn as_float(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some((*f).into()),
            _ => None,
        }
    }

    pub fn as_ordered_float(&self) -> Option<OrderedFloat<f64>> {
        match self {
            Value::Float(f) => Some(*f as OrderedFloat<f64>),
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
        prim.try_into().ok()
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
        prim.try_into().ok()
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

impl From<i32> for Value {
    fn from(value: i32) -> Self {
        Value::Integer(value as i64)
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

impl From<Option<String>> for Value {
    fn from(value: Option<String>) -> Self {
        if let Some(value) = value {
            Value::from(value)
        } else {
            Value::Null
        }
    }
}

impl From<&Option<String>> for Value {
    fn from(value: &Option<String>) -> Self {
        if let Some(value) = value {
            Value::from(value)
        } else {
            Value::Null
        }
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

impl From<Vec<Value>> for Value {
    #[inline]
    fn from(v: Vec<Value>) -> Self {
        Value::Array(v)
    }
}

impl From<IndexMap<String, Value>> for Value {
    #[inline]
    fn from(m: IndexMap<String, Value>) -> Self {
        Value::Object(m)
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

impl Into<serde_json::Value> for Value {
    fn into(self) -> serde_json::Value {
        match self {
            Value::Null => serde_json::Value::Null,
            Value::Bool(b) => serde_json::Value::Bool(b),
            Value::Unsigned(u) => serde_json::Value::Number((u).into()),
            Value::Integer(i) => serde_json::Value::Number((i).into()),
            Value::Float(f) => serde_json::Number::from_f64(*f)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            Value::String(s) => serde_json::Value::String(s),
            Value::Array(a) => serde_json::Value::Array(a.into_iter().map(|v| v.into()).collect()),
            Value::Object(m) => {
                let mut map = serde_json::Map::with_capacity(m.len());
                for (k, v2) in m {
                    map.insert(k.clone(), v2.into());
                }
                serde_json::Value::Object(map)
            }
        }
    }
}

impl Into<Value> for serde_json::Value {
    fn into(self) -> Value {
        match self {
            serde_json::Value::Null => Value::Null,
            serde_json::Value::Bool(b) => Value::Bool(b),
            serde_json::Value::Number(n) => {
                if n.is_u64() {
                    Value::Unsigned(n.as_u64().unwrap())
                } else if n.is_i64() {
                    Value::Integer(n.as_i64().unwrap())
                } else {
                    Value::Float(n.as_f64().unwrap().into())
                }
            }
            serde_json::Value::String(s) => Value::String(s),
            serde_json::Value::Array(a) => Value::Array(a.into_iter().map(|v| v.into()).collect()),
            serde_json::Value::Object(m) => {
                let mut map = indexmap::IndexMap::with_capacity(m.len());
                for (k, v2) in m {
                    map.insert(k.clone(), v2.into());
                }
                Value::Object(map)
            }
        }
    }
}

// Internal helper: parse object key/value pairs
#[macro_export]
macro_rules! __value_obj_kvs {
    // End of object
    ($map:ident ,) => {};
    ($map:ident) => {};

    // ---------- STRING LITERAL KEY ----------
    // value is nested object
    ($map:ident, $key:literal : { $($inner:tt)* } , $($rest:tt)*) => {{
        let v = $crate::to_value!({ $($inner)* });
        $map.insert($key.to_string(), v);
        $crate::__value_obj_kvs!($map, $($rest)*);
    }};
    ($map:ident, $key:literal : { $($inner:tt)* }) => {{
        let v = $crate::to_value!({ $($inner)* });
        $map.insert($key.to_string(), v);
    }};
    // value is array
    ($map:ident, $key:literal : [ $($inner:tt)* ] , $($rest:tt)*) => {{
        let v = $crate::to_value!([ $($inner)* ]);
        $map.insert($key.to_string(), v);
        $crate::__value_obj_kvs!($map, $($rest)*);
    }};
    ($map:ident, $key:literal : [ $($inner:tt)* ]) => {{
        let v = $crate::to_value!([ $($inner)* ]);
        $map.insert($key.to_string(), v);
    }};
    // value is a general expression
    ($map:ident, $key:literal : $val:expr , $($rest:tt)*) => {{
        let v = $crate::to_value!($val);
        $map.insert($key.to_string(), v);
        $crate::__value_obj_kvs!($map, $($rest)*);
    }};
    ($map:ident, $key:literal : $val:expr) => {{
        let v = $crate::to_value!($val);
        $map.insert($key.to_string(), v);
    }};

    // ---------- IDENTIFIER KEY ----------
    ($map:ident, $ident:ident : { $($inner:tt)* } , $($rest:tt)*) => {{
        let v = $crate::to_value!({ $($inner)* });
        $map.insert(::std::stringify!($ident).to_string(), v);
        $crate::__value_obj_kvs!($map, $($rest)*);
    }};
    ($map:ident, $ident:ident : { $($inner:tt)* }) => {{
        let v = $crate::to_value!({ $($inner)* });
        $map.insert(::std::stringify!($ident).to_string(), v);
    }};
    ($map:ident, $ident:ident : [ $($inner:tt)* ] , $($rest:tt)*) => {{
        let v = $crate::to_value!([ $($inner)* ]);
        $map.insert(::std::stringify!($ident).to_string(), v);
        $crate::__value_obj_kvs!($map, $($rest)*);
    }};
    ($map:ident, $ident:ident : [ $($inner:tt)* ]) => {{
        let v = $crate::to_value!([ $($inner)* ]);
        $map.insert(::std::stringify!($ident).to_string(), v);
    }};
    ($map:ident, $ident:ident : $val:expr , $($rest:tt)*) => {{
        let v = $crate::to_value!($val);
        $map.insert(::std::stringify!($ident).to_string(), v);
        $crate::__value_obj_kvs!($map, $($rest)*);
    }};
    ($map:ident, $ident:ident : $val:expr) => {{
        let v = $crate::to_value!($val);
        $map.insert(::std::stringify!($ident).to_string(), v);
    }};

    // ---------- COMPUTED KEY (expr) ----------
    ($map:ident, ( $key:expr ) : { $($inner:tt)* } , $($rest:tt)*) => {{
        let k: String = ::std::convert::Into::into($key);
        let v = $crate::to_value!({ $($inner)* });
        $map.insert(k, v);
        $crate::__value_obj_kvs!($map, $($rest)*);
    }};
    ($map:ident, ( $key:expr ) : { $($inner:tt)* }) => {{
        let k: String = ::std::convert::Into::into($key);
        let v = $crate::to_value!({ $($inner)* });
        $map.insert(k, v);
    }};
    ($map:ident, ( $key:expr ) : [ $($inner:tt)* ] , $($rest:tt)*) => {{
        let k: String = ::std::convert::Into::into($key);
        let v = $crate::to_value!([ $($inner)* ]);
        $map.insert(k, v);
        $crate::__value_obj_kvs!($map, $($rest)*);
    }};
    ($map:ident, ( $key:expr ) : [ $($inner:tt)* ]) => {{
        let k: String = ::std::convert::Into::into($key);
        let v = $crate::to_value!([ $($inner)* ]);
        $map.insert(k, v);
    }};
    ($map:ident, ( $key:expr ) : $val:expr , $($rest:tt)*) => {{
        let k: String = ::std::convert::Into::into($key);
        let v = $crate::to_value!($val);
        $map.insert(k, v);
        $crate::__value_obj_kvs!($map, $($rest)*);
    }};
    ($map:ident, ( $key:expr ) : $val:expr) => {{
        let k: String = ::std::convert::Into::into($key);
        let v = $crate::to_value!($val);
        $map.insert(k, v);
    }};
}

#[macro_export]
macro_rules! to_value {
    (null) => { $crate::value::Value::Null };
    (true) => { $crate::value::Value::Bool(true) };
    (false) => { $crate::value::Value::Bool(false) };

    // Array
    ([ $($elem:tt),* $(,)? ]) => {
        $crate::value::Value::Array(vec![ $( $crate::to_value!($elem) ),* ])
    };

    // Object
    ({ $($rest:tt)* }) => {{
        let mut __map: indexmap::IndexMap<String, $crate::value::Value> = indexmap::IndexMap::new();
        $crate::__value_obj_kvs!(__map, $($rest)*);
        $crate::value::Value::Object(__map)
    }};

    // Fallback
    ($other:expr) => { $crate::value::Value::from($other) };
}

#[derive(Debug, Clone)]
pub enum ValueError {
    InvalidType,
    InvalidValue,
    MissingField,
}

#[cfg(feature = "python")]
mod py {
    use pyo3::{prelude::*, types::PyAny};
    use pyo3_stub_gen::{PyStubType, TypeInfo};

    use super::Value;
    use crate::ffi::py::base::{python_to_value, value_to_python};

    impl<'a, 'py> FromPyObject<'a, 'py> for Value {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            python_to_value(&obj)
        }
    }

    impl<'py> IntoPyObject<'py> for Value {
        type Target = PyAny;
        type Output = Bound<'py, PyAny>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
            value_to_python(py, &self)
        }
    }

    impl PyStubType for Value {
        fn type_output() -> TypeInfo {
            TypeInfo::any()
        }
    }
}

#[cfg(feature = "nodejs")]
mod node {
    use napi::{Env, Result, Unknown, bindgen_prelude::*};

    use super::Value;

    impl FromNapiValue for Value {
        unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> Result<Self> {
            let env = Env::from_raw(env);
            let any = unsafe { Unknown::from_raw_unchecked(env.raw(), napi_val) };
            match any.get_type()? {
                ValueType::Undefined | ValueType::Null => Ok(Value::Null),
                ValueType::Boolean => Ok(Value::Bool(any.coerce_to_bool()?)),
                ValueType::Number => {
                    let num = any.coerce_to_number()?;
                    let f = num.get_double()?;
                    if !f.is_finite() {
                        return Ok(Value::Float(ordered_float::OrderedFloat(f)));
                    }
                    const MAX_SAFE_I: f64 = 9_007_199_254_740_991.0;
                    if f.fract() == 0.0 && f.abs() <= MAX_SAFE_I {
                        if f < 0.0 {
                            Ok(Value::Integer(f as i64))
                        } else {
                            Ok(Value::Unsigned(f as u64))
                        }
                    } else {
                        Ok(Value::Float(ordered_float::OrderedFloat(f)))
                    }
                }
                ValueType::BigInt => {
                    let bi = BigInt::from_unknown(any)?;
                    let (signed, u, lossless) = bi.get_u64();
                    if !signed && lossless {
                        return Ok(Value::Unsigned(u));
                    }
                    let (i, lossless) = bi.get_i64();
                    if lossless {
                        return Ok(Value::Integer(i));
                    }
                    Err(Error::from_reason(
                        "BigInt value is out of supported i64/u64 range",
                    ))
                }
                ValueType::String => Ok(Value::String(
                    any.coerce_to_string()?.into_utf8()?.as_str()?.to_owned(),
                )),
                ValueType::Object => {
                    let obj = any.coerce_to_object()?;

                    if obj.is_array()? {
                        // Array
                        let len = obj.get_array_length()?;
                        let mut out = Vec::with_capacity(len as usize);
                        for i in 0..len {
                            let el: Unknown = obj.get_element(i)?;
                            let v = unsafe { Value::from_napi_value(env.raw(), el.raw()) }?;
                            out.push(v);
                        }
                        Ok(Value::Array(out))
                    } else {
                        // Plain object
                        let keys = obj.get_property_names()?; // JS 배열
                        let klen = keys.get_array_length()?;
                        let mut map = indexmap::IndexMap::with_capacity(klen as usize);

                        for i in 0..klen {
                            let k_any: Unknown = keys.get_element(i)?;
                            let k = k_any.coerce_to_string()?.into_utf8()?.as_str()?.to_owned();

                            let v_any: Unknown = obj.get_named_property(&k)?;
                            let v = unsafe { Value::from_napi_value(env.raw(), v_any.raw()) }?;
                            map.insert(k, v);
                        }
                        Ok(Value::Object(map))
                    }
                }
                ValueType::Symbol
                | ValueType::Function
                | ValueType::External
                | ValueType::Unknown => Err(Error::from_reason("Unsupported JS type for Value")),
            }
        }
    }

    impl ToNapiValue for Value {
        unsafe fn to_napi_value(env: sys::napi_env, this: Self) -> Result<sys::napi_value> {
            let env = Env::from_raw(env);
            Ok(match this {
                Value::Null => ().into_unknown(&env)?,
                Value::Bool(b) => b.into_unknown(&env)?,
                Value::Unsigned(u) => {
                    if u <= u32::MAX as u64 {
                        env.create_uint32(u as u32)?.into_unknown(&env)?
                    } else {
                        BigInt::from(u).into_unknown(&env)? // 권장: Env API 대신 BigInt 변환 사용
                    }
                }
                Value::Integer(i) => env.create_int64(i as i64)?.into_unknown(&env)?,
                Value::Float(f) => env.create_double(*f)?.into_unknown(&env)?,
                Value::String(s) => env.create_string(&s)?.into_unknown(&env)?,
                Value::Array(arr) => {
                    let js_arr = Array::from_vec(&env, arr)?;
                    js_arr.into_unknown(&env)?
                }

                Value::Object(map) => {
                    let mut obj = Object::new(&env)?;
                    for (k, v) in map {
                        let v = unsafe { <Value as ToNapiValue>::to_napi_value(env.raw(), v) }?;
                        let v = unsafe { Unknown::from_raw_unchecked(env.raw(), v) };
                        obj.set_named_property(&k, v)?;
                    }
                    obj.into_unknown(&env)?
                }
            }
            .raw())
        }
    }

    impl TypeName for Value {
        fn type_name() -> &'static str {
            "any"
        }

        fn value_type() -> ValueType {
            ValueType::Unknown
        }
    }

    impl ValidateNapiValue for Value {}
}

#[cfg(feature = "wasm")]
mod wasm {
    use std::convert::TryFrom;

    use js_sys::{Array, BigInt, Object, Reflect};
    use wasm_bindgen::{
        convert::{FromWasmAbi, IntoWasmAbi, OptionFromWasmAbi, OptionIntoWasmAbi},
        describe::WasmDescribe,
        prelude::*,
    };

    use super::Value;

    #[wasm_bindgen(typescript_custom_section)]
    const TS_APPEND_CONTENT: &'static str = dedent::dedent!(
        r#"
        export type Value = string | number | boolean | null | Array<Value> | {[property: string]: Value};
        "#
    );

    impl WasmDescribe for Value {
        fn describe() {
            JsValue::describe()
        }
    }

    impl FromWasmAbi for Value {
        type Abi = <JsValue as FromWasmAbi>::Abi;

        #[inline]
        unsafe fn from_abi(js: Self::Abi) -> Self {
            let js_value = unsafe { JsValue::from_abi(js) };
            Value::try_from(js_value).unwrap()
        }
    }

    impl IntoWasmAbi for Value {
        type Abi = <JsValue as IntoWasmAbi>::Abi;

        #[inline]
        fn into_abi(self) -> Self::Abi {
            let js_value = JsValue::from(self);
            js_value.into_abi()
        }
    }

    impl OptionFromWasmAbi for Value {
        #[inline]
        fn is_none(js: &Self::Abi) -> bool {
            let js_value = unsafe { JsValue::from_abi(*js) };
            let is_none = js_value.is_null() || js_value.is_undefined();
            std::mem::forget(js_value);
            is_none
        }
    }

    impl OptionIntoWasmAbi for Value {
        #[inline]
        fn none() -> Self::Abi {
            JsValue::NULL.into_abi()
        }
    }

    impl TryFrom<JsValue> for Value {
        type Error = js_sys::Error;

        fn try_from(js_val: JsValue) -> Result<Self, Self::Error> {
            // Handle null and undefined
            if js_val.is_null() || js_val.is_undefined() {
                return Ok(Value::Null);
            }

            // Handle boolean
            if let Some(b) = js_val.as_bool() {
                return Ok(Value::Bool(b));
            }

            // Handle string
            if let Some(s) = js_val.as_string() {
                return Ok(Value::String(s));
            }

            // Handle number
            if let Some(n) = js_val.as_f64() {
                if n.is_finite() {
                    if n.fract() == 0.0 {
                        if n >= 0.0 && n <= u64::MAX as f64 {
                            return Ok(Value::Unsigned(n as u64));
                        } else if n >= i64::MIN as f64 && n <= i64::MAX as f64 {
                            return Ok(Value::Integer(n as i64));
                        }
                    }
                    return Ok(Value::Float(ordered_float::OrderedFloat(n)));
                }
                return Err(js_sys::Error::new("Infinity or NaN"));
            }

            // Handle BigInt
            if let Ok(bigint) = js_val.clone().dyn_into::<BigInt>() {
                // Try to convert to u64 first
                if let Ok(s) = bigint.to_string(10) {
                    if let Some(s_str) = s.as_string() {
                        // Try parsing as u64
                        if let Ok(u) = s_str.parse::<u64>() {
                            return Ok(Value::Unsigned(u));
                        }

                        // Try parsing as i64
                        if let Ok(i) = s_str.parse::<i64>() {
                            return Ok(Value::Integer(i));
                        }
                    }
                }
                // BigInt out of range
                return Err(js_sys::Error::new("BigInt out of range"));
            }

            // Handle Array
            if Array::is_array(&js_val) {
                let arr = Array::from(&js_val);
                let len = arr.length();
                let mut vec = Vec::with_capacity(len as usize);

                for i in 0..len {
                    let elem = arr.get(i);
                    vec.push(Value::try_from(elem)?);
                }

                return Ok(Value::Array(vec));
            }

            // Handle Object
            if js_val.is_object() {
                let obj = Object::from(js_val.clone());
                let keys = Object::keys(&obj);
                let len = keys.length();
                let mut map = indexmap::IndexMap::with_capacity(len as usize);

                for i in 0..len {
                    let key_val = keys.get(i);
                    if let Some(key) = key_val.as_string() {
                        if let Ok(val) = Reflect::get(&obj, &key_val) {
                            map.insert(key, Value::try_from(val)?);
                        }
                    }
                }

                return Ok(Value::Object(map));
            }

            // Fallback to null for unsupported types
            Err(js_sys::Error::new("Unknown value type"))
        }
    }

    impl From<Value> for JsValue {
        fn from(value: Value) -> Self {
            match value {
                Value::Null => JsValue::NULL,
                Value::Bool(b) => JsValue::from_bool(b),
                Value::Unsigned(u) => {
                    if u <= (1u64 << 53) {
                        // Safe integer range
                        JsValue::from_f64(u as f64)
                    } else {
                        // Use BigInt for large numbers
                        match BigInt::new(&JsValue::from_str(&u.to_string())) {
                            Ok(bi) => bi.into(),
                            Err(_) => JsValue::from_f64(u as f64),
                        }
                    }
                }
                Value::Integer(i) => {
                    if i.abs() <= (1i64 << 53) {
                        // Safe integer range
                        JsValue::from_f64(i as f64)
                    } else {
                        // Use BigInt for large numbers
                        match BigInt::new(&JsValue::from_str(&i.to_string())) {
                            Ok(bi) => bi.into(),
                            Err(_) => JsValue::from_f64(i as f64),
                        }
                    }
                }
                Value::Float(f) => JsValue::from_f64(*f),
                Value::String(s) => JsValue::from_str(&s),
                Value::Array(arr) => {
                    let js_arr = Array::new_with_length(arr.len() as u32);
                    for (i, val) in arr.into_iter().enumerate() {
                        js_arr.set(i as u32, JsValue::from(val));
                    }
                    js_arr.into()
                }
                Value::Object(map) => {
                    let obj = Object::new();
                    for (key, val) in map {
                        let _ = Reflect::set(&obj, &JsValue::from_str(&key), &JsValue::from(val));
                    }
                    obj.into()
                }
            }
        }
    }
}
