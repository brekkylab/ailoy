//! Crossing as objects: ailoy's types to JavaScript through their serde form, and back.
//!
//! A [`Message`] goes out as `{ role: 'assistant', contents: [{ type: 'text', ... }] }`,
//! exactly the JSON ailoy writes, and comes back from the same shape. Bytes — an embedded
//! image — are a `Buffer` on the JavaScript side rather than the base64 JSON would need.

use ailoy::message::{Message, Part, Role};
use napi::{
    Env, JsValue, ValueType,
    bindgen_prelude::{FromNapiValue, ToNapiValue, Unknown},
    sys,
};
use serde::{Serialize, de::DeserializeOwned};

use crate::error::{Result, invalid};

/// A value that crosses as its serde form, in either direction.
///
/// Converted where napi converts, on the JavaScript thread — so a future can settle with
/// one, and the conversion happens when the promise does.
pub struct Json<T>(pub T);

impl<T: Serialize> ToNapiValue for Json<T> {
    unsafe fn to_napi_value(env: sys::napi_env, val: Self) -> napi::Result<sys::napi_value> {
        Ok(Env::from_raw(env).to_js_value(&val.0)?.raw())
    }
}

impl<T: DeserializeOwned> FromNapiValue for Json<T> {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> napi::Result<Self> {
        let value = unsafe { Unknown::from_napi_value(env, napi_val)? };
        Env::from_raw(env).from_js_value(value).map(Json)
    }
}

/// `INVALID_ARG` naming what was expected, since an object that does not fit says only which
/// field it tripped on.
pub fn from_js<T: DeserializeOwned>(env: &Env, value: Unknown<'_>, what: &str) -> Result<T> {
    env.from_js_value(value)
        .map_err(|e| invalid(format!("not {what}: {}", e.reason)))
}

/// A query as a caller may spell it: a message, or a string as the user's text.
pub fn query(env: &Env, value: Unknown<'_>) -> Result<Message> {
    if value.get_type().map_err(|e| invalid(e.reason))? == ValueType::String {
        let text: String = from_js(env, value, "a string")?;
        return Ok(Message::new(Role::User).with_contents([Part::text(text)]));
    }
    from_js(env, value, "a message")
}
