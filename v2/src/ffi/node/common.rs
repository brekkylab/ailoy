use std::sync::OnceLock;

use napi::{Env, Error, Result as NapiResult, Status, bindgen_prelude::*};

static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

pub fn get_or_create_runtime() -> &'static tokio::runtime::Runtime {
    RUNTIME.get_or_init(|| tokio::runtime::Runtime::new().expect("Failed to create tokio runtime"))
}

pub fn get_property<T: FromNapiValue>(obj: Object, name: &str) -> NapiResult<T> {
    let prop: T = obj
        .get(name)?
        .ok_or_else(|| Error::new(Status::InvalidArg, format!("Missing '{}' field", name)))?;
    Ok(prop)
}

pub fn json_stringify(env: Env, obj: Object) -> NapiResult<String> {
    let json_global = env.get_global()?.get_named_property::<Object>("JSON")?;
    let stringify_fn = json_global.get_named_property::<Function<Object, String>>("stringify")?;
    let str = stringify_fn.call(obj)?;
    Ok(str)
}

pub fn json_parse<'a>(env: Env, str: String) -> NapiResult<Object<'a>> {
    let json_global = env.get_global()?.get_named_property::<Object>("JSON")?;
    let parse_fn = json_global.get_named_property::<Function<String, Object>>("parse")?;
    let obj = parse_fn.call(str)?;
    Ok(obj)
}
