use std::sync::OnceLock;

use napi::{Env, Error, Result as NapiResult, Status, bindgen_prelude::*};

static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

pub fn get_or_create_runtime() -> &'static tokio::runtime::Runtime {
    RUNTIME.get_or_init(|| tokio::runtime::Runtime::new().expect("Failed to create tokio runtime"))
}

#[allow(unused)]
pub fn await_future<T, E: ToString + std::any::Any>(
    fut: impl Future<Output = std::result::Result<T, E>>,
) -> napi::Result<T> {
    let rt = get_or_create_runtime();
    let result = rt.block_on(fut).map_err(|e| {
        if std::any::TypeId::of::<E>() == std::any::TypeId::of::<napi::Error>() {
            // napi::Error is returned as-is
            let any_err = Box::new(e) as Box<dyn std::any::Any>;
            let napi_err = *any_err.downcast::<napi::Error>().unwrap();
            napi_err
        } else {
            // Other errors are converted to napi::Error with GenericFailure
            napi::Error::new(Status::GenericFailure, e.to_string())
        }
    });
    result
}

#[allow(unused)]
pub fn get_property<T: FromNapiValue>(obj: Object, name: &str) -> NapiResult<T> {
    let prop: T = obj
        .get(name)?
        .ok_or_else(|| Error::new(Status::InvalidArg, format!("Missing '{}' field", name)))?;
    Ok(prop)
}

#[allow(unused)]
pub fn json_stringify(env: Env, obj: Object) -> NapiResult<String> {
    let json_global = env.get_global()?.get_named_property::<Object>("JSON")?;
    let stringify_fn = json_global.get_named_property::<Function<Object, String>>("stringify")?;
    let str = stringify_fn.call(obj)?;
    Ok(str)
}

#[allow(unused)]
pub fn json_parse<'a>(env: Env, str: String) -> NapiResult<Object<'a>> {
    let json_global = env.get_global()?.get_named_property::<Object>("JSON")?;
    let parse_fn = json_global.get_named_property::<Function<String, Object>>("parse")?;
    let obj = parse_fn.call(str)?;
    Ok(obj)
}
