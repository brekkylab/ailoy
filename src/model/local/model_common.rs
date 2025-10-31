use std::collections::HashMap;

use crate::value::Value;

pub(crate) fn get_cache_context(device_id: Option<i32>) -> HashMap<String, Value> {
    let mut ctx = HashMap::new();
    if let Some(device_id) = device_id {
        ctx.insert("device_id".to_owned(), Value::integer(device_id.into()));
    }
    ctx
}
