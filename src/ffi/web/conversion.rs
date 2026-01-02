#[allow(dead_code)]
mod conversion {
    use anyhow::{Result, anyhow};
    use wasm_bindgen::prelude::*;

    /// Convert u32 to JsValue
    pub fn u32_to_js(value: u32) -> JsValue {
        JsValue::from_f64(value as f64)
    }

    /// Convert i64 to JsValue
    pub fn i64_to_js(value: i64) -> JsValue {
        JsValue::from_f64(value as f64)
    }

    /// Convert f64 to JsValue
    pub fn f64_to_js(value: f64) -> JsValue {
        JsValue::from_f64(value)
    }

    /// Convert &[u32] to JS Array
    pub fn u32_slice_to_js(slice: &[u32]) -> js_sys::Array {
        let arr = js_sys::Array::new();
        for &val in slice {
            arr.push(&JsValue::from_f64(val as f64));
        }
        arr
    }

    /// Convert &[i32] to JS Array
    pub fn i32_slice_to_js(slice: &[i32]) -> js_sys::Array {
        let arr = js_sys::Array::new();
        for &val in slice {
            arr.push(&JsValue::from_f64(val as f64));
        }
        arr
    }

    /// Convert &[i64] to JS Array
    pub fn i64_slice_to_js(slice: &[i64]) -> js_sys::Array {
        let arr = js_sys::Array::new();
        for &val in slice {
            arr.push(&JsValue::from_f64(val as f64));
        }
        arr
    }

    /// Convert &[f64] to JS Array
    pub fn f64_slice_to_js(slice: &[f64]) -> js_sys::Array {
        let arr = js_sys::Array::new();
        for &val in slice {
            arr.push(&JsValue::from_f64(val));
        }
        arr
    }

    /// Convert JsValue to i64
    pub fn js_to_i64(value: &JsValue) -> Result<i64> {
        value
            .as_f64()
            .map(|f| f as i64)
            .ok_or_else(|| anyhow!("Failed to convert JsValue to i64"))
    }

    /// Convert JsValue to u32
    pub fn js_to_u32(value: &JsValue) -> Result<u32> {
        value
            .as_f64()
            .map(|f| f as u32)
            .ok_or_else(|| anyhow!("Failed to convert JsValue to u32"))
    }

    /// Convert JsValue to f64
    pub fn js_to_f64(value: &JsValue) -> Result<f64> {
        value
            .as_f64()
            .ok_or_else(|| anyhow!("Failed to convert JsValue to f64"))
    }
}

pub use conversion::*;
