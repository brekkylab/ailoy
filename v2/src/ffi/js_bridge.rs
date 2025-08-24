use wasm_bindgen::prelude::*;

#[wasm_bindgen(module = "/shim_js/dist/index.js")]
extern "C" {
    #[wasm_bindgen(js_name = init)]
    pub fn init_js(cache_contents: &js_sys::Object) -> js_sys::Promise;
}
