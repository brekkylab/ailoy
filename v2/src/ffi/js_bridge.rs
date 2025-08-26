use wasm_bindgen::prelude::*;

#[wasm_bindgen(module = "/shim_js/dist/index.js")]
extern "C" {
    #[wasm_bindgen(js_name = init_language_model)]
    pub fn init_language_model_js(cache_contents: &js_sys::Object) -> js_sys::Promise;

    #[wasm_bindgen(js_name = "LanguageModel")]
    pub type JSLanguageModel;

    #[wasm_bindgen(method, js_class = "LanguageModel", js_name = prefill)]
    pub fn prefill(this: &JSLanguageModel, tokens: js_sys::Uint32Array) -> js_sys::Promise;

    #[wasm_bindgen(method, js_class = "LanguageModel", js_name = decode)]
    pub fn decode(this: &JSLanguageModel, last_token: u32) -> js_sys::Promise;

    #[wasm_bindgen(method, js_class = "LanguageModel", js_name = sample)]
    pub fn sample(this: &JSLanguageModel, logits: js_sys::Float32Array) -> u32;
}
