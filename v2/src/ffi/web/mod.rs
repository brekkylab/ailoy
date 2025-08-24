use wasm_bindgen::prelude::*;

use crate::value::Part;

#[wasm_bindgen]
extern "C" {
    // Import a JavaScript function named 'alert'
    fn alert(s: &str);

    // Import a JavaScript function from a specific module
    #[wasm_bindgen(js_namespace = ailoy)]
    fn log(s: &str);
}

#[derive(Clone)]
#[wasm_bindgen]
pub struct WasmPart {
    inner: Part,
}
