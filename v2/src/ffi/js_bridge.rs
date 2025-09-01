use wasm_bindgen::prelude::*;

#[wasm_bindgen(raw_module = "/shim_js/dist/index.js")]
extern "C" {
    //////////////////////
    /// Language Model ///
    //////////////////////

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

    ///////////////////////////
    /// Faiss Index Wrapper ///
    ///////////////////////////

    #[wasm_bindgen(js_name = "FaissIndexSearchResult")]
    pub type FaissIndexSearchResult;

    #[wasm_bindgen(method, getter)]
    pub fn distances(this: &FaissIndexSearchResult) -> js_sys::Float32Array;

    #[wasm_bindgen(method, getter)]
    pub fn indexes(this: &FaissIndexSearchResult) -> js_sys::BigInt64Array;

    #[wasm_bindgen(js_name = "FaissIndexWrapper")]
    pub type FaissIndexWrapper;

    #[wasm_bindgen(js_name = "init_faiss_index_wrapper")]
    pub fn init_faiss_index_wrapper(args: &JsValue) -> js_sys::Promise;

    #[wasm_bindgen(js_name = "load_faiss_wasm_from_bytes")]
    pub fn load_faiss_wasm_from_bytes(bytes: &js_sys::Uint8Array) -> js_sys::Promise;

    // Methods for FaissIndexWrapper
    #[wasm_bindgen(method, js_class = "FaissIndexWrapper", js_name = "get_metric_type")]
    pub fn get_metric_type(this: &FaissIndexWrapper) -> js_sys::JsString;

    #[wasm_bindgen(method, js_class = "FaissIndexWrapper", js_name = "clear")]
    pub fn clear(this: &FaissIndexWrapper);

    #[wasm_bindgen(method, js_class = "FaissIndexWrapper", js_name = "is_trained")]
    pub fn is_trained(this: &FaissIndexWrapper) -> bool;

    #[wasm_bindgen(method, js_class = "FaissIndexWrapper", js_name = "get_dimension")]
    pub fn get_dimension(this: &FaissIndexWrapper) -> i32;

    #[wasm_bindgen(method, js_class = "FaissIndexWrapper", js_name = "get_ntotal")]
    pub fn get_ntotal(this: &FaissIndexWrapper) -> i64;

    #[wasm_bindgen(method, js_class = "FaissIndexWrapper", js_name = "train_index")]
    pub fn train_index(
        this: &FaissIndexWrapper,
        training_vectors: &js_sys::Float32Array,
        num_training_vectors: u32,
    );

    #[wasm_bindgen(
        method,
        js_class = "FaissIndexWrapper",
        js_name = "add_vectors_with_ids"
    )]
    pub fn add_vectors_with_ids(
        this: &FaissIndexWrapper,
        vectors: &js_sys::Float32Array,
        num_vectors: u32,
        ids: &js_sys::BigInt64Array,
    );

    #[wasm_bindgen(method, js_class = "FaissIndexWrapper", js_name = "search_vectors")]
    pub fn search_vectors(
        this: &FaissIndexWrapper,
        query_vectors: &js_sys::Float32Array,
        k: u32,
    ) -> FaissIndexSearchResult;

    #[wasm_bindgen(method, js_class = "FaissIndexWrapper", js_name = "get_by_ids")]
    pub fn get_by_ids(
        this: &FaissIndexWrapper,
        ids: &js_sys::BigInt64Array,
    ) -> js_sys::Float32Array;

    #[wasm_bindgen(method, js_class = "FaissIndexWrapper", js_name = "remove_vectors")]
    pub fn remove_vectors(this: &FaissIndexWrapper, ids: &js_sys::BigInt64Array) -> u32;
}
