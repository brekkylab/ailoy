use wasm_bindgen::prelude::*;

#[wasm_bindgen(raw_module = "./shim_js/dist/index.js")]
extern "C" {
    //////////////////////
    /// Language Model ///
    //////////////////////

    #[wasm_bindgen(js_name = init_tvm_language_model)]
    pub fn init_tvm_language_model_js(cache_contents: &js_sys::Object) -> js_sys::Promise;

    #[wasm_bindgen(js_name = "TVMLanguageModel")]
    pub type JSTVMLanguageModel;

    #[wasm_bindgen(method, js_class = "TVMLanguageModel", js_name = prefill)]
    pub fn prefill(this: &JSTVMLanguageModel, tokens: js_sys::Uint32Array) -> js_sys::Promise;

    #[wasm_bindgen(method, js_class = "TVMLanguageModel", js_name = decode)]
    pub fn decode(this: &JSTVMLanguageModel, last_token: u32) -> js_sys::Promise;

    #[wasm_bindgen(method, js_class = "TVMLanguageModel", js_name = sample)]
    pub fn sample(this: &JSTVMLanguageModel, logits: js_sys::Float32Array) -> u32;

    ///////////////////////
    /// Embedding Model ///
    ///////////////////////

    #[wasm_bindgen(js_name = init_tvm_embedding_model)]
    pub fn init_tvm_embedding_model_js(cache_contents: &js_sys::Object) -> js_sys::Promise;

    #[wasm_bindgen(js_name = "TVMEmbeddingModel")]
    pub type JSTVMEmbeddingModel;

    #[wasm_bindgen(method, js_class = "TVMEmbeddingModel", js_name = infer)]
    pub fn infer(this: &JSTVMEmbeddingModel, tokens: js_sys::Uint32Array) -> js_sys::Promise;

    ///////////////////
    /// Faiss Index ///
    ///////////////////

    #[wasm_bindgen(js_name = "FaissIndexSearchResult")]
    pub type FaissIndexSearchResult;

    #[wasm_bindgen(method, getter)]
    pub fn distances(this: &FaissIndexSearchResult) -> js_sys::Float32Array;

    #[wasm_bindgen(method, getter)]
    pub fn indexes(this: &FaissIndexSearchResult) -> js_sys::BigInt64Array;

    #[wasm_bindgen(js_name = "FaissIndexInner")]
    pub type FaissIndexInner;

    #[wasm_bindgen(catch, js_name = "init_faiss_index_inner")]
    pub fn init_faiss_index_inner(args: &JsValue) -> Result<js_sys::Promise, JsValue>;

    // Methods for FaissIndexInner
    #[wasm_bindgen(method, js_class = "FaissIndexInner", js_name = "get_metric_type")]
    pub fn get_metric_type(this: &FaissIndexInner) -> js_sys::JsString;

    #[wasm_bindgen(method, js_class = "FaissIndexInner", js_name = "is_trained")]
    pub fn is_trained(this: &FaissIndexInner) -> bool;

    #[wasm_bindgen(method, js_class = "FaissIndexInner", js_name = "get_dimension")]
    pub fn get_dimension(this: &FaissIndexInner) -> i32;

    #[wasm_bindgen(method, js_class = "FaissIndexInner", js_name = "get_ntotal")]
    pub fn get_ntotal(this: &FaissIndexInner) -> i64;

    #[wasm_bindgen(method, catch, js_class = "FaissIndexInner", js_name = "train_index")]
    pub fn train_index(
        this: &FaissIndexInner,
        training_vectors: &js_sys::Float32Array,
        num_training_vectors: u32,
    ) -> Result<(), JsValue>;

    #[wasm_bindgen(
        method,
        catch,
        js_class = "FaissIndexInner",
        js_name = "add_vectors_with_ids"
    )]
    pub fn add_vectors_with_ids(
        this: &FaissIndexInner,
        vectors: &js_sys::Float32Array,
        num_vectors: u32,
        ids: &js_sys::BigInt64Array,
    ) -> Result<(), JsValue>;

    #[wasm_bindgen(
        method,
        catch,
        js_class = "FaissIndexInner",
        js_name = "search_vectors"
    )]
    pub fn search_vectors(
        this: &FaissIndexInner,
        query_vectors: &js_sys::Float32Array,
        k: u32,
    ) -> Result<FaissIndexSearchResult, JsValue>;

    #[wasm_bindgen(method, catch, js_class = "FaissIndexInner", js_name = "get_by_ids")]
    pub fn get_by_ids(
        this: &FaissIndexInner,
        ids: &js_sys::BigInt64Array,
    ) -> Result<js_sys::Float32Array, JsValue>;

    #[wasm_bindgen(
        method,
        catch,
        js_class = "FaissIndexInner",
        js_name = "remove_vectors"
    )]
    pub fn remove_vectors(
        this: &FaissIndexInner,
        ids: &js_sys::BigInt64Array,
    ) -> Result<u32, JsValue>;

    #[wasm_bindgen(method, catch, js_class = "FaissIndexInner", js_name = "clear")]
    pub fn clear(this: &FaissIndexInner) -> Result<(), JsValue>;
}
