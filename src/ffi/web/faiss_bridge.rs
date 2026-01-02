use wasm_bindgen::prelude::*;

#[wasm_bindgen(raw_module = "./shim_js/dist/index.js")]
extern "C" {
    ///////////////////
    /// Faiss Index ///
    ///////////////////
    #[wasm_bindgen(js_name = "FaissIndexSearchResult")]
    pub type JsFaissIndexSearchResult;

    #[wasm_bindgen(method, getter)]
    pub fn distances(this: &JsFaissIndexSearchResult) -> js_sys::Float32Array;

    #[wasm_bindgen(method, getter)]
    pub fn indexes(this: &JsFaissIndexSearchResult) -> js_sys::BigInt64Array;

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
        num_training_vectors: usize,
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
        num_vectors: usize,
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
        k: usize,
    ) -> Result<JsFaissIndexSearchResult, JsValue>;

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

    #[wasm_bindgen(catch, js_name = "create_faiss_index")]
    pub async fn create_faiss_index(
        dimension: i32,
        description: String,
        metric: String,
    ) -> Result<FaissIndexInner, JsValue>;

}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum FaissMetricType {
    /// basic metrics
    InnerProduct = 0,
    L2 = 1,
    L1,
    Linf,
    Lp,

    /// some additional metrics defined in scipy.spatial.distance
    Canberra = 20,
    BrayCurtis,
    JensenShannon,

    /// sum_i(min(a_i, b_i)) / sum_i(max(a_i, b_i)) where a_i, b_i > 0
    Jaccard,
    /// Squared Eucliden distance, ignoring NaNs
    NaNEuclidean,
    /// Gower's distance - numeric dimensions are in [0,1] and categorical
    /// dimensions are negative integers
    Gower,
}

impl std::fmt::Display for FaissMetricType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            FaissMetricType::InnerProduct => write!(f, "InnerProduct"),
            FaissMetricType::L2 => write!(f, "L2"),
            FaissMetricType::L1 => write!(f, "L1"),
            FaissMetricType::Linf => write!(f, "Linf"),
            FaissMetricType::Lp => write!(f, "Lp"),
            FaissMetricType::Canberra => write!(f, "Canberra"),
            FaissMetricType::BrayCurtis => write!(f, "BrayCurtis"),
            FaissMetricType::JensenShannon => write!(f, "JensenShannon"),
            FaissMetricType::Jaccard => write!(f, "Jaccard"),
            FaissMetricType::NaNEuclidean => write!(f, "NaNEuclidean"),
            FaissMetricType::Gower => write!(f, "Gower"),
        }
    }
}

impl TryFrom<String> for FaissMetricType {
    type Error = anyhow::Error; // TODO: Define custom error for this.

    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.as_str() {
            "InnerProduct" => Ok(FaissMetricType::InnerProduct),
            "L2" => Ok(FaissMetricType::L2),
            "L1" => Ok(FaissMetricType::L1),
            "Linf" => Ok(FaissMetricType::Linf),
            "Lp" => Ok(FaissMetricType::Lp),
            "Canberra" => Ok(FaissMetricType::Canberra),
            "BrayCurtis" => Ok(FaissMetricType::BrayCurtis),
            "JensenShannon" => Ok(FaissMetricType::JensenShannon),
            "Jaccard" => Ok(FaissMetricType::Jaccard),
            "NaNEuclidean" => Ok(FaissMetricType::NaNEuclidean),
            "Gower" => Ok(FaissMetricType::Gower),
            _ => anyhow::bail!("Unknown metric type: '{}'", value),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FaissIndexSearchResult {
    pub distances: Vec<f32>,
    pub indexes: Vec<i64>,
}

impl From<JsFaissIndexSearchResult> for FaissIndexSearchResult {
    fn from(value: JsFaissIndexSearchResult) -> Self {
        let distances = value.distances().to_vec();
        let indexes = value.indexes().to_vec();
        Self { distances, indexes }
    }
}
