#[cfg(not(target_arch = "wasm32"))]
#[cxx::bridge]
mod ffi {
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum FaissMetricType {
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

    #[derive(Debug, Clone)]
    struct FaissIndexSearchResult {
        pub distances: Vec<f32>,
        pub indexes: Vec<i64>,
    }

    #[namespace = "faiss_bridge"]
    unsafe extern "C++" {
        include!("bridge.hpp");

        type FaissIndexInner;

        // Creator functions
        unsafe fn create_index(
            dimension: i32,
            description: &str,
            metric: FaissMetricType,
        ) -> Result<UniquePtr<FaissIndexInner>>;

        unsafe fn read_index(filename: &str) -> Result<UniquePtr<FaissIndexInner>>;

        // Methods
        fn is_trained(self: &FaissIndexInner) -> bool;
        fn get_ntotal(self: &FaissIndexInner) -> i64;
        fn get_dimension(self: &FaissIndexInner) -> i32;
        fn get_metric_type(self: &FaissIndexInner) -> FaissMetricType;

        unsafe fn train_index(
            self: Pin<&mut FaissIndexInner>,
            training_vectors: &[f32],
            num_training_vectors: usize,
        ) -> Result<()>;

        unsafe fn add_vectors_with_ids(
            self: Pin<&mut FaissIndexInner>,
            vectors: &[f32],
            num_vectors: usize,
            ids: &[i64],
        ) -> Result<()>;

        unsafe fn search_vectors(
            self: &FaissIndexInner,
            query_vectors: &[f32],
            k: usize,
        ) -> Result<FaissIndexSearchResult>;

        unsafe fn get_by_ids(self: &FaissIndexInner, ids: &[i64]) -> Result<Vec<f32>>;

        unsafe fn remove_vectors(self: Pin<&mut FaissIndexInner>, ids: &[i64]) -> Result<usize>;

        unsafe fn clear(self: Pin<&mut FaissIndexInner>) -> Result<()>;

        unsafe fn write_index(self: &FaissIndexInner, filename: &str) -> Result<()>;
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod lib {
    pub use ffi::*;

    impl std::fmt::Display for FaissMetricType {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            match *self {
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
                _ => write!(f, ""),
            }
        }
    }

    impl std::str::FromStr for FaissMetricType {
        type Err = anyhow::Error; // TODO: Define custom error for this.

        fn from_str(s: &str) -> Result<Self, Self::Err> {
            match s {
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
                _ => anyhow::bail!("Unknown metric type: '{}'", s),
            }
        }
    }

    unsafe impl Send for ffi::FaissIndexInner {}

    unsafe impl Sync for ffi::FaissIndexInner {}
}

#[cfg(not(target_arch = "wasm32"))]
pub use lib::*;
