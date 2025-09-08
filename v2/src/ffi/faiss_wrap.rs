use std::fmt;
use std::str::FromStr;

use anyhow::{Context, Result, bail};
use std::sync::atomic::{AtomicI64, Ordering};

#[derive(Debug, Clone, Copy, PartialEq)]
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

impl fmt::Display for FaissMetricType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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

impl FromStr for FaissMetricType {
    type Err = ();

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
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
            _ => Err(()),
        }
    }
}

#[cfg(any(target_family = "unix", target_family = "windows"))]
impl From<crate::ffi::cxx_bridge::FaissMetricType> for FaissMetricType {
    fn from(value: crate::ffi::cxx_bridge::FaissMetricType) -> Self {
        match value {
            crate::ffi::cxx_bridge::FaissMetricType::InnerProduct => FaissMetricType::InnerProduct,
            crate::ffi::cxx_bridge::FaissMetricType::L2 => FaissMetricType::L2,
            crate::ffi::cxx_bridge::FaissMetricType::L1 => FaissMetricType::L1,
            crate::ffi::cxx_bridge::FaissMetricType::Linf => FaissMetricType::Linf,
            crate::ffi::cxx_bridge::FaissMetricType::Lp => FaissMetricType::Lp,
            crate::ffi::cxx_bridge::FaissMetricType::Canberra => FaissMetricType::Canberra,
            crate::ffi::cxx_bridge::FaissMetricType::BrayCurtis => FaissMetricType::BrayCurtis,
            crate::ffi::cxx_bridge::FaissMetricType::JensenShannon => {
                FaissMetricType::JensenShannon
            }
            crate::ffi::cxx_bridge::FaissMetricType::Jaccard => FaissMetricType::Jaccard,
            crate::ffi::cxx_bridge::FaissMetricType::NaNEuclidean => FaissMetricType::NaNEuclidean,
            crate::ffi::cxx_bridge::FaissMetricType::Gower => FaissMetricType::Gower,
            _ => panic!("Undefined Metric Type"),
        }
    }
}

#[cfg(any(target_family = "unix", target_family = "windows"))]
impl Into<crate::ffi::cxx_bridge::FaissMetricType> for FaissMetricType {
    fn into(self) -> crate::ffi::cxx_bridge::FaissMetricType {
        match self {
            FaissMetricType::InnerProduct => crate::ffi::cxx_bridge::FaissMetricType::InnerProduct,
            FaissMetricType::L2 => crate::ffi::cxx_bridge::FaissMetricType::L2,
            FaissMetricType::L1 => crate::ffi::cxx_bridge::FaissMetricType::L1,
            FaissMetricType::Linf => crate::ffi::cxx_bridge::FaissMetricType::Linf,
            FaissMetricType::Lp => crate::ffi::cxx_bridge::FaissMetricType::Lp,
            FaissMetricType::Canberra => crate::ffi::cxx_bridge::FaissMetricType::Canberra,
            FaissMetricType::BrayCurtis => crate::ffi::cxx_bridge::FaissMetricType::BrayCurtis,
            FaissMetricType::JensenShannon => {
                crate::ffi::cxx_bridge::FaissMetricType::JensenShannon
            }
            FaissMetricType::Jaccard => crate::ffi::cxx_bridge::FaissMetricType::Jaccard,
            FaissMetricType::NaNEuclidean => crate::ffi::cxx_bridge::FaissMetricType::NaNEuclidean,
            FaissMetricType::Gower => crate::ffi::cxx_bridge::FaissMetricType::Gower,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FaissIndexSearchResult {
    pub distances: Vec<f32>,
    pub indexes: Vec<i64>,
}

#[cfg(any(target_family = "unix", target_family = "windows"))]
impl From<crate::ffi::cxx_bridge::FaissIndexSearchResult> for FaissIndexSearchResult {
    fn from(value: crate::ffi::cxx_bridge::FaissIndexSearchResult) -> Self {
        FaissIndexSearchResult {
            distances: value.distances.clone(),
            indexes: value.indexes.clone(),
        }
    }
}

#[derive(Debug)]
pub struct FaissIndexBuilder {
    dimension: i32,
    description: String,
    metric: FaissMetricType,
}

impl FaissIndexBuilder {
    pub fn new(dimension: i32) -> Self {
        FaissIndexBuilder {
            dimension,
            description: "IDMap2,Flat".to_owned(),
            metric: FaissMetricType::InnerProduct,
        }
    }

    pub fn description(mut self, description: &str) -> Self {
        self.description = description.to_owned();
        self
    }

    pub fn metric(mut self, metric: FaissMetricType) -> Self {
        self.metric = metric;
        self
    }

    pub async fn build(self) -> Result<FaissIndex> {
        FaissIndex::new(self.dimension, self.description.as_str(), self.metric).await
    }
}

/// Rust wrapper of faiss::Index
pub struct FaissIndex {
    #[cfg(any(target_family = "unix", target_family = "windows"))]
    inner: cxx::UniquePtr<crate::ffi::cxx_bridge::FaissIndexInner>,
    #[cfg(target_family = "wasm")]
    inner: crate::ffi::js_bridge::FaissIndexInner,

    next_id: AtomicI64, // thread-safe ID Generator
}

impl FaissIndex {
    pub async fn new(dimension: i32, description: &str, metric: FaissMetricType) -> Result<Self> {
        #[cfg(any(target_family = "unix", target_family = "windows"))]
        let wrapper =
            unsafe { crate::ffi::cxx_bridge::create_index(dimension, description, metric.into())? };

        #[cfg(target_family = "wasm")]
        let wrapper = {
            use wasm_bindgen::JsCast;

            let obj = js_sys::Object::new();
            js_sys::Reflect::set(&obj, &"dimension".into(), &dimension.into()).unwrap();
            js_sys::Reflect::set(&obj, &"description".into(), &description.into()).unwrap();
            js_sys::Reflect::set(&obj, &"metric".into(), &metric.to_string().into()).unwrap();
            let promise = crate::ffi::js_bridge::init_faiss_index_inner(&obj.into()).unwrap();
            let js_result = wasm_bindgen_futures::JsFuture::from(promise).await.unwrap();
            let js_instance = js_result.unchecked_into::<crate::ffi::js_bridge::FaissIndexInner>();
            js_instance
        };

        Ok(Self {
            inner: wrapper,
            next_id: AtomicI64::new(0),
        })
    }

    #[cfg(any(target_family = "unix", target_family = "windows"))]
    fn inner(&self) -> &crate::ffi::cxx_bridge::FaissIndexInner {
        self.inner.as_ref().unwrap()
    }

    #[cfg(target_family = "wasm")]
    fn inner(&self) -> &crate::ffi::js_bridge::FaissIndexInner {
        &self.inner
    }

    pub fn is_trained(&self) -> bool {
        self.inner().is_trained()
    }

    pub fn ntotal(&self) -> i64 {
        self.inner().get_ntotal()
    }

    pub fn dimension(&self) -> i32 {
        self.inner().get_dimension()
    }

    pub fn metric_type(&self) -> FaissMetricType {
        #[cfg(any(target_family = "unix", target_family = "windows"))]
        {
            self.inner().get_metric_type().into()
        }

        #[cfg(target_family = "wasm")]
        {
            let metric = self.inner().get_metric_type().as_string().unwrap();
            FaissMetricType::from_str(metric.as_str()).unwrap()
        }
    }

    pub fn train(&mut self, training_vectors: &[Vec<f32>]) -> Result<()> {
        if self.is_trained() {
            return Ok(());
        }

        let flattened: Vec<f32> = training_vectors.iter().flatten().cloned().collect();
        let num_vectors = training_vectors.len();

        #[cfg(any(target_family = "unix", target_family = "windows"))]
        unsafe {
            Ok(self.inner.pin_mut().train_index(&flattened, num_vectors)?)
        }

        #[cfg(target_family = "wasm")]
        {
            let arr = js_sys::Float32Array::new_with_length(flattened.len() as u32);
            for (i, &val) in flattened.iter().enumerate() {
                arr.set_index(i as u32, val);
            }
            self.inner().train_index(&arr, num_vectors as u32).unwrap();
            Ok(())
        }
    }

    pub fn add_vector(&mut self, vector: &Vec<f32>) -> Result<String> {
        let ids = self.add_vectors(&[vector.clone()])?;
        Ok(ids.first().unwrap().to_string())
    }

    pub fn add_vectors(&mut self, vectors: &[Vec<f32>]) -> Result<Vec<String>> {
        if vectors.is_empty() {
            return Ok(vec![]);
        }

        let flattened: Vec<f32> = vectors.iter().flatten().cloned().collect();
        let num_vectors = vectors.len();

        let start_id = self.next_id.fetch_add(num_vectors as i64, Ordering::SeqCst);
        let ids: Vec<i64> = (start_id..start_id + num_vectors as i64).collect();

        #[cfg(any(target_family = "unix", target_family = "windows"))]
        unsafe {
            self.inner
                .pin_mut()
                .add_vectors_with_ids(&flattened, num_vectors, &ids)?;
        }

        #[cfg(target_family = "wasm")]
        {
            let vector_arr = js_sys::Float32Array::new_with_length(flattened.len() as u32);
            for (i, &val) in flattened.iter().enumerate() {
                vector_arr.set_index(i as u32, val);
            }
            let ids_arr = js_sys::BigInt64Array::new_with_length(ids.len() as u32);
            for (i, &val) in ids.iter().enumerate() {
                ids_arr.set_index(i as u32, val);
            }

            self.inner()
                .add_vectors_with_ids(&vector_arr, ids.len() as u32, &ids_arr)
                .unwrap();
        }

        Ok(ids.into_iter().map(|id| id.to_string()).collect())
    }

    pub fn search(
        &self,
        query_vectors: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<FaissIndexSearchResult>> {
        if query_vectors.is_empty() {
            return Ok(vec![]);
        }

        let num_queries = query_vectors.len();
        let flattened: Vec<f32> = query_vectors.iter().flatten().cloned().collect();

        let search_result: FaissIndexSearchResult = {
            #[cfg(any(target_family = "unix", target_family = "windows"))]
            unsafe {
                FaissIndexSearchResult::from(self.inner().search_vectors(&flattened, k)?)
            }

            #[cfg(target_family = "wasm")]
            {
                let query_vectors_arr =
                    js_sys::Float32Array::new_with_length(flattened.len() as u32);
                for (i, val) in flattened.into_iter().enumerate() {
                    query_vectors_arr.set_index(i as u32, val);
                }
                let raw_result = self
                    .inner()
                    .search_vectors(&query_vectors_arr, k as u32)
                    .unwrap();
                FaissIndexSearchResult {
                    distances: raw_result.distances().to_vec(),
                    indexes: raw_result.indexes().to_vec(),
                }
            }
        };

        let expected_len = num_queries * k;
        if search_result.indexes.len() != expected_len
            || search_result.distances.len() != expected_len
        {
            bail!(
                "C++ FFI returned mismatched result length. Expected: {}, Got: (indexes: {}, distances: {})",
                expected_len,
                search_result.indexes.len(),
                search_result.distances.len()
            );
        }

        let results: Vec<FaissIndexSearchResult> = search_result
            .distances
            .chunks_exact(k)
            .zip(search_result.indexes.chunks_exact(k))
            .map(|(distances_chunk, indexes_chunk)| FaissIndexSearchResult {
                distances: distances_chunk.to_vec(),
                indexes: indexes_chunk.to_vec(),
            })
            .collect();

        if results.len() != num_queries {
            bail!(
                "Internal logic error: Failed to group search results correctly. Expected {} groups, got {}.",
                num_queries,
                results.len()
            );
        }

        Ok(results)
    }

    /// assume that for every id, there is a vector corresponding to that id.
    /// This should be guaranteed before call this function.
    pub fn get_by_ids(&self, ids: &[&str]) -> Result<Vec<Vec<f32>>> {
        if ids.is_empty() {
            return Ok(vec![]);
        }

        let numeric_ids: Vec<i64> = ids
            .iter()
            .map(|s| s.parse::<i64>())
            .collect::<Result<Vec<i64>, _>>()
            .context("Failed to parse one or more string IDs to integer")?;

        let dimension = self.inner().get_dimension() as usize;
        let expected_len = ids.len() * dimension;

        let flat_results: Vec<f32> = {
            #[cfg(any(target_family = "unix", target_family = "windows"))]
            unsafe {
                self.inner().get_by_ids(&numeric_ids)?
            }

            #[cfg(target_family = "wasm")]
            {
                let ids_arr = js_sys::BigInt64Array::new_with_length(numeric_ids.len() as u32);
                for (i, val) in numeric_ids.into_iter().enumerate() {
                    ids_arr.set_index(i as u32, val);
                }
                let flat_vectors = self.inner().get_by_ids(&ids_arr).unwrap();
                flat_vectors.to_vec()
            }
        };

        if flat_results.len() != expected_len {
            bail!(
                "FFI returned mismatched result length. Expected: {}, Got: {}",
                expected_len,
                flat_results.len()
            );
        }
        let results: Vec<Vec<f32>> = flat_results
            .chunks_exact(dimension)
            .map(|chunk| chunk.to_vec())
            .collect();

        if results.len() != ids.len() {
            bail!(
                "Internal logic error: Failed to group get results correctly. Expected {} groups, got {}.",
                ids.len(),
                results.len()
            );
        }

        Ok(results)
    }

    /// assume that for every id, there is a vector corresponding to that id.
    /// This should be guaranteed before call this function.
    pub fn remove_vectors(&mut self, ids: &[&str]) -> Result<usize> {
        let numeric_ids: Vec<i64> = ids
            .iter()
            .map(|s| s.parse::<i64>())
            .collect::<Result<Vec<i64>, _>>()?;

        #[cfg(any(target_family = "unix", target_family = "windows"))]
        unsafe {
            Ok(self.inner.pin_mut().remove_vectors(&numeric_ids)?)
        }

        #[cfg(target_family = "wasm")]
        {
            let arr = js_sys::BigInt64Array::new_with_length(numeric_ids.len() as u32);
            for (i, val) in numeric_ids.into_iter().enumerate() {
                arr.set_index(i as u32, val);
            }
            Ok(self.inner().remove_vectors(&arr).unwrap() as usize)
        }
    }

    pub fn clear(&mut self) -> Result<()> {
        #[cfg(any(target_family = "unix", target_family = "windows"))]
        unsafe {
            Ok(self.inner.pin_mut().clear()?)
        }

        #[cfg(target_family = "wasm")]
        {
            self.inner().clear().unwrap();
            Ok(())
        }
    }

    #[cfg(any(target_family = "unix", target_family = "windows"))]
    pub fn write_index(&self, filename: &str) -> Result<()> {
        unsafe { Ok(self.inner().write_index(filename)?) }
    }

    #[cfg(any(target_family = "unix", target_family = "windows"))]
    pub fn read_index(filename: &str) -> Result<Self> {
        let wrapper = unsafe { crate::ffi::cxx_bridge::read_index(filename)? };
        let current_total = wrapper.get_ntotal();
        Ok(Self {
            inner: wrapper,
            next_id: AtomicI64::new(current_total),
        })
    }

    // Debug
    pub fn current_id_counter(&self) -> i64 {
        self.next_id.load(Ordering::SeqCst)
    }
}
