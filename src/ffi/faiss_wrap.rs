use std::sync::atomic::{AtomicI64, Ordering};

#[cfg(any(target_family = "unix", target_family = "windows"))]
use ailoy_faiss_sys::{FaissIndexSearchResult, FaissMetricType};
use anyhow::{Context, bail};

#[cfg(target_arch = "wasm32")]
use crate::ffi::web::faiss_bridge::{
    FaissIndexInner, FaissIndexSearchResult, FaissMetricType, create_faiss_index,
};

#[derive(Debug)]
pub struct FaissIndexBuilder {
    dimension: i32,
    description: String,
    metric: FaissMetricType,
}

#[allow(dead_code)]
impl FaissIndexBuilder {
    pub fn new(dimension: i32) -> Self {
        FaissIndexBuilder {
            dimension,
            description: "IDMap2,Flat".to_owned(),
            metric: FaissMetricType::L2,
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

    pub async fn build(self) -> anyhow::Result<FaissIndex> {
        FaissIndex::new(self.dimension, self.description.as_str(), self.metric).await
    }
}

/// Rust wrapper of faiss::Index
pub struct FaissIndex {
    #[cfg(any(target_family = "unix", target_family = "windows"))]
    inner: cxx::UniquePtr<ailoy_faiss_sys::FaissIndexInner>,
    #[cfg(target_family = "wasm")]
    inner: crate::ffi::web::faiss_bridge::FaissIndexInner,

    next_id: AtomicI64, // thread-safe ID Generator
}

#[allow(dead_code)]
impl FaissIndex {
    pub async fn new(
        dimension: i32,
        description: &str,
        metric: FaissMetricType,
    ) -> anyhow::Result<Self> {
        #[cfg(any(target_family = "unix", target_family = "windows"))]
        let wrapper =
            unsafe { ailoy_faiss_sys::create_index(dimension, description, metric.into())? };

        #[cfg(target_family = "wasm")]
        let wrapper = {
            create_faiss_index(dimension, description.into(), metric.to_string())
                .await
                .expect("Failed to create Faiss index")
        };

        Ok(Self {
            inner: wrapper,
            next_id: AtomicI64::new(0),
        })
    }

    #[cfg(any(target_family = "unix", target_family = "windows"))]
    fn inner(&self) -> &ailoy_faiss_sys::FaissIndexInner {
        self.inner.as_ref().unwrap()
    }

    #[cfg(target_family = "wasm")]
    fn inner(&self) -> &FaissIndexInner {
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
            self.inner()
                .get_metric_type()
                .as_string()
                .unwrap()
                .try_into()
                .unwrap()
        }
    }

    pub fn train(&mut self, training_vectors: &[Vec<f32>]) -> anyhow::Result<()> {
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
            self.inner().train_index(&arr, num_vectors).unwrap();
            Ok(())
        }
    }

    pub fn add_vector(&mut self, vector: &Vec<f32>) -> anyhow::Result<String> {
        let ids = self.add_vectors(&[vector.clone()])?;
        Ok(ids.first().unwrap().to_string())
    }

    pub fn add_vectors(&mut self, vectors: &[Vec<f32>]) -> anyhow::Result<Vec<String>> {
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
                .add_vectors_with_ids(&vector_arr, ids.len(), &ids_arr)
                .unwrap();
        }

        Ok(ids.into_iter().map(|id| id.to_string()).collect())
    }

    pub fn search(
        &self,
        query_vectors: &[Vec<f32>],
        k: usize,
    ) -> anyhow::Result<Vec<FaissIndexSearchResult>> {
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
                self.inner()
                    .search_vectors(&query_vectors_arr, k)
                    .expect("Failed to search vector store with query vector")
                    .into()
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
                distances: distances_chunk.into(),
                indexes: indexes_chunk.into(),
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
    pub fn get_by_ids(&self, ids: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
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
    pub fn remove_vectors(&mut self, ids: &[&str]) -> anyhow::Result<usize> {
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

    pub fn clear(&mut self) -> anyhow::Result<()> {
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
    pub fn write_index(&self, filename: &str) -> anyhow::Result<()> {
        unsafe { Ok(self.inner().write_index(filename)?) }
    }

    #[cfg(any(target_family = "unix", target_family = "windows"))]
    pub fn read_index(filename: &str) -> anyhow::Result<Self> {
        let wrapper = unsafe { ailoy_faiss_sys::read_index(filename)? };
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
