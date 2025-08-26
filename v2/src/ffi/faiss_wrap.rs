use anyhow::{Context, Result, bail};
use std::sync::atomic::{AtomicI64, Ordering};

use crate::ffi::{
    FaissIndexSearchResult, FaissIndexWrapper, FaissMetricType, create_index, read_index,
};

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

    pub fn build(self) -> Result<FaissIndex> {
        FaissIndex::new(self.dimension, self.description.as_str(), self.metric)
    }
}

/// Rust wrapper of faiss::Index
pub struct FaissIndex {
    inner: cxx::UniquePtr<FaissIndexWrapper>,
    next_id: AtomicI64, // thread-safe ID Generator
}

impl FaissIndex {
    pub fn new(dimension: i32, description: &str, metric: FaissMetricType) -> Result<Self> {
        let wrapper = unsafe { create_index(dimension, description, metric)? };
        Ok(Self {
            inner: wrapper,
            next_id: AtomicI64::new(0),
        })
    }

    pub fn is_trained(&self) -> bool {
        self.inner.as_ref().unwrap().is_trained()
    }

    pub fn ntotal(&self) -> i64 {
        self.inner.as_ref().unwrap().get_ntotal()
    }

    pub fn dimension(&self) -> i32 {
        self.inner.as_ref().unwrap().get_dimension()
    }

    pub fn metric_type(&self) -> FaissMetricType {
        self.inner.as_ref().unwrap().get_metric_type()
    }

    pub fn train(&mut self, training_vectors: &[Vec<f32>]) -> Result<()> {
        if self.is_trained() {
            return Ok(());
        }

        let flattened: Vec<f32> = training_vectors.iter().flatten().cloned().collect();
        let num_vectors = training_vectors.len();

        unsafe { Ok(self.inner.pin_mut().train_index(&flattened, num_vectors)?) }
    }

    pub fn add_vector(&mut self, vector: &Vec<f32>) -> Result<String> {
        let id = self.ntotal();

        unsafe {
            self.inner
                .pin_mut()
                .add_vectors_with_ids(vector, 1, &[id])?;
        }

        Ok(id.to_string())
    }

    pub fn add_vectors(&mut self, vectors: &[Vec<f32>]) -> Result<Vec<String>> {
        if vectors.is_empty() {
            return Ok(vec![]);
        }

        let flattened: Vec<f32> = vectors.iter().flatten().cloned().collect();
        let num_vectors = vectors.len();

        let start_id = self.next_id.fetch_add(num_vectors as i64, Ordering::SeqCst);
        let ids: Vec<i64> = (start_id..start_id + num_vectors as i64).collect();

        unsafe {
            self.inner
                .pin_mut()
                .add_vectors_with_ids(&flattened, num_vectors, &ids)?;
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

        let search_result = unsafe { self.inner.as_ref().unwrap().search_vectors(&flattened, k)? };

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

        let wrapper_ref = self.inner.as_ref().unwrap();

        let numeric_ids: Vec<i64> = ids
            .iter()
            .map(|s| s.parse::<i64>())
            .collect::<Result<Vec<i64>, _>>()
            .context("Failed to parse one or more string IDs to integer")?;

        let dimension = wrapper_ref.get_dimension() as usize;
        let expected_len = ids.len() * dimension;
        let flat_vectors = unsafe { wrapper_ref.get_by_ids(&numeric_ids)? };
        if flat_vectors.len() != expected_len {
            bail!(
                "C++ FFI returned mismatched result length. Expected: {}, Got: {}",
                expected_len,
                flat_vectors.len()
            );
        }

        let results: Vec<Vec<f32>> = flat_vectors
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
        unsafe { Ok(self.inner.pin_mut().remove_vectors(&numeric_ids)?) }
    }

    pub fn clear(&mut self) -> Result<()> {
        unsafe { Ok(self.inner.pin_mut().clear()?) }
    }

    pub fn write_index(&self, filename: &str) -> Result<()> {
        unsafe { Ok(self.inner.as_ref().unwrap().write_index(filename)?) }
    }

    pub fn read_index(filename: &str) -> Result<Self> {
        let wrapper = unsafe { read_index(filename)? };
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
