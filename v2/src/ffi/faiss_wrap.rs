use crate::ffi::FaissIndex;

// --- 검색 결과를 위한 구조체들 ---
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub matches: Vec<SearchMatch>,
}

#[derive(Debug, Clone)]
pub struct SearchMatch {
    pub id: String,
    pub distance: f32,
}

impl ffi::FaissIndex {
    /// 새로운 FAISS 인덱스를 생성합니다.
    ///
    /// # Arguments
    /// * `dimension` - 벡터의 차원
    /// * `description` - Index factory 문자열 (예: "IVF100,Flat", "HNSW32")
    /// * `metric` - 거리 측정 방식
    pub fn new(dimension: i32, description: &str, metric: ffi::MetricType) -> Result<Self> {
        let wrapper = unsafe { ffi::create_index(dimension, description, metric)? };
        Ok(Self { inner: wrapper })
    }

    /// 인덱스가 학습되었는지 확인합니다.
    pub fn is_trained(&self) -> bool {
        ffi::is_trained(self.inner.as_ref().unwrap())
    }

    /// 인덱스에 저장된 벡터의 총 개수를 반환합니다.
    pub fn ntotal(&self) -> i64 {
        ffi::get_ntotal(self.inner.as_ref().unwrap())
    }

    /// 벡터의 차원을 반환합니다.
    pub fn dimension(&self) -> i32 {
        ffi::get_dimension(self.inner.as_ref().unwrap())
    }

    /// 인덱스를 학습시킵니다 (IVF 계열 등 필요한 경우).
    pub fn train(&mut self, training_vectors: &[Vec<f32>]) -> Result<()> {
        if self.is_trained() {
            return Ok(());
        }

        let flattened: Vec<f32> = training_vectors.iter().flatten().cloned().collect();
        let num_vectors = training_vectors.len();

        unsafe { ffi::train_index(self.inner.as_mut().unwrap(), &flattened, num_vectors) }
    }

    /// 벡터들을 인덱스에 추가하고 할당된 ID들을 문자열로 반환합니다.
    pub fn add_vectors(&mut self, vectors: &[Vec<f32>]) -> Result<Vec<String>> {
        if vectors.is_empty() {
            return Ok(vec![]);
        }

        // 1. 벡터들을 flatten하여 하나의 배열로 만듭니다.
        let flattened: Vec<f32> = vectors.iter().flatten().cloned().collect();
        let num_vectors = vectors.len();

        // 2. ID들을 생성합니다. 현재 ntotal부터 시작합니다.
        let start_id = self.ntotal();
        let ids: Vec<i64> = (start_id..start_id + num_vectors as i64).collect();

        // 3. C++의 add_vectors_with_ids를 호출합니다.
        unsafe {
            ffi::add_vectors_with_ids(self.inner.as_mut().unwrap(), &flattened, num_vectors, &ids)?;
        }

        // 4. ID들을 String으로 변환하여 반환합니다.
        Ok(ids.into_iter().map(|id| id.to_string()).collect())
    }

    /// 쿼리 벡터들과 가장 유사한 k개의 벡터들을 검색합니다.
    ///
    /// # Returns
    /// Vec<SearchResult> - 각 쿼리에 대한 검색 결과
    pub fn search(&self, query_vectors: &[Vec<f32>], k: usize) -> Result<Vec<SearchResult>> {
        if query_vectors.is_empty() {
            return Ok(vec![]);
        }

        let num_queries = query_vectors.len();
        let flattened: Vec<f32> = query_vectors.iter().flatten().cloned().collect();

        // 결과를 저장할 배열들을 준비합니다.
        let mut distances = vec![0.0f32; num_queries * k];
        let mut indices = vec![0i64; num_queries * k];

        unsafe {
            ffi::search_vectors(
                self.inner.as_ref().unwrap(),
                &flattened,
                num_queries,
                k,
                &mut distances,
                &mut indices,
            )?;
        }

        // 결과를 구조화하여 반환합니다.
        let mut results = Vec::with_capacity(num_queries);
        for query_idx in 0..num_queries {
            let start = query_idx * k;
            let end = start + k;

            let query_results: Vec<SearchMatch> = indices[start..end]
                .iter()
                .zip(distances[start..end].iter())
                .map(|(&id, &distance)| SearchMatch {
                    id: id.to_string(),
                    distance,
                })
                .collect();

            results.push(SearchResult {
                matches: query_results,
            });
        }

        Ok(results)
    }
}
