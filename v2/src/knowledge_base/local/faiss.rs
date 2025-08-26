use std::collections::HashMap;

use super::super::{AddInput, Embedding, GetResult, Metadata, RetrieveResult, VectorStore};
use crate::ffi;

use anyhow::Result;
use async_trait::async_trait;

pub struct DocEntry {
    pub document: String,
    pub metadata: Option<Metadata>,
}
type DocStore = HashMap<String, DocEntry>;

pub struct FaissStore {
    index: ffi::FaissIndex,
    doc_store: DocStore,
}

impl FaissStore {
    pub fn new(dim: i32) -> Result<Self> {
        // use only default type("IdMap2,Flat", InnerProduct) Index for now
        let index = ffi::FaissIndexBuilder::new(dim).build().unwrap();
        Ok(Self {
            index,
            doc_store: HashMap::new(),
        })
    }
}

#[async_trait]
impl VectorStore for FaissStore {
    async fn add_vector(&mut self, input: AddInput) -> Result<String> {
        let ids: Vec<String> = self.index.add_vectors(&[input.embedding]).unwrap();
        let id = ids.iter().next().unwrap().clone();
        self.doc_store.insert(
            id.clone(),
            DocEntry {
                document: input.document,
                metadata: input.metadata,
            },
        );
        Ok(id)
    }

    async fn add_vectors(&mut self, inputs: Vec<AddInput>) -> Result<Vec<String>> {
        let (embeddings, entries): (Vec<Embedding>, Vec<DocEntry>) = inputs
            .into_iter()
            .map(|input| {
                (
                    input.embedding,
                    DocEntry {
                        document: input.document,
                        metadata: input.metadata,
                    },
                )
            })
            .unzip();
        let ids: Vec<String> = self.index.add_vectors(&embeddings).unwrap();
        self.doc_store
            .extend(ids.iter().cloned().zip(entries.into_iter()));
        Ok(ids)
    }

    async fn get_by_id(&self, id: &str) -> Result<Option<GetResult>> {
        if !self.doc_store.contains_key(&id.to_string()) {
            return Ok(None);
        }

        let embeddings: Vec<Vec<f32>> = self.index.get_by_ids(&[id]).unwrap();
        let embedding = embeddings.into_iter().next().unwrap();
        let doc_entry = self.doc_store.get(&id.to_string()).unwrap();
        Ok(Some(GetResult {
            id: id.to_string(),
            document: doc_entry.document.clone(),
            metadata: doc_entry.metadata.clone(),
            embedding,
        }))
    }

    async fn get_by_ids(&self, ids: &[&str]) -> Result<Vec<GetResult>> {
        // filter ids to only those that exist in doc_store
        let filtered_ids: Vec<_> = ids
            .iter()
            .filter(|&&id| self.doc_store.contains_key(id))
            .cloned()
            .collect();

        let embeddings: Vec<Vec<f32>> = self.index.get_by_ids(&filtered_ids).unwrap();

        Ok(ids
            .iter()
            .zip(embeddings.into_iter())
            .filter_map(|(id, embedding)| {
                self.doc_store
                    .get(&id.to_string())
                    .map(|doc_entry| GetResult {
                        id: id.to_string(),
                        document: doc_entry.document.clone(),
                        metadata: doc_entry.metadata.clone(),
                        embedding,
                    })
            })
            .collect())
    }

    async fn retrieve(
        &self,
        query_embedding: Embedding,
        top_k: usize,
    ) -> Result<Vec<RetrieveResult>> {
        let index_results = self.index.search(&[query_embedding], top_k).unwrap();
        let index_result = index_results.into_iter().next().unwrap();

        Ok(index_result
            .indexes
            .into_iter()
            .zip(index_result.distances.into_iter())
            .filter_map(|(id_i64, distance)| {
                let id = id_i64.to_string();
                self.doc_store.get(&id).map(|doc_entry| RetrieveResult {
                    id,
                    document: doc_entry.document.clone(),
                    metadata: doc_entry.metadata.clone(),
                    distance,
                })
            })
            .collect::<Vec<_>>())
    }

    async fn batch_retrieve(
        &self,
        query_embeddings: Vec<Embedding>,
        top_k: usize,
    ) -> Result<Vec<Vec<RetrieveResult>>> {
        let index_results = self.index.search(&query_embeddings, top_k).unwrap();

        Ok(index_results
            .into_iter()
            .map(|index_result| {
                index_result
                    .indexes
                    .into_iter()
                    .zip(index_result.distances.into_iter())
                    .filter_map(|(id_i64, distance)| {
                        let id = id_i64.to_string();
                        self.doc_store.get(&id).map(|doc_entry| RetrieveResult {
                            id,
                            document: doc_entry.document.clone(),
                            metadata: doc_entry.metadata.clone(),
                            distance,
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>())
    }

    async fn remove_vector(&mut self, id: &str) -> Result<()> {
        if !self.doc_store.contains_key(&id.to_string()) {
            return Ok(());
        }

        self.index.remove_vectors(&[id]).unwrap();
        self.doc_store.remove(id);
        Ok(())
    }

    async fn remove_vectors(&mut self, ids: &[&str]) -> Result<()> {
        // filter ids to only those that exist in doc_store
        let filtered_ids: Vec<_> = ids
            .iter()
            .filter(|&&id| self.doc_store.contains_key(id))
            .cloned()
            .collect();

        self.index.remove_vectors(&filtered_ids).unwrap();
        for id in filtered_ids {
            self.doc_store.remove(id);
        }

        Ok(())
    }

    async fn clear(&mut self) -> Result<()> {
        self.index.clear().unwrap();
        self.doc_store.clear();
        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        Ok(self.index.ntotal() as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Ok;
    use serde_json::json;
    use tokio;

    use crate::multi_platform_test;

    async fn setup_test_store() -> Result<FaissStore> {
        Ok(FaissStore::new(3).unwrap())
    }

    multi_platform_test! {
        async fn test_add_and_get_vector() -> Result<()> {
            let mut store = setup_test_store().await?;
            let test_embedding = vec![1.1, 2.2, 3.3];
            let test_document = "This is a test document.".to_owned();
            let test_metadata = json!({"source": "test_add_and_get_vector"})
                .as_object()
                .unwrap()
                .clone();

            let input = AddInput {
                embedding: test_embedding.clone(),
                document: test_document.clone(),
                metadata: Some(test_metadata.clone()),
            };

            let added_id = store.add_vector(input).await?;
            let result = store.get_by_id(&added_id).await?;

            assert!(result.is_some(), "Vector should be found by its ID");
            let retrieved = result.unwrap();

            assert_eq!(retrieved.id, added_id);
            assert_eq!(retrieved.document, test_document);
            assert_eq!(retrieved.embedding, test_embedding);
            assert_eq!(retrieved.metadata, Some(test_metadata));

            Ok(())
        }
    }

    multi_platform_test! {
        async fn test_add_vectors_batch() -> Result<()> {
            let mut store = setup_test_store().await?;

            let added_ids = store
                .add_vectors(vec![
                    AddInput {
                        embedding: vec![1.0, 1.0, 1.0],
                        document: "doc1".to_owned(),
                        metadata: Some(json!({"id": 1}).as_object().unwrap().clone()),
                    },
                    AddInput {
                        embedding: vec![2.0, 2.0, 2.0],
                        document: "doc2".to_owned(),
                        metadata: Some(json!({"id": 2}).as_object().unwrap().clone()),
                    },
                ])
                .await?;

            assert_eq!(added_ids.len(), 2);

            let res1 = store.get_by_id(&added_ids[0]).await?.unwrap();
            assert_eq!(res1.document, "doc1");
            assert_eq!(res1.embedding, vec![1.0, 1.0, 1.0]);

            let res2 = store.get_by_id(&added_ids[1]).await?.unwrap();
            assert_eq!(res2.document, "doc2");
            assert_eq!(res2.embedding, vec![2.0, 2.0, 2.0]);

            let res = store
                .get_by_ids(&added_ids.iter().map(|id| id.as_str()).collect::<Vec<_>>())
                .await?;
            assert_eq!(res[0].document, "doc1");
            assert_eq!(res[0].embedding, vec![1.0, 1.0, 1.0]);

            assert_eq!(res[1].document, "doc2");
            assert_eq!(res[1].embedding, vec![2.0, 2.0, 2.0]);

            Ok(())
        }
    }

    multi_platform_test! {
        async fn test_retrieve_similar_vectors() -> Result<()> {
            let mut store = setup_test_store().await?;
            let inputs = vec![
                AddInput {
                    embedding: vec![1.0, 0.0, 0.0],
                    document: "vector one".to_owned(),
                    metadata: None,
                },
                AddInput {
                    embedding: vec![0.0, 1.0, 0.0],
                    document: "vector two".to_owned(),
                    metadata: None,
                },
                AddInput {
                    embedding: vec![0.9, 0.1, 0.0],
                    document: "vector one-ish".to_owned(),
                    metadata: None,
                },
                AddInput {
                    embedding: vec![0.0, 0.9, 0.1],
                    document: "vector two-ish".to_owned(),
                    metadata: None,
                },
            ];
            store.add_vectors(inputs).await?;

            let query_embeddings = vec![vec![0.95, 0.0, 0.0], vec![0.0, 0.95, 0.0]];

            // top_k=2
            let results = store
                .retrieve(query_embeddings.get(0).cloned().unwrap(), 2)
                .await?;

            assert_eq!(results.len(), 2);

            let retrieved_docs: Vec<_> = results.iter().map(|r| r.document.clone()).collect();
            // should exist
            assert!(retrieved_docs.contains(&"vector one".to_owned()));
            assert!(retrieved_docs.contains(&"vector one-ish".to_owned()));
            // should not exist
            assert!(!retrieved_docs.contains(&"vector two".to_owned()));
            assert!(!retrieved_docs.contains(&"vector two-ish".to_owned()));

            // top_k=2
            let batch_results = store.batch_retrieve(query_embeddings, 2).await?;

            for (i, results) in batch_results.iter().enumerate() {
                assert_eq!(results.len(), 2);
                let retrieved_docs: Vec<_> = results.iter().map(|r| r.document.clone()).collect();
                if i == 0 {
                    // should exist
                    assert!(retrieved_docs.contains(&"vector one".to_owned()));
                    assert!(retrieved_docs.contains(&"vector one-ish".to_owned()));
                    // should not exist
                    assert!(!retrieved_docs.contains(&"vector two".to_owned()));
                    assert!(!retrieved_docs.contains(&"vector two-ish".to_owned()));
                } else {
                    // should exist
                    assert!(retrieved_docs.contains(&"vector two".to_owned()));
                    assert!(retrieved_docs.contains(&"vector two-ish".to_owned()));
                    // should not exist
                    assert!(!retrieved_docs.contains(&"vector one".to_owned()));
                    assert!(!retrieved_docs.contains(&"vector one-ish".to_owned()));
                }
            }

            Ok(())
        }
    }

    multi_platform_test! {
        async fn test_remove_vector() -> Result<()> {
            let mut store = setup_test_store().await?;
            let inputs = vec![
                AddInput {
                    embedding: vec![5.5, 6.6, 3.3],
                    document: "to be deleted1".to_owned(),
                    metadata: None,
                },
                AddInput {
                    embedding: vec![4.4, 6.6, 5.5],
                    document: "to be deleted2".to_owned(),
                    metadata: None,
                },
                AddInput {
                    embedding: vec![3.3, 2.2, 6.6],
                    document: "to be deleted3".to_owned(),
                    metadata: None,
                },
            ];
            let mut ids_to_delete = store.add_vectors(inputs).await?;

            let first_id = ids_to_delete.swap_remove(0);

            store.remove_vector(&first_id).await?;
            assert_eq!(store.count().await?, 2);

            let result = store.get_by_id(&first_id).await?;
            assert!(
                result.is_none(),
                "Vector should not be found after deletion"
            );

            store
                .remove_vectors(
                    &ids_to_delete
                        .iter()
                        .map(|id| id.as_str())
                        .collect::<Vec<_>>(),
                )
                .await?;
            assert_eq!(store.count().await?, 0);

            Ok(())
        }
    }

    multi_platform_test! {
        async fn test_get_non_existent_vector() -> Result<()> {
            let store = setup_test_store().await?;
            let non_existent_id = "-1";

            let result = store.get_by_id(&non_existent_id).await?;

            assert!(
                result.is_none(),
                "Getting a non-existent ID should return None"
            );

            Ok(())
        }
    }

    multi_platform_test! {
        async fn test_clear_collection() -> Result<()> {
            let mut store = setup_test_store().await?;
            let inputs = vec![
                AddInput {
                    embedding: vec![1.0, 1.1, 1.2],
                    document: "doc1".to_owned(),
                    metadata: None,
                },
                AddInput {
                    embedding: vec![2.0, 2.1, 2.2],
                    document: "doc2".to_owned(),
                    metadata: None,
                },
            ];
            let added_ids = store.add_vectors(inputs).await?;
            assert_eq!(added_ids.len(), 2);

            store.clear().await?;

            let result = store.get_by_id(&added_ids[0]).await?;
            assert!(
                result.is_none(),
                "Vector should not be found after clearing the collection"
            );

            let count = store.count().await?;
            assert_eq!(count, 0);

            Ok(())
        }
    }
}
