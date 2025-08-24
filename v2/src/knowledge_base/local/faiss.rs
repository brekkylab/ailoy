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
        // Ok(self.add_vectors(vec![input]).await?.into_iter().next().unwrap())
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

    // async fn batch_retrieve(
    //     &self,
    //     query_embeddings: Vec<Embedding>,
    //     top_k: usize,
    // ) -> Result<Vec<Vec<RetrieveResult>>> {
    //     let index_results = self.index.search(&query_embeddings, top_k).unwrap();

    //     Ok(index_results
    //         .into_iter()
    //         .map(|index_result| {
    //             index_result
    //                 .indexes
    //                 .into_iter()
    //                 .zip(index_result.distances.into_iter())
    //                 .filter_map(|(id_i64, distance)| {
    //                     let id = id_i64.to_string();
    //                     self.doc_store.get(&id).map(|doc_entry| RetrieveResult {
    //                         id,
    //                         document: doc_entry.document.clone(),
    //                         metadata: doc_entry.metadata.clone(),
    //                         distance,
    //                     })
    //                 })
    //                 .collect::<Vec<_>>()
    //         })
    //         .collect::<Vec<_>>())
    // }

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
