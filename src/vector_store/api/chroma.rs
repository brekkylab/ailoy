use ailoy_macros::multi_platform_async_trait;
use anyhow::{Context, bail};
use chromadb::{
    client::{ChromaAuthMethod, ChromaClient, ChromaClientOptions},
    collection::{ChromaCollection, CollectionEntries, GetOptions, QueryOptions, QueryResult},
};
use serde_json::{Map, Value as Json};
use uuid::Uuid;

use super::super::base::{
    VectorStoreAddInput, VectorStoreBehavior, VectorStoreGetResult, VectorStoreMetadata,
    VectorStoreRetrieveResult,
};
use crate::value::Embedding;

type ChromaMetadata = Map<String, Json>;

fn into_chroma_metadata(metadata: &VectorStoreMetadata) -> ChromaMetadata {
    let mut map = Map::new();
    for (key, val) in metadata.iter() {
        map.insert(key.clone(), val.clone().into());
    }
    map
}

fn from_chroma_metadata(metadata: &ChromaMetadata) -> VectorStoreMetadata {
    let mut map = VectorStoreMetadata::new();
    for (key, val) in metadata.iter() {
        map.insert(key.clone(), val.clone().into());
    }
    map
}

const CHROMADB_DEFAULT_COLLECTION: &'static str = "default_collection";

pub struct ChromaStore {
    client: ChromaClient,
    collection: ChromaCollection,
}

#[allow(dead_code)]
impl ChromaStore {
    pub async fn new(chroma_url: String, collection_name: Option<String>) -> anyhow::Result<Self> {
        let client = ChromaClient::new(ChromaClientOptions {
            url: Some(chroma_url.to_owned()),
            ..Default::default()
        })
        .await?;
        let collection = client
            .get_or_create_collection(
                &collection_name.unwrap_or(CHROMADB_DEFAULT_COLLECTION.to_owned()),
                None,
            )
            .await?;
        Ok(Self { client, collection })
    }

    pub async fn with_auth(
        chroma_url: &str,
        collection_name: &str,
        auth: ChromaAuthMethod,
    ) -> anyhow::Result<Self> {
        let client = ChromaClient::new(ChromaClientOptions {
            url: Some(chroma_url.to_owned()),
            auth,
            ..Default::default()
        })
        .await?;
        let collection = client
            .get_or_create_collection(collection_name, None)
            .await?;
        Ok(Self { client, collection })
    }

    pub async fn get_collection(&self, collection_name: &str) -> anyhow::Result<ChromaCollection> {
        self.client
            .get_collection(collection_name)
            .await
            .context(format!(
                "Failed to get collection '{}'. It might not exist or an error occurred.",
                collection_name
            ))
    }

    pub async fn create_collection(
        &self,
        collection_name: &str,
        metadata: Option<Map<String, Json>>,
    ) -> anyhow::Result<ChromaCollection> {
        self.client
            .create_collection(collection_name, metadata, true)
            .await
    }

    pub async fn delete_collection(&self, collection_name: &str) -> anyhow::Result<()> {
        self.client.delete_collection(collection_name).await
    }
}

#[multi_platform_async_trait]
impl VectorStoreBehavior for ChromaStore {
    async fn add_vector(&mut self, input: VectorStoreAddInput) -> anyhow::Result<String> {
        let id = Uuid::new_v4().to_string();

        let metadatas = input
            .metadata
            .as_ref()
            .map(|map| vec![into_chroma_metadata(map)]);

        let entry = CollectionEntries {
            ids: vec![id.as_ref()],
            embeddings: Some(vec![input.embedding.into()]),
            documents: Some(vec![input.document.as_ref()]),
            metadatas: metadatas,
        };
        self.collection.upsert(entry, None).await?;
        Ok(id)
    }

    async fn add_vectors(
        &mut self,
        inputs: Vec<VectorStoreAddInput>,
    ) -> anyhow::Result<Vec<String>> {
        let ids: Vec<String> = (0..inputs.len())
            .map(|_| Uuid::new_v4().to_string())
            .collect();
        let embeddings: Vec<Vec<f32>> = inputs.iter().map(|i| i.embedding.clone().into()).collect();
        let documents: Vec<String> = inputs.iter().map(|i| i.document.clone()).collect();
        let metadatas: Option<Vec<Map<String, Json>>> =
            if inputs.iter().any(|i| i.metadata.is_some()) {
                Some(
                    inputs
                        .iter()
                        .map(|i| match &i.metadata {
                            Some(map) => into_chroma_metadata(map),
                            _ => Map::new(),
                        })
                        .collect(),
                )
            } else {
                None
            };
        let entry = CollectionEntries {
            ids: ids.iter().map(|id| id.as_str()).collect(),
            embeddings: Some(embeddings),
            documents: Some(documents.iter().map(|d| d.as_str()).collect()),
            metadatas,
        };
        self.collection.upsert(entry, None).await?;
        Ok(ids)
    }

    async fn get_by_id(&self, id: &str) -> anyhow::Result<Option<VectorStoreGetResult>> {
        let results: Vec<VectorStoreGetResult> = self.get_by_ids(&[id]).await?;
        Ok(results.into_iter().next())
    }

    async fn get_by_ids(&self, ids: &[&str]) -> anyhow::Result<Vec<VectorStoreGetResult>> {
        let opts = GetOptions {
            ids: ids.iter().map(|s| s.to_string()).collect(),
            include: Some(vec![
                "metadatas".to_owned(),
                "documents".to_owned(),
                "embeddings".to_owned(),
            ]),
            ..Default::default()
        };
        let get_results = self.collection.get(opts).await?;
        if get_results.ids.is_empty() {
            return Ok(vec![]);
        }

        if let (Some(documents), Some(metadatas), Some(embeddings)) = (
            get_results.documents,
            get_results.metadatas,
            get_results.embeddings,
        ) {
            let zipped_iter = get_results
                .ids
                .into_iter()
                .zip(documents.into_iter())
                .zip(metadatas.into_iter())
                .zip(embeddings.into_iter())
                .map(|(((id, d), m), e)| (id, d.unwrap(), m, e.unwrap())); // 튜플을 평탄화하여 가독성 개선

            let results: Vec<VectorStoreGetResult> = zipped_iter
                .map(|(id, document, metadata, embedding)| VectorStoreGetResult {
                    id,
                    document,
                    metadata: metadata.map(|metadata| from_chroma_metadata(&metadata)),
                    embedding: embedding.into(),
                })
                .collect();
            return Ok(results);
        } else {
            bail!("Results from get operation are malformed.")
        }
    }

    async fn retrieve(
        &self,
        query: Embedding,
        top_k: usize,
    ) -> anyhow::Result<Vec<VectorStoreRetrieveResult>> {
        let opts = QueryOptions {
            query_embeddings: Some(vec![query.into()]),
            n_results: Some(top_k),
            ..Default::default()
        };
        let QueryResult {
            ids,
            documents,
            metadatas,
            distances,
            embeddings: _,
            ..
        } = self.collection.query(opts, None).await?;
        let out: Vec<VectorStoreRetrieveResult> =
            ids.get(0)
                .and_then(|ids_vec| {
                    distances
                        .as_ref()
                        .and_then(|d| d.get(0))
                        .map(|distances_vec| ids_vec.iter().zip(distances_vec).enumerate())
                })
                .map(|iter| {
                    iter.filter_map(|(i, (id_ref, &distance))| {
                        let id = id_ref.clone();
                        let document = documents
                            .as_ref()
                            .and_then(|d| d.get(0)?.get(i).cloned())
                            .unwrap_or_default();
                        let metadata = metadatas.as_ref().and_then(|m| m.get(0)?.get(i)).and_then(
                            |inner_opt| inner_opt.clone().map(|inner| from_chroma_metadata(&inner)),
                        );

                        Some(VectorStoreRetrieveResult {
                            id,
                            document,
                            metadata,
                            distance: distance as f64,
                        })
                    })
                    .collect()
                })
                .unwrap_or_else(Vec::new);

        Ok(out)
    }

    async fn batch_retrieve(
        &self,
        query_embeddings: Vec<Embedding>,
        top_k: usize,
    ) -> anyhow::Result<Vec<Vec<VectorStoreRetrieveResult>>> {
        let opts = QueryOptions {
            query_embeddings: Some(
                query_embeddings
                    .into_iter()
                    .map(|query| query.into())
                    .collect(),
            ),
            n_results: Some(top_k),
            ..Default::default()
        };
        let QueryResult {
            ids,
            documents,
            metadatas,
            distances,
            embeddings: _,
            ..
        } = self.collection.query(opts, None).await?;
        let out: Vec<Vec<VectorStoreRetrieveResult>> = ids
            .iter()
            .enumerate()
            .filter_map(|(outer_index, ids_vec)| {
                distances
                    .as_ref()
                    .and_then(|d| d.get(outer_index))
                    .map(|distances_vec| {
                        ids_vec
                            .iter()
                            .zip(distances_vec)
                            .enumerate()
                            .filter_map(|(inner_index, (id_ref, &distance))| {
                                let document = documents
                                    .as_ref()
                                    .and_then(|d| d.get(outer_index)?.get(inner_index).cloned())
                                    .unwrap_or_default();
                                let metadata = metadatas
                                    .as_ref()
                                    .and_then(|m| m.get(outer_index)?.get(inner_index))
                                    .and_then(|inner_opt| {
                                        inner_opt.clone().map(|inner| from_chroma_metadata(&inner))
                                    });

                                Some(VectorStoreRetrieveResult {
                                    id: id_ref.clone(),
                                    document,
                                    metadata,
                                    distance: distance as f64,
                                })
                            })
                            .collect()
                    })
            })
            .collect();

        Ok(out)
    }

    async fn remove_vector(&mut self, id: &str) -> anyhow::Result<()> {
        self.collection.delete(Some(vec![id]), None, None).await?;
        Ok(())
    }

    async fn remove_vectors(&mut self, ids: &[&str]) -> anyhow::Result<()> {
        self.collection
            .delete(Some(ids.to_vec()), None, None)
            .await?;
        Ok(())
    }

    async fn clear(&mut self) -> anyhow::Result<()> {
        let all_items = self.collection.get(GetOptions::default()).await?;
        if !all_items.ids.is_empty() {
            self.collection
                .delete(
                    Some(all_items.ids.iter().map(|s| s.as_str()).collect()),
                    None,
                    None,
                )
                .await?;
        }
        Ok(())
    }

    async fn count(&self) -> anyhow::Result<usize> {
        Ok(self.collection.count().await?)
    }
}

#[cfg(test)]
mod tests {
    use ailoy_macros::multi_platform_test;
    use serde_json::json;

    use super::*;

    async fn setup_test_store() -> anyhow::Result<ChromaStore> {
        let client = ChromaClient::new(ChromaClientOptions::default()).await?;
        let collection_name = format!("test-collection-{}", Uuid::new_v4());
        let collection = client
            .get_or_create_collection(&collection_name, None)
            .await?;
        Ok(ChromaStore { client, collection })
    }

    #[multi_platform_test]
    async fn test_add_and_get_vector() -> anyhow::Result<()> {
        let mut store = setup_test_store().await?;
        let test_embedding: Embedding = vec![1.1, 2.2, 3.3].into();
        let test_document = "This is a test document.".to_owned();
        let test_metadata = json!({"source": "test_add_and_get_vector"})
            .as_object()
            .unwrap()
            .clone();

        let input = VectorStoreAddInput {
            embedding: test_embedding.clone(),
            document: test_document.clone(),
            metadata: Some(from_chroma_metadata(&test_metadata)),
        };

        let added_id = store.add_vector(input).await?;
        let result = store.get_by_id(&added_id).await?;

        assert!(result.is_some(), "Vector should be found by its ID");
        let retrieved = result.unwrap();

        assert_eq!(retrieved.id, added_id);
        assert_eq!(retrieved.document, test_document);
        assert_eq!(retrieved.embedding, test_embedding);
        assert_eq!(
            retrieved.metadata,
            Some(from_chroma_metadata(&test_metadata))
        );

        Ok(())
    }

    #[multi_platform_test]
    async fn test_add_vectors_batch() -> anyhow::Result<()> {
        let mut store = setup_test_store().await?;

        let added_ids = store
            .add_vectors(vec![
                VectorStoreAddInput {
                    embedding: vec![1.0, 1.0, 1.0].into(),
                    document: "doc1".to_owned(),
                    metadata: Some(from_chroma_metadata(json!({"id": 1}).as_object().unwrap())),
                },
                VectorStoreAddInput {
                    embedding: vec![2.0, 2.0, 2.0].into(),
                    document: "doc2".to_owned(),
                    metadata: Some(from_chroma_metadata(json!({"id": 2}).as_object().unwrap())),
                },
            ])
            .await?;

        assert_eq!(added_ids.len(), 2);

        let res1 = store.get_by_id(&added_ids[0]).await?.unwrap();
        assert_eq!(res1.document, "doc1");
        assert_eq!(res1.embedding, vec![1.0, 1.0, 1.0].into());

        let res2 = store.get_by_id(&added_ids[1]).await?.unwrap();
        assert_eq!(res2.document, "doc2");
        assert_eq!(res2.embedding, vec![2.0, 2.0, 2.0].into());

        let res = store
            .get_by_ids(&added_ids.iter().map(|id| id.as_str()).collect::<Vec<_>>())
            .await?;
        assert_eq!(res[0].document, "doc1");
        assert_eq!(res[0].embedding, vec![1.0, 1.0, 1.0].into());

        assert_eq!(res[1].document, "doc2");
        assert_eq!(res[1].embedding, vec![2.0, 2.0, 2.0].into());

        Ok(())
    }

    #[multi_platform_test]
    async fn test_retrieve_similar_vectors() -> anyhow::Result<()> {
        let mut store = setup_test_store().await?;
        let inputs = vec![
            VectorStoreAddInput {
                embedding: vec![1.0, 0.0, 0.0].into(),
                document: "vector one".to_owned(),
                metadata: None,
            },
            VectorStoreAddInput {
                embedding: vec![0.0, 1.0, 0.0].into(),
                document: "vector two".to_owned(),
                metadata: None,
            },
            VectorStoreAddInput {
                embedding: vec![0.9, 0.1, 0.0].into(),
                document: "vector one-ish".to_owned(),
                metadata: None,
            },
            VectorStoreAddInput {
                embedding: vec![0.0, 0.9, 0.1].into(),
                document: "vector two-ish".to_owned(),
                metadata: None,
            },
        ];
        store.add_vectors(inputs).await?;

        let query_embeddings = vec![vec![0.95, 0.0, 0.0].into(), vec![0.0, 0.95, 0.0].into()];

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

    #[multi_platform_test]
    async fn test_remove_vector() -> anyhow::Result<()> {
        let mut store = setup_test_store().await?;
        let inputs = vec![
            VectorStoreAddInput {
                embedding: vec![5.5, 6.6, 3.3].into(),
                document: "to be deleted1".to_owned(),
                metadata: None,
            },
            VectorStoreAddInput {
                embedding: vec![4.4, 6.6, 5.5].into(),
                document: "to be deleted2".to_owned(),
                metadata: None,
            },
            VectorStoreAddInput {
                embedding: vec![3.3, 2.2, 6.6].into(),
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

    #[multi_platform_test]
    async fn test_get_non_existent_vector() -> anyhow::Result<()> {
        let store = setup_test_store().await?;
        let non_existent_id = Uuid::new_v4().to_string();

        let result = store.get_by_id(&non_existent_id).await?;

        assert!(
            result.is_none(),
            "Getting a non-existent ID should return None"
        );
        Ok(())
    }

    #[multi_platform_test]
    async fn test_clear_collection() -> anyhow::Result<()> {
        let mut store = setup_test_store().await?;
        let inputs = vec![
            VectorStoreAddInput {
                embedding: vec![1.0, 1.1, 1.2].into(),
                document: "doc1".to_owned(),
                metadata: Some(from_chroma_metadata(
                    &json!({"id": 1}).as_object().unwrap().clone(),
                )),
            },
            VectorStoreAddInput {
                embedding: vec![2.0, 2.1, 2.2].into(),
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
