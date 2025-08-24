use super::super::{AddInput, Embedding, GetResult, RetrieveResult, VectorStore};

use anyhow::{Result, bail};
use async_trait::async_trait;
use chromadb::client::{ChromaClient, ChromaClientOptions};
use chromadb::collection::{
    ChromaCollection, CollectionEntries, GetOptions, QueryOptions, QueryResult,
};
use serde_json::{Map, Value as Json};
use uuid::Uuid;

pub struct ChromaStore {
    collection: ChromaCollection,
}

impl ChromaStore {
    pub async fn new(
        chroma_url: &str,
        collection_name: &str,
        // 필요시 인증옵션 추가 (ex. token)
    ) -> Result<Self> {
        let client = ChromaClient::new(ChromaClientOptions {
            url: Some(chroma_url.to_owned()),
            ..Default::default()
        })
        .await?;
        // existing 또는 새 컬렉션 생성
        let collection = client
            .get_or_create_collection(collection_name, None)
            .await?;
        Ok(Self { collection })
    }
}

#[async_trait]
impl VectorStore for ChromaStore {
    async fn add_vector(&mut self, input: AddInput) -> Result<String> {
        let id = Uuid::new_v4().to_string();

        let metadatas = match &input.metadata {
            Some(Json::Object(map)) => Some(vec![map.clone()]),
            Some(_) => bail!("Metadata must be a JSON object."),
            None => None,
        };

        let entry = CollectionEntries {
            ids: vec![id.as_ref()],
            embeddings: Some(vec![input.embedding]),
            documents: Some(vec![input.document.as_ref()]),
            metadatas: metadatas,
        };
        self.collection.upsert(entry, None).await?;
        Ok(id)
    }

    async fn add_vectors(&mut self, inputs: Vec<AddInput>) -> Result<Vec<String>> {
        let ids: Vec<String> = (0..inputs.len())
            .map(|_| Uuid::new_v4().to_string())
            .collect();
        let embeddings: Vec<Vec<f32>> = inputs.iter().map(|i| i.embedding.clone()).collect();
        let documents: Vec<String> = inputs.iter().map(|i| i.document.clone()).collect();
        let metadatas: Option<Vec<Map<String, Json>>> =
            if inputs.iter().any(|i| i.metadata.is_some()) {
                Some(
                    inputs
                        .iter()
                        .map(|i| match &i.metadata {
                            Some(Json::Object(map)) => map.clone(),
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

    async fn get_by_id(&self, id: &str) -> Result<Option<GetResult>> {
        let results: Vec<GetResult> = self.get_by_ids(&[id]).await?;
        Ok(results.into_iter().next())
    }

    async fn get_by_ids(&self, ids: &[&str]) -> Result<Vec<GetResult>> {
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

            let results: Vec<GetResult> = zipped_iter
                .map(|(id, document, metadata, embedding)| GetResult {
                    id,
                    document,
                    metadata: match metadata {
                        Some(metadata) => Some(Json::Object(metadata)),
                        None => None,
                    },
                    embedding,
                })
                .collect();
            return Ok(results);
        } else {
            bail!("Results from get operation are malformed.")
        }
    }

    async fn retrieve(&self, query: Embedding, top_k: usize) -> Result<Vec<RetrieveResult>> {
        let opts = QueryOptions {
            query_embeddings: Some(vec![query.to_vec()]),
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
        let mut out = vec![];
        if let (Some(ids_vec), Some(distances_vec)) =
            (ids.get(0), distances.as_ref().and_then(|d| d.get(0)))
        {
            for i in 0..ids_vec.len() {
                let id = ids_vec[i].clone();
                let distance = distances_vec[i];
                let document: String = documents
                    .as_ref()
                    .and_then(|d| d.get(0)?.get(i).cloned())
                    .unwrap_or_default();
                let metadata: Option<Json> = metadatas
                    .as_ref()
                    .and_then(|m| m.get(0)?.get(i))
                    .and_then(|inner_opt| inner_opt.as_ref())
                    .map(|map_ref| Json::Object(map_ref.clone()));
                out.push(RetrieveResult {
                    id,
                    document,
                    metadata,
                    distance,
                });
            }
        }

        Ok(out)
    }

    async fn remove_vector(&mut self, id: &str) -> Result<()> {
        self.collection.delete(Some(vec![id]), None, None).await?;
        Ok(())
    }

    async fn remove_vectors(&mut self, ids: &[&str]) -> Result<()> {
        self.collection
            .delete(Some(ids.to_vec()), None, None)
            .await?;
        Ok(())
    }

    async fn clear(&mut self) -> Result<()> {
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

    async fn count(&self) -> Result<usize> {
        Ok(self.collection.count().await?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokio;

    async fn setup_test_store() -> Result<ChromaStore> {
        let client = ChromaClient::new(ChromaClientOptions::default()).await?;
        let collection_name = format!("test-collection-{}", Uuid::new_v4());
        let collection = client
            .get_or_create_collection(&collection_name, None)
            .await?;
        Ok(ChromaStore { collection })
    }

    #[tokio::test]
    async fn test_add_and_get_vector() -> Result<()> {
        let mut store = setup_test_store().await?;
        let test_embedding = vec![1.1, 2.2, 3.3];
        let test_document = "This is a test document.".to_owned();
        let test_metadata = json!({"source": "test_add_and_get_vector"});

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

    #[tokio::test]
    async fn test_add_vectors_batch() -> Result<()> {
        let mut store = setup_test_store().await?;

        let added_ids = store
            .add_vectors(vec![
                AddInput {
                    embedding: vec![1.0, 1.0, 1.0],
                    document: "doc1".to_owned(),
                    metadata: Some(json!({"id": 1})),
                },
                AddInput {
                    embedding: vec![2.0, 2.0, 2.0],
                    document: "doc2".to_owned(),
                    metadata: Some(json!({"id": 2})),
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

    #[tokio::test]
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
        ];
        store.add_vectors(inputs).await?;

        let query_embedding = vec![0.95, 0.0, 0.0];

        // top_k=2
        let results = store.retrieve(query_embedding, 2).await?;

        assert_eq!(results.len(), 2);

        let retrieved_docs: Vec<_> = results.iter().map(|r| r.document.clone()).collect();
        // only these two should be exist
        assert!(retrieved_docs.contains(&"vector one".to_owned()));
        assert!(retrieved_docs.contains(&"vector one-ish".to_owned()));
        // this one should not be exists
        assert!(!retrieved_docs.contains(&"vector two".to_owned()));

        Ok(())
    }

    #[tokio::test]
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

    #[tokio::test]
    async fn test_get_non_existent_vector() -> Result<()> {
        let store = setup_test_store().await?;
        let non_existent_id = Uuid::new_v4().to_string();

        let result = store.get_by_id(&non_existent_id).await?;

        assert!(
            result.is_none(),
            "Getting a non-existent ID should return None"
        );
        Ok(())
    }

    #[tokio::test]
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
