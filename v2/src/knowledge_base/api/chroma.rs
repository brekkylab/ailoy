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
    async fn add_vector(&self, input: AddInput) -> Result<String> {
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

    async fn add_vectors(&self, inputs: Vec<AddInput>) -> Result<Vec<String>> {
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
        let opts = GetOptions {
            ids: vec![id.to_owned()],
            include: Some(vec![
                "metadatas".to_owned(),
                "documents".to_owned(),
                "embeddings".to_owned(),
            ]),
            ..Default::default()
        };
        let results = self.collection.get(opts).await?;
        if results.ids.is_empty() {
            return Ok(None);
        }
        let idx = 0;
        let metadata = results
            .metadatas
            .as_ref()
            .and_then(|vec| vec.get(idx))
            .and_then(|inner_option| inner_option.as_ref())
            .map(|map| Json::Object(map.clone()));
        let embedding = results
            .embeddings
            .as_ref()
            .and_then(|vec| vec.get(idx))
            .and_then(|opt_vec| opt_vec.as_ref())
            .map(|vec_f32| vec_f32.clone());

        Ok(Some(GetResult {
            id: results.ids[idx].clone(),
            document: results
                .documents
                .as_ref()
                .map(|d| d[idx].clone())
                .unwrap_or_default()
                .expect(""),
            metadata: metadata,
            embedding: embedding.unwrap(),
        }))
    }

    async fn retrieve(&self, query: Embedding, top_k: u64) -> Result<Vec<RetrieveResult>> {
        let opts = QueryOptions {
            query_embeddings: Some(vec![query.to_vec()]),
            n_results: Some(top_k as usize),
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
                    distance, // 예시적인 유사도 변환
                });
            }
        }

        Ok(out)
    }

    async fn remove_vector(&self, id: &str) -> Result<()> {
        self.collection.delete(Some(vec![id]), None, None).await?;
        Ok(())
    }

    async fn clear(&self) -> Result<()> {
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
        let store = setup_test_store().await?;
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
        let store = setup_test_store().await?;

        let added_ids = store
            .add_vectors(vec![
                AddInput {
                    embedding: vec![1.0, 1.0],
                    document: "doc1".to_owned(),
                    metadata: Some(json!({"id": 1})),
                },
                AddInput {
                    embedding: vec![2.0, 2.0],
                    document: "doc2".to_owned(),
                    metadata: Some(json!({"id": 2})),
                },
            ])
            .await?;

        assert_eq!(added_ids.len(), 2);

        let res1 = store.get_by_id(&added_ids[0]).await?.unwrap();
        assert_eq!(res1.document, "doc1");
        assert_eq!(res1.embedding, vec![1.0, 1.0]);

        let res2 = store.get_by_id(&added_ids[1]).await?.unwrap();
        assert_eq!(res2.document, "doc2");
        assert_eq!(res2.embedding, vec![2.0, 2.0]);

        Ok(())
    }

    #[tokio::test]
    async fn test_retrieve_similar_vectors() -> Result<()> {
        let store = setup_test_store().await?;
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
        let store = setup_test_store().await?;
        let input = AddInput {
            embedding: vec![5.5, 6.6],
            document: "to be deleted".to_owned(),
            metadata: None,
        };
        let id_to_delete = store.add_vector(input).await?;

        store.remove_vector(&id_to_delete).await?;

        let result = store.get_by_id(&id_to_delete).await?;
        assert!(
            result.is_none(),
            "Vector should not be found after deletion"
        );

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
        let store = setup_test_store().await?;
        let inputs = vec![
            AddInput {
                embedding: vec![1.0],
                document: "doc1".to_owned(),
                metadata: None,
            },
            AddInput {
                embedding: vec![2.0],
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

        let all_items = store.collection.get(GetOptions::default()).await?;
        assert!(
            all_items.ids.is_empty(),
            "Collection should be empty after clear"
        );

        Ok(())
    }
}
