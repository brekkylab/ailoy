//! Provides mechanisms for integrating **external knowledge** into an LLM.
//!
//! The `Knowledge` module enables a language model to reference information
//! outside of its pre-trained parameters — such as documents, facts, or context
//! stored in a **knowledge base**(or **vector store**).
//!
//! # What Is a Knowledge Base?
//!
//! A **knowledge base** is a structured collection of information that can be
//! searched or queried to provide additional context to an LLM.
//! It may contain textual documents, database entries, or pre-curated facts
//! relevant to the application domain (for example, company manuals, product
//! FAQs, or research papers).
//!
//! During inference, when a user asks a question, the LLM can **retrieve** relevant
//! information from the knowledge base — similar to how a search engine finds
//! matching pages — and use that context to generate more accurate, grounded
//! answers.
//!
//! # What Is a Vector Store?
//!
//! A **vector store** is a specialized kind of knowledge base that represents
//! text as numerical vectors (embeddings) in a high-dimensional space.
//! Similar pieces of text have similar vector representations, allowing
//! *semantic search*: finding related content even when the query does not share
//! exact keywords.
//!
//! Vector stores are a key component in modern **RAG (Retrieval-Augmented
//! Generation)** systems. This module provides the [`VectorStoreKnowledge`]
//! implementation that connects any [`crate::vector_store::VectorStore`] backend with an
//! [`crate::model::EmbeddingModel`] to perform such retrievals.
//!
//! # Integration Modes
//!
//! Some large language models support *tools* natively, but most do **not**
//! natively support knowledge retrieval. The `Knowledge` module bridges this gap
//! in two main ways:
//!
//! 1. **Native** - If the LLM natively supports knowledge, it will work out of the box without any extra configuration.
//!    However, if the model does not support knowledge natively, Ailoy also ignores documents by default, without any extra setup.
//!    To make such models reference external document data during inference, you need to use a technique called polyfill.
//!    When a model does not natively accept knowledge inputs, the system can use a polyfill (see [`crate::model::DocumentPolyfill`]) to
//!    inject retrieved documents into the model’s prompt as if they were natively supported.
//!
//! 2. **Tool Mode** – When the model determines that it needs external knowledge,
//!    it can call a retrieval function as a *tool* through the
//!    [`KnowledgeTool`], which implements the [`crate::tool::ToolBehavior`] interface.
//!
//! # Components
//!
//! - [`Knowledge`]: The main API for using knowledge.
//!   Internally, it's unified abstraction that wraps either a [`VectorStoreKnowledge`] or a [`CustomKnowledge`] implementation.
//! - [`KnowledgeBehavior`]: Trait defining how a knowledge source retrieves documents.
//! - [`KnowledgeTool`]: Exposes a retriever as an LLM-callable tool.
//! - [`KnowledgeConfig`]: Retrieval configuration (e.g., `top_k` results).
//!
//! # Example
//!
//! ```rust
//! use crate::knowledge::{Knowledge, KnowledgeConfig};
//! use crate::vector_store::SimpleVectorStore;
//! use crate::model::EmbeddingModel;
//!
//! // Create a vector-store based knowledge source
//! let store = FaissStore::new(3).await.unwrap();
//! let embedding_model = EmbeddingModel::new("BAAI/bge-m3").await;
//! let knowledge = Knowledge::new_vector_store(store, embedding_model);
//!
//! // Retrieve relevant documents for a query
//! let results = knowledge
//!     .retrieve("What is Rust async?".into(), KnowledgeConfig { top_k: 3 })
//!     .await
//!     .unwrap();
//!
//! println!("Retrieved {} documents", results.len());
//! ```
//!
//! # See Also
//! - [`crate::model::DocumentPolyfill`]: For making non-native models accept knowledge documents.
//! - [`crate::tool::ToolBehavior`]: For exposing knowledge as a model-invocable tool.
//! - [`crate::vector_store::VectorStore`]: For building custom vector-based retrieval backends.
mod base;
mod custom_knowledge;
mod vector_store_knowledge;

pub use base::*;
pub use custom_knowledge::*;
pub use vector_store_knowledge::*;
