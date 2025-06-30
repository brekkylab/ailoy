#include "language.hpp"

#include "chromadb_vector_store.hpp"
#include "faiss/faiss_vector_store.hpp"
#include "openai.hpp"
#include "split_text.hpp"
#include "tvm/embedding_model.hpp"
#include "tvm/language_model.hpp"
#include "tvm/model_cache.hpp"

namespace ailoy {

static std::shared_ptr<module_t> language_module = create<module_t>();

std::shared_ptr<const module_t> get_language_module() {
  // Add Component: TVM Embedding model
  if (!language_module->factories.contains("tvm_embedding_model")) {
    language_module->factories.insert_or_assign(
        "tvm_embedding_model", create_tvm_embedding_model_component);
  }

  // Add Component: TVM Language model
  if (!language_module->factories.contains("tvm_language_model")) {
    language_module->factories.insert_or_assign(
        "tvm_language_model", create_tvm_language_model_component);
  }

  // Add Operators: Model Cache
  if (!language_module->ops.contains("list_local_models")) {
    language_module->ops.insert_or_assign(
        "list_local_models",
        create<instant_operator_t>(ailoy::operators::list_local_models));
  }
  if (!language_module->ops.contains("download_model")) {
    language_module->ops.insert_or_assign(
        "download_model",
        create<instant_operator_t>(ailoy::operators::download_model));
  }
  if (!language_module->ops.contains("remove_model")) {
    language_module->ops.insert_or_assign(
        "remove_model",
        create<instant_operator_t>(ailoy::operators::remove_model));
  }

  // Add Operators: Split Text
  if (!language_module->ops.contains("split_text_by_separator")) {
    language_module->ops.insert_or_assign(
        "split_text_by_separator",
        create<instant_operator_t>(split_text_by_separator_op));
  }
  if (!language_module->ops.contains("split_text") ||
      !language_module->ops.contains("split_text_separators_recursively")) {
    language_module->ops.insert_or_assign(
        "split_text_separators_recursively",
        create<instant_operator_t>(split_text_by_separators_recursively_op));
    language_module->ops.insert_or_assign(
        "split_text",
        create<instant_operator_t>(split_text_by_separators_recursively_op));
  }

  // Add Components: Vectorstores
  if (!language_module->factories.contains("faiss_vector_store")) {
    language_module->factories.insert_or_assign(
        "faiss_vector_store",
        create_vector_store_component<faiss_vector_store_t>);
  }
  if (!language_module->factories.contains("chromadb_vector_store")) {
    language_module->factories.insert_or_assign(
        "chromadb_vector_store",
        create_vector_store_component<chromadb_vector_store_t>);
  }

  // Add Component: OpenAI
  if (!language_module->factories.contains("openai")) {
    language_module->factories.insert_or_assign(
        "openai", create_openai_component<openai_llm_engine_t>);
  }

  // Add Component: Gemini
  if (!language_module->factories.contains("gemini")) {
    language_module->factories.insert_or_assign(
        "gemini", create_openai_component<gemini_llm_engine_t>);
  }

  // Add Component: Claude
  if (!language_module->factories.contains("claude")) {
    language_module->factories.insert_or_assign(
        "claude", create_openai_component<claude_llm_engine_t>);
  }

  return language_module;
}

} // namespace ailoy
