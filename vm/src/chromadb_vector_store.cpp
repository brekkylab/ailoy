#include "chromadb_vector_store.hpp"

#include <format>

#include "exception.hpp"
#include "uuid.hpp"

namespace ailoy {

// https://github.com/chroma-core/chroma/blob/main/chromadb/config.py#L91
const std::string DEFAULT_TENANT = "default_tenant";
// https://github.com/chroma-core/chroma/blob/main/chromadb/config.py#L92
const std::string DEFAULT_DATABASE = "default_database";

const std::string COLLECTIONS_BASE_URL =
    std::format("/api/v2/tenants/{}/databases/{}/collections", DEFAULT_TENANT,
                DEFAULT_DATABASE);

chromadb_vector_store_t::chromadb_vector_store_t(
    const std::string &url, const std::string &collection,
    bool delete_collection_on_cleanup)
    : cli_(std::make_shared<httplib::Client>(url)),
      collection_name_(collection),
      delete_collection_on_cleanup_(delete_collection_on_cleanup) {
  _create_collection();
}

void chromadb_vector_store_t::_create_collection() {
  nlohmann::json params;
  params["name"] = collection_name_;
  // use cosine similarity as default
  params["configuration"] =
      nlohmann::json{{"hnsw", nlohmann::json{{"space", "cosine"}}}};

  std::string body = params.dump();
  httplib::Result create_result = cli_->Post(COLLECTIONS_BASE_URL, body.data(),
                                             body.size(), "application/json");

  // Request has been failed
  if (!create_result)
    throw ailoy::runtime_error("[Chromadb] Failed to request: " +
                               httplib::to_string(create_result.error()));

  // Created the collection successfully
  if (create_result->status == httplib::OK_200) {
    auto j = nlohmann::json::parse(create_result->body);
    collection_id_ = j["id"];
    return;
  }

  // The collection is already created. Get existing collection's id.
  if (create_result->status == httplib::Conflict_409) {
    httplib::Result get_result =
        cli_->Get(std::format("{}/{}", COLLECTIONS_BASE_URL, collection_name_));
    if (get_result->status != httplib::OK_200) {
      throw ailoy::runtime_error(
          "[Chromadb] Failed to get existing collection: " +
          std::string(httplib::status_message(get_result->status)));
    }
    auto j = nlohmann::json::parse(get_result->body);
    collection_id_ = j["id"];
    return;
  }

  throw ailoy::runtime_error(
      "[Chromadb] Failed to create collection: " +
      std::string(httplib::status_message(create_result->status)));
}

void chromadb_vector_store_t::_delete_collection() {
  httplib::Result result = cli_->Delete(
      std::format("{}/{}", COLLECTIONS_BASE_URL, collection_name_));
  if (result->status != httplib::OK_200) {
    throw ailoy::runtime_error(
        "[Chromadb] Failed to delete collection: " +
        std::string(httplib::status_message(result->status)));
  }
}

chromadb_vector_store_t::chromadb_vector_store_t(
    std::shared_ptr<const value_t> attrs) {
  if (!attrs->is_type_of<map_t>()) {
    throw ailoy::runtime_error("[Chromadb] component attrs should be map type");
  }
  auto attrs_map = attrs->as<map_t>();
  std::string url = CHROMADB_DEFAULT_URL;
  std::string collection = CHROMADB_DEFAULT_COLLECTION;
  if (attrs_map->contains("url")) {
    auto attr_url = attrs_map->at("url");
    if (!attr_url->is_type_of<string_t>()) {
      throw ailoy::runtime_error("[Chromadb] url should be a type of string");
    }
    url = *attr_url->as<string_t>();
  }
  if (attrs_map->contains("collection")) {
    auto attr_collection = attrs_map->at("collection");
    if (!attr_collection->is_type_of<string_t>()) {
      throw ailoy::runtime_error(
          "[Chromadb] collection should be a type of string");
    }
    collection = *attr_collection->as<string_t>();
  }
  cli_ = std::make_shared<httplib::Client>(url);
  collection_name_ = collection;
  delete_collection_on_cleanup_ = false;
  _create_collection();
}

chromadb_vector_store_t::~chromadb_vector_store_t() {
  if (delete_collection_on_cleanup_)
    _delete_collection();
}

std::string
chromadb_vector_store_t::add_vector(const vector_store_add_input_t &input) {
  std::string id = generate_uuid();

  nlohmann::json params;
  params["ids"] = std::vector<std::string>{id};
  params["embeddings"] = std::vector<std::vector<float>>{
      input.embedding->operator std::vector<float>()};
  params["documents"] = std::vector<std::string>{input.document};
  params["metadatas"] = nlohmann::json::array();
  if (input.metadata.has_value())
    params["metadatas"].push_back(input.metadata.value());
  else {
    // metadata is null
    params["metadatas"].push_back(nlohmann::json({}));
  }

  std::string body = params.dump();
  httplib::Result result =
      cli_->Post(std::format("{}/{}/add", COLLECTIONS_BASE_URL, collection_id_),
                 body.data(), body.size(), "application/json");
  if (result->status != httplib::Created_201) {
    throw ailoy::runtime_error(
        "[Chromadb] Failed to add vector to collection: " +
        std::string(httplib::status_message(result->status)));
  }

  return id;
}

std::vector<std::string> chromadb_vector_store_t::add_vectors(
    std::vector<vector_store_add_input_t> &inputs) {
  nlohmann::json params;
  params["ids"] = nlohmann::json::array();
  params["embeddings"] = nlohmann::json::array();
  params["documents"] = nlohmann::json::array();
  params["metadatas"] = nlohmann::json::array();

  for (const auto &input : inputs) {
    params["ids"].push_back(generate_uuid());
    params["embeddings"].push_back(
        input.embedding->operator std::vector<float>());
    params["documents"].push_back(input.document);
    if (input.metadata.has_value())
      params["metadatas"].push_back(input.metadata.value());
    else
      params["metadatas"].push_back(nlohmann::json({}));
  }

  std::string body = params.dump();
  httplib::Result result =
      cli_->Post(std::format("{}/{}/add", COLLECTIONS_BASE_URL, collection_id_),
                 body.data(), body.size(), "application/json");
  if (result->status != httplib::Created_201) {
    throw ailoy::runtime_error(
        "[Chromadb] Failed to add vectors to collection: " +
        std::string(httplib::status_message(result->status)));
  }

  return params["ids"].get<std::vector<std::string>>();
}

std::optional<vector_store_get_result_t>
chromadb_vector_store_t::get_by_id(const std::string &id) {
  nlohmann::json params;
  params["ids"] = std::vector<std::string>{id};
  params["include"] =
      std::vector<std::string>{"embeddings", "documents", "metadatas"};

  std::string body = params.dump();
  httplib::Result result =
      cli_->Post(std::format("{}/{}/get", COLLECTIONS_BASE_URL, collection_id_),
                 body.data(), body.size(), "application/json");
  if (result->status != httplib::OK_200) {
    return std::nullopt;
  }

  auto j = nlohmann::json::parse(result->body);
  auto document = j["documents"][0].get<std::string>();
  auto metadata = j["metadatas"][0];
  auto embedding = j["embeddings"][0].get<std::vector<float>>();

  auto ndarray = ailoy::create<ailoy::ndarray_t>();
  ndarray->shape.push_back(embedding.size());
  ndarray->dtype = {.code = kDLFloat, .bits = 32, .lanes = 1};
  ndarray->data.resize(sizeof(float) * embedding.size());
  memcpy(ndarray->data.data(), embedding.data(),
         sizeof(float) * embedding.size());

  return vector_store_get_result_t{
      .id = id,
      .document = document,
      .metadata = metadata,
      .embedding = std::move(ndarray),
  };
}

std::vector<vector_store_retrieve_result_t>
chromadb_vector_store_t::retrieve(embedding_t query_embedding, uint64_t k) {
  nlohmann::json params;
  params["query_embeddings"] = std::vector<std::vector<float>>{
      query_embedding->operator std::vector<float>()};
  params["include"] =
      std::vector<std::string>{"documents", "metadatas", "distances"};
  params["n_results"] = k;

  std::string body = params.dump();
  httplib::Result result = cli_->Post(
      std::format("{}/{}/query", COLLECTIONS_BASE_URL, collection_id_),
      body.data(), body.size(), "application/json");
  if (result->status != httplib::OK_200) {
    throw ailoy::runtime_error(
        "[Chromadb] Failed to get query results: " +
        std::string(httplib::status_message(result->status)));
  }

  std::vector<vector_store_retrieve_result_t> results;
  auto res_body = nlohmann::json::parse(result->body);
  auto ids = res_body["ids"][0].get<std::vector<std::string>>();
  auto documents = res_body["documents"][0].get<std::vector<std::string>>();
  auto metadatas = res_body["metadatas"][0];
  auto distances = res_body["distances"][0].get<std::vector<float>>();
  for (int i = 0; i < ids.size(); i++) {
    results.push_back({.id = ids[i],
                       .document = documents[i],
                       .metadata = metadatas[i],
                       .similarity = 1 - distances[i]});
  }
  return results;
}

void chromadb_vector_store_t::remove_vector(const std::string &id) {
  nlohmann::json params;
  params["ids"] = std::vector<std::string>{id};

  std::string body = params.dump();
  httplib::Result result = cli_->Post(
      std::format("{}/{}/delete", COLLECTIONS_BASE_URL, collection_id_),
      body.data(), body.size(), "application/json");
  if (result->status != httplib::OK_200) {
    throw ailoy::runtime_error(
        "[Chromadb] Failed to delete embedding: " +
        std::string(httplib::status_message(result->status)));
  }
}
void chromadb_vector_store_t::clear() {
  // delete and recreate collection
  _delete_collection();
  _create_collection();
}

} // namespace ailoy