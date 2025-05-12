#pragma once

#include <vector>

#include <nlohmann/json.hpp>

#include "module.hpp"
#include "value.hpp"

namespace ailoy {

using embedding_t = std::shared_ptr<const ailoy::ndarray_t>;
using metadata_t = std::optional<nlohmann::json>;

struct vector_store_add_input_t {
  embedding_t embedding;
  std::string document;
  metadata_t metadata = std::nullopt;
};

struct vector_store_get_result_t {
  std::string id;
  std::string document;
  metadata_t metadata;
  embedding_t embedding;
};

struct vector_store_retrieve_result_t {
  std::string id;
  std::string document;
  metadata_t metadata;
  // similarity with query vector, ranges from 0 to 1.
  float similarity;
};

class vector_store_t : public object_t {
public:
  /**
   * @brief Add a vector to vector store.
   * @param embedding Embedding vector to store
   * @param document Document related to embedding
   * @param metadata Additional metadata
   * @return Unique identifier of the added vector
   */
  virtual std::string add_vector(const vector_store_add_input_t &input) = 0;

  /**
   * @brief Add vectors to vector store at once.
   * @param embedding Embedding vectors to store
   * @param documents Documents related to embeddings
   * @param metadata Additional metadatas
   * @return Unique identifiers of the added vectors
   */
  virtual std::vector<std::string>
  add_vectors(std::vector<vector_store_add_input_t> &inputs) = 0;

  /**
   * @brief Get a vector by id if exists
   * @param id Unique identifier of the vector
   * @return id, text, metadata and vector of found item, nullopt otherwise
   */
  virtual std::optional<vector_store_get_result_t>
  get_by_id(const std::string &id) = 0;

  /**
   * @brief Retrieve similar vectors with query embedding up to n vectors.
   * @param query_embedding Query embedding
   * @param n Number of results to retrieve.
   * @return Retrieved results of text, metadata and similarity score.
   */
  virtual std::vector<vector_store_retrieve_result_t>
  retrieve(embedding_t query_embedding, uint64_t k) = 0;

  /**
   * @brief Remove a vector from vector store.
   * @param id Unique identifier of the vector
   */
  virtual void remove_vector(const std::string &id) = 0;

  /**
   * @brief Clear all stored vectors from vector store.
   */
  virtual void clear() = 0;
};

template <typename derived_vector_store_t>
  requires std::is_base_of_v<vector_store_t, derived_vector_store_t>
component_or_error_t
create_vector_store_component(std::shared_ptr<const value_t> attrs) {
  auto insert = create<instant_method_operator_t>(
      [](std::shared_ptr<component_t> component,
         std::shared_ptr<const value_t> inputs) -> value_or_error_t {
        if (!inputs->is_type_of<map_t>())
          return error_output_t(type_error("Vector Store: insert", "inputs",
                                           "map_t", inputs->get_type()));
        auto input_map = inputs->as<map_t>();

        if (!input_map->contains("embedding") ||
            !input_map->at("embedding")->is_type_of<ndarray_t>())
          return error_output_t(
              type_error("Vector Store: insert", "embedding", "ndarray_t",
                         input_map->at("embedding")->get_type()));
        auto embedding = input_map->at<ndarray_t>("embedding");

        if (!input_map->contains("document") ||
            !input_map->at("document")->is_type_of<string_t>())
          return error_output_t(
              type_error("Vector Store: insert", "document", "string_t",
                         input_map->at("embedding")->get_type()));
        std::string document = *input_map->at<string_t>("document");

        nlohmann::json metadata;
        if (input_map->contains("metadata")) {
          auto metadata_val = input_map->at("metadata");
          if (metadata_val->is_type_of<map_t>() ||
              metadata_val->is_type_of<null_t>())
            metadata = *metadata_val;
          else
            return error_output_t(type_error("Vector Store: insert", "metadata",
                                             "map_t | null_t",
                                             metadata_val->get_type()));
        } else
          metadata = nlohmann::json({});

        auto add_input = vector_store_add_input_t{
            .embedding = std::move(embedding),
            .document = document,
            .metadata = metadata,
        };

        auto outputs = create<map_t>();
        auto id = component->get_obj("vector_store")
                      ->as<vector_store_t>()
                      ->add_vector(add_input);
        outputs->insert_or_assign("id", create<string_t>(id));
        return outputs;
      });

  auto insert_many = create<instant_method_operator_t>(
      [](std::shared_ptr<component_t> component,
         std::shared_ptr<const value_t> inputs) -> value_or_error_t {
        if (!inputs->is_type_of<array_t>()) {
          return error_output_t(type_error("Vector Store: insert_many",
                                           "inputs", "array_t",
                                           inputs->get_type()));
        }
        auto input_arr = inputs->as<array_t>();

        std::vector<vector_store_add_input_t> add_inputs;
        for (auto input_item : *input_arr) {
          if (!input_item->is_type_of<map_t>()) {
            return error_output_t(type_error("Vector Store: insert_many",
                                             "inputs.*", "map_t",
                                             input_item->get_type()));
          }
          auto input_item_map = input_item->as<map_t>();

          if (!input_item_map->contains("embedding") ||
              !input_item_map->at("embedding")->is_type_of<ndarray_t>())
            return error_output_t(type_error(
                "Vector Store: insert_many", "embedding", "ndarray_t",
                input_item_map->at("embedding")->get_type()));
          auto embedding = input_item_map->at<ndarray_t>("embedding");

          if (!input_item_map->contains("document") ||
              !input_item_map->at("document")->is_type_of<string_t>())
            return error_output_t(
                type_error("Vector Store: insert_many", "document", "string_t",
                           input_item_map->at("embedding")->get_type()));
          std::string document = *input_item_map->at<string_t>("document");

          nlohmann::json metadata;
          if (input_item_map->contains("metadata")) {
            auto metadata_val = input_item_map->at("metadata");
            if (metadata_val->is_type_of<map_t>() ||
                metadata_val->is_type_of<null_t>())
              metadata = *metadata_val;
            else
              return error_output_t(type_error("Vector Store: insert_many",
                                               "metadata", "map_t | null_t",
                                               metadata_val->get_type()));
          } else
            metadata = nlohmann::json({});

          add_inputs.push_back({
              .embedding = std::move(embedding),
              .document = document,
              .metadata = metadata,
          });
        }

        auto ids = component->get_obj("vector_store")
                       ->as<vector_store_t>()
                       ->add_vectors(add_inputs);

        auto outputs = create<map_t>();
        outputs->insert_or_assign("ids", create<array_t>());
        for (const auto &id : ids) {
          outputs->at<array_t>("ids")->push_back(create<string_t>(id));
        }
        return outputs;
      });

  auto get_by_id = create<instant_method_operator_t>(
      [](std::shared_ptr<component_t> component,
         std::shared_ptr<const value_t> inputs) -> value_or_error_t {
        if (!inputs->is_type_of<map_t>()) {
          return error_output_t(type_error("Vector Store: get_by_id", "inputs",
                                           "map_t", inputs->get_type()));
        }
        auto inputs_map = inputs->as<map_t>();

        if (!inputs_map->contains("id"))
          return error_output_t(range_error("Vector Store: get_by_id", "id"));
        if (!inputs_map->at("id")->is_type_of<string_t>())
          return error_output_t(type_error("Vector Store: get_by_id", "id",
                                           "string_t",
                                           inputs_map->at("id")->get_type()));
        std::string id = *inputs_map->at<string_t>("id");

        auto get_result = component->get_obj("vector_store")
                              ->as<vector_store_t>()
                              ->get_by_id(id);

        auto outputs = create<map_t>();
        if (get_result.has_value()) {
          outputs->insert_or_assign("id",
                                    create<string_t>(get_result.value().id));
          outputs->insert_or_assign(
              "embedding",
              std::const_pointer_cast<ndarray_t>(get_result.value().embedding));
          outputs->insert_or_assign(
              "document", create<string_t>(get_result.value().document));
          if (get_result.value().metadata.has_value())
            outputs->insert_or_assign(
                "metadata",
                from_nlohmann_json(get_result.value().metadata.value()));
          else
            outputs->insert_or_assign("metadata", create<null_t>());
          return outputs;
        } else {
          return create<null_t>();
        }
      });

  auto retrieve = create<instant_method_operator_t>(
      [](std::shared_ptr<component_t> component,
         std::shared_ptr<const value_t> inputs) -> value_or_error_t {
        if (!inputs->is_type_of<map_t>()) {
          return error_output_t(type_error("Vector Store: retrieve", "inputs",
                                           "map_t", inputs->get_type()));
        }
        auto inputs_map = inputs->as<map_t>();

        if (!inputs_map->contains("query_embedding"))
          return error_output_t(
              range_error("Vector Store: retrieve", "query_embedding"));
        if (!inputs_map->at("query_embedding")->is_type_of<ndarray_t>())
          return error_output_t(type_error(
              "Vector Store: retrieve", "query_embedding", "ndarray_t",
              inputs_map->at("query_embedding")->get_type()));
        auto query_embedding = inputs_map->at<ndarray_t>("query_embedding");

        uint64_t k;
        if (!inputs_map->contains("k"))
          return error_output_t(
              range_error("Vector Store: retrieve", "query_embedding"));
        if (inputs_map->at("k")->is_type_of<uint_t>())
          k = *inputs_map->at<uint_t>("k");
        else if (inputs_map->at("k")->is_type_of<int_t>())
          k = *inputs_map->at<int_t>("k");
        else
          return error_output_t(type_error("Vector Store: retrieve", "k",
                                           "uint_t | int_t",
                                           inputs_map->at("k")->get_type()));

        auto retrieve_results = component->get_obj("vector_store")
                                    ->as<vector_store_t>()
                                    ->retrieve(query_embedding, k);

        auto results = create<array_t>();
        for (const auto &result : retrieve_results) {
          auto retrieve_result = create<map_t>();
          retrieve_result->insert_or_assign("id", create<string_t>(result.id));
          retrieve_result->insert_or_assign("document",
                                            create<string_t>(result.document));
          if (result.metadata.has_value())
            retrieve_result->insert_or_assign(
                "metadata", from_nlohmann_json(result.metadata.value()));
          else
            retrieve_result->insert_or_assign("metadata", create<null_t>());
          retrieve_result->insert_or_assign("similarity",
                                            create<float_t>(result.similarity));
          results->push_back(retrieve_result);
        }
        auto outputs = create<map_t>();
        outputs->insert_or_assign("results", results);
        return outputs;
      });

  auto remove = create<instant_method_operator_t>(
      [](std::shared_ptr<component_t> component,
         std::shared_ptr<const value_t> inputs) -> value_or_error_t {
        if (!inputs->is_type_of<map_t>()) {
          return error_output_t(type_error("Vector Store: remove", "inputs",
                                           "map_t", inputs->get_type()));
        }
        auto inputs_map = inputs->as<map_t>();

        if (!inputs_map->contains("id"))
          return error_output_t(range_error("Vector Store: remove", "id"));
        if (!inputs_map->at("id")->is_type_of<string_t>())
          return error_output_t(type_error("Vector Store: remove", "id",
                                           "string_t",
                                           inputs_map->at("id")->get_type()));
        std::string id = *inputs_map->at<string_t>("id");

        component->get_obj("vector_store")
            ->as<vector_store_t>()
            ->remove_vector(id);
        return create<bool_t>(true);
      });

  auto clear = create<instant_method_operator_t>(
      [](std::shared_ptr<component_t> component,
         std::shared_ptr<const value_t> inputs) -> value_or_error_t {
        component->get_obj("vector_store")->as<vector_store_t>()->clear();
        return create<bool_t>(true);
      });

  auto vector_store = create<derived_vector_store_t>(attrs);
  auto component = create<component_t>(
      std::initializer_list<
          std::pair<const std::string, std::shared_ptr<method_operator_t>>>{
          {"insert", insert},
          {"insert_many", insert_many},
          {"get_by_id", get_by_id},
          {"retrieve", retrieve},
          {"remove", remove},
          {"clear", clear},
      });
  component->set_obj("vector_store", vector_store);
  return component;
}

} // namespace ailoy