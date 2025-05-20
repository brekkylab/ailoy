#include <format>
#include <random>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "chromadb_vector_store.hpp"
#include "language.hpp"

std::string dump_vector(std::vector<float> &vec) {
  std::stringstream ss;
  ss << "[";
  for (int i = 0; i < vec.size(); i++) {
    ss << vec[i];
    if (i < vec.size() - 1) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}

std::shared_ptr<ailoy::ndarray_t> get_random_normalized_vector(size_t dim) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  std::vector<float> vec(dim);
  for (int i = 0; i < dim; i++) {
    vec[i] = dist(gen);
  }

  // normalize
  float magnitude = 0.0f;
  for (float value : vec) {
    magnitude += value * value;
  }
  magnitude = std::sqrt(magnitude);
  for (float &value : vec) {
    value /= magnitude;
  }

  auto ndarray = ailoy::create<ailoy::ndarray_t>();
  ndarray->shape.push_back(dim);
  ndarray->dtype = {
      .code = kDLFloat,
      .bits = 32,
      .lanes = 1,
  };
  ndarray->data.resize(sizeof(float) * dim);
  memcpy(ndarray->data.data(), vec.data(), sizeof(float) * dim);

  return ndarray;
}

TEST(VectorStoreTest, Chromadb_CreateAddRetrieve) {
  size_t dimension = 10;
  std::shared_ptr<ailoy::chromadb_vector_store_t> vs;
  const std::string url = "http://localhost:8000";
  try {
    vs = std::make_shared<ailoy::chromadb_vector_store_t>(url);
  } catch (const ailoy::runtime_error e) {
    GTEST_SKIP() << std::format(
        "Chromadb is not available on {}. Skip testing..", url);
  }

  size_t num_vectors = 10;
  std::vector<ailoy::vector_store_add_input_t> add_inputs;
  for (int i = 0; i < num_vectors; i++) {
    auto vec = get_random_normalized_vector(dimension);
    auto item = ailoy::vector_store_add_input_t{
        .embedding = vec,
        .document = "document" + std::to_string(i),
        .metadata = nlohmann::json{{"value", i}},
    };
    add_inputs.push_back(item);
  }
  auto vec_ids = vs->add_vectors(add_inputs);

  std::string test_id = vec_ids[0];

  // test get_by_id
  auto item = vs->get_by_id(test_id).value();
  ASSERT_EQ(item.document, add_inputs[0].document);
  ASSERT_EQ(item.metadata.value(), add_inputs[0].metadata.value());

  // test retrieve
  auto results = vs->retrieve(add_inputs[0].embedding, 1);
  ASSERT_EQ(results.size(), 1);
  auto result = results[0];
  ASSERT_EQ(result.id, test_id);
  ASSERT_EQ(result.document, add_inputs[0].document);
  ASSERT_EQ(result.metadata.value(), add_inputs[0].metadata.value());
  ASSERT_FLOAT_EQ(result.similarity, 1.0f);
}

TEST(VectorStoreTest, ChromadbComponent_CreateAddRetrieve) {
  size_t dimension = 10;
  auto create_vectorstore =
      ailoy::get_language_module()->factories.at("chromadb_vector_store");
  auto attrs = ailoy::create<ailoy::map_t>();
  const std::string url = "http://localhost:8000";
  attrs->insert_or_assign("url", ailoy::create<ailoy::string_t>(url));

  ailoy::component_or_error_t vectorstore_opt;
  try {
    vectorstore_opt = create_vectorstore(attrs);
    ASSERT_EQ(vectorstore_opt.index(), 0);
  } catch (const ailoy::runtime_error e) {
    GTEST_SKIP() << std::format(
        "Chromadb is not available on {}. Skip testing..", url);
  }

  auto vectorstore = std::get<0>(vectorstore_opt);
  auto insert_op = vectorstore->get_operator("insert");
  auto insert_many_op = vectorstore->get_operator("insert_many");
  auto get_by_id_op = vectorstore->get_operator("get_by_id");
  auto retrieve_op = vectorstore->get_operator("retrieve");
  auto remove_op = vectorstore->get_operator("remove");
  auto clear_op = vectorstore->get_operator("clear");

  size_t num_vectors = 10;
  std::vector<ailoy::vector_store_add_input_t> insert_inputs;
  std::vector<std::string> vec_ids;
  for (int i = 0; i < num_vectors; i++) {
    auto vec = get_random_normalized_vector(dimension);
    auto insert_input = ailoy::vector_store_add_input_t{
        .embedding = vec,
        .document = "document" + std::to_string(i),
        .metadata = nlohmann::json{{"value", i}},
    };
    insert_inputs.push_back(insert_input);
  }

  // test insert
  {
    auto item0 = insert_inputs[0];
    auto in1 = ailoy::create<ailoy::map_t>();
    in1->insert_or_assign("embedding",
                          ailoy::create<ailoy::ndarray_t>(*item0.embedding));
    in1->insert_or_assign("document",
                          ailoy::create<ailoy::string_t>(item0.document));
    in1->insert_or_assign("metadata",
                          ailoy::from_nlohmann_json(item0.metadata));
    insert_op->initialize(in1);
    auto insert_outputs_opt = insert_op->step();
    ASSERT_EQ(insert_outputs_opt.index(), 0);
    auto insert_outputs = std::get<0>(insert_outputs_opt);
    auto vec_id =
        insert_outputs.val->as<ailoy::map_t>()->at<ailoy::string_t>("id");
    vec_ids.push_back(*vec_id);
  }

  // test insert_many
  {
    auto items = ailoy::create<ailoy::array_t>();
    for (int i = 1; i < num_vectors; i++) {
      auto in = ailoy::create<ailoy::map_t>();
      in->insert_or_assign("embedding", ailoy::create<ailoy::ndarray_t>(
                                            *insert_inputs[i].embedding));
      in->insert_or_assign("document", ailoy::create<ailoy::string_t>(
                                           insert_inputs[i].document));
      in->insert_or_assign(
          "metadata", ailoy::from_nlohmann_json(insert_inputs[i].metadata));
      items->push_back(in);
    }

    insert_many_op->initialize(items);
    auto insert_many_outputs_opt = insert_many_op->step();
    ASSERT_EQ(insert_many_outputs_opt.index(), 0);
    auto insert_many_outputs = std::get<0>(insert_many_outputs_opt);
    auto ids =
        insert_many_outputs.val->as<ailoy::map_t>()->at<ailoy::array_t>("ids");
    for (auto vec_id : *ids) {
      vec_ids.push_back(*vec_id->as<ailoy::string_t>());
    }
  }

  std::string test_id = vec_ids[0];

  // test get_by_id
  auto in2 = ailoy::create<ailoy::map_t>();
  in2->insert_or_assign("id", ailoy::create<ailoy::string_t>(test_id));
  auto init_result = get_by_id_op->initialize(in2);
  ASSERT_FALSE(init_result.has_value());
  auto out2_opt = get_by_id_op->step();
  ASSERT_EQ(out2_opt.index(), 0);
  auto out2 = std::get<0>(out2_opt).val->as<ailoy::map_t>();
  ASSERT_EQ(*out2->at<ailoy::string_t>("document"), insert_inputs[0].document);
  ASSERT_EQ(out2->at<ailoy::value_t>("metadata")->to_nlohmann_json(),
            insert_inputs[0].metadata.value());

  // test retrieve
  auto in3 = ailoy::create<ailoy::map_t>();
  in3->insert_or_assign("query_embedding", ailoy::create<ailoy::ndarray_t>(
                                               *insert_inputs[0].embedding));
  in3->insert_or_assign("top_k", ailoy::create<ailoy::uint_t>(1));
  retrieve_op->initialize(in3);
  auto out3_opt = retrieve_op->step();
  ASSERT_EQ(out3_opt.index(), 0);
  auto out3 = std::get<0>(out3_opt).val->as<ailoy::map_t>()->at<ailoy::array_t>(
      "results");
  ASSERT_EQ(out3->size(), 1);
  auto result = out3->at<ailoy::map_t>(0);
  ASSERT_EQ(*result->at<ailoy::string_t>("id"), test_id);
  ASSERT_EQ(*result->at<ailoy::string_t>("document"),
            insert_inputs[0].document);
  ASSERT_EQ(result->at<ailoy::value_t>("metadata")->to_nlohmann_json(),
            insert_inputs[0].metadata.value());
  ASSERT_FLOAT_EQ(*result->at<ailoy::float_t>("similarity"), 1.0f);

  // test remove
  auto in4 = ailoy::create<ailoy::map_t>();
  in4->insert_or_assign("id", ailoy::create<ailoy::string_t>(vec_ids[0]));
  remove_op->initialize(in4);
  auto out4_opt = remove_op->step();
  ASSERT_EQ(out4_opt.index(), 0);

  // test clear
  clear_op->initialize();
  auto out5_opt = clear_op->step();
  ASSERT_EQ(out5_opt.index(), 0);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
