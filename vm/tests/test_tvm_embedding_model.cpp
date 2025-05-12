#include <gtest/gtest.h>

#include "language.hpp"
#include "mlc_llm/embedding_model.hpp"
#include "module.hpp"

TEST(EmbeddingModelTest, TestTokenize) {
  auto create_tvm_embedding_model =
      ailoy::get_language_module()->factories.at("tvm_embedding_model");
  auto attrs = ailoy::create<ailoy::map_t>();
  auto embedding_model = std::get<0>(create_tvm_embedding_model(attrs));

  auto tokenize_op = embedding_model->get_operator("tokenize");
  {
    auto in = ailoy::create<ailoy::map_t>();
    in->insert_or_assign("prompt",
                         ailoy::create<ailoy::string_t>("What is BGE M3?"));
    tokenize_op->initialize(in);
    auto tokens = std::get<0>(tokenize_op->step())
                      .val->as<ailoy::map_t>()
                      ->at<ailoy::array_t>("tokens");
    ASSERT_EQ(tokens->size(), 9);
    ASSERT_EQ(*tokens->at<ailoy::int_t>(0), 0); // cls token
    ASSERT_EQ(*tokens->at<ailoy::int_t>(8), 2); // eos token
  }
}

TEST(EmbeddingModelTest, TestInfer) {
  auto create_tvm_embedding_model =
      ailoy::get_language_module()->factories.at("tvm_embedding_model");
  auto attrs = ailoy::create<ailoy::map_t>();
  auto embedding_model = std::get<0>(create_tvm_embedding_model(attrs));

  auto infer_op = embedding_model->get_operator("infer");
  {
    auto in = ailoy::create<ailoy::map_t>();
    in->insert_or_assign("prompt",
                         ailoy::create<ailoy::string_t>("What is BGE M3?"));
    infer_op->initialize(in);
    auto out = std::get<0>(infer_op->step())
                   .val->as<ailoy::map_t>()
                   ->at<ailoy::ndarray_t>("embedding");
    auto shape = out->shape;
    std::vector<float> embedding = *out;
    ASSERT_EQ(embedding.size(), 1024);
  }
}

float dot(std::vector<float> a, std::vector<float> b) {
  float res = 0;
  if (a.size() != b.size())
    throw ailoy::exception("cannot dot product vectors of different sizes");
  return std::inner_product(std::begin(a), std::end(a), std::begin(b), 0.0);
}

TEST(EmbeddingModelTest, TestInferResult) {
  auto normalized_data = [](std::shared_ptr<ailoy::ndarray_t> ndarr) {
    size_t dimension = ndarr->shape.back();
    std::vector<float> data = ndarr->operator std::vector<float>();

    if (data.size() % dimension != 0)
      throw ailoy::exception(
          "Size of data is not a multiple of the dimension.");

    // calculate magnitude in unit of separated vectors
    for (auto begin = data.begin(); begin < data.end(); begin += dimension) {
      std::span<float> view(begin, begin + dimension);

      float magnitude = 0.0f;
      for (float value : view)
        magnitude += value * value;
      magnitude = std::sqrt(magnitude);

      for (float &value : view)
        value /= magnitude;
    }
    return data;
  };

  auto create_tvm_embedding_model =
      ailoy::get_language_module()->factories.at("tvm_embedding_model");
  auto attrs = ailoy::create<ailoy::map_t>();
  auto embedding_model = std::get<0>(create_tvm_embedding_model(attrs));

  auto infer_op = embedding_model->get_operator("infer");
  std::vector<float> embedding0;
  std::string prompt0 = "What is BGE M3?";
  {
    auto in = ailoy::create<ailoy::map_t>();
    in->insert_or_assign("prompt", ailoy::create<ailoy::string_t>(prompt0));
    infer_op->initialize(in);
    auto out = std::get<0>(infer_op->step())
                   .val->as<ailoy::map_t>()
                   ->at<ailoy::ndarray_t>("embedding");
    auto shape = out->shape;
    embedding0 = normalized_data(out);
  }
  std::vector<float> embedding1;
  std::string prompt1 = "Defination of BM25";
  {
    auto in = ailoy::create<ailoy::map_t>();
    in->insert_or_assign("prompt", ailoy::create<ailoy::string_t>(prompt1));
    infer_op->initialize(in);
    auto out = std::get<0>(infer_op->step())
                   .val->as<ailoy::map_t>()
                   ->at<ailoy::ndarray_t>("embedding");
    auto shape = out->shape;
    embedding1 = normalized_data(out);
  }
  std::vector<float> embedding2;
  std::string prompt2 =
      "BGE M3 is an embedding model supporting dense retrieval, lexical "
      "matching and multi-vector interaction.";
  {
    auto in = ailoy::create<ailoy::map_t>();
    in->insert_or_assign("prompt", ailoy::create<ailoy::string_t>(prompt2));
    infer_op->initialize(in);
    auto out = std::get<0>(infer_op->step())
                   .val->as<ailoy::map_t>()
                   ->at<ailoy::ndarray_t>("embedding");
    auto shape = out->shape;
    embedding2 = normalized_data(out);
  }
  std::vector<float> embedding3;
  std::string prompt3 =
      "BM25 is a bag-of-words retrieval function that ranks a set of documents "
      "based on the query terms appearing in each document";
  {
    auto in = ailoy::create<ailoy::map_t>();
    in->insert_or_assign("prompt", ailoy::create<ailoy::string_t>(prompt3));
    infer_op->initialize(in);
    auto out = std::get<0>(infer_op->step())
                   .val->as<ailoy::map_t>()
                   ->at<ailoy::ndarray_t>("embedding");
    auto shape = out->shape;
    embedding3 = normalized_data(out);
  }
  ASSERT_GT(dot(embedding0, embedding2), dot(embedding0, embedding3));
  ASSERT_LT(dot(embedding1, embedding2), dot(embedding1, embedding3));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
