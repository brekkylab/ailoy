#include "tokenizer.hpp"

#include <tokenizers_c.h>

#include "filesystem.hpp"
#include "model_cache.hpp"

namespace ailoy {

/* tokenizer_t */

tokenizer_t::tokenizer_t(const ailoy::fs::path_t &json_file_path) {
  auto contents = ailoy::fs::read_file_text(json_file_path).unwrap();
  handle_ = tokenizers_new_from_str(contents.data(), contents.size());
}

tokenizer_t::~tokenizer_t() { tokenizers_free(handle_); }

size_t tokenizer_t::get_vocab_size() {
  size_t rv;
  tokenizers_get_vocab_size(handle_, &rv);
  return rv;
}

std::vector<tokenizer_t::token_t> tokenizer_t::encode(const std::string &text,
                                                      bool add_special_token) {
  TokenizerEncodeResult result;
  tokenizers_encode(handle_, text.data(), text.length(),
                    static_cast<int>(add_special_token), &result);

  std::vector<tokenizer_t::token_t> rv(result.token_ids,
                                       result.token_ids + result.len);
  tokenizers_free_encode_results(&result, 1);
  return rv;
}

std::string tokenizer_t::decode(const std::vector<tokenizer_t::token_t> &ids,
                                bool skip_special_tokens) {
  size_t ids_size = ids.size();
  std::vector<uint32_t> ids_data(ids_size);
  // uint32_t ids_data[ids_size];
  for (size_t i = 0; i < ids_size; i++)
    ids_data[i] = static_cast<uint32_t>(ids.at(i));
  tokenizers_decode(handle_, ids_data.data(), ids_size,
                    static_cast<int>(skip_special_tokens));

  char *rv_data;
  size_t len;
  tokenizers_get_decode_str(handle_, const_cast<const char **>(&rv_data), &len);

  std::string rv(rv_data, len);
  return rv;
}

tokenizer_t::token_t
tokenizer_t::token_str_to_id(const std::string &token_str) {
  const char *token_c_str = token_str.c_str();
  token_t token_id;
  tokenizers_token_to_id(handle_, token_c_str, token_str.length(), &token_id);
  return token_id;
}

std::string tokenizer_t::token_id_to_str(token_t token_id) {
  const char *token_str;
  size_t token_str_len;
  tokenizers_id_to_token(handle_, token_id, &token_str, &token_str_len);
  return std::string(token_str, token_str_len);
}

component_or_error_t
create_tokenizer_component(std::shared_ptr<const value_t> inputs) {
  if (!inputs->is_type_of<map_t>())
    return error_output_t(
        type_error("Tokenizer: create", "inputs", "map_t", inputs->get_type()));

  auto inputs_map = inputs->as<map_t>();

  if (!inputs_map->contains("model"))
    return error_output_t(range_error("Tokenizer: create", "model"));
  if (!inputs_map->at("model")->is_type_of<string_t>())
    return error_output_t(type_error("Tokenizer: create", "model", "string_t",
                                     inputs_map->at("model")->get_type()));
  std::string model = *inputs_map->at<string_t>("model");

  std::string quantization;
  if (inputs_map->contains("quantization")) {
    if (inputs_map->at("quantization")->is_type_of<string_t>())
      quantization = *inputs_map->at<string_t>("quantization");
    else
      return error_output_t(
          type_error("Tokenizer: create", "quantization", "string_t",
                     inputs_map->at("quantization")->get_type()));
  } else
    quantization = "q4f16_1";

  auto model_path =
      get_cache_root() / get_model_base_path(model) / quantization;
  if (!fs::directory_exists(model_path))
    return error_output_t(
        std::format("Tokenizer: model \"{}\"(quantization: {}) does not "
                    "exist. Download the model first.",
                    model, quantization));

  auto tokenizer_json_path = model_path / "tokenizer.json";
  if (!fs::file_exists(tokenizer_json_path))
    return error_output_t("Tokenizer: tokenizer.json does not exist.");

  auto tokenizer = create<tokenizer_t>(tokenizer_json_path);

  auto encode = [](std::shared_ptr<component_t> component,
                   std::shared_ptr<const value_t> inputs) -> value_or_error_t {
    if (!inputs->is_type_of<map_t>())
      return error_output_t(type_error("Tokenizer: encode", "inputs", "map_t",
                                       inputs->get_type()));

    auto inputs_map = inputs->as<map_t>();

    // Get text
    if (!inputs_map->contains("text"))
      return error_output_t(range_error("Tokenizer: encode", "text"));
    if (!inputs_map->at("text")->is_type_of<string_t>())
      return error_output_t(type_error("Tokenizer: encode", "text", "string_t",
                                       inputs_map->at("text")->get_type()));
    std::string text = *inputs_map->at<string_t>("text");

    auto t = component->get_obj("tokenizer")->as<tokenizer_t>();
    auto encoded = t->encode(text);

    auto tokens = create<array_t>();
    for (const auto &token : encoded) {
      tokens->push_back(create<int_t>(token));
    }

    auto res = create<map_t>();
    res->insert_or_assign("tokens", tokens);
    return res;
  };

  auto decode = [](std::shared_ptr<component_t> component,
                   std::shared_ptr<const value_t> inputs) -> value_or_error_t {
    if (!inputs->is_type_of<map_t>())
      return error_output_t(type_error("Tokenizer: decode", "inputs", "map_t",
                                       inputs->get_type()));

    auto inputs_map = inputs->as<map_t>();

    // Get tokens
    if (!inputs_map->contains("tokens"))
      return error_output_t(range_error("Tokenizer: decode", "tokens"));
    if (!inputs_map->at("tokens")->is_type_of<array_t>())
      return error_output_t(type_error("Tokenizer: decode", "tokens", "array_t",
                                       inputs_map->at("tokens")->get_type()));
    auto tokens = inputs_map->at<array_t>("tokens");

    std::vector<tokenizer_t::token_t> tokens_;
    for (const auto &token :
         tokens->to_nlohmann_json().get<std::vector<int>>()) {
      tokens_.push_back(token);
    }

    auto t = component->get_obj("tokenizer")->as<tokenizer_t>();
    auto decoded = t->decode(tokens_, false);

    auto res = create<map_t>();
    res->insert_or_assign("text", create<string_t>(decoded));
    return res;
  };

  auto comp = create<component_t>(
      std::initializer_list<
          std::pair<const std::string, std::shared_ptr<method_operator_t>>>{
          {"encode", create<instant_method_operator_t>(encode)},
          {"decode", create<instant_method_operator_t>(decode)}});
  comp->set_obj("tokenizer", tokenizer);
  return comp;
}

} // namespace ailoy
