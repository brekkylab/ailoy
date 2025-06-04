#include "tokenizer.hpp"

#include <filesystem>

#include <tokenizers_c.h>

#include "../file_util.hpp"

namespace ailoy {

/* tokenizer_t */

tokenizer_t::tokenizer_t(const std::filesystem::path &json_file_path) {
  auto contents = utils::LoadBytesFromFile(json_file_path);
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

} // namespace ailoy
