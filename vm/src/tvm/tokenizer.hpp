#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "object.hpp"

namespace ailoy {

class tokenizer_t : public object_t {
  using token_t = int32_t;

public:
  tokenizer_t(const std::filesystem::path &json_file_path);

  ~tokenizer_t();

  size_t get_vocab_size();

  std::vector<token_t> encode(const std::string &text,
                              bool add_special_token = true);

  std::string decode(const std::vector<token_t> &ids,
                     bool skip_special_tokens = true);

  token_t token_str_to_id(const std::string &token_str);

  std::string token_id_to_str(token_t token_id);

private:
  /**
   * @brief tokenizer handle
   */
  void *handle_;
};

} // namespace ailoy
