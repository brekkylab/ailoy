#pragma once

#include <minijinja.h>
#include <nlohmann/json.hpp>

#include "filesystem.hpp"
#include "module.hpp"

namespace ailoy {

class chat_manager_t : public object_t {
public:
  chat_manager_t(const std::string &chat_template, const std::string &bos_token,
                 const std::string &eos_token,
                 const std::string &botc_token = "",
                 const std::string &eotc_token = "")
      : minijinja_env_(mj_env_new()), bos_token_(bos_token),
        eos_token_(eos_token), botc_token_(botc_token),
        eotc_token_(eotc_token) {
    bool ok =
        mj_env_add_template(minijinja_env_, "template", chat_template.c_str());
    if (!ok) {
      throw std::runtime_error("[chat_manager_t] Failed to add chat template");
    }
  }

  ~chat_manager_t() {
    if (minijinja_env_) {
      mj_env_free(minijinja_env_);
      minijinja_env_ = nullptr;
    }
  }

  static std::shared_ptr<chat_manager_t>
  make_from_config_file(ailoy::fs::path_t config_file_path);

  const std::string
  apply_chat_template(std::shared_ptr<const value_t> conversation,
                      std::shared_ptr<const value_t> tools = nullptr,
                      const bool reasoning = false,
                      const bool add_generation_prompt = true);

  const std::string &get_bos_token() const { return bos_token_; }

  const std::string &get_eos_token() const { return eos_token_; }

  const std::string &get_botc_token() const { return botc_token_; }

  const std::string &get_eotc_token() const { return eotc_token_; }

  bool is_bos_token(const std::string &token) const {
    return token == bos_token_;
  }

  bool is_eos_token(const std::string &token) const {
    return token == eos_token_;
  }

  bool is_botc_token(const std::string &token) const {
    return token == botc_token_;
  }

  bool is_eotc_token(const std::string &token) const {
    return token == eotc_token_;
  }

  const std::optional<std::string>
  get_json_str_if_valid(const std::vector<std::string> &tokens);

private:
  struct mj_env *minijinja_env_;

  const std::string bos_token_;

  const std::string eos_token_;

  const std::string botc_token_;

  const std::string eotc_token_;
};

std::shared_ptr<value_t> remove_tool_call_id(std::shared_ptr<const value_t> in);

std::shared_ptr<value_t>
put_default_reasoning(std::shared_ptr<const value_t> in,
                      const std::string &content = "\n\n");

/**
  Melt the `reasoning` field into the output `content`

  Before:
  ```
  "role": "assistant",
  "reasoning": [
    {"type": "text", "text": "reasoning..."}
  ],
  "content": [
    {"type": "text", "text": "Based on reasoning, It's foo"}
  ]
  ```

  After:
  ```
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "<think>reasoning...</think>"
    },
    {
      "type": "text",
      "text": "Based on reasoning, It's foo"
    }
  ]
  ```

  Reasoning field always attached to first element of the content.
 */
std::shared_ptr<value_t>
melt_reasoning(std::shared_ptr<const value_t> in,
               const std::string &bor_delimiter = "<think>",
               const std::string &eor_delimiter = "</think>\n\n");

/**
  Merge successive `text` to one data, in `content` or `reasoning`

  Before:
  ```
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "I'm a foo"
    },
    {
      "type": "text",
      "text": "I'm a bar and foobar"
    }
  ]
  ```

  After:
  ```
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "I'm a fooI'm a bar and foobar"
    }
  ]
  ```
 */
std::shared_ptr<value_t> merge_text_data(std::shared_ptr<const value_t> in,
                                         const std::string &delimiter = "");

/**
   Melt `content` text to a single string
   Before:
  ```
  "role": "user",
  "content": [
    {"type": "text", "text": "This is user text!"}
  ]
  ```
   After:
  ```
  "role": "user",
  "content": "This is user text!"
  ```
  The length of `content` must be 1.
 */
std::shared_ptr<value_t> melt_content_text(std::shared_ptr<const value_t> in);

} // namespace ailoy
