/**
 * The schema of the `message_t` extends the HuggingFace `transformer`s' chat
 * templating convention while maintaining full compatibility.
 *
 * In the HuggingFace transformers library, there is no strict format for
 * messages. Users typically pass a model-specific dictionary to the
 * `apply_chat_template` function. While the function works if the format
 * matches the model's expectations, it may raise errors when used with a
 * different model. This inconsistency arises because each model has its own
 * templating logic and assumptions about the input format.
 *
 * However, most models follow a general "templating convention," which implies
 * a loosely shared structure for input dictionaries. This schema aims to
 * formalize and unify those conventions into a consistent standard across
 * models, while also extending it to support multimodal content and tool
 * usage.
 *
 * Example:
 * ```json
 * messages = [
 *   {
 *     "role": "system",
 *     "content": [{"type": "text", "text": "<SYSTEM_MESSAGE>"}]
 *   },
 *   {
 *     "role": "user",
 *     "content": [
 *       {
 *          "type": "image",
 *          "url": "http://images.cocodataset.org/val2017/000000039769.jpg"
 *       },
 *       {"type": "text", "text": "What are these?"}
 *     ]
 *   },
 *   {
 *     "role": "assistant",
 *     "reasoning": [
 *       {"type": "text", "text": "<REASONINGS>"}
 *     ],
 *     "content": [
 *       {"type": "text", "text": "<OUTPUT_TEXT>"}
 *     ],
 *     "tool_calls": [
 *       {"type": "json", "json": <SINGLE_TOOL_CALL1>},
 *       {"type": "json", "json": <SINGLE_TOOL_CALL2>}
 *     ]
 *   },
 *   {
 *     "role": "tool",
 *     "content": [
 *       {"type": "text", "text": "<TOOL_RESULTS>"}
 *     ]
 *   }
 * ]
 * ```
 *
 **/
#pragma once

#include <ostream>
#include <string>
#include <vector>

#ifdef USE_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#endif

namespace ailoy {

enum class content_datatype_t { text, image, audio };

std::ostream &operator<<(std::ostream &os, const content_datatype_t &v);

struct content_t {
  content_t() {}

  content_t(content_datatype_t dt, const std::string key,
            const std::string &value)
      : ty(dt), key(key), value(value) {}

  content_datatype_t ty;
  std::string key;
  std::string value;
};

#ifdef USE_NLOHMANN_JSON
void to_json(nlohmann::json &j, const content_t &v);
void from_json(const nlohmann::json &j, content_t &v);
#endif

std::ostream &operator<<(std::ostream &os, const content_t &v);

enum class role_t { system, user, assistant, tool };

std::ostream &operator<<(std::ostream &os, const role_t &v);

enum class content_category_t { content, reasoning, tool_call };

std::ostream &operator<<(std::ostream &os, const content_category_t &v);

struct message_t {
  message_t() {}

  message_t(role_t role) : role(role), key(content_category_t::content) {}

  message_t(role_t role, const std::string &content_text) : message_t(role) {
    push_content_text(content_text);
  }

  message_t(role_t role, content_category_t category)
      : role(role), key(category) {}

  message_t(role_t role, content_category_t category,
            const std::string &content_text)
      : message_t(role, category) {
    push_content_text(content_text);
  }

  void push_content_text(const std::string &text) {
    content_t content{content_datatype_t::text, "text", text};
    value.push_back(std::move(content));
  }

  role_t role;
  content_category_t key;
  std::vector<content_t> value;
};

std::ostream &operator<<(std::ostream &os, const message_t &v);

using messages_t = std::vector<message_t>;

std::ostream &operator<<(std::ostream &os, const messages_t &v);

#ifdef USE_NLOHMANN_JSON
void to_json(nlohmann::json &j, const message_t &v);
void from_json(const nlohmann::json &j, message_t &v);
#endif

} // namespace ailoy
