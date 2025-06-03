#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>

#include "exception.hpp"
#include "module.hpp"
#include "object.hpp"

namespace ailoy {

struct tvm_model_t;

struct chat_manager_t;

struct tokenizer_t;

struct kv_cache_t;

struct tokenizer_info_t;

struct grammar_t;

struct grammar_matcher_t;

struct context_length_limit {
  context_length_limit() = default;

  static constexpr const char *what() noexcept {
    return "Context length limit exceeded.";
  }
};

static_assert(is_exception_reason<context_length_limit>);

class tvm_language_model_t : public object_t {
public:
  struct config_t {
    double temperature;
    double top_p;
  };

  /**
   * The term "stream mode" refers to a way of indicating that the model is in a
   * specific state during decoding. For example, when a `<tool_call>` token is
   * generated during inference, we can assume that the model is about to begin
   * generating a (formatted) tool calling request, starting from the next
   * token. Same behavior can also be applied to <reasoning> token or even
   * user-defined patterns. Stream mode serves as a marker for this state.
   *
   * Stream mode is useful for restricting the output format. When the model is
   * in "tool calling mode", its output should conform to a predefined schema.
   * This can be enforced by applying a corresponding grammar.
   */
  struct stream_mode_t {
    stream_mode_t(tvm_language_model_t *model,
                  const std::string &open_indicator,
                  const std::string &close_indicator);

    /**
     * @brief Whether the currently generated token (history) matches the
     * indicator
     * @param indicator_type "open" or "close"
     * @param history History of generated token
     */
    bool check_indicator(const std::string &indicator_type,
                         const std::vector<int32_t> &history) const;

    std::vector<int32_t> open_indicator;

    std::vector<int32_t> close_indicator;

    /**
     * The grammar applied to a stream mode (`set_grammar`).
     */
    std::shared_ptr<grammar_t> grammar;

    /**
     * Created when the model enters grammar-enabled mode.
     */
    std::shared_ptr<grammar_matcher_t> matcher;
  };

  /**
   * Constructor
   */
  tvm_language_model_t(const std::string &model,
                       const std::string &quantization, DLDevice device);

  void clear();

  /**
   * Apply chat template
   */
  std::string
  apply_chat_template(std::shared_ptr<const value_t> conversation,
                      std::shared_ptr<const value_t> tools = nullptr,
                      bool enable_reasoning = false,
                      bool add_generation_prompt = true) const;

  /** Begin of reasoning */
  bool is_bor(const std::string &tok) const;

  /** Begin of reasoning */
  bool is_bor(int32_t tok) const;

  /** End of reasoning */
  bool is_eor(const std::string &tok) const;

  /** End of reasoning */
  bool is_eor(int32_t tok) const;

  /** Begin of sentence */
  bool is_bos(const std::string &tok) const;

  /** End of reasoning */
  bool is_bos(int32_t tok) const;

  /** End of sentence */
  bool is_eos(const std::string &tok) const;

  /** End of reasoning */
  bool is_eos(int32_t tok) const;

  /** Begin of tool call */
  bool is_botc(const std::string &tok) const;

  /** Begin of tool call */
  bool is_botc(int32_t tok) const;

  /** End of tool call */
  bool is_eotc(const std::string &tok) const;

  /** End of tool call */
  bool is_eotc(int32_t tok) const;

  /** Tokenize prompts, before running prefill */
  std::vector<int32_t> tokenize(const std::string &prompt) const;

  /** Prefill */
  int32_t prefill(const std::vector<int32_t> &tokens);

  /** Decode */
  int32_t decode(int32_t last_token);

  /** Output token to string. It can have no value when the incompleted unicode
   * string generated */
  std::optional<std::string> detokenize(int32_t token);

  const std::string &get_current_stream_mode() const;

  const stream_mode_t &get_stream_mode(std::string mode_name) const;

  void add_stream_mode(std::string mode_name, const std::string &open_indicator,
                       const std::string &close_indicator);

  void remove_stream_mode(std::string mode_name);

  std::shared_ptr<grammar_matcher_t> get_current_grammar_matcher();

  void set_builtin_grammar(const std::string &mode_name,
                           const std::string &grammar_type);

  void set_json_schema_grammar(const std::string &mode_name,
                               const std::string &json_schema);

  void set_regex_grammar(const std::string &mode_name,
                         const std::string &regex);

  void set_ebnf_grammar(const std::string &mode_name, const std::string &ebnf);

  void reset_grammar(const std::string &mode_name);

  config_t config;

  const config_t &get_default_config() const { return default_config_; }

private:
  std::shared_ptr<tvm_model_t> model_;

  std::shared_ptr<chat_manager_t> template_engine_;

  std::shared_ptr<tokenizer_t> tokenizer_;

  std::shared_ptr<kv_cache_t> kv_cache_;

  std::shared_ptr<tokenizer_info_t> tokenizer_info_;

  config_t default_config_;

  std::vector<int32_t> history_;

  std::vector<int32_t> output_stream_;

  std::string current_stream_mode_;

  std::unordered_map<std::string, stream_mode_t> stream_modes_;

  tvm::runtime::PackedFunc fembed_;

  tvm::runtime::PackedFunc fprefill_;

  tvm::runtime::PackedFunc fdecode_;

  tvm::runtime::PackedFunc fapply_bitmask_inplace_;

  tvm::runtime::PackedFunc fsample_top_p_from_logits_;
};

component_or_error_t
create_tvm_language_model_component(std::shared_ptr<const value_t> inputs);

} // namespace ailoy
