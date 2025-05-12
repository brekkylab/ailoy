#pragma once

#include <string>

#include <json_ffi/openai_api_protocol.h>
#include <serve/engine.h>
#include <tokenizers/streamer.h>
#include <tokenizers/tokenizers.h>

#include "chat_template_engine.hpp"
#include "model_cache.hpp"
#include "module.hpp"

namespace ailoy {

class mlc_llm_engine_t : public object_t {
public:
  enum class output_mode : uint8_t {
    text = 0,
    tool_call = 1,
    reasoning = 2,
  };

  struct request_state_t {
    std::string model;
    std::vector<mlc::llm::TextStreamer> streamer;
  };

  struct response_state_t {
    nlohmann::json message;
    std::optional<std::string> finish_reason = std::nullopt;
  };

  mlc_llm_engine_t(const std::string &model, const std::string &quantization,
                   DLDevice device, const std::string &mode);

  void initialize_chat_completion(const std::string &request_id,
                                  const std::string &prompt);

  std::optional<response_state_t>
  step_chat_completion(const std::string &request_id);

  const std::optional<std::string> &get_last_error() const {
    return last_error_;
  }

  std::filesystem::path get_model_path() const { return model_path_; }

  void set_chat_template_engine(
      std::shared_ptr<chat_template_engine_t> template_engine) {
    template_engine_ = template_engine;
  }

private:
  std::vector<mlc::llm::json_ffi::ChatCompletionStreamResponse>
  get_response_from_stream_output(
      tvm::Array<mlc::llm::serve::RequestStreamOutput> delta_outputs);

  DLDevice device_;

  std::unique_ptr<mlc::llm::serve::Engine> engine_;

  std::shared_ptr<chat_template_engine_t> template_engine_;

  std::filesystem::path model_path_;

  output_mode mode_ = output_mode::text;

  std::vector<std::string> tool_call_tokens_{};

  mlc::llm::serve::GenerationConfig default_generation_config_;

  std::vector<std::string> stop_str_;

  std::vector<int> stop_token_ids_;

  mlc::llm::Tokenizer tokenizer_;

  std::unordered_map<std::string, request_state_t> request_map_;

  std::unordered_map<std::string, response_state_t> response_map_;

  std::optional<std::string> last_error_;
};

std::shared_ptr<mlc_llm_engine_t>
get_mlc_llm_engine(const std::string &model_name,
                   const std::string &quantization, DLDevice device,
                   const std::string &mode = "interactive");

} // namespace ailoy
