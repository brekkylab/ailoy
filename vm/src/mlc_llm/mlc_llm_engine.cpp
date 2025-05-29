#include "mlc_llm_engine.hpp"

#include <json_ffi/conv_template.h>
#include <support/json_parser.h>

#include "logging.hpp"
#include "uuid.hpp"

using namespace mlc::llm;
using namespace mlc::llm::serve;
using namespace mlc::llm::json_ffi;

namespace ailoy {

mlc_llm_engine_t::mlc_llm_engine_t(const std::string &model_name,
                                   const std::string &quantization,
                                   DLDevice device, const std::string &mode) {
  device_ = device;
  auto device_type_str = tvm::runtime::DLDeviceType2Str(device.device_type);
  debug("Using device {}:{}", device_type_str, device_.device_id);

  // Download model
  debug("Downloading model: {}, quantization: {}, device: {}", model_name,
        quantization, device_type_str);
  model_cache_download_result_t download_model_result =
      download_model(model_name, quantization, device_type_str);
  if (!download_model_result.success) {
    throw ailoy::runtime_error(download_model_result.error_message.value());
  }

  model_path_ = download_model_result.model_path.value();
  debug("Model downloaded to {}", model_path_.string());

  // Configuration for engine creation
  picojson::object engine_config_json;
  engine_config_json["model"] = picojson::value(model_path_.string());
  engine_config_json["model_lib"] =
      picojson::value(download_model_result.model_lib_path.value().string());
  engine_config_json["mode"] = picojson::value(mode);

  // Create engine
  EngineCreationOutput engine_creation_output =
      Engine::Create(
          picojson::value(engine_config_json).serialize(), device,
          PackedFunc([&](TVMArgs args, TVMRetValue *ret) {
            ICHECK_EQ(args.size(), 1);
            Array<RequestStreamOutput> delta_outputs = args[0];

            auto responses = get_response_from_stream_output(delta_outputs);
            for (auto response : responses) {
              if (response.choices.size() == 0)
                continue;
              auto &choice = response.choices[0];
              auto &resp = response_map_.at(response.id);

              if (choice.finish_reason.has_value()) {
                if (choice.finish_reason.value() ==
                    mlc::llm::json_ffi::FinishReason::stop) {
                  resp.message["role"] = choice.delta.role;
                  resp.message["content"] = "\n";
                  resp.finish_reason = "stop";
                } else if (choice.finish_reason.value() ==
                           mlc::llm::json_ffi::FinishReason::length) {
                  resp.message["role"] = choice.delta.role;
                  resp.message["content"] = "\n";
                  resp.finish_reason = "length";
                } else if (choice.finish_reason.value() ==
                           mlc::llm::json_ffi::FinishReason::error) {
                  resp.message["role"] = choice.delta.role;
                  resp.message["content"] = "\n";
                  resp.finish_reason = "error";
                } else if (choice.finish_reason.value() ==
                           mlc::llm::json_ffi::FinishReason::tool_calls) {
                  auto tool_call_json =
                      nlohmann::json::parse(choice.delta.content.Text());
                  auto put_tool_calls = [&resp](nlohmann::json function,
                                                int index = 0) {
                    resp.message["tool_calls"].push_back(nlohmann::json{
                        {"index", index},
                        {"id", std::format("call_{}", generate_uuid())},
                        {"type", "function"},
                        {"function", function},
                        // if arguments should be string
                        //  {"function",
                        //   {{"name", function["name"]},
                        //    {"arguments", function["arguments"].dump()}},
                    });
                  };
                  resp.message["role"] = choice.delta.role;
                  resp.message["content"] = nullptr;
                  if (tool_call_json.is_array()) {
                    for (int i = 0; i < tool_call_json.size(); i++) {
                      put_tool_calls(tool_call_json.at(i), i);
                    }
                  } else {
                    put_tool_calls(tool_call_json);
                  }
                  resp.finish_reason = "tool_calls";
                }
              } else if (choice.delta.content.IsText()) {
                resp.message["role"] = choice.delta.role;
                resp.message["content"] = choice.delta.content.Text();
                if (mode_ == output_mode::reasoning)
                  resp.message["reasoning"] = true;
              }
            }
          }),
          NullOpt)
          .Unwrap();
  engine_ = std::move(engine_creation_output.reloaded_engine);
  auto engine_config =
      std::move(engine_creation_output.completed_engine_config);
  default_generation_config_ =
      std::move(engine_creation_output.default_generation_cfg);
  debug("Engine created");

  // Load conversation template.
  Result<picojson::object> model_config_json =
      serve::Model::LoadModelConfig(engine_config->model);
  if (model_config_json.IsErr()) {
    last_error_ = model_config_json.UnwrapErr();
    return;
  }
  const picojson::object &model_config_json_unwrapped =
      model_config_json.Unwrap();
  Result<Conversation> conv_template =
      Conversation::FromJSON(mlc::llm::json::Lookup<picojson::object>(
          model_config_json_unwrapped, "conv_template"));
  if (conv_template.IsErr()) {
    last_error_ =
        "Invalid conversation template JSON: " + conv_template.UnwrapErr();
    return;
  }
  auto conv_template_ = conv_template.Unwrap();
  stop_str_ = conv_template_.stop_str;
  stop_token_ids_ = conv_template_.stop_token_ids;
  debug("Conv template loaded");

  tokenizer_ = Tokenizer::FromPath(engine_config->model);
  debug("Tokenizer loaded");
}

void mlc_llm_engine_t::initialize_chat_completion(const std::string &request_id,
                                                  const std::string &prompt) {
  mlc::llm::json_ffi::ChatCompletionRequest request;

  // n is always 1
  request.n = 1;
  // stream is always true
  request.stream = true;

  auto inputs = std::vector<Data>();
  inputs.push_back(mlc::llm::serve::TextData(prompt));
  Array<String> stop_strs;
  stop_strs.reserve(stop_str_.size());
  for (const std::string &stop_str : stop_str_) {
    stop_strs.push_back(stop_str);
  }
  if (request.stop.has_value()) {
    stop_strs.reserve(stop_strs.size() + request.stop.value().size());
    for (const std::string &stop_str : request.stop.value()) {
      stop_strs.push_back(stop_str);
    }
  }

  // create a generation config from request
  const auto &default_gen_cfg = default_generation_config_;
  auto gen_cfg = tvm::runtime::make_object<GenerationConfigNode>();
  gen_cfg->n = request.n;
  gen_cfg->temperature =
      request.temperature.value_or(default_gen_cfg->temperature);
  gen_cfg->top_p = request.top_p.value_or(default_gen_cfg->top_p);
  gen_cfg->frequency_penalty =
      request.frequency_penalty.value_or(default_gen_cfg->frequency_penalty);
  gen_cfg->presence_penalty =
      request.presence_penalty.value_or(default_gen_cfg->presence_penalty);
  gen_cfg->logprobs = request.logprobs;
  gen_cfg->top_logprobs = request.top_logprobs;
  gen_cfg->logit_bias =
      request.logit_bias.value_or(default_gen_cfg->logit_bias);
  gen_cfg->seed = request.seed.value_or(std::random_device{}());
  gen_cfg->max_tokens =
      request.max_tokens.value_or(default_gen_cfg->max_tokens);
  gen_cfg->stop_strs = std::move(stop_strs);
  gen_cfg->stop_token_ids = stop_token_ids_;
  gen_cfg->response_format = request.response_format.value_or(ResponseFormat());
  gen_cfg->debug_config = request.debug_config.value_or(DebugConfig());

  Result<GenerationConfig> res_gen_config =
      GenerationConfig::Validate(GenerationConfig(gen_cfg));
  if (res_gen_config.IsErr()) {
    last_error_ = res_gen_config.UnwrapErr();
    return;
  }

  Request engine_request(request_id, inputs, res_gen_config.Unwrap());

  // setup request state
  request_state_t rstate;
  rstate.model = request.model.value_or(request_id);
  rstate.streamer.reserve(gen_cfg->n);
  for (int i = 0; i < gen_cfg->n; ++i) {
    rstate.streamer.push_back(TextStreamer(tokenizer_));
  }
  request_map_[request_id] = std::move(rstate);

  engine_->AddRequest(engine_request);
  debug("Chat template initialized");
}

std::optional<mlc_llm_engine_t::response_state_t>
mlc_llm_engine_t::step_chat_completion(const std::string &request_id) {
  response_map_.insert_or_assign(request_id, response_state_t{});
  engine_->Step();
  debug("Chat template stepped");
  if (get_last_error().has_value()) {
    response_map_.erase(request_id);
    return std::nullopt;
  }

  auto resp = response_map_.at(request_id);
  // `resp.message` may be null if prefill has not been finished yet
  if (resp.message.is_null()) {
    response_map_.erase(request_id);
    return std::nullopt;
  }

  response_map_.erase(request_id);
  return resp;
}

std::vector<ChatCompletionStreamResponse>
mlc_llm_engine_t::get_response_from_stream_output(
    Array<RequestStreamOutput> delta_outputs) {
  std::vector<ChatCompletionStreamResponse> responses;
  for (const auto &delta_output : delta_outputs) {
    std::string request_id = delta_output->request_id;
    auto request_state_it = request_map_.find(request_id);
    if (request_state_it == request_map_.end())
      continue;
    request_state_t &rstate = request_state_it->second;

    // build the final usage messages
    // invariant, we can always let other messages to come first
    // then the final usage messages, as final usage is always last
    if (delta_output->request_final_usage_json_str.defined()) {
      ChatCompletionStreamResponse response;
      response.id = request_id;
      response.model = rstate.model;
      response.system_fingerprint = "";
      std::string usage_json_str =
          delta_output->request_final_usage_json_str.value();
      picojson::value usage_json;
      std::string err = picojson::parse(usage_json, usage_json_str);
      if (!err.empty()) {
        last_error_ = err;
      } else {
        response.usage = usage_json;
      }
      responses.push_back(response);
      request_map_.erase(request_state_it);
      continue;
    }
    ICHECK_NE(delta_output->group_finish_reason.size(), 0);
    ICHECK_EQ(delta_output->group_delta_token_ids.size(),
              delta_output->group_finish_reason.size());
    ICHECK_EQ(delta_output->group_delta_token_ids.size(),
              rstate.streamer.size());

    ChatCompletionStreamResponse response;
    response.id = request_id;
    response.model = rstate.model;
    response.system_fingerprint = "";

    for (size_t i = 0; i < delta_output->group_finish_reason.size(); ++i) {
      // choice
      ChatCompletionStreamResponseChoice choice;
      Optional<String> finish_reason = delta_output->group_finish_reason[i];
      if (finish_reason.defined()) {
        if (finish_reason.value() == "stop") {
          choice.finish_reason = FinishReason::stop;
        } else if (finish_reason.value() == "length") {
          choice.finish_reason = FinishReason::length;
        } else if (finish_reason.value() == "tool_calls") {
          choice.finish_reason = FinishReason::tool_calls;
        } else if (finish_reason.value() == "error") {
          choice.finish_reason = FinishReason::error;
        }
      } else {
        choice.finish_reason = std::nullopt;
      }
      choice.index = static_cast<int>(i);

      // Size of delta_output->group_delta_token_ids Array should be 1
      const IntTuple &delta_token_ids = delta_output->group_delta_token_ids[i];
      std::vector<int32_t> delta_token_ids_vec(delta_token_ids.begin(),
                                               delta_token_ids.end());
      std::string content = rstate.streamer[i]->Put(delta_token_ids_vec);
      if (finish_reason.defined()) {
        content += rstate.streamer[i]->Finish();
      }

      if (template_engine_->is_botc_token(content)) {
        mode_ = output_mode::tool_call;
      } else if (template_engine_->is_eotc_token(content)) {
        auto tool_call_json_str_opt =
            template_engine_->get_json_str_if_valid(tool_call_tokens_);
        if (tool_call_json_str_opt.has_value()) {
          // if valid JSON is made, set the string as the content
          choice.delta.content = tool_call_json_str_opt.value();
          choice.finish_reason = FinishReason::tool_calls;
        }
        // exit tool call mode no matter valid JSON is made
        mode_ = output_mode::text;
        tool_call_tokens_.clear();
      } else if (mode_ == output_mode::tool_call) {
        tool_call_tokens_.push_back(content);
        auto tool_call_json_str_opt =
            template_engine_->get_json_str_if_valid(tool_call_tokens_);
        if (tool_call_json_str_opt.has_value()) {
          // if valid JSON is made, set the string as the content
          choice.delta.content = tool_call_json_str_opt.value();
          choice.finish_reason = FinishReason::tool_calls;
          // exit tool call mode only if valid JSON is made
          mode_ = output_mode::text;
          tool_call_tokens_.clear();
        }
      } else if (!content.empty()) {
        // Process for reasoning outputs
        // TODO: consider other kinds of reasoning part distinguisher
        if (content == "<think>") {
          mode_ = output_mode::reasoning;
          choice.delta.content = ChatCompletionMessageContent();
        } else if (content == "</think>") {
          mode_ = output_mode::text;
          choice.delta.content = ChatCompletionMessageContent();
        } else
          choice.delta.content = ChatCompletionMessageContent(content);
      }

      choice.delta.role = std::string("assistant");
      if (!choice.delta.content.IsNull() || choice.finish_reason.has_value()) {
        response.choices.push_back(choice);
      }
    }

    // if it is not the usage block, choices cannot be empty
    if (!response.choices.empty()) {
      responses.push_back(response);
    }
  }
  return responses;
}

} // namespace ailoy
