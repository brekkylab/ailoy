#include "language_model.hpp"

#include <random>

#include <dlpack/dlpack.h>
#include <nlohmann/json.hpp>
#include <xgrammar/xgrammar.h>

#include "../chat_manager.hpp"
#include "../file_util.hpp"
#include "model_cache.hpp"
#include "tokenizer.hpp"
#include "tvm_model.hpp"

#include <iostream>

using namespace tvm;
using namespace tvm::runtime;

namespace ailoy {

double random_float(double min, double max) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(min, max);
  return dis(gen);
}

constexpr size_t page_size = 16;

struct kv_cache_t {
public:
  kv_cache_t(std::shared_ptr<ailoy::tvm_model_t> engine) {
    auto fn = engine->get_vm_function("create_tir_paged_kv_cache");
    if (!fn.defined())
      throw exception("create_tir_paged_kv_cache not defined");
    kv_cache_ = fn(
        IntTuple{1}, // max_num_sequence
        IntTuple{engine->get_metadata()["context_window_size"].operator int()},
        IntTuple{engine->get_metadata()["prefill_chunk_size"].operator int()},
        IntTuple{page_size}, // page size
        IntTuple{engine->get_metadata()["sliding_window_size"].operator int() !=
                 -1});
    fkv_state_clear_ = engine->get_function("vm.builtin.kv_state_clear");
    fkv_state_add_sequence_ =
        engine->get_function("vm.builtin.kv_state_add_sequence");
    fkv_state_remove_sequence_ =
        engine->get_function("vm.builtin.kv_state_remove_sequence");
    fkv_state_fork_sequence_ =
        engine->get_function("vm.builtin.kv_state_fork_sequence");
    fkv_state_begin_forward_ =
        engine->get_function("vm.builtin.kv_state_begin_forward");
    fkv_state_end_forward_ =
        engine->get_function("vm.builtin.kv_state_end_forward");
    fkv_state_popn_ = engine->get_function("vm.builtin.kv_state_popn");
    fkv_cache_get_num_available_pages_ = engine->get_function(
        "vm.builtin.attention_kv_cache_get_num_available_pages");
    fkv_cache_get_total_sequence_length_ = engine->get_function(
        "vm.builtin.attention_kv_cache_get_total_sequence_length");

    // Register sequence index
    add_sequence();
  }

  ~kv_cache_t() { remove_sequence(); }

  ObjectRef get() { return kv_cache_; }

  void clear() {
    fkv_state_clear_(kv_cache_);
    add_sequence();
  }

  void add_sequence() {
    fkv_state_add_sequence_(kv_cache_, 0 /* Sequence ID */);
  }

  void remove_sequence() {
    fkv_state_remove_sequence_(kv_cache_, 0 /* Sequence ID */);
  }

  void begin_forward(size_t sequence_length) {
    fkv_state_begin_forward_(kv_cache_, IntTuple{0 /* Sequence ID */},
                             IntTuple{static_cast<int32_t>(sequence_length)});
  }

  void end_forward() { fkv_state_end_forward_(kv_cache_); }

  void popn(size_t num_tokens) {
    fkv_state_popn_(kv_cache_, 0 /* Sequence ID */, (int)(num_tokens));
  }

  int get_num_available_pages() {
    return fkv_cache_get_num_available_pages_(kv_cache_);
  }

  int get_total_sequence_length() {
    return fkv_cache_get_total_sequence_length_(kv_cache_);
  }

private:
  ObjectRef kv_cache_;

  PackedFunc fkv_state_clear_;

  PackedFunc fkv_state_add_sequence_;

  PackedFunc fkv_state_fork_sequence_;

  PackedFunc fkv_state_remove_sequence_;

  PackedFunc fkv_state_begin_forward_;

  PackedFunc fkv_state_end_forward_;

  PackedFunc fkv_state_popn_;

  PackedFunc fkv_cache_get_num_available_pages_;

  PackedFunc fkv_cache_get_total_sequence_length_;
};

struct tokenizer_info_t {
  tokenizer_info_t(xgrammar::TokenizerInfo &&inner_)
      : inner(std::move(inner_)) {}

  xgrammar::TokenizerInfo inner;
};

struct grammar_t {
  grammar_t(xgrammar::CompiledGrammar &&inner_) : inner(std::move(inner_)) {}

  xgrammar::CompiledGrammar inner;
};

struct grammar_matcher_t {
  grammar_matcher_t(xgrammar::GrammarMatcher &&inner_)
      : inner(std::move(inner_)) {}

  xgrammar::GrammarMatcher inner;
};

tvm_language_model_t::stream_mode_t::stream_mode_t(
    tvm_language_model_t *model, const std::string &open_indicator_,
    const std::string &close_indicator_) {
  open_indicator = model->tokenizer_->encode(open_indicator_);
  close_indicator = model->tokenizer_->encode(close_indicator_);
}

bool tvm_language_model_t::stream_mode_t::check_indecator(
    const std::string &indicator_type,
    const std::vector<int32_t> &history) const {
  const auto &indicator =
      indicator_type == "open" ? open_indicator : close_indicator;
  if (history.size() < indicator.size())
    return false;
  auto it1 = history.rbegin();
  auto it2 = indicator.rbegin();
  while (it1 != history.rend() && it2 != indicator.rend()) {
    if (*it1 != *it2)
      return false;
    ++it1;
    ++it2;
  }
  return true;
}

tvm_language_model_t::tvm_language_model_t(const std::string &model,
                                           const std::string &quantization,
                                           DLDevice device) {
  model_ = create<tvm_model_t>(model, quantization, device);
  template_engine_ = chat_manager_t::make_from_config_file(
      model_->get_model_path() / "chat-template-config.json");
  tokenizer_ = create<tokenizer_t>(model_->get_model_path() / "tokenizer.json");
  kv_cache_ = create<kv_cache_t>(model_);
  config = config_t{.temperature = model_->get_mlc_chat_config()["temperature"],
                    .top_p = model_->get_mlc_chat_config()["top_p"]};
  default_config_ = config;
  current_stream_mode_ = "output_text";

  // Initiating tokenizer info
  {
    // 1. vocab.json → encoded_vocab
    std::vector<std::string> vocabs(tokenizer_->get_vocab_size());
    for (int32_t i = 0; i < vocabs.size(); i++)
      vocabs[i] = tokenizer_->token_id_to_str(i);

    // 2. tokenizer.json → backend_str
    std::string backend_str =
        utils::LoadBytesFromFile(model_->get_model_path() / "tokenizer.json");

    // 3. Create tokenizer info
    tokenizer_info_ =
        create<tokenizer_info_t>(std::move(xgrammar::TokenizerInfo(vocabs)));
  }

  // Add default modes
  stream_modes_.insert_or_assign("output_text", stream_mode_t(this, "", ""));
  stream_modes_.insert_or_assign("reasoning",
                                 stream_mode_t(this, "<think>", "</think>"));
  stream_modes_.insert_or_assign(
      "tool_call", stream_mode_t(this, template_engine_->get_botc_token(),
                                 template_engine_->get_eotc_token()));

  // Packed functions
  fembed_ = model_->get_vm_function("embed");
  if (!fembed_.defined())
    throw exception("Cannot find embed function");
  fprefill_ = model_->get_vm_function("prefill");
  if (!fprefill_.defined())
    throw exception("Cannot find embed function");
  fdecode_ = model_->get_vm_function("decode");
  if (!fdecode_.defined())
    throw exception("Cannot find embed function");
  fapply_bitmask_inplace_ =
      model_->get_vm_function("apply_bitmask_inplace", true);
  if (!fapply_bitmask_inplace_.defined())
    throw exception("Cannot find embed function");
  fsample_top_p_from_logits_ =
      model_->get_function("vm.builtin.sample_top_p_from_logits");
  if (!fsample_top_p_from_logits_.defined())
    throw exception("Cannot find embed function");
}

void tvm_language_model_t::clear() {
  kv_cache_->clear();
  history_.clear();
}

std::string tvm_language_model_t::apply_chat_template(
    std::shared_ptr<const value_t> conversation,
    std::shared_ptr<const value_t> tools, bool enable_reasoning,
    bool add_generation_prompt) const {
  return template_engine_->apply_chat_template(
      conversation, tools, enable_reasoning, add_generation_prompt);
}

bool tvm_language_model_t::is_bor(const std::string &tok) const {
  return tok == "<think>";
}

bool tvm_language_model_t::is_bor(int32_t tok) const {
  return is_bor(tokenizer_->token_id_to_str(tok));
}

bool tvm_language_model_t::is_eor(const std::string &tok) const {
  return tok == "</think>";
}

bool tvm_language_model_t::is_eor(int32_t tok) const {
  return is_eor(tokenizer_->token_id_to_str(tok));
}

bool tvm_language_model_t::is_bos(const std::string &tok) const {
  return template_engine_->is_bos_token(tok);
}

bool tvm_language_model_t::is_eos(const std::string &tok) const {
  return template_engine_->is_eos_token(tok);
}

bool tvm_language_model_t::is_botc(const std::string &tok) const {
  return template_engine_->is_botc_token(tok);
}

bool tvm_language_model_t::is_botc(int32_t tok) const {
  return is_botc(tokenizer_->token_id_to_str(tok));
}

bool tvm_language_model_t::is_eotc(const std::string &tok) const {
  return template_engine_->is_eotc_token(tok);
}

bool tvm_language_model_t::is_eotc(int32_t tok) const {
  return is_eotc(tokenizer_->token_id_to_str(tok));
}

std::vector<int32_t>
tvm_language_model_t::tokenize(const std::string &prompt) const {
  return std::move(tokenizer_->encode(prompt));
}

int32_t tvm_language_model_t::prefill(const std::vector<int32_t> &tokens) {
  if (tokens.empty())
    throw exception("Token must not be empty");

  // Make sure that kv-cache and history is sync
  if (kv_cache_->get_total_sequence_length() != history_.size())
    this->clear();

  // The longest common prefix (LCP) between inputs & previous conversations
  size_t lcp_index = 0;
  while (lcp_index < history_.size() && lcp_index < tokens.size()) {
    if (history_[lcp_index] != tokens[lcp_index])
      break;
    ++lcp_index;
  }

  // Rewind the head of kv-cache to the LCP
  if (lcp_index < history_.size()) {
    kv_cache_->popn(history_.size() - lcp_index);
  }

  // Tokens to be added (wihout common prefixes)
  std::vector<int32_t> new_tokens(tokens.begin() + lcp_index, tokens.end());
  if (new_tokens.empty())
    return *history_.rbegin();

  // Calculate remaining space in KV cache
  if (new_tokens.size() >= kv_cache_->get_num_available_pages() * page_size) {
    throw exception<context_length_limit>();
  }

  // Chunk size to be split
  size_t prefill_chunk_size = model_->get_metadata()["prefill_chunk_size"];
  for (size_t i = 0; i < new_tokens.size(); i += prefill_chunk_size) {
    // Prefill i to j
    size_t j = (i + prefill_chunk_size < new_tokens.size())
                   ? i + prefill_chunk_size
                   : new_tokens.size();
    int32_t length = j - i;
    DLDataType I32 = DLDataType{.code = kDLInt, .bits = 32, .lanes = 1};

    // Input NDArray
    NDArray input = NDArray::Empty({length}, I32, model_->get_device());
    input.CopyFromBytes(new_tokens.data(), new_tokens.size() * sizeof(int32_t));

    // Embedding of the input
    NDArray embedding =
        model_->get_vm_function("embed")(input, model_->get_params());
    NDArray embedding_reshaped = embedding.CreateView(
        ShapeTuple{1, embedding->shape[0], embedding->shape[1]},
        embedding.DataType());

    // Forward prefill
    kv_cache_->begin_forward(length);
    fprefill_(embedding_reshaped, kv_cache_->get(), model_->get_params());
    kv_cache_->end_forward();
  }

  // Update history
  history_ = tokens;

  // We reset the stream mode, since we consider `prefill` as the begin of a new
  // inference,
  current_stream_mode_ = "output_text";

  // Returns the last token, so that the `decode` can start with it.
  return *new_tokens.rbegin();
}

int32_t tvm_language_model_t::decode(int32_t last_token) {
  DLDataType U32 = DLDataType{.code = kDLUInt, .bits = 32, .lanes = 1};
  DLDataType I32 = DLDataType{.code = kDLInt, .bits = 32, .lanes = 1};
  DLDataType F32 = DLDataType{.code = kDLFloat, .bits = 32, .lanes = 1};

  // Calculate remaining space in KV cache
  if (kv_cache_->get_num_available_pages() < 1) {
    throw exception<context_length_limit>();
  }

  // Input NDArray
  NDArray token_ids = NDArray::Empty({1}, I32, model_->get_device());
  token_ids.CopyFromBytes(&last_token, sizeof(int32_t));

  // Embed
  NDArray embed =
      model_->get_vm_function("embed")(token_ids, model_->get_params());
  tvm::runtime::NDArray embed_reshaped = embed.CreateView(
      tvm::ShapeTuple{1, 1, embed->shape[1]}, embed.DataType());

  // In decode, the sequence length of new tokens are always 1
  kv_cache_->begin_forward(1);
  // Forward decode (output: [logits, kv_caches])
  ObjectRef output =
      fdecode_(embed_reshaped, kv_cache_->get(), model_->get_params());
  kv_cache_->end_forward();

  // Extract logits (1 x seq_len x vocab_size)
  NDArray logits = Downcast<Array<NDArray>>(output)[0];
  const int seq_len = logits.Shape()[1];
  auto vocab_size = logits.Shape()[2];

  // Sampling
  if (get_current_grammar_matcher()) {
    auto &matcher = get_current_grammar_matcher()->inner;

    // Create bitmask
    int bitmask_len = (vocab_size + 31) / 32;
    NDArray bitmask_cpu = NDArray::Empty(
        {bitmask_len}, I32, DLDevice{.device_type = kDLCPU, .device_id = 0});

    // Apply matcher
    matcher.FillNextTokenBitmask(&bitmask_cpu.ToDLPack()->dl_tensor);

    // Copy bitmask to GPU
    NDArray bitmask = NDArray::Empty({bitmask_len}, I32, model_->get_device());
    bitmask.CopyFrom(bitmask_cpu);

    // Create seq_id
    NDArray seq_ids_cpu = NDArray::Empty(
        {1}, I32, DLDevice{.device_type = kDLCPU, .device_id = 0});
    auto seq_ids_cpu_data = static_cast<int32_t *>(seq_ids_cpu->data);
    seq_ids_cpu_data[0] = 0;
    NDArray seq_ids = NDArray::Empty({1}, I32, model_->get_device());
    seq_ids.CopyFrom(seq_ids_cpu);

    // Apply bitmask to logits
    fapply_bitmask_inplace_(logits.CreateView({1, vocab_size}, F32),  // logits
                            seq_ids,                                  // seq_ids
                            bitmask.CreateView({1, bitmask_len}, I32) // bitmask
    );
  }

  // Sample token from logits
  int32_t sampled_token = fsample_top_p_from_logits_(
      logits, config.temperature, config.top_p, random_float(0.0, 1.0));

  // Register it to history
  history_.push_back(sampled_token);

  // Update matcher
  if (get_current_grammar_matcher()) {
    auto &matcher = get_current_grammar_matcher()->inner;
    matcher.AcceptToken(sampled_token);
    if (matcher.IsTerminated())
      stream_modes_.at(current_stream_mode_).matcher = nullptr;
  }

  // Update streaming mode
  if (current_stream_mode_ == "output_text") {
    for (auto [name, mode] : stream_modes_) {
      if (name == "output_text")
        continue;
      if (!mode.check_indecator("open", history_))
        continue;
      if (mode.grammar) {
        auto matcher =
            xgrammar::GrammarMatcher(mode.grammar->inner, mode.close_indicator);
        mode.matcher = create<grammar_matcher_t>(std::move(matcher));
      }
      current_stream_mode_ = name;
      break;
    }
  } else {
    const auto &current_mode = stream_modes_.at(current_stream_mode_);
    if (current_mode.check_indecator("close", history_)) {
      auto name = current_stream_mode_;
      if (stream_modes_.at(name).grammar)
        stream_modes_.at(name).matcher = nullptr;
      current_stream_mode_ = "output_text";
    }
  }

  return sampled_token;
}

std::optional<std::string> tvm_language_model_t::detokenize(int32_t token) {
  output_stream_.push_back(token);
  auto str = tokenizer_->decode(output_stream_, false);
  if (str.ends_with("�"))
    return std::nullopt;
  else {
    output_stream_.clear();
    return str;
  }
}

const std::string &tvm_language_model_t::get_current_stream_mode() const {
  return current_stream_mode_;
}

const tvm_language_model_t::stream_mode_t &
tvm_language_model_t::get_stream_mode(std::string mode_name) const {
  return stream_modes_.at(mode_name);
}

void tvm_language_model_t::add_stream_mode(std::string mode_name,
                                           const std::string &open_indicator,
                                           const std::string &close_indicator) {
  stream_modes_.insert_or_assign(
      mode_name, stream_mode_t(this, open_indicator, close_indicator));
}

void tvm_language_model_t::remove_stream_mode(std::string mode_name) {
  stream_modes_.erase(mode_name);
}

std::shared_ptr<grammar_matcher_t>
tvm_language_model_t::get_current_grammar_matcher() {
  auto current_stream_mode = stream_modes_.at(current_stream_mode_);
  return current_stream_mode.matcher;
}

void tvm_language_model_t::set_builtin_grammar(
    const std::string &mode_name, const std::string &grammar_type) {
  if (grammar_type == "json") {
    auto grammar = xgrammar::GrammarCompiler(tokenizer_info_->inner)
                       .CompileBuiltinJSONGrammar();
    stream_modes_.at(mode_name).grammar = create<grammar_t>(std::move(grammar));
  } else {
    throw exception("Unknown grammer type");
  }
}

void tvm_language_model_t::set_json_schema_grammar(
    const std::string &mode_name, const std::string &json_schema) {
  auto grammar = xgrammar::GrammarCompiler(tokenizer_info_->inner)
                     .CompileJSONSchema(json_schema);
  stream_modes_.at(mode_name).grammar = create<grammar_t>(std::move(grammar));
}

void tvm_language_model_t::set_regex_grammar(const std::string &mode_name,
                                             const std::string &regex) {
  auto grammar =
      xgrammar::GrammarCompiler(tokenizer_info_->inner).CompileRegex(regex);
  stream_modes_.at(mode_name).grammar = create<grammar_t>(std::move(grammar));
}

void tvm_language_model_t::set_ebnf_grammar(const std::string &mode_name,
                                            const std::string &ebnf) {
  auto grammar = xgrammar::GrammarCompiler(tokenizer_info_->inner)
                     .CompileGrammar(xgrammar::Grammar::FromEBNF(ebnf));
  stream_modes_.at(mode_name).grammar = create<grammar_t>(std::move(grammar));
}

void tvm_language_model_t::reset_grammar(const std::string &mode_name) {
  if (!stream_modes_.contains(mode_name))
    return;
  stream_modes_.at(mode_name).grammar = nullptr;
  stream_modes_.at(mode_name).matcher = nullptr;
}

/**
 * Internal function to validate language model messages
 */
std::optional<error_output_t>
_validate_language_model_input(const std::string &context,
                               std::shared_ptr<const map_t> input_map) {
  if (!input_map->contains("messages"))
    return error_output_t(range_error(context, "messages"));
  if (!input_map->at("messages")->is_type_of<array_t>())
    return error_output_t(type_error(context, "messages", "array_t",
                                     input_map->at("messages")->get_type()));
  auto messages = input_map->at("messages")->as<array_t>();
  for (auto msg_val : *messages) {
    if (!msg_val->is_type_of<map_t>())
      return error_output_t(
          type_error(context, "messages.*", "map_t", msg_val->get_type()));
    auto msg = msg_val->as<map_t>();
    if (!msg->contains("role"))
      return error_output_t(range_error(context, "role"));
    if (!msg->at("role")->is_type_of<string_t>())
      return error_output_t(
          type_error(context, "role", "string_t", msg->at("role")->get_type()));
    if (*msg->at<string_t>("role") != "system" &&
        *msg->at<string_t>("role") != "user" &&
        *msg->at<string_t>("role") != "assistant" &&
        *msg->at<string_t>("role") != "tool")
      return error_output_t(value_error(context, "role",
                                        "system | user | assistant | tool",
                                        *msg->at<string_t>("role")));
    for (auto key : {"reasoning", "content", "tool_calls"}) {
      if (!msg->contains(key))
        continue;
      if (msg->at(key)->is_type_of<null_t>()) {
        msg->erase(key);
        continue;
      }
      if (!msg->at(key)->is_type_of<array_t>()) {
        return error_output_t(
            type_error(context, key, "array", msg->at(key)->get_type()));
      }
      for (size_t i = 0; i < msg->at<array_t>(key)->size(); i++) {
        auto data_val = msg->at<array_t>(key)->at(i);
        if (!data_val->is_type_of<map_t>()) {
          return error_output_t(type_error(context,
                                           std::format("{}/{}", key, i),
                                           "map_t", msg->at(key)->get_type()));
        }
        auto data = data_val->as<map_t>();
        if (!data->contains("type"))
          return error_output_t(
              std::format("Field not exists: {}/{}/type", key, i));
        if (!data->at("type")->is_type_of<string_t>())
          return error_output_t(type_error(context, "type", "string",
                                           data->at("type")->get_type()));
        if (*data->at<string_t>("type") == "text") {
          if (!data->contains("text"))
            return error_output_t(
                std::format("Field not exists: {}/{}/text", key, i));
          if (!data->at("text")->is_type_of<string_t>())
            return error_output_t(type_error(context, "type", "string",
                                             data->at("type")->get_type()));
        } else if (*data->at<string_t>("type") == "function") {
          if (!data->contains("function"))
            return error_output_t(
                std::format("Field not exists: {}/{}/function", key, i));
        }
      }
    }
  }
  if (input_map->contains("temperature"))
    if (!input_map->at("temperature")->is_type_of<double_t>())
      return error_output_t(
          type_error(context, "temperature", "double_t",
                     input_map->at("temperature")->get_type()));

  if (input_map->contains("top_p"))
    if (!input_map->at("top_p")->is_type_of<double_t>())
      return error_output_t(type_error(context, "top_p", "double_t",
                                       input_map->at("top_p")->get_type()));

  return std::nullopt;
}

/**
 * Create component for tvm_language_model
 */
component_or_error_t
create_tvm_language_model_component(std::shared_ptr<const value_t> inputs) {
  if (!inputs->is_type_of<map_t>())
    return error_output_t(type_error("TVM Language Model: create", "inputs",
                                     "map_t", inputs->get_type()));

  auto input_map = inputs->as<map_t>();

  // Parse model name
  if (!input_map->contains("model"))
    return error_output_t(range_error("TVM Language Model: create", "model"));
  if (!input_map->at("model")->is_type_of<string_t>())
    return error_output_t(type_error("TVM Language Model: create", "model",
                                     "string_t",
                                     input_map->at("model")->get_type()));
  std::string model = *input_map->at<string_t>("model");

  // Parse quantization(optional)
  std::string quantization;
  if (input_map->contains("quantization")) {
    if (input_map->at("quantization")->is_type_of<string_t>())
      quantization = *input_map->at<string_t>("quantization");
    else
      return error_output_t(
          type_error("TVM Language Model: create", "quantization", "string_t",
                     input_map->at("quantization")->get_type()));
  } else
    quantization = "q4f16_1";

  // Parse device
  int32_t device_id;
  if (input_map->contains("device")) {
    if (input_map->at("device")->is_type_of<int_t>())
      device_id = *input_map->at<int_t>("device");
    else if (input_map->at("device")->is_type_of<uint_t>())
      device_id = *input_map->at<uint_t>("device");
    else
      return error_output_t(type_error("TVM Language Model: create", "device",
                                       "int_t | uint_t",
                                       input_map->at("device")->get_type()));
  } else
    device_id = 0;

  auto device_opt = get_tvm_device(device_id);
  if (!device_opt.has_value())
    return error_output_t(
        runtime_error("No supported device is detected for your system."));
  auto device = device_opt.value();

  // Internal model
  std::shared_ptr<tvm_language_model_t> tvm_language_model;
  try {
    // Initialize model
    tvm_language_model =
        create<tvm_language_model_t>(model, quantization, device);
  } catch (const exception_t<runtime_error> &e) {
    return error_output_t(e.what());
  }

  //
  // Define inference op
  //
  auto infer = ailoy::create<iterative_method_operator_t>(
      //
      // Init function (first call)
      //
      [](std::shared_ptr<component_t> component,
         std::shared_ptr<const value_t> inputs) -> value_or_error_t {
        auto model = component->get_obj("model")->as<tvm_language_model_t>();

        if (!inputs->is_type_of<map_t>())
          return error_output_t(type_error("TVM Language Model: infer",
                                           "inputs", "map_t",
                                           inputs->get_type()));
        auto input_map = inputs->as<map_t>();

        // Get input messages
        auto error = _validate_language_model_input("TVM Language Model: infer",
                                                    input_map);
        if (error.has_value())
          return error.value();
        auto messages = input_map->at<array_t>("messages");

        // Get tools (optional)
        std::shared_ptr<const value_t> tools;
        if (input_map->contains("tools")) {
          if (input_map->at("tools")->is_type_of<string_t>()) {
            const std::string tools_str = *input_map->at<string_t>("tools");
            if (!nlohmann::json::accept(tools_str))
              return error_output_t(
                  value_error("[TVM Language Model: apply_chat_template] "
                              "Invalid JSON string in tools: " +
                              tools_str));
            tools = decode(tools_str, encoding_method_t::json);
          } else if (input_map->at("tools")->is_type_of<array_t>()) {
            tools = input_map->at<array_t>("tools");
          } else {
            return error_output_t(type_error(
                "TVM Language Model: apply_chat_template", "tools",
                "string_t | array_t", input_map->at("tools")->get_type()));
          }
        }

        // Get reasoning (optional)
        bool enable_reasoning = false;
        if (input_map->contains("enable_reasoning") &&
            !input_map->at("enable_reasoning")->is_type_of<null_t>())
          enable_reasoning = *input_map->at<bool_t>("enable_reasoning");

        // Get ignore_reasoning (optional)
        bool ignore_reasoning_messages = false;
        if (input_map->contains("ignore_reasoning_messages") &&
            !input_map->at("ignore_reasoning_messages")->is_type_of<null_t>())
          ignore_reasoning_messages =
              *input_map->at<bool_t>("ignore_reasoning_messages");

        // Get temperature (optional)
        model->config.temperature = model->get_default_config().temperature;
        if (input_map->contains("temperature") &&
            !input_map->at("temperature")->is_type_of<null_t>())
          model->config.temperature = *input_map->at<double_t>("temperature");

        // Get top-p (optional)
        model->config.top_p = model->get_default_config().top_p;
        if (input_map->contains("top_p") &&
            !input_map->at("top_p")->is_type_of<null_t>())
          model->config.top_p = *input_map->at<double_t>("top_p");

        // Apply chat template on messages
        auto prompt =
            model->apply_chat_template(messages, tools, enable_reasoning);

        // Tokenize
        auto tokens = model->tokenize(prompt);

        // Prefill
        int32_t current_token;
        std::string finish_reason;
        try {
          current_token = model->prefill(tokens);
          finish_reason = "stop"; // Assumes default finish reason here
        } catch (const exception_t<context_length_limit> &e) {
          current_token = -1;
          finish_reason = "length";
        }

        auto rv = create<map_t>();
        rv->insert_or_assign("current_token", create<int_t>(current_token));
        rv->insert_or_assign("finish_reason", create<string_t>(finish_reason));
        rv->insert_or_assign("ignore_reasoning_messages",
                             create<ailoy::bool_t>(ignore_reasoning_messages));
        return rv;
      },
      //
      // Step function
      //
      [](std::shared_ptr<component_t> component,
         std::shared_ptr<value_t> state) -> output_t {
        // Get saved values & objects
        auto model = component->get_obj("model")->as<tvm_language_model_t>();
        auto current_token = state->as<map_t>()->at<int_t>("current_token");
        auto finish_reason = state->as<map_t>()->at<string_t>("finish_reason");
        bool ignore_reasoning_messages =
            *state->as<map_t>()->at<bool_t>("ignore_reasoning_messages");

        // If current_token == -1: Raised in prefill
        if (current_token < 0) {
          if (finish_reason) {
            auto resp = create<map_t>();
            resp->insert_or_assign("message", create<map_t>());
            resp->insert_or_assign("finish_reason",
                                   create<string_t>(*finish_reason));
            return ok_output_t(resp, true);
          }
          return error_output_t("Unknown error");
        }

        // Repeat steps until valid output comes or it finished.
        auto resp = create<map_t>();
        auto delta = create<map_t>();
        resp->insert_or_assign("message", delta);
        auto insert_to_delta = [&](const std::string &key,
                                   const std::string &datatype,
                                   std::shared_ptr<value_t> data) {
          delta->insert_or_assign(key, create<array_t>());
          auto delta_data = create<map_t>();
          delta_data->insert_or_assign("type", create<string_t>(datatype));
          delta_data->insert_or_assign(datatype, data);
          delta->at<array_t>(key)->push_back(delta_data);
        };
        try {
          std::string agg_token_str;
          while (true) {
            *current_token = model->decode(*current_token);
            auto current_stream_mode = model->get_current_stream_mode();
            auto opt = model->detokenize(*current_token);
            if (!opt.has_value())
              continue;
            std::string current_token_str = opt.value();
            if (current_stream_mode == "tool_call") { // Tool call mode
              if (model->is_botc(current_token_str))
                // BOTC token generated
                state->as<map_t>()->insert_or_assign(
                    "finish_reason", create<string_t>("tool_calls"));
              else
                agg_token_str += current_token_str;
              continue;
            } else if (current_stream_mode == "reasoning") { // Reasoning mode
              if (ignore_reasoning_messages)
                continue;
              if (model->is_bor(current_token_str))
                // BOR token generated
                continue;
              insert_to_delta("reasoning", "text",
                              create<string_t>(current_token_str));
              return ok_output_t(resp, false);
            } else { // Default mode
              if (model->is_eos(current_token_str)) {
                // EOS token generated
                resp->insert_or_assign("finish_reason", finish_reason);
                return ok_output_t(resp, true);
              } else if (model->is_eotc(current_token_str)) {
                // EOTC(</tool_call>) token generated
                try {
                  // Try to decode
                  auto decoded = decode(agg_token_str, encoding_method_t::json);
                  insert_to_delta("tool_calls", "function", decoded);
                } catch (const std::exception &e) {
                  // Decode fail -> invalid_tool_call
                  insert_to_delta(
                      "error", "text",
                      create<string_t>("Invalid tool_call created"));
                  resp->insert_or_assign("finish_reason",
                                         create<string_t>("invalid_tool_call"));
                  return ok_output_t(resp, true);
                }
                agg_token_str = "";
                return ok_output_t(resp, false);
              } else if (model->is_eor(current_token_str)) {
                // EOR(</think>) token generated
                continue;
              }
              // Default case
              insert_to_delta("content", "text",
                              create<string_t>(current_token_str));
              return ok_output_t(resp, false);
            }
          }
        } catch (const exception_t<context_length_limit> &e) {
          resp->insert_or_assign("finish_reason", create<string_t>("length"));
          return ok_output_t(resp, true);
        };
      });

  //
  // Define apply_chat_template op
  //
  auto apply_chat_template = ailoy::create<instant_method_operator_t>(
      [](std::shared_ptr<component_t> component,
         std::shared_ptr<const value_t> inputs) -> value_or_error_t {
        // Validate inputs
        if (!inputs->is_type_of<map_t>())
          return error_output_t(
              type_error("TVM Language Model: apply_chat_template", "inputs",
                         "map_t", inputs->get_type()));

        auto input_map = inputs->as<map_t>();

        // Get input messages
        auto error = _validate_language_model_input(
            "TVM Language Model: apply_chat_template", input_map);
        if (error.has_value())
          return error.value();
        auto messages = input_map->at<array_t>("messages");

        // Get tools (optional)
        std::shared_ptr<const value_t> tools;
        if (input_map->contains("tools")) {
          if (input_map->at("tools")->is_type_of<string_t>()) {
            const std::string tools_str = *input_map->at<string_t>("tools");
            if (!nlohmann::json::accept(tools_str))
              return error_output_t(
                  value_error("[TVM Language Model: apply_chat_template] "
                              "Invalid JSON string in tools: " +
                              tools_str));
            tools = decode(tools_str, encoding_method_t::json);
          } else if (input_map->at("tools")->is_type_of<array_t>()) {
            tools = input_map->at<array_t>("tools");
          } else {
            return error_output_t(type_error(
                "TVM Language Model: apply_chat_template", "tools",
                "string_t | array_t", input_map->at("tools")->get_type()));
          }
        }

        // Get reasoning (optional)
        bool enable_reasoning = false;
        if (input_map->contains("enable_reasoning"))
          if (input_map->at("enable_reasoning")->is_type_of<bool_t>())
            enable_reasoning = *input_map->at<bool_t>("enable_reasoning");

        // Apply chat template on messages
        auto prompt =
            component->get_obj("model")
                ->as<tvm_language_model_t>()
                ->apply_chat_template(messages, tools, enable_reasoning);

        auto outputs = create<map_t>();
        outputs->insert_or_assign("prompt", create<string_t>(prompt));
        return outputs;
      });

  //
  // Define clear op
  //
  auto clear = ailoy::create<instant_method_operator_t>(
      [](std::shared_ptr<component_t> component,
         std::shared_ptr<const value_t> inputs) -> value_or_error_t {
        auto model = component->get_obj("model")->as<tvm_language_model_t>();
        model->clear();
        return create<null_t>();
      });

  // Create component
  auto ops = std::initializer_list<
      std::pair<const std::string, std::shared_ptr<method_operator_t>>>{
      {"infer", infer},
      {"apply_chat_template", apply_chat_template},
      {"clear", clear},
  };
  auto rv = create<component_t>(ops);
  rv->set_obj("model", tvm_language_model);
  return rv;
};

} // namespace ailoy
