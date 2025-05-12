#include "chat_template_engine.hpp"

#include "../file_util.hpp"
#include "../string_util.hpp"

#include "module.hpp"

namespace ailoy {

std::shared_ptr<chat_template_engine_t>
chat_template_engine_t::make_from_config_file(
    std::filesystem::path config_file_path) {
  nlohmann::json chat_template_config =
      nlohmann::json::parse(utils::LoadBytesFromFile(config_file_path));
  auto chat_template_content =
      utils::LoadBytesFromFile(config_file_path.replace_filename(
          chat_template_config.at("template_file")));
  auto template_engine = create<chat_template_engine_t>(
      chat_template_content, chat_template_config.at("bos_token"),
      chat_template_config.at("eos_token"),
      chat_template_config.value("botc_token", ""),
      chat_template_config.value("eotc_token", ""));
  return template_engine;
}

const std::string chat_template_engine_t::apply_chat_template(
    const nlohmann::json &messages, const nlohmann::json &tools,
    const bool reasoning, const bool add_generation_prompt) {
  minja::chat_template_inputs inputs;
  inputs.messages = messages;
  if (!tools.is_null())
    inputs.tools = tools;
  inputs.add_generation_prompt = add_generation_prompt;
  // TODO: consider other ways of enable/disable reasoning
  //       & models without reasoning
  if (!reasoning)
    inputs.extra_context = {{"enable_thinking", false}};
  minja::chat_template_options options;
  return template_->apply(inputs, options);
}

const std::optional<std::string> chat_template_engine_t::get_json_str_if_valid(
    const std::vector<std::string> &tokens) {
  // TODO: validate whether the json has valid 'tool call' format
  std::string tool_call_string = utils::join("", tokens);
  utils::trim(tool_call_string);
  if (nlohmann::json::accept(tool_call_string))
    return tool_call_string;
  return std::nullopt;
}

} // namespace ailoy
