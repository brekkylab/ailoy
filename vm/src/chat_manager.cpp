#include "chat_manager.hpp"

#include "file_util.hpp"
#include "string_util.hpp"

#include "module.hpp"

namespace ailoy {

std::shared_ptr<chat_manager_t>
chat_manager_t::make_from_config_file(std::filesystem::path config_file_path) {
  nlohmann::json chat_template_config =
      nlohmann::json::parse(utils::LoadBytesFromFile(config_file_path));
  auto chat_template_content =
      utils::LoadBytesFromFile(config_file_path.replace_filename(
          chat_template_config.at("template_file")));
  auto template_engine = create<chat_manager_t>(
      chat_template_content, chat_template_config.at("bos_token"),
      chat_template_config.at("eos_token"),
      chat_template_config.value("botc_token", ""),
      chat_template_config.value("eotc_token", ""));
  return template_engine;
}

const std::string
chat_manager_t::apply_chat_template(std::shared_ptr<const value_t> conversation,
                                    std::shared_ptr<const value_t> tools,
                                    const bool enable_reasoning,
                                    const bool add_generation_prompt) {
  // @jhlee TODO Different conversion passes to be applied for each model. We'll
  // consider to make configuration for them
  conversation = put_default_reasoning(conversation, "\n\n");
  conversation = melt_reasoning(conversation);
  conversation = merge_content_text(conversation);
  conversation = merge_content_json(conversation);
  conversation = melt_content_data(conversation);

  minja::chat_template_inputs inputs;
  inputs.messages = conversation->operator nlohmann::json();
  if (tools)
    inputs.tools = tools->operator nlohmann::json();
  inputs.add_generation_prompt = add_generation_prompt;
  // TODO: consider other ways of enable/disable reasoning
  //       & models without reasoning
  if (!enable_reasoning)
    inputs.extra_context = {{"enable_thinking", false}};
  minja::chat_template_options options;
  return template_->apply(inputs, options);
}

const std::optional<std::string>
chat_manager_t::get_json_str_if_valid(const std::vector<std::string> &tokens) {
  // TODO: validate whether the json has valid 'tool call' format
  std::string tool_call_string = utils::join("", tokens);
  utils::trim(tool_call_string);
  if (nlohmann::json::accept(tool_call_string))
    return tool_call_string;
  return std::nullopt;
}

std::shared_ptr<value_t>
put_default_reasoning(std::shared_ptr<const value_t> in,
                      const std::string &content) {
  auto out =
      decode(in->encode(encoding_method_t::json), encoding_method_t::json);
  for (auto message_value : *out->as<array_t>()) {
    auto message = message_value->as<map_t>();
    if (*message->at<string_t>("role") != "assistant")
      continue;
    if (message->contains("content") && !message->contains("reasoning")) {
      auto reasoning = create<array_t>();
      auto reasoning_data = create<map_t>();
      reasoning_data->insert_or_assign("type", create<string_t>("text"));
      reasoning_data->insert_or_assign("text", create<string_t>(content));
      reasoning->push_back(reasoning_data);
      message->insert_or_assign("reasoning", reasoning);
    }
  }
  return out;
}

std::shared_ptr<value_t> melt_reasoning(std::shared_ptr<const value_t> in,
                                        const std::string &bor_delimiter,
                                        const std::string &eor_delimiter) {
  auto out = create<array_t>();
  for (auto message_value : *in->as<array_t>()) {
    auto message = message_value->as<map_t>();
    auto message_out = create<map_t>();

    // Insert other fields
    for (auto [key, content_value] : *message) {
      if (key == "reasoning" || key == "content")
        continue;
      message_out->insert_or_assign(
          key, decode(content_value->encode(encoding_method_t::json),
                      encoding_method_t::json));
    }

    // Parse reasoning field
    std::string reasoning_str;
    if (message->contains("reasoning")) {
      reasoning_str = *message->at<array_t>("reasoning")
                           ->at<map_t>(0)
                           ->at<string_t>("text");
      reasoning_str = bor_delimiter + reasoning_str + eor_delimiter;
    }
    auto reasoning_data = create<map_t>();
    reasoning_data->insert_or_assign("type", create<string_t>("text"));
    reasoning_data->insert_or_assign("text", create<string_t>(reasoning_str));

    // Initialize content_field
    if (message->contains("content"))
      message_out->insert_or_assign(
          "content",
          decode(message->at("content")->encode(encoding_method_t::json),
                 encoding_method_t::json));
    else
      message_out->insert_or_assign("content", create<array_t>());

    message_out->at<array_t>("content")->insert(
        message_out->at<array_t>("content")->begin(),
        reasoning_data->as<value_t>());

    out->push_back(message_out);
  }
  return out;
}

std::shared_ptr<value_t> merge_content_text(std::shared_ptr<const value_t> in,
                                            const std::string &delimiter) {
  auto out = create<array_t>();
  for (auto message_value : *in->as<array_t>()) {
    auto message = message_value->as<map_t>();
    auto message_out = create<map_t>();
    for (auto [key, content_value] : *message) {
      if (key == "role") {
        message_out->insert_or_assign(
            key, create<string_t>(*content_value->as<string_t>()));
        continue;
      }
      std::shared_ptr<array_t> content = content_value->as<array_t>();

      // Empty or just one content -> nothing to merge
      if (content->size() < 2) {
        message_out->insert_or_assign(
            key, decode(content_value->encode(encoding_method_t::json),
                        encoding_method_t::json));
        continue;
      }

      // Check all content data is text
      bool is_content_all_text = true;
      for (auto content_data : *content) {
        if (*content_data->as<map_t>()->at<string_t>("type") != "text") {
          is_content_all_text = false;
        }
      }
      if (!is_content_all_text) {
        message_out->insert_or_assign(
            key, decode(content_value->encode(encoding_method_t::json),
                        encoding_method_t::json));
        continue;
      }
      std::string content_str;
      for (auto content_data : *content) {
        if (!content_str.empty())
          content_str += delimiter;
        content_str += *content_data->as<map_t>()->at<string_t>("text");
      }
      auto content_arr_out = create<array_t>();
      auto content_out = create<map_t>();
      content_out->insert_or_assign("type", create<string_t>("text"));
      content_out->insert_or_assign("text", create<string_t>(content_str));
      content_arr_out->push_back(content_out);
      message_out->insert_or_assign(key, content_arr_out);
    }
    out->push_back(message_out);
  }
  return out;
}

std::shared_ptr<value_t> merge_content_json(std::shared_ptr<const value_t> in) {
  auto out = create<array_t>();
  for (auto message_value : *in->as<array_t>()) {
    auto message = message_value->as<map_t>();
    auto message_out = create<map_t>();
    for (auto [key, content_value] : *message) {
      if (key == "role") {
        message_out->insert_or_assign(
            key, create<string_t>(*content_value->as<string_t>()));
        continue;
      }
      std::shared_ptr<array_t> content = content_value->as<array_t>();

      // Empty -> nothing to merge
      if (content->empty()) {
        message_out->insert_or_assign(
            key, decode(content_value->encode(encoding_method_t::json),
                        encoding_method_t::json));
        continue;
      }

      // Check all content data is json
      bool is_content_all_json = true;
      for (auto content_data : *content) {
        if (*content_data->as<map_t>()->at<string_t>("type") != "json") {
          is_content_all_json = false;
        }
      }
      if (!is_content_all_json) {
        message_out->insert_or_assign(
            key, decode(content_value->encode(encoding_method_t::json),
                        encoding_method_t::json));
        continue;
      }

      // Merge json
      auto content_json = create<array_t>();
      for (auto content_data : *content) {
        if (content_data->is_type_of<array_t>()) {
          // If array -> merge array
          for (auto content_data_elem : *content_data->as<array_t>())
            content_json->push_back(
                decode(content_data_elem->as<map_t>()->at("json")->encode(
                           encoding_method_t::json),
                       encoding_method_t::json));
        } else
          // If not array -> push back to merged content
          content_json->push_back(
              decode(content_data->as<map_t>()->at("json")->encode(
                         encoding_method_t::json),
                     encoding_method_t::json));
      }
      auto content_arr_out = create<array_t>();
      auto content_out = create<map_t>();
      content_out->insert_or_assign("type", create<string_t>("json"));
      content_out->insert_or_assign("json", content_json);
      content_arr_out->push_back(content_out);
      message_out->insert_or_assign(key, content_arr_out);
    }
    out->push_back(message_out);
  }
  return out;
}

std::shared_ptr<value_t> melt_content_data(std::shared_ptr<const value_t> in) {
  auto out = create<array_t>();
  for (auto message_value : *in->as<array_t>()) {
    auto message = message_value->as<map_t>();
    auto message_out = create<map_t>();
    for (auto [key, content_value] : *message) {
      if (key == "role") {
        message_out->insert_or_assign(
            key, create<string_t>(*content_value->as<string_t>()));
        continue;
      }
      std::shared_ptr<array_t> content = content_value->as<array_t>();
      std::shared_ptr<value_t> content_out;
      if (content->size() != 1)
        throw exception("content length != 1");
      auto content_data = content->at<map_t>(0);
      auto content_type = content_data->at<string_t>("type");
      if (*content_type == "text") {
        content_out = create<string_t>(*content_data->at<string_t>("text"));
      } else if (*content_type == "json") {
        content_out =
            decode(content_data->at("json")->encode(encoding_method_t::json),
                   encoding_method_t::json);
      } else {
        throw exception("Unknown content data type");
      }
      message_out->insert_or_assign(key, content_out);
    }
    out->push_back(message_out);
  }
  return out;
}

} // namespace ailoy
