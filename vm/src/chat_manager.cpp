#include "chat_manager.hpp"

#include "module.hpp"
#include "string_util.hpp"

namespace ailoy {

std::shared_ptr<chat_manager_t>
chat_manager_t::make_from_config_file(ailoy::fs::path_t config_file_path) {
  nlohmann::json chat_template_config = nlohmann::json::parse(
      ailoy::fs::read_file_text(config_file_path).unwrap());
  std::string chat_template_content = ailoy::fs::read_file_text(
      config_file_path / ".." / chat_template_config.at("template_file"));
  auto template_engine = create<chat_manager_t>(
      chat_template_content, chat_template_config.at("bos_token"),
      chat_template_config.at("eos_token"),
      chat_template_config.value("botc_token", ""),
      chat_template_config.value("eotc_token", ""));
  return template_engine;
}

mj_value value_to_mj_val(std::shared_ptr<const value_t> val) {
  if (!val || val->is_type_of<null_t>()) {
    return mj_value_new_none();
  } else if (val->is_type_of<map_t>()) {
    mj_value obj = mj_value_new_object();
    for (const auto &[key, value] : *val->as<map_t>()) {
      mj_value_set_string_key(&obj, key.c_str(), value_to_mj_val(value));
    }
    return obj;
  } else if (val->is_type_of<array_t>()) {
    mj_value arr = mj_value_new_list();
    for (const auto &value : *val->as<array_t>()) {
      mj_value_append(&arr, value_to_mj_val(value));
    }
    return arr;
  } else if (val->is_type_of<string_t>()) {
    return mj_value_new_string(val->as<string_t>()->c_str());
  } else if (val->is_type_of<bool_t>()) {
    return mj_value_new_bool(*val->as<bool_t>());
  } else if (val->is_type_of<bytes_t>()) {
    auto bytes = val->as<bytes_t>();
    return mj_value_new_bytes(reinterpret_cast<const char *>(bytes->data()),
                              bytes->size());
  } else if (val->is_type_of<float_t>()) {
    return mj_value_new_f32(*val->as<float_t>());
  } else if (val->is_type_of<double_t>()) {
    return mj_value_new_f64(*val->as<double_t>());
  } else if (val->is_type_of<int_t>()) {
    return mj_value_new_i32(*val->as<int_t>());
  } else if (val->is_type_of<uint_t>()) {
    return mj_value_new_u32(*val->as<uint_t>());
  } else {
    throw std::runtime_error(
        std::format("Undefined value type: {}", val->get_type()));
  }
}

const std::string
chat_manager_t::apply_chat_template(std::shared_ptr<const value_t> conversation,
                                    std::shared_ptr<const value_t> tools,
                                    const bool reasoning,
                                    const bool add_generation_prompt) {
  // @jhlee TODO Different conversion passes to be applied for each model. We'll
  // consider to make configuration for them
  conversation = remove_tool_call_id(conversation);
  conversation = put_default_reasoning(conversation, "\n\n");
  conversation = melt_reasoning(conversation);
  conversation = merge_text_data(conversation);
  conversation = melt_content_text(conversation);

  mj_value ctx = mj_value_new_object();
  mj_value_set_string_key(&ctx, "messages", value_to_mj_val(conversation));

  if (tools) {
    mj_value_set_string_key(&ctx, "tools", value_to_mj_val(tools));
  }
  mj_value_set_string_key(&ctx, "add_generation_prompt",
                          mj_value_new_bool(add_generation_prompt));
  if (!reasoning) {
    mj_value_set_string_key(&ctx, "enable_thinking", mj_value_new_bool(false));
  }

  char *rv = mj_env_render_template(minijinja_env_, "template", ctx);
  if (!rv) {
    mj_err_print();
    return "";
  }

  std::string str = rv;
  mj_str_free(rv);
  return str;
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
remove_tool_call_id(std::shared_ptr<const value_t> in) {
  auto out =
      decode(in->encode(encoding_method_t::json), encoding_method_t::json);
  for (auto message_value : *out->as<array_t>()) {
    auto message = message_value->as<map_t>();
    if (*message->at<string_t>("role") == "assistant") {
      if (!message->contains("tool_calls"))
        continue;
      auto tool_calls = message->at<array_t>("tool_calls");
      for (auto tool_call_data : *tool_calls) {
        auto tool_call = tool_call_data->as<map_t>();
        if (tool_call->contains("id"))
          tool_call->erase("id");
      }
    } else if (*message->at<string_t>("role") == "tool") {
      if (!message->contains("content"))
        continue;
      auto contents = message->at<array_t>("content");
      for (auto content : *contents) {
        auto tool_call = content->as<map_t>();
        if (tool_call->contains("tool_call_id"))
          tool_call->erase("tool_call_id");
      }
    }
  }
  return out;
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
    if ((message->contains("content") || message->contains("tool_calls")) &&
        !message->contains("reasoning")) {
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

std::shared_ptr<value_t> merge_text_data(std::shared_ptr<const value_t> in,
                                         const std::string &delimiter) {
  auto out =
      decode(in->encode(encoding_method_t::json), encoding_method_t::json);
  for (auto message_value : *out->as<array_t>()) {
    auto message = message_value->as<map_t>();
    for (std::string key : {"content", "reasoning"}) {
      if (!message->contains(key))
        continue;
      std::shared_ptr<array_t> content = message->at<array_t>(key);
      auto content_new = create<array_t>();

      // Iterate over content
      for (auto data_value : *content) {
        auto data = data_value->as<map_t>();
        if (!content_new->empty()) {
          std::shared_ptr<map_t> data_new_last =
              (*content_new->rbegin())->as<map_t>();
          if (*data_new_last->at<string_t>("type") == "text" &&
              *data->at<string_t>("type") == "text") {
            *data_new_last->at<string_t>("text") += *data->at<string_t>("text");
            continue;
          }
        }
        content_new->push_back(data);
      }
      message->insert_or_assign(key, content_new);
    }
  }
  return out;
}

std::shared_ptr<value_t> melt_content_text(std::shared_ptr<const value_t> in) {
  auto out =
      decode(in->encode(encoding_method_t::json), encoding_method_t::json);
  for (auto message_value : *out->as<array_t>()) {
    auto message = message_value->as<map_t>();
    if (!message->contains("content"))
      continue;
    if (!message->at("content")->is_type_of<array_t>())
      continue;
    auto content = message->at<array_t>("content");
    if (content->size() != 1)
      throw exception("content length != 1");
    if (!content->at(0)->is_type_of<map_t>())
      continue;
    if (!content->at<map_t>(0)->contains("type"))
      continue;
    auto content_data = content->at<map_t>(0);
    auto content_type = content_data->at<string_t>("type");
    if (*content_type == "text") {
      if (!content->at<map_t>(0)->contains("text"))
        continue;
      message->insert_or_assign(
          "content", create<string_t>(*content_data->at<string_t>("text")));
    }
  }
  return out;
}

} // namespace ailoy
