#include <gtest/gtest.h>

#include "logging.hpp"
#include "tvm/language_model.hpp"
#include "value.hpp"

#include <iostream>

std::shared_ptr<ailoy::component_t> get_model() {
  static std::shared_ptr<ailoy::component_t> model;
  if (!model) {
    auto in = ailoy::create<ailoy::map_t>();
    in->insert_or_assign("model",
                         ailoy::create<ailoy::string_t>("Qwen/Qwen3-0.6B"));
    auto model_opt = ailoy::create_tvm_language_model_component(in);
    if (model_opt.index() != 0)
      return nullptr;
    model = std::get<0>(model_opt);
  }
  return model;
}

std::string infer(std::shared_ptr<ailoy::component_t> model,
                  std::shared_ptr<ailoy::value_t> messages,
                  std::shared_ptr<ailoy::value_t> tools = nullptr,
                  bool enable_reasoning = false,
                  bool ignore_reasoning_messages = false) {
  auto in = ailoy::create<ailoy::map_t>();
  in->insert_or_assign("messages", messages);
  if (tools)
    in->insert_or_assign("tools", tools);
  in->insert_or_assign("enable_reasoning",
                       ailoy::create<ailoy::bool_t>(enable_reasoning));
  in->insert_or_assign("ignore_reasoning_messages",
                       ailoy::create<ailoy::bool_t>(ignore_reasoning_messages));
  in->insert_or_assign("temperature", ailoy::create<ailoy::double_t>(0.));
  in->insert_or_assign("top_p", ailoy::create<ailoy::double_t>(0.));
  auto init_out_opt = model->get_operator("infer")->initialize(in);
  if (init_out_opt.has_value())
    return init_out_opt.value().reason;
  std::string agg_out = "";
  while (true) {
    ailoy::output_t out_opt = model->get_operator("infer")->step();
    if (out_opt.index() != 0)
      throw ailoy::exception(std::get<1>(out_opt).reason);
    auto out = std::get<0>(out_opt);
    auto resp = out.val->as<ailoy::map_t>();
    if (resp->contains("finish_reason"))
      break;
    if (!resp->contains("message"))
      continue;
    auto message = resp->at<ailoy::map_t>("message");
    if (message->contains("reasoning"))
      agg_out += *message->at<ailoy::array_t>("reasoning")
                      ->at<ailoy::map_t>(0)
                      ->at<ailoy::string_t>("text");
    if (message->contains("content"))
      agg_out += *message->at<ailoy::array_t>("content")
                      ->at<ailoy::map_t>(0)
                      ->at<ailoy::string_t>("text");
    if (message->contains("tool_calls"))
      agg_out += message->at<ailoy::array_t>("tool_calls")
                     ->at<ailoy::map_t>(0)
                     ->at("function")
                     ->encode(ailoy::encoding_method_t::json)
                     ->operator std::string();
    if (out.finish)
      break;
  }
  return agg_out;
}

TEST(TestTVMLanguageModel, TestSimple) {
  auto messages_str = R"([
  {
    "role": "user",
    "content": [
      {
        "type": "text",
        "text": "Introduce yourself in one sentence."
      }
    ]
  }
])";

  std::shared_ptr<ailoy::component_t> model = get_model();
  auto messages = ailoy::decode(messages_str, ailoy::encoding_method_t::json);
  auto out = infer(model, messages);
  ASSERT_EQ(
      out,
      R"(I am a language model, and I am here to assist you with language learning and other tasks.)");
}

TEST(TestTVMLanguageModel, TestMultiTurn) {
  auto messages_str1 = R"([
  {
    "role": "system",
    "content": [
      {
        "type": "text",
        "text": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
      }
    ]
  },
  {
    "role": "user",
    "content": [
      {
        "type": "text",
        "text": "Who are you? Answer it simply."
      }
    ]
  }
])";
  std::shared_ptr<ailoy::component_t> model = get_model();
  auto messages1 = ailoy::decode(messages_str1, ailoy::encoding_method_t::json);
  auto answer1 = infer(model, messages1);
  ASSERT_EQ(answer1,
            "I am Qwen, a helpful assistant created by Alibaba Cloud.");

  auto messages_str2 = R"([
  {
    "role": "system",
    "content": [
      {
        "type": "text",
        "text": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
      }
    ]
  },
  {
    "role": "user",
    "content": [
      {
        "type": "text",
        "text": "Who are you? Answer it simply."
      }
    ]
  },
  {
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "I am Qwen, a helpful assistant created by Alibaba Cloud."
      }
    ]
  },
  {
    "role": "user",
    "content": [
      {
        "type": "text",
        "text": "Repeat it."
      }
    ]
  }
])";
  auto messages2 = ailoy::decode(messages_str2, ailoy::encoding_method_t::json);
  auto answer2 = infer(model, messages2);
  ASSERT_EQ(answer2,
            "I am Qwen, a helpful assistant created by Alibaba Cloud.");
}

TEST(TestTVMLanguageModel, TestToolCall) {
  const std::string messages_str = R"([
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "How is the current weather in Seoul?"}
      ]
    }
  ])";

  const std::string tools_str = R"([
    {
      "type": "function",
      "function": {
        "name": "get_current_temperature",
        "description": "Get the current temperature at a location.",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The location to get the temperature for, in the format \"City, Country\""
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"],
              "description": "The unit to return the temperature in."
            }
          },
          "required": ["location", "unit"]
        },
        "return": {
          "type": "number",
          "description": "The current temperature at the specified location in the specified units, as a float."
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "get_current_wind_speed",
        "description": "Get the current wind speed in km/h at a given location.",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The location to get the temperature for, in the format \"City, Country\""
            }
          },
          "required": ["location"]
        },
        "return": {
          "type": "number",
          "description": "The current wind speed at the given location in km/h, as a float."
        }
      }
    }
  ])";

  std::shared_ptr<ailoy::component_t> model = get_model();
  auto messages = ailoy::decode(messages_str, ailoy::encoding_method_t::json);
  auto tools = ailoy::decode(tools_str, ailoy::encoding_method_t::json);

  // model->set_builtin_grammar("tool_call", "json");
  auto answer = infer(model, messages, tools);
  ailoy::debug(answer);
  // model->reset_grammar("tool_call");

  const std::string messages2_str = R"([
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "How is the current weather in Seoul?"}
      ]
    },
    {
      "role": "assistant",
      "tool_calls": [
        {"function":{"arguments":{"location":"Seoul","unit":"celsius"},"name":"get_current_temperature"},"type":"function"}
      ]
    },
    {
      "role": "tool",
      "content": [
        {"type": "text", "text": "20.5"}
      ]
    }
  ])";
  auto messages2 = ailoy::decode(messages2_str, ailoy::encoding_method_t::json);
  auto answer2 = infer(model, messages2, tools, false);
  // std::cout << answer2 << std::endl;
}

TEST(TestTVMLanguageModel, TestReasoning) {
  auto messages_str = R"([
  {"role": "user", "content": [{"type": "text","text": "Introduce yourself."}]}
])";

  std::shared_ptr<ailoy::component_t> model = get_model();
  auto messages = ailoy::decode(messages_str, ailoy::encoding_method_t::json);
  auto out = infer(model, messages, nullptr, true, false);
  ailoy::debug(out);

  auto messages_str2 = R"([
  {"role": "user", "content": [{"type": "text","text": "Introduce yourself."}]},
  {
    "role": "assistant",
    "reasoning": [
      {
        "type": "text",
        "text": "\nOkay, the user wants me to introduce myself. Let me start by acknowledging their request. I should be friendly and open. I can say something like, \"Hi, I'm an AI assistant here. I'm here to help you with your questions!\" That's a good start. Now, I need to add some personal details or a brief introduction. Maybe mention my name or a trait, like being a language model. Let me check if that's needed. Oh, the user might want to know more about their role. So, include that. Make sure it's concise and positive. Alright, that should cover it.\n"
      }
    ],
    "content": [
      {
        "type": "text",
        "text": "Hi! I'm an AI assistant here, and I'm excited to help you with your questions. I'm designed to support you in various ways, and I'm here to be your guide! Let me know how I can assist you! 😊"
      }
    ]
  },
  {"role": "user", "content": [{"type": "text","text": "I love you."}]}
])";
  auto messages2 = ailoy::decode(messages_str2, ailoy::encoding_method_t::json);
  auto out2 = infer(model, messages2, nullptr, true, true);
  ailoy::debug(out2);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
