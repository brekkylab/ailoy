#include <gtest/gtest.h>

#include "language.hpp"
#include "module.hpp"

static const std::string QWEN_MESSAGES = R"([
  {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate."},
  {"role": "user", "content":  "who are you?"}
])";

static const std::string QWEN_MULTI_TURN_MESSAGES = R"([
  {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
  {"role": "user", "content": "Tell me about constant pi in a sentence."},
  {"role": "assistant", "content": "Pi(π) is a mathematical constant representing the ratio of a circle's circumference to its diameter, approximately equal to 3.14159."},
  {"role": "user", "content": "Please give me a little more accurate value."}
])";

static const std::string TOOLS = R"([
  {"type": "function", "function": {"name": "get_current_temperature", "description": "Get the current temperature at a location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \"City, Country\""}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in."}}, "required": ["location", "unit"]}, "return": {"type": "number", "description": "The current temperature at the specified location in the specified units, as a float."}}},
  {"type": "function", "function": {"name": "get_current_wind_speed", "description": "Get the current wind speed in km/h at a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \"City, Country\""}}, "required": ["location"]}, "return": {"type": "number", "description": "The current wind speed at the given location in km/h, as a float."}}}
])";

static const std::string QWEN_MESSAGES_TOOL_CALL = R"([
  {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
  {"role": "user", "content": "Hello, how is the current weather in my city Seoul?"},
  {"role": "assistant", "content": "I can certainly help with that! Would you like to know the current temperature or the current wind speed in Seoul, South Korea?"},
  {"role": "user", "content": "Yes, both please!"}
])";

static const std::string QWEN_MESSAGES_TOOL_RESULTS = R"([
  {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
  {"role": "user", "content": "Hello, how is the current weather in my city Seoul?"},
  {"role": "assistant", "content": "I can certainly help with that! Would you like to know the current temperature or the current wind speed in Seoul, South Korea?"},
  {"role": "user", "content": "Yes, both please!"},
  {"role": "assistant", "content": "", "tool_calls": [{"function":{"arguments":{"location":"Seoul, South Korea","unit":"celsius"},"name":"get_current_temperature"},"id":"call_44ae1b2d-5b95-471c-9d91-999e35b5b7fc","index":0,"type":"function"}]},
  {"role": "assistant", "content": "", "tool_calls": [{"function":{"arguments":{"location":"Seoul, South Korea"},"name":"get_current_wind_speed"},"id":"call_b1a472e3-1418-4e95-abc4-502be5cc590d","index":0,"type":"function"}]},
  {"role": "tool", "name": "get_current_temperature", "tool_call_id": "call_44ae1b2d-5b95-471c-9d91-999e35b5b7fc", "content": "25"},
  {"role": "tool", "name": "get_current_wind_speed", "tool_call_id": "call_b1a472e3-1418-4e95-abc4-502be5cc590d", "content": "18"}
])";

void print_llm_output(std::shared_ptr<ailoy::map_t> delta) {
  if (delta->contains("content") &&
      delta->at("content")->is_type_of<ailoy::string_t>())
    std::cout << *delta->at<ailoy::string_t>("content") << std::flush;
  else if (delta->contains("tool_calls"))
    std::cout << delta->at<ailoy::array_t>("tool_calls")->to_nlohmann_json()
              << std::endl;
}

void print_llm_output(std::shared_ptr<ailoy::map_t> delta, bool &reasoning) {
  if (!reasoning && (*delta->at<ailoy::bool_t>("reasoning", false))) {
    reasoning = true;
    std::cout << "<REASONING>" << std::endl;
  }
  if (reasoning && !(*delta->at<ailoy::bool_t>("reasoning", false))) {
    reasoning = false;
    std::cout << "</REASONING>" << std::endl;
  }
  print_llm_output(delta);
}

TEST(LanguageModelTest, TestInfer) {
  std::shared_ptr<ailoy::component_t> language_model;
  {
    auto in = ailoy::create<ailoy::map_t>();
    in->insert_or_assign("model",
                         ailoy::create<ailoy::string_t>("Qwen/Qwen3-8B"));
    auto language_model_opt =
        ailoy::get_language_module()->factories.at("tvm_language_model")(in);
    ASSERT_EQ(language_model_opt.index(), 0);
    language_model = std::get<0>(language_model_opt);
  }
  {
    auto in = ailoy::create<ailoy::map_t>();
    auto messages =
        ailoy::decode(QWEN_MESSAGES, ailoy::encoding_method_t::json);
    in->insert_or_assign("messages", messages);
    auto init_out = language_model->get_operator("infer")->initialize(in);
    for (size_t i = 0; i < 1024; i++) {
      auto out_opt = language_model->get_operator("infer")->step();
      ASSERT_EQ(out_opt.index(), 0);
      auto out = std::get<0>(out_opt);
      auto delta = out.val->as<ailoy::map_t>()->at<ailoy::map_t>("message");
      print_llm_output(delta);
      if (out.finish) {
        std::cout << "(" << i << " tokens)" << std::endl;
        break;
      }
    }
  }
}

TEST(LanguageModelTest, TestMultiTurnChat) {
  auto create_language_model =
      ailoy::get_language_module()->factories.at("tvm_language_model");
  auto attrs = ailoy::create<ailoy::map_t>();
  attrs->insert_or_assign("model",
                          ailoy::create<ailoy::string_t>("Qwen/Qwen3-8B"));
  auto language_model_opt = create_language_model(attrs);
  ASSERT_EQ(language_model_opt.index(), 0);
  auto language_model = std::get<0>(language_model_opt);

  {
    auto in = ailoy::create<ailoy::map_t>();
    auto messages =
        ailoy::decode(QWEN_MULTI_TURN_MESSAGES, ailoy::encoding_method_t::json);
    in->insert_or_assign("messages", messages);
    auto init_out = language_model->get_operator("infer")->initialize(in);
    for (size_t i = 0; i < 1024; i++) {
      auto out_opt = language_model->get_operator("infer")->step();
      ASSERT_EQ(out_opt.index(), 0);
      auto out = std::get<0>(out_opt);
      auto delta = out.val->as<ailoy::map_t>()->at<ailoy::map_t>("message");
      print_llm_output(delta);
      if (out.finish) {
        std::cout << "(" << i << " tokens)" << std::endl;
        break;
      }
    }
  }
}

TEST(LanguageModelTest, TestChatReasoning) {
  auto create_language_model =
      ailoy::get_language_module()->factories.at("tvm_language_model");
  auto attrs = ailoy::create<ailoy::map_t>();
  attrs->insert_or_assign("model",
                          ailoy::create<ailoy::string_t>("Qwen/Qwen3-8B"));
  auto language_model_opt = create_language_model(attrs);
  ASSERT_EQ(language_model_opt.index(), 0);
  auto language_model = std::get<0>(language_model_opt);

  {
    auto in = ailoy::create<ailoy::map_t>();
    auto messages =
        ailoy::decode(QWEN_MULTI_TURN_MESSAGES, ailoy::encoding_method_t::json);
    in->insert_or_assign("messages", messages);
    in->insert_or_assign("reasoning", ailoy::create<ailoy::bool_t>(true));
    auto init_out = language_model->get_operator("infer")->initialize(in);
    bool reasoning = false;
    for (size_t i = 0; i < 1024; i++) {
      auto out_opt = language_model->get_operator("infer")->step();
      ASSERT_EQ(out_opt.index(), 0);
      auto out = std::get<0>(out_opt);
      auto delta = out.val->as<ailoy::map_t>()->at<ailoy::map_t>("message");
      print_llm_output(delta, reasoning);
      if (out.finish) {
        std::cout << "(" << i << " tokens)" << std::endl;
        break;
      }
    }
  }
}

TEST(LanguageModelTest, TestChatReasoningIgnored) {
  auto create_language_model =
      ailoy::get_language_module()->factories.at("tvm_language_model");
  auto attrs = ailoy::create<ailoy::map_t>();
  attrs->insert_or_assign("model",
                          ailoy::create<ailoy::string_t>("Qwen/Qwen3-8B"));
  auto language_model_opt = create_language_model(attrs);
  ASSERT_EQ(language_model_opt.index(), 0);
  auto language_model = std::get<0>(language_model_opt);

  {
    auto in = ailoy::create<ailoy::map_t>();
    auto messages =
        ailoy::decode(QWEN_MULTI_TURN_MESSAGES, ailoy::encoding_method_t::json);
    in->insert_or_assign("messages", messages);
    in->insert_or_assign("reasoning", ailoy::create<ailoy::bool_t>(true));
    in->insert_or_assign("ignore_reasoning",
                         ailoy::create<ailoy::bool_t>(true));
    auto init_out = language_model->get_operator("infer")->initialize(in);
    for (size_t i = 0; i < 1024; i++) {
      auto out_opt = language_model->get_operator("infer")->step();
      ASSERT_EQ(out_opt.index(), 0);
      auto out = std::get<0>(out_opt);
      auto delta = out.val->as<ailoy::map_t>()->at<ailoy::map_t>("message");
      ASSERT_FALSE(delta->contains("reasoning"));
      print_llm_output(delta);
      if (out.finish) {
        std::cout << "(" << i << " tokens)" << std::endl;
        break;
      }
    }
  }
}

TEST(LanguageModelTest, TestChatTemplateTools) {
  std::shared_ptr<ailoy::component_t> language_model;
  {
    auto in = ailoy::create<ailoy::map_t>();
    in->insert_or_assign("model",
                         ailoy::create<ailoy::string_t>("Qwen/Qwen3-8B"));
    auto language_model_opt =
        ailoy::get_language_module()->factories.at("tvm_language_model")(in);
    ASSERT_EQ(language_model_opt.index(), 0);
    language_model = std::get<0>(language_model_opt);
  }
  {
    auto in = ailoy::create<ailoy::map_t>();
    auto messages =
        ailoy::decode(QWEN_MESSAGES_TOOL_CALL, ailoy::encoding_method_t::json);
    in->insert_or_assign("messages", messages);
    in->insert_or_assign("tools", ailoy::create<ailoy::string_t>(TOOLS));

    language_model->get_operator("apply_chat_template")->initialize(in);
    auto prompt =
        std::get<0>(language_model->get_operator("apply_chat_template")->step())
            .val->as<ailoy::map_t>()
            ->at<ailoy::string_t>("prompt");

    std::cout << *prompt << std::endl;
  }
}

TEST(LanguageModelTest, TestQwenToolCall) {
  std::shared_ptr<ailoy::component_t> language_model;
  {
    auto in = ailoy::create<ailoy::map_t>();
    in->insert_or_assign("model",
                         ailoy::create<ailoy::string_t>("Qwen/Qwen3-8B"));
    auto language_model_opt =
        ailoy::get_language_module()->factories.at("tvm_language_model")(in);
    ASSERT_EQ(language_model_opt.index(), 0);
    language_model = std::get<0>(language_model_opt);
  }
  {
    auto in = ailoy::create<ailoy::map_t>();
    auto messages =
        ailoy::decode(QWEN_MESSAGES_TOOL_CALL, ailoy::encoding_method_t::json);
    in->insert_or_assign("messages", messages);
    in->insert_or_assign("tools", ailoy::create<ailoy::string_t>(TOOLS));
    auto init_out = language_model->get_operator("infer")->initialize(in);
    for (size_t i = 0; i < 1024; i++) {
      auto out_opt = language_model->get_operator("infer")->step();
      ASSERT_EQ(out_opt.index(), 0);
      auto out = std::get<0>(out_opt);
      auto delta = out.val->as<ailoy::map_t>()->at<ailoy::map_t>("message");
      std::cout << delta->to_nlohmann_json() << std::endl;
      print_llm_output(delta);
      if (out.finish) {
        std::cout << "(" << i << " tokens)" << std::endl;
        break;
      }
    }
  }
}

TEST(LanguageModelTest, TestQwenToolCallReasoning) {
  std::shared_ptr<ailoy::component_t> language_model;
  {
    auto in = ailoy::create<ailoy::map_t>();
    in->insert_or_assign("model",
                         ailoy::create<ailoy::string_t>("Qwen/Qwen3-8B"));
    auto language_model_opt =
        ailoy::get_language_module()->factories.at("tvm_language_model")(in);
    ASSERT_EQ(language_model_opt.index(), 0);
    language_model = std::get<0>(language_model_opt);
  }
  {
    auto in = ailoy::create<ailoy::map_t>();
    auto messages =
        ailoy::decode(QWEN_MESSAGES_TOOL_CALL, ailoy::encoding_method_t::json);
    in->insert_or_assign("messages", messages);
    in->insert_or_assign("tools", ailoy::create<ailoy::string_t>(TOOLS));
    in->insert_or_assign("reasoning", ailoy::create<ailoy::bool_t>(true));
    auto init_out = language_model->get_operator("infer")->initialize(in);
    bool reasoning = false;
    for (size_t i = 0; i < 1024; i++) {
      auto out_opt = language_model->get_operator("infer")->step();
      ASSERT_EQ(out_opt.index(), 0);
      auto out = std::get<0>(out_opt);
      auto delta = out.val->as<ailoy::map_t>()->at<ailoy::map_t>("message");
      print_llm_output(delta, reasoning);
      if (out.finish) {
        std::cout << "(" << i << " tokens)" << std::endl;
        break;
      }
    }
  }
}

TEST(LanguageModelTest, TestQwenToolCallResult) {
  std::shared_ptr<ailoy::component_t> language_model;
  {
    auto in = ailoy::create<ailoy::map_t>();
    in->insert_or_assign("model",
                         ailoy::create<ailoy::string_t>("Qwen/Qwen3-8B"));
    auto language_model_opt =
        ailoy::get_language_module()->factories.at("tvm_language_model")(in);
    ASSERT_EQ(language_model_opt.index(), 0);
    language_model = std::get<0>(language_model_opt);
  }
  {
    auto in = ailoy::create<ailoy::map_t>();
    auto messages = ailoy::decode(QWEN_MESSAGES_TOOL_RESULTS,
                                  ailoy::encoding_method_t::json);
    in->insert_or_assign("messages", messages);
    in->insert_or_assign("tools", ailoy::create<ailoy::string_t>(TOOLS));
    auto init_out = language_model->get_operator("infer")->initialize(in);
    for (size_t i = 0; i < 1024; i++) {
      auto out_opt = language_model->get_operator("infer")->step();
      ASSERT_EQ(out_opt.index(), 0);
      auto out = std::get<0>(out_opt);
      auto delta = out.val->as<ailoy::map_t>()->at<ailoy::map_t>("message");
      print_llm_output(delta);
      if (out.finish) {
        std::cout << "(" << i << " tokens)" << std::endl;
        break;
      }
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
