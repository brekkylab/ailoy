#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "language.hpp"
#include "openai.hpp"

class GeminiTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    char const *gemini_api_key = std::getenv("GEMINI_API_KEY");
    if (gemini_api_key == NULL) {
      GTEST_SKIP() << "GEMINI_API_KEY environment variable is not set. Skip "
                      "OpenAI test suite..";
      return;
    }

    auto attrs = ailoy::create<ailoy::map_t>();
    attrs->insert_or_assign("api_key",
                            ailoy::create<ailoy::string_t>(gemini_api_key));
    attrs->insert_or_assign("model",
                            ailoy::create<ailoy::string_t>("gemini-2.0-flash"));
    auto gemini_opt =
        ailoy::get_language_module()->factories.at("gemini")(attrs);
    ASSERT_EQ(gemini_opt.index(), 0);

    auto gemini = std::get<0>(gemini_opt);
    comp_ = gemini;
  }

  static void TestDownTestSuite() {}

  static std::shared_ptr<ailoy::component_t> comp_;
};

std::shared_ptr<ailoy::component_t> GeminiTest::comp_ = nullptr;

TEST_F(GeminiTest, SimpleChat) {
  auto infer = comp_->get_operator("infer");

  auto input = ailoy::create<ailoy::map_t>();
  auto messages = nlohmann::json::array();
  messages.push_back({{"role", "user"}, {"content", nlohmann::json::array()}});
  messages[0]["content"].push_back(nlohmann::json::object());
  messages[0]["content"][0]["type"] = "text";
  messages[0]["content"][0]["text"] =
      "Who is the president of US in 2021? Just answer in two words.";
  input->insert_or_assign("messages", ailoy::from_nlohmann_json(messages));

  infer->initialize(input);
  auto res = infer->step();
  if (res.index() == 1) {
    std::cout << std::get<1>(res).reason << std::endl;
    return;
  }
  ASSERT_EQ(res.index(), 0);
  auto out = std::get<0>(res).val;
  ASSERT_EQ(out->is_type_of<ailoy::map_t>(), true);
  auto out_map = out->as<ailoy::map_t>();

  /* Expected out:
     {
       "finish_reason": "stop",
       "message": {
         "annotations": [],
         "content": "Joe Biden",
         "refusal": null,
         "role": "assistant"
       }
     }
  */
  ASSERT_EQ(*out_map->at<ailoy::string_t>("finish_reason"), "stop");
  ASSERT_EQ(*out_map->at<ailoy::map_t>("message")->at<ailoy::string_t>("role"),
            "assistant");
  ASSERT_TRUE(out_map->at<ailoy::map_t>("message")
                  ->at("content")
                  ->is_type_of<ailoy::array_t>());
  ASSERT_EQ(out_map->at<ailoy::map_t>("message")
                ->at<ailoy::array_t>("content")
                ->size(),
            1);
  ASSERT_TRUE(out_map->at<ailoy::map_t>("message")
                  ->at<ailoy::array_t>("content")
                  ->at(0)
                  ->is_type_of<ailoy::map_t>());
  ASSERT_TRUE(out_map->at<ailoy::map_t>("message")
                  ->at<ailoy::array_t>("content")
                  ->at<ailoy::map_t>(0)
                  ->contains("type"));
  ASSERT_TRUE(out_map->at<ailoy::map_t>("message")
                  ->at<ailoy::array_t>("content")
                  ->at<ailoy::map_t>(0)
                  ->at("type")
                  ->is_type_of<ailoy::string_t>());
  ASSERT_TRUE(out_map->at<ailoy::map_t>("message")
                  ->at<ailoy::array_t>("content")
                  ->at<ailoy::map_t>(0)
                  ->contains("text"));
  ASSERT_TRUE(out_map->at<ailoy::map_t>("message")
                  ->at<ailoy::array_t>("content")
                  ->at<ailoy::map_t>(0)
                  ->at("text")
                  ->is_type_of<ailoy::string_t>());
  EXPECT_THAT(*out_map->at<ailoy::map_t>("message")
                   ->at<ailoy::array_t>("content")
                   ->at<ailoy::map_t>(0)
                   ->at<ailoy::string_t>("text"),
              ::testing::HasSubstr("Joe Biden"));
}

TEST_F(GeminiTest, ToolCall) {
  auto infer = comp_->get_operator("infer");

  auto input = ailoy::create<ailoy::map_t>();
  auto messages = nlohmann::json::array();
  messages.push_back({{"role", "user"}, {"content", nlohmann::json::array()}});
  messages[0]["content"].push_back(nlohmann::json::object());
  messages[0]["content"][0]["type"] = "text";
  messages[0]["content"][0]["text"] =
      "What is the weather like in Paris today?";
  input->insert_or_assign("messages", ailoy::from_nlohmann_json(messages));

  auto tools = nlohmann::json::parse(R"(
    [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current temperature for a given location.",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "City and country e.g. Bogotá, Colombia"
            }
          },
          "required": ["location"],
          "additionalProperties": false
        },
        "strict": true
      }
    }]
  )");
  input->insert_or_assign("tools", ailoy::from_nlohmann_json(tools));

  infer->initialize(input);
  auto res = infer->step();
  ASSERT_EQ(res.index(), 0);
  auto out = std::get<0>(res).val;
  ASSERT_EQ(out->is_type_of<ailoy::map_t>(), true);
  auto out_map = out->as<ailoy::map_t>()->to_nlohmann_json();

  /* Expected out:
     {
       "finish_reason": "tool_calls",
       "message": {
         "content": null,
         "role": "assistant",
         "tool_calls": [
           {
             "function": {
               "arguments": {"location":"Paris, France"},
               "name": "get_weather"
             },
             "id": "call_xxxx",
             "type": "function"
           }
         ]
       }
     }
  */
  ASSERT_EQ(out_map["finish_reason"], "tool_calls");
  ASSERT_EQ(out_map["message"]["role"], "assistant");
  auto tool_call = out_map["message"]["tool_calls"][0];
  ASSERT_EQ(tool_call["type"], "function");
  ASSERT_EQ(tool_call["function"]["name"], "get_weather");
  auto expected_arguments = nlohmann::json{{"location", "Paris, France"}};
  ASSERT_EQ(tool_call["function"]["arguments"].dump(),
            expected_arguments.dump());

  // Infer again with tool call result
  auto input2 = ailoy::create<ailoy::map_t>();
  // append tool_call message from assistant
  messages.push_back(out_map["message"]);
  // append tool_call result message from tool
  messages.push_back({{"role", "tool"},
                      {"tool_call_id", tool_call["id"]},
                      {"content", "14°C"}});
  input2->insert_or_assign("messages", ailoy::from_nlohmann_json(messages));
  input2->insert_or_assign("tools", ailoy::from_nlohmann_json(tools));

  infer->initialize(input2);
  auto res2 = infer->step();
  ASSERT_EQ(res2.index(), 0);
  auto out2 = std::get<0>(res2).val;
  ASSERT_EQ(out2->is_type_of<ailoy::map_t>(), true);
  auto out2_map = out2->as<ailoy::map_t>()->to_nlohmann_json();

  /* Expected out:
    {
      "finish_reason": "stop",
      "message": {
        "content": "The current temperature in Paris, France is 14°C",
        "role": "assistant"
      }
    }
  */
  ASSERT_EQ(out2_map["finish_reason"], "stop");
  ASSERT_EQ(out2_map["message"]["role"], "assistant");
  EXPECT_THAT(out2_map["message"]["content"][0]["text"],
              ::testing::HasSubstr("14°C"));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
