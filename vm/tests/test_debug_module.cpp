#include <array>
#include <string>

#include <gtest/gtest.h>

#include "module.hpp"

TEST(DebugModuleTest, TestEcho) {
  auto default_operators = ailoy::get_debug_module()->ops;
  auto echo = default_operators.at("echo");
  auto text = "Hello world";
  auto inputs = ailoy::create<ailoy::map_t>();
  inputs->insert_or_assign("text", ailoy::create<ailoy::string_t>(text));
  echo->initialize(inputs);
  auto output_opt = echo->step();
  ASSERT_EQ(output_opt.index(), 0);
  auto output = std::get<0>(output_opt);
  auto echo_text = output.val->as<ailoy::map_t>()->at<ailoy::string_t>("text");
  ASSERT_EQ(text, *echo_text);
}

TEST(DebugModuleTest, TestSpell) {
  auto default_operators = ailoy::get_debug_module()->ops;
  auto spell = default_operators.at("spell");
  std::string text = "Hello world";
  auto inputs = ailoy::create<ailoy::map_t>();
  inputs->insert_or_assign("text", ailoy::create<ailoy::string_t>(text));
  spell->initialize(inputs);
  int i = 0;
  while (true) {
    auto output_opt = spell->step();
    ASSERT_EQ(output_opt.index(), 0);
    auto output = std::get<0>(output_opt);
    std::string spelled_text =
        *output.val->as<ailoy::map_t>()->at<ailoy::string_t>("text");
    ASSERT_EQ(spelled_text.size(), 1);
    ASSERT_EQ(text[i], spelled_text[0]);
    if (output.finish)
      break;
    i++;
  }
}

TEST(DebugModuleTest, TestAccumulator) {
  const std::string base_str = "BASE";
  const std::array<std::string, 2> put_str = {"-AAA", "-bbb"};

  auto default_component_creators = ailoy::get_debug_module()->factories;
  auto create_accumulator = default_component_creators.at("accumulator");
  auto attrs = ailoy::create<ailoy::map_t>();
  attrs->insert_or_assign("base", ailoy::create<ailoy::string_t>(base_str));
  auto accumulator_opt = create_accumulator(attrs);
  ASSERT_EQ(accumulator_opt.index(), 0);
  auto accumulator = std::get<0>(accumulator_opt);
  auto put_op = accumulator->get_operator("put");
  auto get_op = accumulator->get_operator("get");
  auto count_op = accumulator->get_operator("count");
  {
    get_op->initialize();
    auto out0_opt = get_op->step();
    ASSERT_EQ(out0_opt.index(), 0);
    auto out0 = std::get<0>(out0_opt).val->as<ailoy::map_t>();
    ASSERT_EQ(out0->contains("text"), true);
    auto out0_text = out0->at("text");
    ASSERT_EQ(out0_text->is_type_of<ailoy::string_t>(), true);
    ASSERT_EQ(base_str, *out0_text->as<ailoy::string_t>());
  }
  {
    auto in1 = ailoy::create<ailoy::map_t>();
    in1->insert_or_assign("s", ailoy::create<ailoy::string_t>(put_str[0]));
    put_op->initialize(in1);
    put_op->step();
  }
  {
    auto in2 = ailoy::create<ailoy::map_t>();
    in2->insert_or_assign("s", ailoy::create<ailoy::string_t>(put_str[1]));
    put_op->initialize(in2);
    put_op->step();
  }
  {
    get_op->initialize();
    auto out3_opt = get_op->step();
    ASSERT_EQ(out3_opt.index(), 0);
    auto out3 = std::get<0>(out3_opt).val->as<ailoy::map_t>();
    ASSERT_EQ(out3->contains("text"), true);
    auto out3_text = out3->at("text");
    ASSERT_EQ(out3_text->is_type_of<ailoy::string_t>(), true);
    ASSERT_EQ(base_str + put_str[0] + put_str[1],
              *out3_text->as<ailoy::string_t>());
  }
  {
    count_op->initialize();
    auto out4_opt = count_op->step();
    ASSERT_EQ(out4_opt.index(), 0);
    auto out4 = std::get<0>(out4_opt).val->as<ailoy::map_t>();
    ASSERT_EQ(out4->contains("count"), true);
    auto out4_count = out4->at("count");
    ASSERT_EQ(out4_count->is_type_of<ailoy::uint_t>(), true);
    ASSERT_EQ(put_str.size(),
              static_cast<size_t>(*out4_count->as<ailoy::uint_t>()));
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
