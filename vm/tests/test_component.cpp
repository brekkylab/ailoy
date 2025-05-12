#include <array>
#include <string>

#include <gtest/gtest.h>

#include "module.hpp"

TEST(ComponentTest, TestEcho) {
  auto default_operators = ailoy::get_default_module()->ops;
  auto echo = default_operators.at("echo");
  auto input = ailoy::create<ailoy::string_t>("Hello world");
  echo->initialize(input);
  auto output_opt = echo->step();
  ASSERT_EQ(output_opt.index(), 0);
  auto output = std::get<0>(output_opt);
  std::cout << *(output.val->as<ailoy::string_t>()) << std::endl;
}

TEST(ComponentTest, TestSpell) {
  auto default_operators = ailoy::get_default_module()->ops;
  auto spell = default_operators.at("spell");
  auto input = ailoy::create<ailoy::string_t>("Hello world");
  spell->initialize(input);
  while (true) {
    auto output_opt = spell->step();
    ASSERT_EQ(output_opt.index(), 0);
    auto output = std::get<0>(output_opt);
    std::cout << *(output.val->as<ailoy::string_t>()) << std::endl;
    if (output.finish)
      break;
  }
}

TEST(ComponentTest, TestAccumulator) {
  const std::string base_str = "BASE";
  const std::array<std::string, 2> put_str = {"-AAA", "-bbb"};

  auto default_component_creators = ailoy::get_default_module()->factories;
  auto create_accumulator = default_component_creators.at("accumulator");
  auto attrs = ailoy::create<ailoy::string_t>(base_str);
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
    auto out0 = std::get<0>(out0_opt).val->as<ailoy::string_t>();
    std::cout << *out0 << std::endl;
    ASSERT_EQ(base_str, *out0);
  }
  {
    auto in1 = ailoy::create<ailoy::string_t>(put_str[0]);
    put_op->initialize(in1);
    put_op->step();
  }
  {
    auto in2 = ailoy::create<ailoy::string_t>(put_str[1]);
    put_op->initialize(in2);
    put_op->step();
  }
  {
    get_op->initialize();
    auto out3_opt = get_op->step();
    ASSERT_EQ(out3_opt.index(), 0);
    auto out3 = std::get<0>(out3_opt).val->as<ailoy::string_t>();
    std::cout << *out3 << std::endl;
    ASSERT_EQ(base_str + put_str[0] + put_str[1], *out3);
  }
  {
    auto out4_opt = count_op->step();
    ASSERT_EQ(out4_opt.index(), 0);
    auto out4 = std::get<0>(out4_opt).val->as<ailoy::uint_t>();
    std::cout << out4->operator unsigned long long() << std::endl;
    ASSERT_EQ(put_str.size(), static_cast<size_t>(*out4));
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
