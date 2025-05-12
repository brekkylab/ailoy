#include <map>

#include <gtest/gtest.h>

#include <tinyexpr.h>

#include "module.hpp"

static const std::map<std::string, double> EXPR_EVAL_MAP = {
    {"1+((2-3*4)/5)^6", 65},
    {"1234567890%3", 0},
    {"0.5+1/3", 5.0 / 6},
    {"3^2+4^2", 25},
    {"sqrt(3^2+4^2)", 5},
    {"floor(ln(exp(e))+cos(2*pi))", 3},
    {"1397.73 * 100", 139773},
    {"log10(10)", 1},
    {"log(e)", 1},
    {"ln(e)", 1},
    {"pi", 3.14159265358979323846},
    {"e", 2.71828182845904523536},
    {"fac 5", 120},
    {"ncr(6,2)", 15},
    {"npr(6,2)", 30},
    {"sin(pi/2)", 1},
    {"atan 1", 3.14159265358979323846 / 4},
};

TEST(CalculatorTest, TestCalculatorOp) {
  auto default_operators = ailoy::get_default_module()->ops;
  auto calculator = default_operators.at("calculator");
  for (auto [expression, expected] : EXPR_EVAL_MAP) {
    auto in = ailoy::create<ailoy::map_t>();
    in->insert_or_assign("expression",
                         ailoy::create<ailoy::string_t>(expression));
    calculator->initialize(in);
    double actual = *std::get<0>(calculator->step())
                         .val->as<ailoy::map_t>()
                         ->at<ailoy::double_t>("value");
    ASSERT_DOUBLE_EQ(actual, expected);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
