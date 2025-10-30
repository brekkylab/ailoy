#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>

#include <gtest/gtest.h>

#include "value.hpp"

using namespace std::chrono_literals;

TEST(AiloyValueTest, TestIsTypeOf) {
  std::pair<std::string, std::shared_ptr<ailoy::value_t>> values[] = {
      {typeid(ailoy::null_t).name(), ailoy::create<ailoy::null_t>()},
      {typeid(ailoy::bool_t).name(), ailoy::create<ailoy::bool_t>(true)},
      {typeid(ailoy::uint_t).name(), ailoy::create<ailoy::uint_t>(1)},
      {typeid(ailoy::int_t).name(), ailoy::create<ailoy::int_t>(-1)},
      {typeid(ailoy::float_t).name(), ailoy::create<ailoy::float_t>(1.0)},
      {typeid(ailoy::double_t).name(), ailoy::create<ailoy::double_t>(1.0)},
      {typeid(ailoy::string_t).name(), ailoy::create<ailoy::string_t>("")},
      {typeid(ailoy::bytes_t).name(), ailoy::create<ailoy::bytes_t>("")},
      {typeid(ailoy::array_t).name(), ailoy::create<ailoy::array_t>()},
      {typeid(ailoy::map_t).name(), ailoy::create<ailoy::map_t>()},
      {typeid(ailoy::ndarray_t).name(), ailoy::create<ailoy::ndarray_t>()},
  };
  for (auto [type_str, value] : values) {
    ASSERT_EQ(type_str, value->get_type());
  }
  ASSERT_TRUE(ailoy::create<ailoy::null_t>()->is_type_of<ailoy::null_t>());
  ASSERT_FALSE(ailoy::create<ailoy::null_t>()->is_type_of<ailoy::bool_t>());
  ASSERT_TRUE(ailoy::create<ailoy::int_t>(0)->is_type_of<ailoy::int_t>());
  ASSERT_FALSE(ailoy::create<ailoy::int_t>(0)->is_type_of<ailoy::map_t>());
}

TEST(AiloyValueTest, TestSerialize) {
  std::shared_ptr<ailoy::bytes_t> v1_se;
  auto v1 = ailoy::create<ailoy::map_t>();
  v1->insert_or_assign("str", ailoy::create<ailoy::string_t>("hello world"));
  auto arr1 = ailoy::create<ailoy::array_t>();
  v1->insert_or_assign("list", arr1);
  arr1->push_back(ailoy::create<ailoy::bool_t>(false));
  arr1->push_back(ailoy::create<ailoy::float_t>(1.0));
  auto ndarr = ailoy::create<ailoy::ndarray_t>();
  v1->insert_or_assign("ndarr", ndarr);
  ndarr->shape.push_back(2);
  ndarr->shape.push_back(2);
  ndarr->dtype.code = kDLFloat;
  ndarr->dtype.bits = 32;
  ndarr->dtype.lanes = 1;
  for (size_t i = 0; i < 4 * sizeof(float_t); i++)
    ndarr->data.push_back(0);
  v1_se = v1->encode(ailoy::encoding_method_t::cbor);
  std::cout << *v1_se << std::endl;
  auto v2 = ailoy::decode(v1_se, ailoy::encoding_method_t::cbor);
  ASSERT_TRUE(v2);
  auto v2_map = v2->as<ailoy::map_t>();
  for (auto &[k, v] : *v2_map)
    std::cout << k << std::endl;
  std::cout << *v2_map->at<ailoy::string_t>("str") << std::endl;
}

TEST(AiloyValueTest, TestFromJsonString) {
  auto array = ailoy::decode(R"([
    null, true, false, 1, -1, 1.0, "AAA", [2, -2, 2.0, "BBB"], {"A": 3, "B":
    -3, "C": 3.0, "D":"CCC"}
  ])",
                             ailoy::encoding_method_t::json)
                   ->as<ailoy::array_t>();
  ASSERT_TRUE(array->at(0)->is_type_of<ailoy::null_t>());
  ASSERT_EQ(*array->at<ailoy::bool_t>(1), true);
  ASSERT_EQ(*array->at<ailoy::bool_t>(2), false);
  ASSERT_EQ(*array->at<ailoy::uint_t>(3), 1);
  ASSERT_EQ(*array->at<ailoy::int_t>(4), -1);
  ASSERT_EQ(*array->at<ailoy::double_t>(5), 1.0);
  ASSERT_EQ(*array->at<ailoy::string_t>(6), "AAA");
  auto arr_in_arr = array->at<ailoy::array_t>(7);
  ASSERT_EQ(*arr_in_arr->at<ailoy::uint_t>(0), 2);
  ASSERT_EQ(*arr_in_arr->at<ailoy::int_t>(1), -2);
  ASSERT_EQ(*arr_in_arr->at<ailoy::double_t>(2), 2.0);
  ASSERT_EQ(*arr_in_arr->at<ailoy::string_t>(3), "BBB");
  auto map_in_arr = array->at<ailoy::map_t>(8);
  ASSERT_EQ(*map_in_arr->at<ailoy::uint_t>("A"), 3);
  ASSERT_EQ(*map_in_arr->at<ailoy::int_t>("B"), -3);
  ASSERT_EQ(*map_in_arr->at<ailoy::double_t>("C"), 3.0);
  ASSERT_EQ(*map_in_arr->at<ailoy::string_t>("D"), "CCC");

  auto map = ailoy::decode(R"({
  "null": null,
  "true": true,
  "false": false,
  "uint": 1,
  "int": -1,
  "double": 1.0,
  "string": "AAA",
  "array": [2, -2, 2.0, "BBB"],
  "map": {"A": 3, "B": -3, "C": 3.0, "D":"CCC"}
})",
                           ailoy::encoding_method_t::json)
                 ->as<ailoy::map_t>();
  ASSERT_TRUE(map->at("null")->is_type_of<ailoy::null_t>());
  ASSERT_EQ(*map->at<ailoy::bool_t>("true"), true);
  ASSERT_EQ(*map->at<ailoy::bool_t>("false"), false);
  ASSERT_EQ(*map->at<ailoy::uint_t>("uint"), 1);
  ASSERT_EQ(*map->at<ailoy::int_t>("int"), -1);
  ASSERT_EQ(*map->at<ailoy::double_t>("double"), 1.0);
  ASSERT_EQ(*map->at<ailoy::string_t>("string"), "AAA");
  auto arr_in_map = map->at<ailoy::array_t>("array");
  ASSERT_EQ(*arr_in_map->at<ailoy::uint_t>(0), 2);
  ASSERT_EQ(*arr_in_map->at<ailoy::int_t>(1), -2);
  ASSERT_EQ(*arr_in_map->at<ailoy::double_t>(2), 2.0);
  ASSERT_EQ(*arr_in_map->at<ailoy::string_t>(3), "BBB");
  auto map_in_map = map->at<ailoy::map_t>("map");
  ASSERT_EQ(*map_in_map->at<ailoy::uint_t>("A"), 3);
  ASSERT_EQ(*map_in_map->at<ailoy::int_t>("B"), -3);
  ASSERT_EQ(*map_in_map->at<ailoy::double_t>("C"), 3.0);
  ASSERT_EQ(*map_in_map->at<ailoy::string_t>("D"), "CCC");
}

TEST(AiloyValueTest, TestToJsonString) {
  auto array = ailoy::decode(R"([
  null, true, false, 1, -1, 1.0, "AAA", [2, -2, 2.0, "BBB"], {"A": 3, "B":
  -3, "C": 3.0, "D":"CCC"}
])",
                             ailoy::encoding_method_t::json);
  std::cout << array->operator nlohmann::json() << std::endl;

  auto map = ailoy::decode(R"({
  "null": null,
  "true": true,
  "false": false,
  "uint": 1,
  "int": -1,
  "double": 1.0,
  "string": "AAA",
  "array": [2, -2, 2.0, "BBB"],
  "map": {"A": 3, "B": -3, "C": 3.0, "D":"CCC"}
})",
                           ailoy::encoding_method_t::json);
  std::cout << array->operator nlohmann::json() << std::endl;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
