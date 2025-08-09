#pragma once

#include <string>
#include <unordered_map>

#include <rust/cxx.h>

namespace ailoy {

struct cache_t {
  cache_t() = default;

  const std::string &read(const std::string &key) const;

  std::unique_ptr<std::string> read_and_remove(const std::string &key);

  void write(const std::string &key, std::string value);

  void write_from_rs(rust::String key, rust::String value);

  void write_binary_from_rs(rust::String key, rust::Vec<uint8_t> value);

  std::unordered_map<std::string, std::string> inner;
};

std::unique_ptr<cache_t> create_cache();

} // namespace ailoy