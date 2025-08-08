#include "cache.hpp"

namespace ailoy {

const std::string &cache_t::read(const std::string &key) const {
  if (!inner.contains(key))
    throw std::runtime_error("File not exists");
  return inner.at(key);
}

std::unique_ptr<std::string> cache_t::read_and_remove(const std::string &key) {
  std::string rv = read(key);
  inner.erase(key);
  return std::make_unique<std::string>(std::move(rv));
}

void cache_t::write(const std::string &key, std::string value) {
  inner.insert_or_assign(key, std::move(value));
}

void cache_t::write_from_rs(rust::String key, rust::String value) {
  inner.insert_or_assign(std::string(key), std::string(value));
}

void cache_t::write_binary_from_rs(rust::String key, rust::Vec<uint8_t> value) {
  std::string binary(reinterpret_cast<const char *>(value.data()),
                     value.size());
  inner.insert_or_assign(std::string(key), std::move(binary));
}

std::unique_ptr<cache_t> create_cache() { return std::make_unique<cache_t>(); }

} // namespace ailoy
