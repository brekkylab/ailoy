#include <iostream>
#include <sstream>

#include "value.hpp"

namespace ailoy {

std::ostream &operator<<(std::ostream &os, const ailoy::bytes_t &v) {
  for (int i = 0; i < v.size(); i++) {
    std::cout << "0x" << std::setfill('0') << std::setw(2) << std::uppercase
              << std::hex << (0xFF & v[i]) << " " << std::dec;
  }
  return os;
}

std::shared_ptr<bytes_t> value_t::encode(encoding_method_t method) const {
  std::shared_ptr<bytes_t> rv = std::make_shared<bytes_t>();
  if (method == encoding_method_t::cbor) {
    auto v = nlohmann::json::to_cbor(operator nlohmann::json());
    rv->resize(v.size());
    memcpy(rv->data(), v.data(), v.size());
  } else if (method == encoding_method_t::json) {
    auto v = operator nlohmann::json().dump();
    rv->resize(v.size());
    memcpy(rv->data(), v.data(), v.size());
  } else {
    throw exception("Encode method not supported");
  }
  return rv;
}

nlohmann::json ndarray_t::to_nlohmann_json() const {
  size_t ndim_bytes_nbytes = sizeof(uint32_t);
  size_t shape_bytes_nbytes = sizeof(uint32_t) * shape.size();
  size_t dl_datatype_nbytes = sizeof(DLDataType);
  size_t ndatalen_bytes_nbytes = sizeof(uint64_t);
  size_t data_nbytes = data.size() * sizeof(decltype(data)::value_type);
  size_t nbytes = ndim_bytes_nbytes + shape_bytes_nbytes + dl_datatype_nbytes +
                  ndatalen_bytes_nbytes + data_nbytes;

  // Allocate vector
  std::vector<uint8_t> rv(nbytes);
  // Write pointer
  auto it = rv.begin();

  // Write ndim
  reinterpret_cast<uint32_t &>(*it) = shape.size();
  it += sizeof(uint32_t);

  // Write shape
  for (auto &v : shape) {
    reinterpret_cast<uint32_t &>(*it) = v;
    it += sizeof(uint32_t);
  }

  // Write dtype
  reinterpret_cast<DLDataType &>(*it) = dtype;
  it += sizeof(DLDataType);

  // Write ndatalen
  reinterpret_cast<uint64_t &>(*it) = data.size();
  it += sizeof(uint64_t);

  // Write data
  std::copy(data.begin(), data.end(), it);

  // Return nlohmann::json obj
  return nlohmann::json::binary(rv, 1801);
}

std::shared_ptr<ndarray_t>
ndarray_t::from_nlohmann_json(const nlohmann::json::binary_t &j) {
  auto it = j.begin();
  auto rv = create<ndarray_t>();

  // Parse ndim
  rv->shape.resize(reinterpret_cast<const uint32_t &>(*it));
  it += sizeof(uint32_t);

  // Parse shape
  for (size_t i = 0; i < rv->shape.size(); i++) {
    rv->shape[i] = reinterpret_cast<const uint32_t &>(*it);
    it += sizeof(uint32_t);
  }

  // Parse dtype
  rv->dtype = reinterpret_cast<const DLDataType &>(*it);
  it += sizeof(DLDataType);

  // Parse ndatalen
  rv->data.resize(reinterpret_cast<const uint64_t &>(*it));
  it += sizeof(uint64_t);

  // Parse data
  std::copy(it, it + rv->data.size(), rv->data.data());

  return rv;
}

std::string ndarray_t::shape_str() const {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < shape.size(); i++) {
    ss << shape[i];
    if (i < shape.size() - 1)
      ss << ", ";
  }
  ss << "]";
  return ss.str();
}

std::shared_ptr<value_t> from_nlohmann_json(const nlohmann::json &j) {
  std::shared_ptr<value_t> value = nullptr;
  switch (j.type()) {
  case nlohmann::detail::value_t::null:
    value = create<null_t>();
    break;
  case nlohmann::detail::value_t::boolean:
    value = create<bool_t>(j.get<bool>());
    break;
  case nlohmann::detail::value_t::string:
    value = create<string_t>(j.get<std::string>());
    break;
  case nlohmann::detail::value_t::binary: {
    auto bin = j.get_binary();
    if (bin.has_subtype()) {
      if (bin.subtype() == 1801) {
        value = ndarray_t::from_nlohmann_json(bin);
      } else {
        throw exception("Cannot handle code");
      }
    } else {
      value = create<bytes_t>(bin.begin(), bin.end());
    }
    break;
  }
  case nlohmann::detail::value_t::number_integer:
    value = create<int_t>(j.get<long long>());
    break;
  case nlohmann::detail::value_t::number_unsigned:
    value = create<uint_t>(j.get<unsigned long long>());
    break;
  case nlohmann::detail::value_t::number_float:
    value = create<double_t>(j.get<double>());
    break;
  case nlohmann::detail::value_t::array:
    value = create<array_t>();
    for (auto &[key, val] : j.items())
      value->as<array_t>()->push_back(from_nlohmann_json(val));
    break;
  case nlohmann::detail::value_t::object:
    value = create<map_t>();
    for (auto &[key, val] : j.items())
      value->as<map_t>()->insert_or_assign(key, from_nlohmann_json(val));
    break;
  default:
    break;
  }
  return value;
}

std::shared_ptr<value_t> decode(std::shared_ptr<bytes_t> bytes,
                                encoding_method_t method) {
  nlohmann::json j;
  if (method == encoding_method_t::cbor) {
    std::vector<uint8_t> v(bytes->size());
    std::copy(bytes->begin(), bytes->end(),
              reinterpret_cast<int8_t *>(v.data()));
    j = nlohmann::json::from_cbor(v, true, true,
                                  nlohmann::json::cbor_tag_handler_t::store);
  } else if (method == encoding_method_t::json) {
    j = nlohmann::json::parse(bytes->begin(), bytes->end());
  } else {
    throw exception("Encode method not supported");
  }
  return from_nlohmann_json(j);
}

std::shared_ptr<value_t> decode(const std::string &bytes,
                                encoding_method_t method) {
  return decode(create<bytes_t>(bytes), method);
}

} // namespace ailoy
