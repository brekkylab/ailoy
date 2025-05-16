#pragma once

#include "value.hpp"

namespace ailoy {

using tx_id_t = std::string;

using channel_t = std::string;

enum class packet_type : uint8_t {
  connect = 0,
  disconnect = 1,
  subscribe = 2,
  unsubscribe = 3,
  execute = 4,
  respond = 16,
  respond_execute = 17,
};

std::ostream &operator<<(std::ostream &, const packet_type &);

enum class instruction_type : uint8_t {
  call_function = 0,
  define_component = 1,
  delete_component = 2,
  call_method = 3,
};

std::ostream &operator<<(std::ostream &, const instruction_type &);

struct packet_t : public object_t {
  packet_t(packet_type ptype)
      : ptype(ptype), itype(std::nullopt), headers(ailoy::create<array_t>()),
        body(ailoy::create<map_t>()) {}

  packet_t(packet_type ptype, instruction_type itype)
      : ptype(ptype), itype(itype), headers(ailoy::create<array_t>()),
        body(ailoy::create<map_t>()) {}

  packet_t(packet_type ptype, std::optional<instruction_type> itype,
           std::shared_ptr<ailoy::array_t> headers,
           std::shared_ptr<ailoy::map_t> body)
      : ptype(ptype), itype(itype), headers(headers), body(body) {}

  virtual ~packet_t() = default;

  tx_id_t get_tx_id() const;

  channel_t get_channel() const;

  packet_type ptype;
  std::optional<instruction_type> itype;
  std::shared_ptr<ailoy::array_t> headers;
  std::shared_ptr<ailoy::map_t> body;

  operator std::string() const;
};

/**
 * Internal helper function
 */
constexpr auto bool_value = []<typename t>(t value) -> std::shared_ptr<bool_t> {
  if constexpr (std::is_convertible_v<std::decay_t<t>, bool>) {
    return ailoy::create<bool_t>(value);
  } else if constexpr (is_shared_ptr_v<std::decay_t<t>> &&
                       std::is_same_v<bool_t,
                                      typename std::decay_t<t>::element_type>) {
    return value;
  } else {
    static_assert(false_type_v<t>, "Unsupported type in push_to_header");
    throw ailoy::exception("Unsupported type in push_to_header");
  }
};

/**
 * Internal helper function
 */
constexpr auto uint_value = []<typename t>(t value) -> std::shared_ptr<uint_t> {
  if constexpr (std::is_integral_v<std::decay_t<t>>) {
    return ailoy::create<uint_t>(value);
  } else if constexpr (is_shared_ptr_v<std::decay_t<t>> &&
                       std::is_same_v<uint_t,
                                      typename std::decay_t<t>::element_type>) {
    return value;
  } else {
    static_assert(false_type_v<t>, "Unsupported type in push_to_header");
    throw ailoy::exception("Unsupported type in push_to_header");
  }
};

/**
 * Internal helper function
 */
constexpr auto string_value =
    []<typename t>(t value) -> std::shared_ptr<string_t> {
  if constexpr (std::is_convertible_v<std::decay_t<t>, std::string>) {
    return ailoy::create<string_t>(value);
  } else if constexpr (is_shared_ptr_v<std::decay_t<t>> &&
                       std::is_same_v<string_t,
                                      typename std::decay_t<t>::element_type>) {
    return value;
  } else {
    static_assert(false_type_v<t>, "Unsupported type in push_to_header");
    throw ailoy::exception("Unsupported type in push_to_header");
  }
};

std::shared_ptr<bytes_t> dump_packet(std::shared_ptr<packet_t> packet);

template <packet_type ptype, typename... args_t>
  requires(ptype == packet_type::connect) || (ptype == packet_type::disconnect)
std::shared_ptr<bytes_t> dump_packet(args_t... args) {
  auto args_tuple = std::forward_as_tuple(std::forward<args_t>(args)...);
  auto packet = ailoy::create<packet_t>(ptype);

  // Insert tx_id
  packet->headers->push_back(string_value(std::get<0>(args_tuple)));

  // Insert version (connect)
  if constexpr (ptype == packet_type::connect)
    packet->headers->push_back(ailoy::create<string_t>("1"));

  // Body(empty map)

  // Return
  return dump_packet(packet);
}

template <packet_type ptype, instruction_type itype, typename... args_t>
  requires(ptype == packet_type::subscribe ||
           ptype == packet_type::unsubscribe ||
           ptype == packet_type::execute) &&
          (itype == instruction_type::call_function ||
           itype == instruction_type::define_component ||
           itype == instruction_type::delete_component ||
           itype == instruction_type::call_method)
std::shared_ptr<bytes_t> dump_packet(args_t... args) {
  auto args_tuple = std::forward_as_tuple(std::forward<args_t>(args)...);
  auto packet = ailoy::create<packet_t>(ptype, itype);

  // Insert tx_id
  packet->headers->push_back(string_value(std::get<0>(args_tuple)));

  if constexpr (itype == instruction_type::call_function) {
    // Insert opname
    packet->headers->push_back(string_value(std::get<1>(args_tuple)));
    if constexpr (ptype == packet_type::execute) {
      if constexpr (ptype == packet_type::execute && sizeof...(args) >= 3)
        // Add body {"in": in}
        packet->body->insert_or_assign("in", std::get<2>(args_tuple));
      else
        // Add body {"in": null}
        packet->body->insert_or_assign("in", ailoy::create<ailoy::null_t>());
    }
  } else if constexpr (itype == instruction_type::define_component) {
    // Insert comptype
    packet->headers->push_back(string_value(std::get<1>(args_tuple)));
    if constexpr (ptype == packet_type::execute) {
      // Add body {"name": compname, "in": in}
      packet->body->insert_or_assign("name",
                                     string_value(std::get<2>(args_tuple)));
      if constexpr (sizeof...(args) == 3)
        packet->body->insert_or_assign("in", ailoy::create<null_t>());
      else
        packet->body->insert_or_assign("in", std::get<3>(args_tuple));
    }
  } else if constexpr (itype == instruction_type::delete_component) {
    // Insert compname
    packet->headers->push_back(string_value(std::get<1>(args_tuple)));
  } else {
    // Insert compname
    packet->headers->push_back(string_value(std::get<1>(args_tuple)));
    // Insert op(method)name
    packet->headers->push_back(string_value(std::get<2>(args_tuple)));
    if constexpr (ptype == packet_type::execute && sizeof...(args) >= 4)
      // Add body {"in": in}
      packet->body->insert_or_assign("in", std::get<3>(args_tuple));
    else
      // Add body {"in": null}
      packet->body->insert_or_assign("in", ailoy::create<ailoy::null_t>());
  }
  return dump_packet(packet);
}

template <packet_type ptype, bool status, typename... args_t>
  requires((ptype == packet_type::respond) && status)
std::shared_ptr<bytes_t> dump_packet(args_t... args) {
  auto args_tuple = std::forward_as_tuple(std::forward<args_t>(args)...);
  auto packet = ailoy::create<packet_t>(ptype);

  // Insert tx_id
  packet->headers->push_back(string_value(std::get<0>(args_tuple)));

  // Setup body
  packet->body->insert_or_assign("status", ailoy::create<bool_t>(true));

  // Return
  return dump_packet(packet);
}

template <packet_type ptype, bool status, typename... args_t>
  requires((ptype == packet_type::respond_execute) && status)
std::shared_ptr<bytes_t> dump_packet(args_t... args) {
  auto args_tuple = std::forward_as_tuple(std::forward<args_t>(args)...);
  auto packet = ailoy::create<packet_t>(ptype);

  // Insert tx_id
  packet->headers->push_back(string_value(std::get<0>(args_tuple)));

  // Insert sequence
  packet->headers->push_back(uint_value(std::get<1>(args_tuple)));

  // Insert finish
  packet->headers->push_back(bool_value(std::get<2>(args_tuple)));

  // Insert status
  packet->body->insert_or_assign("status", ailoy::create<bool_t>(true));

  // Insert out data
  if constexpr (ptype == packet_type::respond_execute)
    packet->body->insert_or_assign("out", std::get<3>(args_tuple));

  // Return
  return dump_packet(packet);
}

template <packet_type ptype, bool status, typename... args_t>
  requires((ptype == packet_type::respond ||
            ptype == packet_type::respond_execute) &&
           !status)
std::shared_ptr<bytes_t> dump_packet(args_t... args) {
  auto args_tuple = std::forward_as_tuple(std::forward<args_t>(args)...);
  auto packet = ailoy::create<packet_t>(ptype);

  // Insert tx_id
  packet->headers->push_back(string_value(std::get<0>(args_tuple)));

  if constexpr (ptype == packet_type::respond_execute)
    // Insert sequence
    packet->headers->push_back(uint_value(std::get<1>(args_tuple)));

  // Insert finished
  if constexpr (ptype == packet_type::respond_execute)
    packet->headers->push_back(bool_value(true));

  // Setup body (status & reason)
  packet->body->insert_or_assign("status", ailoy::create<bool_t>(false));
  if constexpr (ptype == packet_type::respond_execute)
    packet->body->insert_or_assign("reason",
                                   string_value(std::get<2>(args_tuple)));
  else
    packet->body->insert_or_assign("reason",
                                   string_value(std::get<1>(args_tuple)));

  // Return
  return dump_packet(packet);
}

std::shared_ptr<packet_t> load_packet(std::shared_ptr<bytes_t> packet,
                                      bool skip_body = false);

// tx_id_t get_tx_id(std::shared_ptr<const packet_t> packet);

// channel_t get_channel(std::shared_ptr<const packet_t> packet);

} // namespace ailoy
