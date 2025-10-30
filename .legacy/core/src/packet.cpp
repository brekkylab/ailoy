#include "packet.hpp"

#include <format>
#include <sstream>

#include <magic_enum/magic_enum.hpp>

namespace ailoy {

tx_id_t packet_t::get_tx_id() const {
  if (headers->as<array_t>()->size() < 0)
    return tx_id_t();
  return *headers->at<string_t>(0);
}

channel_t packet_t::get_channel() const {
  if (!itype.has_value())
    throw exception("This type of packet does not have instruction");
  switch (itype.value()) {
  case instruction_type::call_function:
  case instruction_type::define_component:
  case instruction_type::delete_component:
    return std::to_string(static_cast<uint8_t>(itype.value())) + "/" +
           *(headers->at(1)->as<string_t>());
  case instruction_type::call_method:
    return std::to_string(static_cast<uint8_t>(itype.value())) + "/" +
           *(headers->at(1)->as<string_t>()) + "/" +
           *(headers->at(2)->as<string_t>());
  default:
    throw exception("Cannot hanle instruction type");
  }
}

std::ostream &operator<<(std::ostream &os, const packet_type &ptype) {
  switch (ptype) {
  case packet_type::connect:
    os << "connect";
    return os;
  case packet_type::disconnect:
    os << "disconnect";
    return os;
  case packet_type::subscribe:
    os << "subscribe";
    return os;
  case packet_type::unsubscribe:
    os << "unsubscribe";
    return os;
  case packet_type::execute:
    os << "execute";
    return os;
  case packet_type::respond:
    os << "respond";
    return os;
  case packet_type::respond_execute:
    os << "respond_execute";
    return os;
  default:
    return os;
  }
}

std::ostream &operator<<(std::ostream &os, const instruction_type &itype) {
  switch (itype) {
  case instruction_type::call_function:
    os << "call_function";
    return os;
  case instruction_type::define_component:
    os << "define_component";
    return os;
  case instruction_type::delete_component:
    os << "delete_component";
    return os;
  case instruction_type::call_method:
    os << "call_method";
    return os;
  default:
    return os;
  }
}

packet_t::operator std::string() const {
  switch (ptype) {
  case packet_type::connect:
  case packet_type::disconnect:
    return std::format("{} {}", get_tx_id(), magic_enum::enum_name(ptype));
  case packet_type::subscribe:
  case packet_type::unsubscribe:
  case packet_type::execute:
    return std::format("{} {} {}", get_tx_id(), magic_enum::enum_name(ptype),
                       magic_enum::enum_name(itype.value()));
  case packet_type::respond:
    return std::format("{} {}", get_tx_id(), magic_enum::enum_name(ptype));
  case packet_type::respond_execute:
    return std::format("{} {} idx {} fin {}", get_tx_id(),
                       magic_enum::enum_name(ptype),
                       std::to_string(*headers->at<uint_t>(1)),
                       std::to_string(*headers->at<bool_t>(2)));
  default:
    return std::string();
  }
}

std::shared_ptr<bytes_t> dump_packet(std::shared_ptr<packet_t> packet) {
  packet_type ptype = packet->ptype;
  std::optional<instruction_type> itype = packet->itype;
  std::shared_ptr<array_t> headers = packet->headers;
  std::shared_ptr<value_t> body = packet->body;

  size_t preamble_bytes_size = sizeof(packet_type);
  if (itype.has_value())
    preamble_bytes_size += sizeof(instruction_type);
  std::shared_ptr<bytes_t> header_bytes = nullptr;
  size_t header_bytes_size = 0;
  if (headers) {
    header_bytes = headers->encode(encoding_method_t::cbor);
    header_bytes_size += header_bytes->size();
  }
  std::shared_ptr<bytes_t> body_bytes = nullptr;
  size_t body_bytes_size = 0;
  if (body) {
    body_bytes = body->encode(encoding_method_t::cbor);
    body_bytes_size += body_bytes->size();
  }
  auto bytes_size = preamble_bytes_size + sizeof(uint16_t) + header_bytes_size +
                    sizeof(uint32_t) + body_bytes_size;
  auto rv = create<bytes_t>(bytes_size);
  auto it = rv->begin();

  reinterpret_cast<packet_type &>(*it) = ptype;
  it += sizeof(packet_type);
  if (itype.has_value()) {
    reinterpret_cast<instruction_type &>(*it) = itype.value();
    it += sizeof(instruction_type);
  }

  if (header_bytes) {
    reinterpret_cast<uint16_t &>(*it) =
        reinterpret_cast<uint16_t &>(header_bytes_size);
    it += sizeof(uint16_t);
    std::copy(header_bytes->begin(), header_bytes->end(), it);
    it += header_bytes_size;
  }

  if (body_bytes) {
    reinterpret_cast<uint32_t &>(*it) = body_bytes_size;
    it += sizeof(uint32_t);
    std::copy(body_bytes->begin(), body_bytes->end(), it);
    it += body_bytes_size;
  }
  return rv;
}

std::shared_ptr<packet_t> load_packet(std::shared_ptr<bytes_t> packet_bytes,
                                      bool skip_body) {
  auto it = packet_bytes->begin();

  packet_type ptype = reinterpret_cast<packet_type &>(*it);
  it += sizeof(packet_type);

  std::optional<instruction_type> itype;
  if (ptype == packet_type::subscribe || ptype == packet_type::unsubscribe ||
      ptype == packet_type::execute) {
    instruction_type itype_value;
    itype_value = reinterpret_cast<instruction_type &>(*it);
    it += sizeof(instruction_type);
    itype = itype_value;
  }

  uint16_t header_bytes_size = reinterpret_cast<uint16_t &>(*it);
  it += sizeof(uint16_t);
  auto header_bytes = ailoy::create<bytes_t>(header_bytes_size);
  std::copy(it, it + header_bytes_size, header_bytes->data());
  it += header_bytes_size;
  auto headers =
      ailoy::decode(header_bytes, encoding_method_t::cbor)->as<array_t>();

  std::shared_ptr<map_t> body = nullptr;
  if (!skip_body) {
    uint32_t body_bytes_size = reinterpret_cast<uint32_t &>(*it);
    it += sizeof(uint32_t);
    if (body_bytes_size > 0) {
      auto body_bytes = ailoy::create<bytes_t>(body_bytes_size);
      std::copy(it, it + body_bytes_size, body_bytes->data());
      body = ailoy::decode(body_bytes, encoding_method_t::cbor)->as<map_t>();
    }
  }

  return ailoy::create<packet_t>(ptype, itype, headers, body);
}

} // namespace ailoy
