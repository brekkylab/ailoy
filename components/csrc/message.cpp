#include "message.hpp"

#include <magic_enum/magic_enum.hpp>
#include <nlohmann/json.hpp>

#include "exception.hpp"

namespace ailoy {

std::ostream &operator<<(std::ostream &os, const content_datatype_t &v) {
  os << magic_enum::enum_name(v);
  return os;
}

void to_json(nlohmann::json &j, const content_t &v) {
  j = nlohmann::json::object();
  j["type"] = magic_enum::enum_name(v.ty);
  j[v.key] = v.value;
}

void from_json(const nlohmann::json &j, content_t &v) {
  if (!j.is_object())
    throw exception("Error while parsing message: content is not dict");
  if (!j.contains("type"))
    throw exception(
        "Error while parsing message: content does not contains type");

  auto ty1 = j.at("type");
  if (!ty1.is_string())
    throw exception("Error while parsing message: content type is not string");
  auto ty2 = magic_enum::enum_cast<content_datatype_t>(ty1.get<std::string>());
  if (!ty2.has_value())
    throw exception("Error while parsing message: unknown content type " +
                    ty1.get<std::string>());
  v.ty = ty2.value();

  for (auto it = j.begin(); it != j.end(); ++it) {
    if (it.key() == "type")
      continue;
    v.key = it.key();
    v.value = it.value();
    break;
  }
}

std::ostream &operator<<(std::ostream &os, const content_t &v) {
  nlohmann::json j = v;
  os << j.dump();
  return os;
}

std::ostream &operator<<(std::ostream &os, const role_t &v) {
  os << magic_enum::enum_name(v);
  return os;
}

std::ostream &operator<<(std::ostream &os, const content_category_t &v) {
  os << magic_enum::enum_name(v);
  return os;
}

void to_json(nlohmann::json &j, const message_t &m) {
  j = nlohmann::json::object();
  j["role"] = magic_enum::enum_name(m.role);
  j[magic_enum::enum_name(m.key)] = m.value;
}

void from_json(const nlohmann::json &j, message_t &m) {
  if (!j.is_object())
    throw exception("Error while parsing message: message is not dict");
  if (!j.contains("role"))
    throw exception(
        "Error while parsing message: message does not contains role");

  auto role = j.at("role");
  if (!role.is_string())
    throw exception("Error while parsing message: role is not string");
  auto role2 = magic_enum::enum_cast<role_t>(role.get<std::string>());
  if (!role2.has_value())
    throw exception("Error while parsing message: unknown role type " +
                    role.get<std::string>());
  m.role = role2.value();

  for (auto it = j.begin(); it != j.end(); ++it) {
    if (it.key() == "role")
      continue;
    m.key = magic_enum::enum_cast<content_category_t>(it.key()).value();
    m.value = std::move(it.value().get<std::vector<content_t>>());
    break;
  }
}

std::ostream &operator<<(std::ostream &os, const message_t &v) {
  nlohmann::json j = v;
  os << j.dump();
  return os;
}

std::ostream &operator<<(std::ostream &os, const messages_t &v) {
  nlohmann::json j = v;
  os << j.dump();
  return os;
}

} // namespace ailoy
