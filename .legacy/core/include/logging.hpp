#pragma once

#include <format>

namespace ailoy {

void set_log_level(const std::string &level);

void set_log_format(const std::string &fmt);

void debug(const std::string &msg);

template <typename... args_t>
inline void debug(const std::format_string<std::type_identity_t<args_t>...> fmt,
                  args_t &&...args) {
  return debug(std::format(fmt, std::forward<args_t>(args)...));
}

void info(const std::string &msg);

template <typename... args_t>
inline void info(const std::format_string<std::type_identity_t<args_t>...> fmt,
                 args_t &&...args) {
  return info(std::format(fmt, std::forward<args_t>(args)...));
}

void warn(const std::string &msg);

template <typename... args_t>
inline void warn(const std::format_string<std::type_identity_t<args_t>...> fmt,
                 args_t &&...args) {
  return warn(std::format(fmt, std::forward<args_t>(args)...));
}

void error(const std::string &msg);

template <typename... args_t>
inline void error(const std::format_string<std::type_identity_t<args_t>...> fmt,
                  args_t &&...args) {
  return error(std::format(fmt, std::forward<args_t>(args)...));
}

void critical(const std::string &msg);

template <typename... args_t>
inline void
critical(const std::format_string<std::type_identity_t<args_t>...> fmt,
         args_t &&...args) {
  return critical(std::format(fmt, std::forward<args_t>(args)...));
}

} // namespace ailoy
