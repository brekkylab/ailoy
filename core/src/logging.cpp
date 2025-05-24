#include "logging.hpp"

#include <algorithm>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "exception.hpp"

namespace ailoy {

std::mutex m;

std::shared_ptr<spdlog::logger> get_logger() {
  static std::shared_ptr<spdlog::logger> logger = nullptr;

  if (!logger) {
    logger = spdlog::stdout_color_mt("AILOY");

    const char *env = std::getenv("AILOY_LOG_LEVEL");
    if (env) {
      // @jhlee: We cannot reuse `set_log_level` since it occurs infinite loop
      std::string lvl = env;
      std::transform(lvl.begin(), lvl.end(), lvl.begin(), ::tolower);
      if (lvl == "debug")
        logger->set_level(spdlog::level::debug);
      else if (lvl == "info")
        logger->set_level(spdlog::level::info);
      else if (lvl == "warn" || lvl == "warning")
        logger->set_level(spdlog::level::warn);
      else if (lvl == "error")
        logger->set_level(spdlog::level::err);
      else if (lvl == "critical")
        logger->set_level(spdlog::level::critical);
      else
        logger->warn("Unknown log level: {}", lvl);
    }
  }

  return logger;
}

void set_log_level(const std::string &level) {
  std::string lvl = level;
  std::transform(lvl.begin(), lvl.end(), lvl.begin(), ::tolower);
  if (lvl == "debug")
    get_logger()->set_level(spdlog::level::debug);
  else if (lvl == "info")
    get_logger()->set_level(spdlog::level::info);
  else if (lvl == "warn" || lvl == "warning")
    get_logger()->set_level(spdlog::level::warn);
  else if (lvl == "error")
    get_logger()->set_level(spdlog::level::err);
  else if (lvl == "critical")
    get_logger()->set_level(spdlog::level::critical);
  else
    get_logger()->warn("Unknown log level: {}", level);
}

void set_log_format(const std::string &fmt) { get_logger()->set_pattern(fmt); }

void debug(const std::string &msg) { get_logger()->debug(msg); }

void info(const std::string &msg) { get_logger()->info(msg); }

void warn(const std::string &msg) { get_logger()->warn(msg); }

void error(const std::string &msg) { get_logger()->error(msg); }

void critical(const std::string &msg) { get_logger()->critical(msg); }

} // namespace ailoy
