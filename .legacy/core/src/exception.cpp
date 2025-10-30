#include "exception.hpp"

#ifndef EMSCRIPTEN
#include <cpptrace/cpptrace.hpp>
#endif
#include <format>

namespace ailoy {

std::string build_errstr(const char *what) {
#ifdef EMSCRIPTEN
  // TODO: add traceback on Emscripten
  // https://github.com/jeremy-rifkin/cpptrace/issues/175
  return std::format("\033[1;31m{}\033[0m", what);
#else
  auto trace = cpptrace::generate_trace();
  auto &frames = trace.frames;
  while (true) {
    if (frames.empty())
      break;
    auto front_symbol = frames.at(0).symbol;
    if (front_symbol.starts_with("ailoy::build_errstr"))
      frames.erase(frames.begin());
    else
      break;
  }
  return std::format("\033[1;31m{}\033[0m\n{}", what, trace.to_string(true));
#endif
}

not_implemented::not_implemented() : errstr_(std::format("Not implemented:")) {}

not_implemented::not_implemented(const std::string &what)
    : errstr_(std::format("Not implemented: {}", what)) {}

const char *not_implemented::what() const noexcept { return errstr_.c_str(); }

} // namespace ailoy
