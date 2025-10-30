#include "string_util.hpp"

namespace ailoy {
namespace utils {

std::vector<std::string> split_text(const std::string &s,
                                    const std::string &delimiter) {
  std::vector<std::string> result;
  for (auto &&subrange : std::ranges::split_view(s, delimiter))
    result.emplace_back(subrange.begin(), subrange.end());
  return result;
}

} // namespace utils
} // namespace ailoy
