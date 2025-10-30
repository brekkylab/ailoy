#include <algorithm>
#include <concepts>
#include <ranges>
#include <sstream>
#include <string_view>
#include <vector>

namespace ailoy {
namespace utils {

std::vector<std::string> split_text(const std::string &s,
                                    const std::string &delimiter);

/**
 * chunks can be one of
 *  - C-style array
 *  - std::vector
 *  - std::deque
 *  - std::initializer_list
 *  - std::ranges::views
 *  - etc.
 * of {std::string, std:string_view, const char*}.
 */
template <std::ranges::input_range R>
  requires std::convertible_to<std::ranges::range_reference_t<R>,
                               std::string_view>
std::string join(std::string_view delimiter, R &&chunks) {
  auto begin = std::ranges::begin(chunks);
  auto end = std::ranges::end(chunks);

  if (begin == end)
    return "";

  std::ostringstream oss;
  oss << std::string_view(*begin++);
  for (; begin != end; ++begin)
    oss << delimiter << std::string_view(*begin);
  return oss.str();
}

inline std::string &ltrim(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
          }));
  return s;
}

inline std::string &rtrim(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(),
                       [](unsigned char ch) { return !std::isspace(ch); })
              .base(),
          s.end());
  return s;
}

inline std::string &trim(std::string &s) { return ltrim(rtrim(s)); }

} // namespace utils
} // namespace ailoy
