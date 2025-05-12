#include <vector>

#include "component.hpp"

namespace ailoy {

std::vector<std::string>
split_text_by_separator(const std::string text, const size_t chunk_size = 4000,
                        const size_t chunk_overlap = 200,
                        const std::string separator = "\n\n",
                        const std::string length_function = "default");

std::vector<std::string> split_text_by_separators_recursively(
    const std::string text, const size_t chunk_size = 4000,
    const size_t chunk_overlap = 200,
    const std::vector<std::string> separator = {"\n\n", "\n", " ", ""},
    const std::string length_function = "default");
} // namespace ailoy
