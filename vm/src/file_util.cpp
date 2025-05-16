#include "file_util.hpp"

#include <fstream>

namespace ailoy {
namespace utils {

std::string LoadBytesFromFile(const std::filesystem::path &path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    throw std::invalid_argument{"Cannot open " + path.string()};
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

} // namespace utils
} // namespace ailoy
