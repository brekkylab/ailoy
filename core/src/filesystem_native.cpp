#ifndef EMSCRIPTEN

#include <filesystem>
#include <fstream>
#include <sstream>

#include "filesystem.hpp"

namespace ailoy {

namespace fs {

// ============================================================================
// Directory Operations
// ============================================================================

result_t create_directory(const path_t &path, bool recursive) {
  try {
    std::filesystem::path p(path);
    if (recursive) {
      std::filesystem::create_directories(p);
    } else {
      if (!std::filesystem::create_directory(p)) {
        if (std::filesystem::exists(p)) {
          return result_t(error_code_t::AlreadyExists,
                          "Directory already exists");
        }
        return result_t(error_code_t::IOError, "Failed to create directory");
      }
    }
    return result_t();
  } catch (const std::filesystem::filesystem_error &e) {
    return result_t(error_code_t::IOError, e.what());
  }
}

result_t delete_directory(const path_t &path, bool recursive) {
  try {
    std::filesystem::path p(path);
    if (recursive) {
      std::filesystem::remove_all(p);
    } else {
      if (!std::filesystem::remove(p)) {
        return result_t(error_code_t::NotFound, "Directory not found");
      }
    }
    return result_t();
  } catch (const std::filesystem::filesystem_error &e) {
    return result_t(error_code_t::IOError, e.what());
  }
}

result_value_t<bool> directory_exists(const path_t &path) {
  try {
    return result_value_t<bool>(std::filesystem::is_directory(path.string()));
  } catch (const std::filesystem::filesystem_error &e) {
    return result_value_t<bool>(error_code_t::IOError, e.what());
  }
}

result_value_t<std::vector<dir_entry_t>> list_directory(const path_t &path) {
  try {
    std::vector<dir_entry_t> files;
    for (const auto &entry :
         std::filesystem::directory_iterator(path.string())) {
      dir_entry_t info;
      info.name = entry.path().filename().string();
      info.path = entry.path().string();
      info.type = entry.is_directory()      ? file_type_t::Directory
                  : entry.is_regular_file() ? file_type_t::Regular
                                            : file_type_t::Unknown;
      info.size = entry.is_regular_file() ? entry.file_size() : 0;
      files.push_back(info);
    }
    return result_value_t<std::vector<dir_entry_t>>(files);
  } catch (const std::filesystem::filesystem_error &e) {
    return result_value_t<std::vector<dir_entry_t>>(error_code_t::IOError,
                                                    e.what());
  }
}

// ============================================================================
// File Operations
// ============================================================================

result_t create_file(const path_t &path) {
  std::ofstream file(path);
  if (!file) {
    return result_t(error_code_t::IOError, "Failed to create file");
  }
  return result_t();
}

result_t delete_file(const path_t &path) {
  try {
    if (!std::filesystem::remove(path.string())) {
      return result_t(error_code_t::NotFound, "File not found");
    }
    return result_t();
  } catch (const std::filesystem::filesystem_error &e) {
    return result_t(error_code_t::IOError, e.what());
  }
}

result_value_t<bool> file_exists(const path_t &path) {
  try {
    return result_value_t<bool>(
        std::filesystem::is_regular_file(path.string()));
  } catch (const std::filesystem::filesystem_error &e) {
    return result_value_t<bool>(error_code_t::IOError, e.what());
  }
}

result_value_t<size_t> get_file_size(const path_t &path) {
  try {
    return result_value_t<size_t>(std::filesystem::file_size(path.string()));
  } catch (const std::filesystem::filesystem_error &e) {
    return result_value_t<size_t>(error_code_t::IOError, e.what());
  }
}

// ============================================================================
// Directory/File Common Operations
// ============================================================================

result_value_t<bool> exists(const path_t &path) {
  auto exists_ = directory_exists(path).unwrap() || file_exists(path).unwrap();
  return result_value_t<bool>(exists_);
}

// ============================================================================
// Read/Write Operations
// ============================================================================

result_t write_file(const path_t &path, const std::string &content,
                    bool append) {
  std::ofstream file(path, append ? std::ios::app : std::ios::trunc);
  if (!file) {
    return result_t(error_code_t::IOError, "Failed to open file for writing");
  }
  file << content;
  if (!file) {
    return result_t(error_code_t::IOError, "Failed to write to file");
  }
  return result_t();
}

result_t write_file(const path_t &path, const std::vector<uint8_t> &data,
                    bool append) {
  std::ofstream file(path, std::ios::binary |
                               (append ? std::ios::app : std::ios::trunc));
  if (!file) {
    return result_t(error_code_t::IOError, "Failed to open file for writing");
  }
  file.write(reinterpret_cast<const char *>(data.data()), data.size());
  if (!file) {
    return result_t(error_code_t::IOError, "Failed to write to file");
  }
  return result_t();
}

result_value_t<std::string> read_file_text(const path_t &path) {
  std::ifstream file(path);
  if (!file) {
    return result_value_t<std::string>(error_code_t::IOError,
                                       "Failed to open file");
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return result_value_t<std::string>(buffer.str());
}

result_value_t<std::vector<uint8_t>> read_file_bytes(const path_t &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return result_value_t<std::vector<uint8_t>>(error_code_t::IOError,
                                                "Failed to open file");
  }
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
  return result_value_t<std::vector<uint8_t>>(std::move(data));
}

} // namespace fs

} // namespace ailoy

#endif