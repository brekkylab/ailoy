#ifndef EMSCRIPTEN

#include <filesystem>
#include <fstream>
#include <sstream>

#include "filesystem.hpp"

namespace ailoy {

namespace fs {

result_t create_directory(const std::string &path, bool recursive) {
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

result_t delete_directory(const std::string &path, bool recursive) {
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

result_value_t<bool> directory_exists(const std::string &path) {
  try {
    return result_value_t<bool>(std::filesystem::is_directory(path));
  } catch (const std::filesystem::filesystem_error &e) {
    return result_value_t<bool>(error_code_t::IOError, e.what());
  }
}

result_value_t<std::vector<file_info_t>>
list_directory(const std::string &path) {
  try {
    std::vector<file_info_t> files;
    for (const auto &entry : std::filesystem::directory_iterator(path)) {
      file_info_t info;
      info.name = entry.path().filename().string();
      info.path = entry.path().string();
      info.type = entry.is_directory()      ? file_type_t::Directory
                  : entry.is_regular_file() ? file_type_t::Regular
                                            : file_type_t::Unknown;
      info.size = entry.is_regular_file() ? entry.file_size() : 0;
      files.push_back(info);
    }
    return result_value_t<std::vector<file_info_t>>(std::move(files));
  } catch (const std::filesystem::filesystem_error &e) {
    return result_value_t<std::vector<file_info_t>>(error_code_t::IOError,
                                                    e.what());
  }
}

result_t create_file(const std::string &path) {
  std::ofstream file(path);
  if (!file) {
    return result_t(error_code_t::IOError, "Failed to create file");
  }
  return result_t();
}

result_t delete_file(const std::string &path) {
  try {
    if (!std::filesystem::remove(path)) {
      return result_t(error_code_t::NotFound, "File not found");
    }
    return result_t();
  } catch (const std::filesystem::filesystem_error &e) {
    return result_t(error_code_t::IOError, e.what());
  }
}

result_value_t<bool> file_exists(const std::string &path) {
  try {
    return result_value_t<bool>(std::filesystem::is_regular_file(path));
  } catch (const std::filesystem::filesystem_error &e) {
    return result_value_t<bool>(error_code_t::IOError, e.what());
  }
}

result_value_t<size_t> get_file_size(const std::string &path) {
  try {
    return result_value_t<size_t>(std::filesystem::file_size(path));
  } catch (const std::filesystem::filesystem_error &e) {
    return result_value_t<size_t>(error_code_t::IOError, e.what());
  }
}

result_t write_file(const std::string &path, const std::string &content,
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

result_t write_file(const std::string &path, const std::vector<uint8_t> &data,
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

result_value_t<std::string> read_file_text(const std::string &path) {
  std::ifstream file(path);
  if (!file) {
    return result_value_t<std::string>(error_code_t::IOError,
                                       "Failed to open file");
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return result_value_t<std::string>(buffer.str());
}

result_value_t<std::vector<uint8_t>> read_file_bytes(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return result_value_t<std::vector<uint8_t>>(error_code_t::IOError,
                                                "Failed to open file");
  }
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
  return result_value_t<std::vector<uint8_t>>(std::move(data));
}

std::string get_absolute_path(const std::string &path) {
  return std::filesystem::absolute(path).string();
}

std::string get_parent_path(const std::string &path) {
  return std::filesystem::path(path).parent_path().string();
}

std::string get_file_name(const std::string &path) {
  return std::filesystem::path(path).filename().string();
}

std::string join_path(const std::string &path1, const std::string &path2) {
  return (std::filesystem::path(path1) / path2).string();
}

bool is_absolute(const std::string &path) {
  return std::filesystem::path(path).is_absolute();
}

} // namespace fs

} // namespace ailoy

#endif