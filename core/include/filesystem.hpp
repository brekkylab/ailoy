#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ailoy {

namespace fs {

class path_t {
public:
  path_t() = default;
  path_t(const std::string &p) : path_(p) {}
  path_t(const char *p) : path_(p) {}

  // Allow implicit conversion to string
  operator std::string() const { return path_; }
  const std::string &string() const { return path_; }

  // Implicit operator/ for path concatenation
  path_t operator/(const std::string &other) const { return join_paths(other); }
  path_t operator/(const char *other) const {
    return join_paths(std::string(other));
  }
  path_t operator/(const path_t &other) const {
    return join_paths(other.path_);
  }

  // Assignment operators
  path_t &operator/=(const std::string &other) {
    path_ = join_paths(other);
    return *this;
  }
  path_t &operator/=(const char *other) {
    path_ = join_paths(std::string(other));
    return *this;
  }
  path_t &operator/=(const path_t &other) {
    path_ = join_paths(other.path_);
    return *this;
  }

  // Comparison operators
  bool operator==(const path_t &other) const { return path_ == other.path_; }
  bool operator!=(const path_t &other) const { return path_ != other.path_; }
  bool operator<(const path_t &other) const { return path_ < other.path_; }

  // stream output
  friend std::ostream &operator<<(std::ostream &os, const path_t &p) {
    return os << p.path_;
  }

  path_t parent() const {
    if (path_.empty() || path_ == "/")
      return path_t("/");

    // Remove trailing slash if present (except for root)
    std::string working_path = path_;
    if (working_path.length() > 1 && working_path.back() == '/') {
      working_path.pop_back();
    }

    // Find the last slash
    size_t last_slash = working_path.find_last_of('/');
    if (last_slash == std::string::npos) {
      // No slash found, return empty or current directory
      return path_t("");
    }

    if (last_slash == 0) {
      // Root directory case
      return path_t("/");
    }

    // Return everything up to (but not including) the last slash
    return path_t(working_path.substr(0, last_slash));
  }

  std::string filename() const {
    if (path_.empty()) {
      return "";
    }

    // Handle root path
    if (path_ == "/") {
      return "";
    }

    // Remove trailing slashes (except for root)
    std::string working_path = path_;
    while (working_path.length() > 1 && working_path.back() == '/') {
      working_path.pop_back();
    }

    // Find the last slash
    size_t last_slash = working_path.find_last_of('/');
    if (last_slash == std::string::npos) {
      // No slash found, entire path is the filename
      return working_path;
    }

    // Return everything after the last slash
    return working_path.substr(last_slash + 1);
  }

  std::string extension() const {
    std::string fname = filename();

    // Empty filename or filename starting with '.' (hidden files)
    if (fname.empty() || fname == "." || fname == "..") {
      return "";
    }

    // If filename starts with '.' and has no other dots, no extension
    if (fname[0] == '.' && fname.find('.', 1) == std::string::npos) {
      return "";
    }

    // Find the last dot
    size_t last_dot = fname.find_last_of('.');
    if (last_dot == std::string::npos || last_dot == 0) {
      return "";
    }

    // Return everything from the last dot onwards
    return fname.substr(last_dot);
  }

  std::string stem() const {
    std::string fname = filename();

    // Empty filename or special cases
    if (fname.empty() || fname == "." || fname == "..") {
      return fname;
    }

    // If filename starts with '.' and has no other dots, return as is
    if (fname[0] == '.' && fname.find('.', 1) == std::string::npos) {
      return fname;
    }

    // Find the last dot
    size_t last_dot = fname.find_last_of('.');
    if (last_dot == std::string::npos || last_dot == 0) {
      return fname;
    }

    // Return everything before the last dot
    return fname.substr(0, last_dot);
  }

private:
  std::string path_;

  path_t join_paths(std::string other) const {
    if (path_.empty())
      return other;
    if (other.empty())
      return *this;

    // Handle ".." - go to parent directory
    if (other == "..") {
      return parent();
    }

    // Normal path joining
    if (path_.back() == '/')
      return path_ + other;
    return path_ + "/" + other;
  }
};

// Global operator/ for string / path_t combinations
inline path_t operator/(const std::string &left, const path_t &right) {
  return path_t(left) / right;
}
inline path_t operator/(const char *left, const path_t &right) {
  return path_t(left) / right;
}

enum class file_type_t { Regular, Directory, Unknown };

struct dir_entry_t {
  std::string name;
  path_t path;
  file_type_t type;
  size_t size;

  bool is_regular_file() const { return type == file_type_t::Regular; }
  bool is_directory() const { return type == file_type_t::Directory; }
};

enum class error_code_t {
  Success,
  NotFound,
  AlreadyExists,
  PermissionDenied,
  InvalidPath,
  IOError,
  NotSupported,
  Unknown
};

struct result_t {
  error_code_t code;
  std::string message;

  result_t(error_code_t c = error_code_t::Success, const std::string &msg = "")
      : code(c), message(msg) {}

  bool success() const { return code == error_code_t::Success; }
  operator bool() const { return success(); }
};

template <typename T> struct result_value_t {
  result_t result_;
  std::optional<T> value_;

  result_value_t(error_code_t code, const std::string &msg = "")
      : result_(code, msg) {}

  result_value_t(T &&val)
      : result_(error_code_t::Success), value_(std::move(val)) {}

  result_value_t(const T &val) : result_(error_code_t::Success), value_(val) {}

  T unwrap() const {
    if (!value_.has_value()) {
      throw std::runtime_error("Attemting to convert failed result to value: " +
                               result_.message);
    }
    return value_.value();
  }

  T &unwrap() {
    if (!value_.has_value()) {
      throw std::runtime_error("Attemting to convert failed result to value: " +
                               result_.message);
    }
    return value_.value();
  }

  operator T() const {
    if (!value_.has_value()) {
      throw std::runtime_error("Attemting to convert failed result to value: " +
                               result_.message);
    }
    return value_.value();
  }

  operator T &() {
    if (!value_.has_value()) {
      throw std::runtime_error("Attemting to convert failed result to value: " +
                               result_.message);
    }
    return value_.value();
  }

  operator const T &() const {
    if (!value_.has_value()) {
      throw std::runtime_error("Attemting to convert failed result to value: " +
                               result_.message);
    }
    return value_.value();
  }
};

// Directory operations
result_t create_directory(const path_t &path, bool recursive = false);
result_t delete_directory(const path_t &path, bool recursive = false);
result_value_t<bool> directory_exists(const path_t &path);
result_value_t<std::vector<dir_entry_t>> list_directory(const path_t &path);

// File operations
result_t create_file(const path_t &path);
result_t delete_file(const path_t &path);
result_value_t<bool> file_exists(const path_t &path);
result_value_t<size_t> get_file_size(const path_t &path);

// Directory/File common operations
result_value_t<bool> exists(const path_t &path);

// Read/Write operations
result_t write_file(const path_t &path, const std::string &content,
                    bool append = false);
result_t write_file(const path_t &path, const std::vector<uint8_t> &data,
                    bool append = false);
result_value_t<std::string> read_file_text(const path_t &path);
result_value_t<std::vector<uint8_t>> read_file_bytes(const path_t &path);

} // namespace fs

} // namespace ailoy