#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ailoy {

namespace fs {

enum class file_type_t { Regular, Directory, Unknown };

struct file_info_t {
  std::string name;
  std::string path;
  file_type_t type;
  size_t size;
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

// Path class for operator/ support
class path_t {
public:
  path_t(const std::string &p) : path_(p) {}
  path_t(const char *p) : path_(p) {}

  // Allow implicit conversion to string
  operator std::string() const { return path_; }
  const std::string &string() const { return path_; }

  // Implicit operator/ for path concatenation
  path_t operator/(const std::string &other) const {
    return path_t(join_paths(path_, other));
  }
  path_t operator/(const char *other) const {
    return path_t(join_paths(path_, std::string(other)));
  }
  path_t operator/(const path_t &other) const {
    return path_t(join_paths(path_, other.path_));
  }

  // Assignment operators
  path_t &operator/=(const std::string &other) {
    path_ = join_paths(path_, other);
    return *this;
  }
  path_t &operator/=(const char *other) {
    path_ = join_paths(path_, std::string(other));
    return *this;
  }
  path_t &operator/=(const path_t &other) {
    path_ = join_paths(path_, other.path_);
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

private:
  std::string path_;

  static std::string join_paths(const std::string &path1,
                                const std::string &path2) {
    if (path1.empty())
      return path2;
    if (path2.empty())
      return path1;

    // Handle ".." - go to parent directory
    if (path2 == "..") {
      return get_parent_path(path1);
    }

    // Normal path joining
    if (path1.back() == '/')
      return path1 + path2;
    return path1 + "/" + path2;
  }

  static std::string get_parent_path(const std::string &path) {
    if (path.empty() || path == "/")
      return "/";

    // Remove trailing slash if present (except for root)
    std::string working_path = path;
    if (working_path.length() > 1 && working_path.back() == '/') {
      working_path.pop_back();
    }

    // Find the last slash
    size_t last_slash = working_path.find_last_of('/');
    if (last_slash == std::string::npos) {
      // No slash found, return empty or current directory
      return "";
    }

    if (last_slash == 0) {
      // Root directory case
      return "/";
    }

    // Return everything up to (but not including) the last slash
    return working_path.substr(0, last_slash);
  }
};

// Global operator/ for string / path_t combinations
inline path_t operator/(const std::string &left, const path_t &right) {
  return path_t(left) / right;
}
inline path_t operator/(const char *left, const path_t &right) {
  return path_t(left) / right;
}

// Directory operations
result_t create_directory(const std::string &path, bool recursive = false);
result_t delete_directory(const std::string &path, bool recursive = false);
result_value_t<bool> directory_exists(const std::string &path);
result_value_t<std::vector<file_info_t>>
list_directory(const std::string &path);

// File operations
result_t create_file(const std::string &path);
result_t delete_file(const std::string &path);
result_value_t<bool> file_exists(const std::string &path);
result_value_t<size_t> get_file_size(const std::string &path);

// Read/Write operations
result_t write_file(const std::string &path, const std::string &content,
                    bool append = false);
result_t write_file(const std::string &path, const std::vector<uint8_t> &data,
                    bool append = false);
result_value_t<std::string> read_file_text(const std::string &path);
result_value_t<std::vector<uint8_t>> read_file_bytes(const std::string &path);

// Path operations
std::string get_absolute_path(const std::string &path);
std::string get_parent_path(const std::string &path);
std::string get_file_name(const std::string &path);
std::string join_path(const std::string &path1, const std::string &path2);
bool is_absolute(const std::string &path);

} // namespace fs

} // namespace ailoy