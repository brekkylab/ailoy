#pragma once

#include <fstream>
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

class stream_base {
public:
  virtual ~stream_base() = default;
  virtual bool is_open() const = 0;
  virtual void close() = 0;
  virtual bool good() const = 0;
  virtual bool eof() const = 0;
  virtual bool fail() const = 0;
  virtual bool bad() const = 0;
  virtual operator bool() const { return good(); }
  virtual bool operator!() const { return !good(); }

protected:
  path_t path_;
  bool is_open_ = false;
  bool good_ = true;
  bool eof_ = false;
  bool fail_ = false;
  bool bad_ = false;
};

class ifstream_ : public stream_base {
public:
  ifstream_() = default;
  explicit ifstream_(const path_t &path);
  virtual ~ifstream_() = default;

  // Open/close operations
  virtual result_t open(const path_t &path) = 0;
  virtual void close() override = 0;

  // Read operations
  virtual ifstream_ &read(char *buffer, std::streamsize count) = 0;
  virtual ifstream_ &getline(std::string &line, char delim = '\n') = 0;
  virtual std::string read_all() = 0;
  virtual std::vector<uint8_t> read_all_bytes() = 0;

  // Position operations
  virtual std::streampos tellg() = 0;
  virtual ifstream_ &seekg(std::streampos pos) = 0;
  virtual ifstream_ &seekg(std::streamoff off, std::ios_base::seekdir dir) = 0;

  // Character operations
  virtual int get() = 0;
  virtual ifstream_ &get(char &c) = 0;
  virtual int peek() = 0;
  virtual ifstream_ &unget() = 0;
  virtual std::streamsize gcount() const = 0;

  // Stream extraction operators
  virtual ifstream_ &operator>>(std::string &str) = 0;
  virtual ifstream_ &operator>>(int &val) = 0;
  virtual ifstream_ &operator>>(double &val) = 0;
  virtual ifstream_ &operator>>(float &val) = 0;
  virtual ifstream_ &operator>>(long &val) = 0;
  virtual ifstream_ &operator>>(char &c) = 0;
};

class ofstream_ : public stream_base {
public:
  ofstream_() = default;
  explicit ofstream_(const path_t &path, bool append = false);
  virtual ~ofstream_() = default;

  // Open/close operations
  virtual result_t open(const path_t &path, bool append = false) = 0;
  virtual void close() override = 0;

  // Write operations
  virtual ofstream_ &write(const char *buffer, std::streamsize count) = 0;
  virtual ofstream_ &write(const std::string &str) = 0;
  virtual ofstream_ &write(const std::vector<uint8_t> &data) = 0;
  virtual result_t flush() = 0;

  // Position operations
  virtual std::streampos tellp() = 0;
  virtual ofstream_ &seekp(std::streampos pos) = 0;
  virtual ofstream_ &seekp(std::streamoff off, std::ios_base::seekdir dir) = 0;

  // Character operations
  virtual ofstream_ &put(char c) = 0;

  // Stream insertion operators
  virtual ofstream_ &operator<<(const std::string &str) = 0;
  virtual ofstream_ &operator<<(const char *str) = 0;
  virtual ofstream_ &operator<<(char c) = 0;
  virtual ofstream_ &operator<<(int val) = 0;
  virtual ofstream_ &operator<<(double val) = 0;
  virtual ofstream_ &operator<<(float val) = 0;
  virtual ofstream_ &operator<<(long val) = 0;
};

std::unique_ptr<ifstream_> ifstream(const path_t &path);
std::unique_ptr<ofstream_> ofstream(const path_t &path, bool append = false);

} // namespace fs

} // namespace ailoy