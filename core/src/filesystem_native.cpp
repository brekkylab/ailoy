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

class native_ifstream : public ifstream_ {
private:
  std::ifstream file_;

public:
  native_ifstream() = default;
  explicit native_ifstream(const path_t &path) { open(path); }
  ~native_ifstream() override { close(); }

  // Open/close operations
  result_t open(const path_t &path) override {
    path_ = path;
    file_.open(path.string(), std::ios::binary);

    if (!file_.is_open()) {
      fail_ = true;
      return result_t(error_code_t::IOError,
                      "Failed to open file: " + path.string());
    }

    is_open_ = true;
    good_ = true;
    return result_t();
  }
  void close() override {
    if (file_.is_open()) {
      file_.close();
    }
    is_open_ = false;
  }
  bool is_open() const override { return file_.is_open(); }
  bool good() const override { return file_.good(); }
  bool eof() const override { return file_.eof(); }
  bool fail() const override { return file_.fail(); }
  bool bad() const override { return file_.bad(); }

  // Read operations
  ifstream_ &read(char *buffer, std::streamsize count) override {
    file_.read(buffer, count);
    return *this;
  }
  ifstream_ &getline(std::string &line, char delim = '\n') override {
    std::getline(file_, line, delim);
    return *this;
  }
  std::string read_all() override {
    std::ostringstream ss;
    ss << file_.rdbuf();
    return ss.str();
  }
  std::vector<uint8_t> read_all_bytes() override {
    file_.seekg(0, std::ios::end);
    size_t size = file_.tellg();
    file_.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    file_.read(reinterpret_cast<char *>(buffer.data()), size);
    return buffer;
  }

  // Position operations
  std::streampos tellg() override { return file_.tellg(); }
  ifstream_ &seekg(std::streampos pos) override {
    file_.seekg(pos);
    return *this;
  }
  ifstream_ &seekg(std::streamoff off, std::ios_base::seekdir dir) override {
    file_.seekg(off, dir);
    return *this;
  }

  // Character operations
  int get() override { return file_.get(); }
  ifstream_ &get(char &c) override {
    file_.get(c);
    return *this;
  }
  int peek() override { return file_.peek(); }
  ifstream_ &unget() override {
    file_.unget();
    return *this;
  }
  std::streamsize gcount() const override { return file_.gcount(); }

  // Stream extraction operators
  ifstream_ &operator>>(std::string &str) override {
    file_ >> str;
    return *this;
  }
  ifstream_ &operator>>(int &val) override {
    file_ >> val;
    return *this;
  }
  ifstream_ &operator>>(double &val) override {
    file_ >> val;
    return *this;
  }
  ifstream_ &operator>>(float &val) override {
    file_ >> val;
    return *this;
  }
  ifstream_ &operator>>(long &val) override {
    file_ >> val;
    return *this;
  }
  ifstream_ &operator>>(char &c) override {
    file_ >> c;
    return *this;
  }
};

class native_ofstream : public ofstream_ {
private:
  std::ofstream file_;

public:
  native_ofstream() = default;
  explicit native_ofstream(const path_t &path, bool append = false) {
    open(path, append);
  }
  ~native_ofstream() override { close(); }

  // Implementation of base class methods
  result_t open(const path_t &path, bool append = false) override {
    path_ = path;
    auto mode = std::ios::binary;
    if (append) {
      mode |= std::ios::app;
    }

    file_.open(path.string(), mode);

    if (!file_.is_open()) {
      fail_ = true;
      return result_t(error_code_t::IOError,
                      "Failed to open file: " + path.string());
    }

    is_open_ = true;
    good_ = true;
    return result_t();
  }
  void close() override {
    if (file_.is_open()) {
      file_.close();
    }
    is_open_ = false;
  }
  bool is_open() const override { return file_.is_open(); }
  bool good() const override { return file_.good(); }
  bool eof() const override { return file_.eof(); }
  bool fail() const override { return file_.fail(); }
  bool bad() const override { return file_.bad(); }

  // Write operations
  ofstream_ &write(const char *buffer, std::streamsize count) override {
    file_.write(buffer, count);
    return *this;
  }
  ofstream_ &write(const std::string &str) override {
    file_.write(str.c_str(), str.size());
    return *this;
  }
  ofstream_ &write(const std::vector<uint8_t> &data) override {
    file_.write(reinterpret_cast<const char *>(data.data()), data.size());
    return *this;
  }
  result_t flush() override {
    file_.flush();
    return result_t();
  }

  // Position operations
  std::streampos tellp() override { return file_.tellp(); }
  ofstream_ &seekp(std::streampos pos) override {
    file_.seekp(pos);
    return *this;
  }
  ofstream_ &seekp(std::streamoff off, std::ios_base::seekdir dir) override {
    file_.seekp(off, dir);
    return *this;
  }

  // Character operations
  ofstream_ &put(char c) override {
    file_.put(c);
    return *this;
  }

  // Stream insertion operators
  ofstream_ &operator<<(const std::string &str) override {
    file_ << str;
    return *this;
  }
  ofstream_ &operator<<(const char *str) override {
    file_ << str;
    return *this;
  }
  ofstream_ &operator<<(char c) override {
    file_ << c;
    return *this;
  }
  ofstream_ &operator<<(int val) override {
    file_ << val;
    return *this;
  }
  ofstream_ &operator<<(double val) override {
    file_ << val;
    return *this;
  }
  ofstream_ &operator<<(float val) override {
    file_ << val;
    return *this;
  }
  ofstream_ &operator<<(long val) override {
    file_ << val;
    return *this;
  }
};

std::unique_ptr<ifstream_> ifstream(const path_t &path) {
  return std::make_unique<native_ifstream>(path);
}

std::unique_ptr<ofstream_> ofstream(const path_t &path, bool append) {
  return std::make_unique<native_ofstream>(path, append);
}

} // namespace fs

} // namespace ailoy

#endif