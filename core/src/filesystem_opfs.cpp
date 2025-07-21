#ifdef EMSCRIPTEN

#include <iostream>
#include <sstream>

#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <magic_enum/magic_enum.hpp>

#include "filesystem.hpp"

using namespace emscripten;

namespace ailoy {
namespace fs {

// A utility to convert JS error names to our C++ error codes
error_code_t js_error_to_code(val error) {
  std::string name = error["name"].as<std::string>();
  if (name == "NotFoundError")
    return error_code_t::NotFound;
  if (name == "NotAllowedError")
    return error_code_t::PermissionDenied;
  if (name == "TypeMismatchError")
    return error_code_t::InvalidPath;
  if (name == "InvalidModificationError")
    return error_code_t::AlreadyExists;
  return error_code_t::Unknown;
}

// ============================================================================
// Directory Operations
// ============================================================================

EM_ASYNC_JS(EM_VAL, js_create_directory,
            (const char *parent_path_str, const char *dir_name, bool recursive),
            {
              // clang-format off
              try {
                const parentPath = UTF8ToString(parent_path_str);
                const pathParts = parentPath.split("/");
                let handle = await navigator.storage.getDirectory();

                for (let i = 0; i < pathParts.length; i++) {
                    const part = pathParts[i];
                    if (part === "")
                        continue;

                    // Create parent dirs if recursive enabled
                    handle = await handle.getDirectoryHandle(part, {
                        create: recursive,
                    });
                }

                const dirName = UTF8ToString(dir_name);
                await handle.getDirectoryHandle(dirName, {create: true});

                return Emval.toHandle(undefined);
              } catch (error) {
                return Emval.toHandle({name: error.name, message: error.message});
              }
              // clang-format on
            });

result_t create_directory(const path_t &path, bool recursive) {
  path_t parent_path = path.parent();
  std::string dir_name = path.filename();
  val result = val::take_ownership(js_create_directory(
      parent_path.string().c_str(), dir_name.c_str(), recursive));

  if (result.instanceof(val::global("Error"))) {
    return result_t(js_error_to_code(result),
                    result["message"].as<std::string>());
  }

  return result_t();
}

EM_ASYNC_JS(EM_VAL, js_delete_directory,
            (const char *parent_path_str, const char *dir_name, bool recursive),
            {
              // clang-format off
              try {
                const parentPath = UTF8ToString(parent_path_str);
                const pathParts = parentPath.split("/");
                let handle = await navigator.storage.getDirectory();

                for (let i = 0; i < pathParts.length; i++) {
                    const part = pathParts[i];
                    if (part === "")
                        continue;

                    handle = await handle.getDirectoryHandle(part);
                }

                const dirName = UTF8ToString(dir_name);
                await handle.removeEntry(dirName, {
                    recursive: recursive,
                });

                return Emval.toHandle(undefined);
              } catch (error) {
                return Emval.toHandle({name: error.name, message: error.message});
              }
              // clang-format on
            });

result_t delete_directory(const path_t &path, bool recursive) {
  path_t parent_path = path.parent();
  std::string dir_name = path.filename();
  val result = val::take_ownership(js_delete_directory(
      parent_path.string().c_str(), dir_name.data(), recursive));

  if (result.instanceof(val::global("Error"))) {
    return result_t(js_error_to_code(result),
                    result["message"].as<std::string>());
  }

  return result_t();
}

EM_ASYNC_JS(EM_VAL, js_directory_exists, (const char *path_str), {
  // clang-format off
    try {
        const path = UTF8ToString(path_str);
        const pathParts = path.split("/");
        let handle = await navigator.storage.getDirectory();

        for (let i = 0; i < pathParts.length; i++) {
            const part = pathParts[i];
            if (part === "")
                continue;

            handle = await handle.getDirectoryHandle(part);
        }

        return Emval.toHandle(true);
    } catch (error) {
        if (error.name === "NotFoundError") {
            return Emval.toHandle(false);
        } else {
            return Emval.toHandle({name: error.name, message: error.message});
        }
    }
  // clang-format on
});

result_value_t<bool> directory_exists(const path_t &path) {
  val result = val::take_ownership(js_directory_exists(path.string().c_str()));

  if (result.instanceof(val::global("Error"))) {
    return result_value_t<bool>(js_error_to_code(result),
                                result["message"].as<std::string>());
  }

  return result_value_t<bool>(result.as<bool>());
}

EM_ASYNC_JS(EM_VAL, js_list_directory, (const char *path_str), {
  // clang-format off
    try {
        const path = UTF8ToString(path_str);
        const pathParts = path.split("/");
        let handle = await navigator.storage.getDirectory();

        for (let i = 0; i < pathParts.length; i++) {
            const part = pathParts[i];
            if (part === "")
                continue;
            
            handle = await handle.getDirectoryHandle(part);
        }

        const entries = [];
        for await (let [name, handle_] of handle) {
            const entry = {
                name,
                path: `${path}/${name}`,
                type: handle_.kind === "file" ? "Regular" : "Directory",
            };
            if (handle_.kind === "file") {
                const file = await handle_.getFile();
                entry["size"] = file.size;
            } else {
                entry["size"] = 0;
            }
            entries.push(entry);
        }

        return Emval.toHandle(entries);
    } catch (error) {
        return Emval.toHandle({name: error.name, message: error.message});
    }
  // clang-format on
});

result_value_t<std::vector<dir_entry_t>> list_directory(const path_t &path) {
  val result = val::take_ownership(js_list_directory(path.string().c_str()));

  if (result.instanceof(val::global("Error"))) {
    return result_value_t<std::vector<dir_entry_t>>(
        js_error_to_code(result), result["message"].as<std::string>());
  }

  std::vector<dir_entry_t> entries;
  int length = result["length"].as<int>();
  entries.reserve(length);

  for (int i = 0; i < length; i++) {
    val entry = result[i];

    std::string name = entry["name"].as<std::string>();
    path_t path = entry["path"].as<std::string>();
    file_type_t type =
        magic_enum::enum_cast<file_type_t>(entry["type"].as<std::string>())
            .value();
    size_t size = entry["size"].as<size_t>();

    entries.push_back({
        .name = name,
        .path = path,
        .type = type,
        .size = size,
    });
  }

  return result_value_t<std::vector<dir_entry_t>>(entries);
}

// ============================================================================
// File Operations
// ============================================================================

EM_ASYNC_JS(EM_VAL, js_create_file,
            (const char *parent_path_str, const char *file_name), {
              // clang-format off
              try {
                const parentPath = UTF8ToString(parent_path_str);
                const pathParts = parentPath.split("/");
                let handle = await navigator.storage.getDirectory();

                for (let i = 0; i < pathParts.length; i++) {
                    const part = pathParts[i];
                    if (part === "")
                        continue;

                    // Parent dirs should be exist
                    handle = await handle.getDirectoryHandle(part);
                }

                const fileName = UTF8ToString(file_name);
                await handle.getFileHandle(fileName, {create: true});

                return Emval.toHandle(undefined);
              } catch (error) {
                return Emval.toHandle({name: error.name, message: error.message});
              }
              // clang-format on
            });

result_t create_file(const path_t &path) {
  path_t parent_path = path.parent();
  std::string file_name = path.filename();
  val result = val::take_ownership(
      js_create_file(parent_path.string().c_str(), file_name.c_str()));

  if (result.instanceof(val::global("Error"))) {
    return result_t(js_error_to_code(result),
                    result["message"].as<std::string>());
  }

  return result_t();
}

EM_ASYNC_JS(EM_VAL, js_delete_file,
            (const char *parent_path_str, const char *file_name), {
              // clang-format off
              try {
                const parentPath = UTF8ToString(parent_path_str);
                const pathParts = parentPath.split("/");
                let handle = await navigator.storage.getDirectory();

                for (let i = 0; i < pathParts.length; i++) {
                    const part = pathParts[i];
                    if (part === "")
                        continue;

                    // Parent dirs should be exist
                    handle = await handle.getDirectoryHandle(part);
                }

                const fileName = UTF8ToString(file_name);
                await handle.removeEntry(fileName);

                return Emval.toHandle(undefined);
              } catch (error) {
                return Emval.toHandle({name: error.name, message: error.message});
              }
              // clang-format on
            });

result_t delete_file(const path_t &path) {
  path_t parent_path = path.parent();
  std::string file_name = path.filename();
  val result = val::take_ownership(
      js_delete_file(parent_path.string().c_str(), file_name.data()));

  if (result.instanceof(val::global("Error"))) {
    return result_t(js_error_to_code(result),
                    result["message"].as<std::string>());
  }

  return result_t();
}

EM_ASYNC_JS(EM_VAL, js_file_exists,
            (const char *parent_path_str, const char *file_name), {
              // clang-format off
              try {
                const parentPath = UTF8ToString(parent_path_str);
                const pathParts = parentPath.split("/");
                let handle = await navigator.storage.getDirectory();

                for (let i = 0; i < pathParts.length; i++) {
                    const part = pathParts[i];
                    if (part === "")
                        continue;

                    // Parent dirs should be exist
                    handle = await handle.getDirectoryHandle(part);
                }

                const fileName = UTF8ToString(file_name);
                await handle.getFileHandle(fileName);

                return Emval.toHandle(true);
              } catch (error) {
                if (error.name === "NotFoundError") {
                    return Emval.toHandle(false);
                } else {
                    return Emval.toHandle({name: error.name, message: error.message});
                }
              }
              // clang-format on
            });

result_value_t<bool> file_exists(const path_t &path) {
  path_t parent_path = path.parent();
  std::string file_name = path.filename();
  val result = val::take_ownership(
      js_file_exists(parent_path.string().c_str(), file_name.c_str()));

  if (result.instanceof(val::global("Error"))) {
    return result_value_t<bool>(js_error_to_code(result),
                                result["message"].as<std::string>());
  }

  return result_value_t<bool>(result.as<bool>());
}

EM_ASYNC_JS(EM_VAL, js_get_file_size,
            (const char *parent_path_str, const char *file_name), {
              // clang-format off
              try {
                const parentPath = UTF8ToString(parent_path_str);
                const pathParts = parentPath.split("/");
                let handle = await navigator.storage.getDirectory();

                for (let i = 0; i < pathParts.length; i++) {
                    const part = pathParts[i];
                    if (part === "")
                        continue;

                    // Parent dirs should be exist
                    handle = await handle.getDirectoryHandle(part);
                }

                const fileName = UTF8ToString(file_name);
                handle = await handle.getFileHandle(fileName);
                const file = await handle.getFile();

                return Emval.toHandle(file.size);
              } catch (error) {
                return Emval.toHandle({name: error.name, message: error.message});
              }
              // clang-format on
            });

result_value_t<size_t> get_file_size(const path_t &path) {
  path_t parent_path = path.parent();
  std::string file_name = path.filename();
  val result = val::take_ownership(
      js_get_file_size(parent_path.string().c_str(), file_name.c_str()));

  if (result.instanceof(val::global("Error"))) {
    return result_value_t<size_t>(js_error_to_code(result),
                                  result["message"].as<std::string>());
  }

  return result_value_t<size_t>(result.as<size_t>());
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

EM_ASYNC_JS(EM_VAL, js_write_file,
            (const char *path_str, const uint8_t *data_ptr, size_t data_size,
             bool append),
            {
              // clang-format off
              try {
                const path = UTF8ToString(path_str);
                const pathParts = path.split("/");
                let handle = await navigator.storage.getDirectory();

                for (let i = 0; i < pathParts.length; i++) {
                    const part = pathParts[i];
                    if (part === "")
                        continue;
                    const isLastPart = (i === pathParts.length - 1);

                    const options = {create: true};
                    if (isLastPart) {
                        // Last part is a file
                        handle = await handle.getFileHandle(part, options);
                    } else {
                        // This is an intermediate directory
                        handle = await handle.getDirectoryHandle(part, options);
                    }
                }

                const options = {keepExistingData: append};
                const writable = await handle.createWritable(options);

                if (append) {
                  const file = await handle.getFile();
                  await writable.seek(file.size);
                }

                // Copy the data immediately to avoid buffer invalidation issues 
                const dataArray = new Uint8Array(data_size);
                dataArray.set(HEAPU8.subarray(data_ptr, data_ptr + data_size));

                await writable.write(dataArray);
                await writable.close();

                return Emval.toHandle(undefined);
              } catch (error) {
                return Emval.toHandle(error);
              }
              // clang-format on
            });

result_t write_file(const path_t &path, const std::vector<uint8_t> &data,
                    bool append) {
  val result = val::take_ownership(
      js_write_file(path.string().c_str(), data.data(), data.size(), append));

  if (result.instanceof(val::global("Error"))) {
    return result_t(js_error_to_code(result),
                    result["message"].as<std::string>());
  }

  return result_t();
}

result_t write_file(const path_t &path, const std::string &content,
                    bool append) {
  return write_file(path, std::vector<uint8_t>(content.begin(), content.end()),
                    append);
}

EM_ASYNC_JS(EM_VAL, js_read_file_bytes,
            (const char *path_str, uint8_t **data_ptr, size_t *data_size), {
              // clang-format off
              try {
                  const path = UTF8ToString(path_str);
                  const pathParts = path.split("/");
                  let handle = await navigator.storage.getDirectory();
  
                  for (let i = 0; i < pathParts.length; i++) {
                      const part = pathParts[i];
                      if (part === "") continue;
                      const isLastPart = (i === pathParts.length - 1);

                      const options = {create: false};
                      if (isLastPart) {
                          // Last part is a file
                          handle = await handle.getFileHandle(part, options);
                      } else {
                          // This is an intermediate directory
                          handle = await handle.getDirectoryHandle(part, options);
                      }
                  }
  
                  // Read the file
                  const file = await handle.getFile();
                  const arrayBuffer = await file.arrayBuffer();
                  const uint8Array = new Uint8Array(arrayBuffer);
  
                  // Allocate memory for the data
                  const dataSize = uint8Array.length;
                  const dataPtr = _malloc(dataSize);
  
                  // Copy data to allocated memory
                  HEAPU8.set(uint8Array, dataPtr);
  
                  // Set output parameters
                  setValue(data_ptr, dataPtr, 'i32');
                  setValue(data_size, dataSize, 'i32');
  
                  return Emval.toHandle(undefined);
              } catch (error) {
                  return Emval.toHandle(error);
              }
              // clang-format on
            });

result_value_t<std::vector<uint8_t>> read_file_bytes(const path_t &path) {
  uint8_t *data_ptr = nullptr;
  size_t data_size = 0;

  val result = val::take_ownership(
      js_read_file_bytes(path.string().c_str(), &data_ptr, &data_size));

  if (result.instanceof(val::global("Error"))) {
    return result_value_t<std::vector<uint8_t>>(
        js_error_to_code(result), result["message"].as<std::string>());
  }

  std::vector<uint8_t> data(data_ptr, data_ptr + data_size);
  free(data_ptr);

  return result_value_t<std::vector<uint8_t>>(data);
}

result_value_t<std::string> read_file_text(const path_t &path) {
  auto result = read_file_bytes(path);
  if (!result.result_.success()) {
    return result_value_t<std::string>(result.result_.code,
                                       result.result_.message);
  }

  std::vector<uint8_t> data = result.value_.value();
  return result_value_t<std::string>(
      std::string(reinterpret_cast<const char *>(data.data()), data.size()));
}

class opfs_ifstream : public ifstream_ {
private:
  std::vector<uint8_t> buffer_;
  size_t position_;
  bool loaded_;
  int last_read_count_;

  result_t load_file() {
    auto result = read_file_bytes(path_);
    if (!result.result_.success()) {
      return result.result_;
    }

    buffer_ = result.unwrap();
    loaded_ = true;
    return result_t();
  }

public:
  opfs_ifstream() = default;
  explicit opfs_ifstream(const path_t &path)
      : position_(0), loaded_(false), last_read_count_(0) {
    open(path);
  }
  ~opfs_ifstream() override = default;

  // Implementation of base class methods
  result_t open(const path_t &path) override {
    path_ = path;
    position_ = 0;
    loaded_ = false;
    last_read_count_ = 0;

    auto result = load_file();
    if (!result.success()) {
      fail_ = true;
      return result;
    }

    is_open_ = true;
    good_ = true;
    return result_t();
  }
  void close() override {
    buffer_.clear();
    position_ = 0;
    loaded_ = false;
    last_read_count_ = 0;
    is_open_ = false;
  }
  bool is_open() const override { return is_open_; }
  bool good() const override { return good_ && !fail_ && !bad_; }
  bool eof() const override { return position_ >= buffer_.size(); }
  bool fail() const override { return fail_; }
  bool bad() const override { return bad_; }

  // Read operations
  ifstream_ &read(char *buffer, std::streamsize count) override {
    if (!loaded_ || position_ >= buffer_.size()) {
      fail_ = true;
      last_read_count_ = 0;
      return *this;
    }

    size_t available =
        std::min(static_cast<size_t>(count), buffer_.size() - position_);

    std::copy(buffer_.begin() + position_,
              buffer_.begin() + position_ + available, buffer);

    position_ += available;
    last_read_count_ = available;

    return *this;
  }
  ifstream_ &getline(std::string &line, char delim = '\n') override {
    line.clear();
    size_t initial_pos = position_;

    while (position_ < buffer_.size()) {
      char c = static_cast<char>(buffer_[position_++]);
      if (c == delim) {
        break;
      }
      line += c;
    }

    last_read_count_ = position_ - initial_pos;
    return *this;
  }
  std::string read_all() override {
    if (!loaded_) {
      return "";
    }
    return std::string(buffer_.begin() + position_, buffer_.end());
  }
  std::vector<uint8_t> read_all_bytes() override {
    if (!loaded_) {
      return {};
    }
    return std::vector<uint8_t>(buffer_.begin() + position_, buffer_.end());
  }

  // Position operations
  std::streampos tellg() override { return position_; }
  ifstream_ &seekg(std::streampos pos) override {
    position_ = std::min(static_cast<size_t>(pos), buffer_.size());
    return *this;
  }
  ifstream_ &seekg(std::streamoff off, std::ios_base::seekdir dir) override {
    switch (dir) {
    case std::ios::beg:
      position_ = std::max(0, static_cast<int>(off));
      break;
    case std::ios::cur:
      position_ = std::max(0, static_cast<int>(position_ + off));
      break;
    case std::ios::end:
      position_ = std::max(0, static_cast<int>(buffer_.size() + off));
      break;
    }
    position_ = std::min(position_, buffer_.size());
    return *this;
  }

  // Character operations
  int get() override {
    if (position_ >= buffer_.size()) {
      return EOF;
    }
    return static_cast<int>(buffer_[position_++]);
  }
  ifstream_ &get(char &c) override {
    if (position_ >= buffer_.size()) {
      fail_ = true;
      last_read_count_ = 0;
      return *this;
    }
    c = static_cast<char>(buffer_[position_++]);
    last_read_count_ = 1;
    return *this;
  }
  int peek() override {
    if (position_ >= buffer_.size()) {
      return EOF;
    }
    return static_cast<int>(buffer_[position_]);
  }
  ifstream_ &unget() override {
    if (position_ > 0) {
      position_--;
    }
    return *this;
  }
  std::streamsize gcount() const override { return last_read_count_; }

  // Stream extraction operators
  ifstream_ &operator>>(std::string &str) override {
    str.clear();

    // Skip whitespace
    while (position_ < buffer_.size() && std::isspace(buffer_[position_])) {
      position_++;
    }

    if (position_ >= buffer_.size()) {
      fail_ = true;
      last_read_count_ = 0;
      return *this;
    }

    size_t start_pos = position_;

    // Read until whitespace or end
    while (position_ < buffer_.size() && !std::isspace(buffer_[position_])) {
      str += static_cast<char>(buffer_[position_++]);
    }

    last_read_count_ = position_ - start_pos;
    return *this;
  }
  ifstream_ &operator>>(int &val) override {
    std::string str;
    *this >> str;

    if (!str.empty()) {
      try {
        val = std::stoi(str);
      } catch (const std::exception &) {
        fail_ = true;
      }
    } else {
      fail_ = true;
    }

    return *this;
  }
  ifstream_ &operator>>(double &val) override {
    std::string str;
    *this >> str;

    if (!str.empty()) {
      try {
        val = std::stod(str);
      } catch (const std::exception &) {
        fail_ = true;
      }
    } else {
      fail_ = true;
    }

    return *this;
  }
  ifstream_ &operator>>(float &val) override {
    std::string str;
    *this >> str;

    if (!str.empty()) {
      try {
        val = std::stof(str);
      } catch (const std::exception &) {
        fail_ = true;
      }
    } else {
      fail_ = true;
    }

    return *this;
  }
  ifstream_ &operator>>(long &val) override {
    std::string str;
    *this >> str;

    if (!str.empty()) {
      try {
        val = std::stol(str);
      } catch (const std::exception &) {
        fail_ = true;
      }
    } else {
      fail_ = true;
    }

    return *this;
  }
  ifstream_ &operator>>(char &c) override {
    // Skip whitespace for char extraction (standard behavior)
    while (position_ < buffer_.size() && std::isspace(buffer_[position_])) {
      position_++;
    }

    if (position_ >= buffer_.size()) {
      fail_ = true;
      last_read_count_ = 0;
      return *this;
    }

    c = static_cast<char>(buffer_[position_++]);
    last_read_count_ = 1;
    return *this;
  }
};
class opfs_ofstream : public ofstream_ {
private:
  std::vector<uint8_t> buffer_;
  bool append_mode_;
  bool dirty_;

  result_t flush_to_file() {
    if (!dirty_) {
      return result_t();
    }

    auto result = write_file(path_, buffer_);
    if (result.success()) {
      dirty_ = false;
    }
    return result;
  }

public:
  opfs_ofstream() = default;
  explicit opfs_ofstream(const path_t &path, bool append = false)
      : append_mode_(append), dirty_(false) {
    open(path, append);
  }
  ~opfs_ofstream() override { close(); }

  // Implementation of base class methods
  result_t open(const path_t &path, bool append = false) override {
    path_ = path;
    append_mode_ = append;
    dirty_ = false;

    if (append_mode_) {
      // Load existing file content if appending
      auto result = read_file_bytes(path_);
      if (result.result_.success()) {
        buffer_ = result.unwrap();
      }
    }

    is_open_ = true;
    good_ = true;
    return result_t();
  }
  void close() override {
    if (dirty_) {
      flush_to_file();
    }
    buffer_.clear();
    is_open_ = false;
  }
  bool is_open() const override { return is_open_; }
  bool good() const override { return good_ && is_open_; }
  bool eof() const override {
    // For output streams, EOF is typically not relevant
    // Return false since we can always write more data
    return false;
  }
  bool fail() const override { return !good_; }
  bool bad() const override {
    // For this implementation, bad() is the same as fail()
    // In a more sophisticated implementation, you might distinguish
    // between recoverable errors (fail) and unrecoverable errors (bad)
    return !good_;
  }

  // Write operations
  ofstream_ &write(const char *buffer, std::streamsize count) override {
    buffer_.insert(buffer_.end(), reinterpret_cast<const uint8_t *>(buffer),
                   reinterpret_cast<const uint8_t *>(buffer) + count);
    dirty_ = true;
    return *this;
  }
  ofstream_ &write(const std::string &str) override {
    buffer_.insert(buffer_.end(), str.begin(), str.end());
    dirty_ = true;
    return *this;
  }
  ofstream_ &write(const std::vector<uint8_t> &data) override {
    buffer_.insert(buffer_.end(), data.begin(), data.end());
    dirty_ = true;
    return *this;
  }
  result_t flush() override { return flush_to_file(); }

  // Position operations
  std::streampos tellp() override {
    return static_cast<std::streampos>(buffer_.size());
  }
  ofstream_ &seekp(std::streampos pos) override {
    if (pos < 0 || static_cast<size_t>(pos) > buffer_.size()) {
      good_ = false;
      return *this;
    }

    // Resize buffer if seeking beyond current size
    if (static_cast<size_t>(pos) > buffer_.size()) {
      buffer_.resize(static_cast<size_t>(pos));
    }

    // For output streams, seeking sets the write position
    // We don't need to store position separately since we always append
    // But we can truncate the buffer to the seek position
    buffer_.resize(static_cast<size_t>(pos));
    dirty_ = true;
    return *this;
  }
  ofstream_ &seekp(std::streamoff off, std::ios_base::seekdir dir) override {
    std::streampos new_pos;

    switch (dir) {
    case std::ios_base::beg:
      new_pos = off;
      break;
    case std::ios_base::cur:
      new_pos = static_cast<std::streampos>(buffer_.size()) + off;
      break;
    case std::ios_base::end:
      new_pos = static_cast<std::streampos>(buffer_.size()) + off;
      break;
    default:
      good_ = false;
      return *this;
    }

    return seekp(new_pos);
  }

  // Character operations
  ofstream_ &put(char c) override {
    buffer_.push_back(static_cast<uint8_t>(c));
    dirty_ = true;
    return *this;
  }

  // Stream insertion operators
  ofstream_ &operator<<(const std::string &str) override { return write(str); }
  ofstream_ &operator<<(const char *str) override {
    return write(std::string(str));
  }
  ofstream_ &operator<<(char c) override { return put(c); }
  ofstream_ &operator<<(int val) override { return write(std::to_string(val)); }
  ofstream_ &operator<<(double val) override {
    return write(std::to_string(val));
  }
  ofstream_ &operator<<(float val) override {
    return write(std::to_string(val));
  }
  ofstream_ &operator<<(long val) override {
    return write(std::to_string(val));
  }
};

std::unique_ptr<ifstream_> ifstream(const path_t &path) {
  return std::make_unique<opfs_ifstream>(path);
}

std::unique_ptr<ofstream_> ofstream(const path_t &path, bool append) {
  return std::make_unique<opfs_ofstream>(path, append);
}

} // namespace fs
} // namespace ailoy

#endif