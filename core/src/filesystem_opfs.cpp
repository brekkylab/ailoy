#ifdef EMSCRIPTEN

#include <iostream>

#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/val.h>

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

EM_JS(EM_VAL, js_get_file_handle,
      (EM_VAL current_handle, const char *part, bool create), {
        // clang-format off
        return Asyncify.handleAsync(async() => {
          try {
            const handle = Emval.toValue(current_handle);
            const options = {create: create};
            const result =
                await handle.getFileHandle(UTF8ToString(part), options);
            return Emval.toHandle(result);
          } catch (error) {
            // Convert JavaScript exception to Emscripten exception
            return Emval.toHandle(error);
          }
        });
        // clang-format on
      });

EM_JS(EM_VAL, js_get_directory_handle,
      (EM_VAL current_handle, const char *part, bool create), {
        // clang-format off
        return Asyncify.handleAsync(async() => {
          try {
            const handle = Emval.toValue(current_handle);
            const options = {create: create};
            const result =
                await handle.getDirectoryHandle(UTF8ToString(part), options);
            return Emval.toHandle(result);
          } catch (error) {
            // Convert JavaScript exception to Emscripten exception
            return Emval.toHandle(error);
          }
        });
        // clang-format on
      });

// Helper to get a handle (file or directory) from a path.
// Throws val on error.
val get_handle(const std::string &path_str, bool is_dir,
               bool create_if_not_exists) {
  val root =
      val::global("navigator")["storage"].call<val>("getDirectory").await();
  val path_parts = val::global("String")(path_str).call<val>("split", val("/"));
  int num_parts = path_parts["length"].as<int>();
  val current_handle = root;

  for (int i = 0; i < num_parts; ++i) {
    std::string part = path_parts[i].as<std::string>();
    if (part.empty())
      continue;

    bool is_last_part = (i == num_parts - 1);

    if (is_last_part && !is_dir) {
      val result = val::take_ownership(js_get_file_handle(
          current_handle.as_handle(), part.c_str(), create_if_not_exists));
      if (result.instanceof(val::global("Error"))) {
        throw result;
      }
      current_handle = result;
    } else {
      val result = val::take_ownership(js_get_directory_handle(
          current_handle.as_handle(), part.c_str(), create_if_not_exists));
      if (result.instanceof(val::global("Error"))) {
        throw result;
      }
      current_handle = result;
    }
  }

  return current_handle;
}

// ============================================================================
// Directory Operations
// ============================================================================

result_t create_directory(const path_t &path, bool recursive) {
  try {
    get_handle(path, true, true);
    return result_t();
  } catch (const val &e) {
    return result_t(js_error_to_code(e), e["message"].as<std::string>());
  }
}

result_t delete_directory(const path_t &path, bool recursive) {
  try {
    path_t parent_path = path.parent();
    std::string dir_name = path.filename();

    val parent_handle = get_handle(parent_path, true, false);
    val options = val::object();
    options.set("recursive", recursive);
    parent_handle.call<void>("removeEntry", val(dir_name), options);
    return result_t();
  } catch (const val &e) {
    return result_t(js_error_to_code(e), e["message"].as<std::string>());
  }
}

result_value_t<bool> directory_exists(const path_t &path) {
  try {
    get_handle(path, true, false);
    return result_value_t<bool>(true);
  } catch (const val &e) {
    if (js_error_to_code(e) == error_code_t::NotFound) {
      return result_value_t<bool>(false);
    }
    return result_value_t<bool>(js_error_to_code(e),
                                e["message"].as<std::string>());
  }
}

result_value_t<std::vector<dir_entry_t>> list_directory(const path_t &path) {
  try {
    val dir_handle = get_handle(path, true, false);
    val iterator = dir_handle.call<val>("values").await();
    std::vector<dir_entry_t> entries;

    while (true) {
      val next = iterator.call<val>("next").await();
      if (next["done"].as<bool>()) {
        break;
      }
      val entry = next["value"];
      dir_entry_t info;
      info.name = entry["name"].as<std::string>();
      info.path = path / info.name;
      std::string kind = entry["kind"].as<std::string>();
      if (kind == "file") {
        info.type = file_type_t::Regular;
        val file = entry.call<val>("getFile").await();
        info.size = file["size"].as<size_t>();
      } else {
        info.type = file_type_t::Directory;
        info.size = 0; // Directories have no size in this model
      }
      entries.push_back(info);
    }
    return result_value_t<std::vector<dir_entry_t>>(entries);
  } catch (const val &e) {
    return result_value_t<std::vector<dir_entry_t>>(
        js_error_to_code(e), e["message"].as<std::string>());
  }
}

// ============================================================================
// File Operations
// ============================================================================

result_t create_file(const path_t &path) {
  try {
    get_handle(path, false, true); // Get handle with create=true
    return result_t();
  } catch (const val &e) {
    return result_t(js_error_to_code(e), e["message"].as<std::string>());
  }
}

result_t delete_file(const path_t &path) {
  try {
    path_t parent_path = path.parent();
    std::string file_name = path.filename();

    val parent_handle = get_handle(parent_path, true, false);
    parent_handle.call<void>("removeEntry", val(file_name));
    return result_t();
  } catch (const val &e) {
    return result_t(js_error_to_code(e), e["message"].as<std::string>());
  }
}

result_value_t<bool> file_exists(const path_t &path) {
  try {
    get_handle(path, false, false);
    return result_value_t<bool>(true);
  } catch (const val &e) {
    if (js_error_to_code(e) == error_code_t::NotFound) {
      return result_value_t<bool>(false);
    }
    return result_value_t<bool>(js_error_to_code(e),
                                e["message"].as<std::string>());
  }
}

result_value_t<size_t> get_file_size(const path_t &path) {
  try {
    val file_handle = get_handle(path, false, false);
    val file = file_handle.call<val>("getFile").await();
    return result_value_t<size_t>(file["size"].as<size_t>());
  } catch (const val &e) {
    return result_value_t<size_t>(js_error_to_code(e),
                                  e["message"].as<std::string>());
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
  return write_file(path, std::vector<uint8_t>(content.begin(), content.end()),
                    append);
}

result_t write_file(const path_t &path, const std::vector<uint8_t> &data,
                    bool append) {
  try {
    val file_handle = get_handle(path, false, true);
    val options = val::object();
    options.set("keepExistingData", append);
    val writable = file_handle.call<val>("createWritable", options).await();

    if (append) {
      val file = file_handle.call<val>("getFile").await();
      size_t size = file["size"].as<size_t>();
      writable.call<val>("seek", size).await();
    }

    val data_array =
        val::global("Uint8Array")
            .new_(val(emscripten::typed_memory_view(data.size(), data.data())));
    writable.call<val>("write", data_array).await();
    writable.call<val>("close").await();
    return result_t();
  } catch (const val &e) {
    return result_t(js_error_to_code(e), e["message"].as<std::string>());
  }
}

result_value_t<std::string> read_file_text(const path_t &path) {
  try {
    val file_handle = get_handle(path, false, false);
    val file = file_handle.call<val>("getFile").await();
    std::string content = file.call<val>("text").await().as<std::string>();
    return result_value_t<std::string>(content);
  } catch (const val &e) {
    return result_value_t<std::string>(js_error_to_code(e),
                                       e["message"].as<std::string>());
  }
}

result_value_t<std::vector<uint8_t>> read_file_bytes(const path_t &path) {
  try {
    val file_handle = get_handle(path, false, false);
    val file = file_handle.call<val>("getFile").await();
    val array_buffer = file.call<val>("arrayBuffer").await();

    // Use `vecFromJSArray` which is a convenient Emscripten helper
    val uint8_array_val = val::global("Uint8Array").new_(array_buffer);
    std::vector<uint8_t> data =
        emscripten::vecFromJSArray<uint8_t>(uint8_array_val);

    return result_value_t<std::vector<uint8_t>>(std::move(data));
  } catch (const val &e) {
    return result_value_t<std::vector<uint8_t>>(js_error_to_code(e),
                                                e["message"].as<std::string>());
  }
}

} // namespace fs
} // namespace ailoy

#endif