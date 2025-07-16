#ifdef EMSCRIPTEN

#include <iostream>

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
    std::cout << "part: " << part << std::endl;
    if (part.empty())
      continue;

    bool is_last_part = (i == num_parts - 1);
    try {
      if (is_last_part && !is_dir) {
        std::cout << "get file: " << part << std::endl;
        val options = val::object();
        options.set("create", create_if_not_exists);
        current_handle =
            current_handle.call<val>("getFileHandle", val(part), options)
                .await();
      } else {
        std::cout << "get directory: " << part << std::endl;
        val options = val::object();
        options.set("create", create_if_not_exists);
        current_handle =
            current_handle.call<val>("getDirectoryHandle", val(part), options)
                .await();

        std::string name = current_handle["name"].as<std::string>();
        std::cout << "name: " << name << std::endl;
      }
    } catch (const val &e) {
      std::cout << "SSIbal" << std::endl;
      // Re-throw to be caught by the calling C++ function
      throw e;
    }
  }

  return current_handle;
}

// ============================================================================
// Directory Operations
// ============================================================================

result_t create_directory(const std::string &path, bool recursive) {
  try {
    get_handle(path, true, true);
    return result_t();
  } catch (const val &e) {
    return result_t(js_error_to_code(e), e["message"].as<std::string>());
  }
}

result_t delete_directory(const std::string &path, bool recursive) {
  try {
    std::string parent_path_str = get_parent_path(path);
    std::string dir_name = get_file_name(path);

    val parent_handle = get_handle(parent_path_str, true, false);
    val options = val::object();
    options.set("recursive", recursive);
    parent_handle.call<void>("removeEntry", val(dir_name), options);
    return result_t();
  } catch (const val &e) {
    return result_t(js_error_to_code(e), e["message"].as<std::string>());
  }
}

result_value_t<bool> directory_exists(const std::string &path) {
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

result_value_t<std::vector<file_info_t>>
list_directory(const std::string &path) {
  try {
    val dir_handle = get_handle(path, true, false);
    val iterator = dir_handle.call<val>("values").await();
    std::vector<file_info_t> entries;

    while (true) {
      val next = iterator.call<val>("next").await();
      if (next["done"].as<bool>()) {
        break;
      }
      val entry = next["value"];
      file_info_t info;
      info.name = entry["name"].as<std::string>();
      info.path = join_path(path, info.name);
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
    return result_value_t<std::vector<file_info_t>>(entries);
  } catch (const val &e) {
    return result_value_t<std::vector<file_info_t>>(
        js_error_to_code(e), e["message"].as<std::string>());
  }
}

// ============================================================================
// File Operations
// ============================================================================

result_t create_file(const std::string &path) {
  try {
    get_handle(path, false, true); // Get handle with create=true
    return result_t();
  } catch (const val &e) {
    return result_t(js_error_to_code(e), e["message"].as<std::string>());
  }
}

result_t delete_file(const std::string &path) {
  try {
    std::string parent_path_str = get_parent_path(path);
    std::string file_name = get_file_name(path);

    val parent_handle = get_handle(parent_path_str, true, false);
    parent_handle.call<void>("removeEntry", val(file_name));
    return result_t();
  } catch (const val &e) {
    return result_t(js_error_to_code(e), e["message"].as<std::string>());
  }
}

result_value_t<bool> file_exists(const std::string &path) {
  try {
    get_handle(path, false, false);
    return result_value_t<bool>(true);
  } catch (const val &e) {
    std::cout << "[file_exists] error: "
              << magic_enum::enum_name(js_error_to_code(e)) << std::endl;
    if (js_error_to_code(e) == error_code_t::NotFound) {
      return result_value_t<bool>(false);
    }
    return result_value_t<bool>(js_error_to_code(e),
                                e["message"].as<std::string>());
  }
}

result_value_t<size_t> get_file_size(const std::string &path) {
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
// Read/Write Operations
// ============================================================================

result_t write_file(const std::string &path, const std::string &content,
                    bool append) {
  return write_file(path, std::vector<uint8_t>(content.begin(), content.end()),
                    append);
}

result_t write_file(const std::string &path, const std::vector<uint8_t> &data,
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

result_value_t<std::string> read_file_text(const std::string &path) {
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

result_value_t<std::vector<uint8_t>> read_file_bytes(const std::string &path) {
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

// ============================================================================
// Path Operations
// ============================================================================
// These are pure string manipulations and don't need JS interop.

std::string get_absolute_path(const std::string &path) {
  // In OPFS, all paths are relative to the origin's root, so they are
  // effectively "absolute" within that context. This function can be used to
  // clean up paths, e.g., resolving ".." or "." For this draft, we'll just
  // return the path, assuming it's clean.
  return path;
}

std::string get_parent_path(const std::string &path) {
  if (path.empty())
    return "";
  auto pos = path.find_last_of('/');
  if (pos == std::string::npos)
    return "";
  if (pos == 0)
    return "/"; // Parent of "/foo" is "/"
  return path.substr(0, pos);
}

std::string get_file_name(const std::string &path) {
  if (path.empty())
    return "";
  auto pos = path.find_last_of('/');
  if (pos == std::string::npos)
    return path;
  return path.substr(pos + 1);
}

std::string join_path(const std::string &path1, const std::string &path2) {
  if (path1.empty() || path1 == "/")
    return path2;
  if (path2.empty())
    return path1;
  if (path1.back() == '/')
    return path1 + path2;
  return path1 + "/" + path2;
}

bool is_absolute(const std::string &path) {
  // In OPFS, we can consider any path starting with / as absolute,
  // or any non-empty path as absolute from the context root.
  return !path.empty() && path.front() == '/';
}

} // namespace fs
} // namespace ailoy

#endif