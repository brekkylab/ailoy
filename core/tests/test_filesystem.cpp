#include <iostream>
#include <string>
#include <vector>

#include "filesystem.hpp"

void check(const ailoy::fs::result_t &res, const std::string &message) {
  if (res.success()) {
    std::cout << "[SUCCESS] " << message << std::endl;
  } else {
    std::cout << "[FAILED] " << message << " - Error: " << res.message
              << std::endl;
    exit(1);
  }
}

template <typename T>
void check(const ailoy::fs::result_value_t<T> &res,
           const std::string &message) {
  if (res.result_.success()) {
    std::cout << "[SUCCESS] " << message << std::endl;
  } else {
    std::cout << "[FAILED]  " << message << " - Error: " << res.result_.message
              << std::endl;
    exit(1);
  }
}

int main() {
  std::cout << "--- Starting Ailoy OPFS Filesystem Test ---" << std::endl;

  // Use the path_t class for easy path joining
  ailoy::fs::path_t base_dir = "/test_project";
  ailoy::fs::path_t src_dir = base_dir / "src";
  ailoy::fs::path_t file_path = src_dir / "main.cpp";

  std::cout << "\n1. Creating directories..." << std::endl;
  check(ailoy::fs::create_directory(src_dir, true),
        "Recursively created " + src_dir.string());

  auto dir_exists = ailoy::fs::directory_exists(src_dir);
  check(dir_exists, "Checking if directory exists");
  if (!dir_exists.value_.value_or(false)) {
    std::cout << "Verification failed: Directory does not exist after creation."
              << std::endl;
    return 1;
  }

  std::cout << "\n2. Writing a file..." << std::endl;
  std::string file_content = R"(#include <iostream>

int main() {
    std::cout << "Hello, OPFS!" << std::endl;
    return 0;
}
)";
  check(ailoy::fs::write_file(file_path, file_content),
        "Wrote content to " + file_path.string());

  std::cout << "\n3. Reading the file back..." << std::endl;
  auto read_result = ailoy::fs::read_file_text(file_path);
  check(read_result, "Read content from " + file_path.string());

  if (read_result.value_.value_or("") == file_content) {
    std::cout << "[SUCCESS] File content matches!" << std::endl;
  } else {
    std::cout << "[FAILED]  File content does not match!" << std::endl;
    return 1;
  }

  std::cout << "\n4. Listing directory contents..." << std::endl;
  auto list_result = ailoy::fs::list_directory(base_dir);
  check(list_result, "Listed contents of " + base_dir.string());

  std::cout << "Contents of '" << base_dir << "':" << std::endl;
  for (const auto &info : list_result.value_.value()) {
    std::cout << "  - Name: " << info.name << ", Type: "
              << (info.type == ailoy::fs::file_type_t::Directory ? "Directory"
                                                                 : "File")
              << std::endl;
  }

  std::cout << "\n5. Cleaning up..." << std::endl;
  check(ailoy::fs::delete_file(file_path),
        "Deleted file " + file_path.string());
  check(ailoy::fs::delete_directory(base_dir, true),
        "Recursively deleted directory " + base_dir.string());

  auto file_gone = ailoy::fs::file_exists(file_path);
  std::cout << "file_gone" << std::endl;
  if (!file_gone.result_.success() &&
      file_gone.result_.code == ailoy::fs::error_code_t::NotFound) {
    std::cout << "[SUCCESS] File no longer exists." << std::endl;
  } else {
    std::cout << "[FAILED]  File still exists or an error occurred."
              << std::endl;
  }

  std::cout << "\n--- Test Finished Successfully! --- 🚀" << std::endl;

  return 0;
}