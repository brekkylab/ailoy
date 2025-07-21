#include <iostream>
#include <string>
#include <vector>

#include <magic_enum/magic_enum.hpp>

#include "filesystem.hpp"
#include "http.hpp"
#include "model_cache.hpp"

int main() {
  ailoy::fs::delete_directory("/ailoy/tvm-models/BAAI--bge-m3");

  ailoy::download_model("BAAI/bge-m3", "q4f16_1", "metal", std::nullopt, false);

  std::function<void(ailoy::fs::dir_entry_t, int)> print_directory_entries =
      [&](ailoy::fs::dir_entry_t entry, int tab) {
        std::cout << std::string(tab, ' ') << entry.name
                  << (entry.is_directory() ? "/" : "")
                  << (entry.is_regular_file()
                          ? "\t" + std::to_string(entry.size) + "B"
                          : "")
                  << std::endl;
        if (entry.is_directory()) {
          auto entries = ailoy::fs::list_directory(entry.path).unwrap();
          for (const auto &subentry : entries) {
            print_directory_entries(subentry, tab + 2);
          }
        }
      };

  ailoy::fs::path_t base_dir = "/ailoy";
  auto list_result = ailoy::fs::list_directory(base_dir);
  std::cout << base_dir.string() << "/" << std::endl;
  for (const auto &info : list_result.unwrap()) {
    print_directory_entries(info, 2);
  }

  return 0;
}