#include <dlpack/dlpack.h>
#include <string>
#include <unordered_map>

namespace ailoy {

class tvm_runtime_t {
public:
  tvm_runtime_t(
      const std::string &lib_filename,
      const std::unordered_map<std::string, std::string> file_contents,
      DLDevice device);

private:
};

} // namespace ailoy

extern "C" {
struct ailoy_file_contents_t;

int ailoy_file_contents_create(ailoy_file_contents_t **out);

void ailoy_file_contents_destroy(ailoy_file_contents_t *contents);

int ailoy_file_contents_insert(ailoy_file_contents_t *contents,
                               char const *filename, size_t len,
                               char const *content);

int ailoy_tvm_runtime_create(char const *lib_filename,
                             ailoy_file_contents_t const *contents,
                             ailoy::tvm_runtime_t **out);

void ailoy_tvm_runtime_destroy(ailoy::tvm_runtime_t *model);
}
