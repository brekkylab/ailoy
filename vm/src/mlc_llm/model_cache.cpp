#include "model_cache.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <csignal>
#include <fstream>
#include <regex>
#include <string>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__EMSCRIPTEN__)
#include <emscripten.h>
#else
#include <sys/utsname.h>
#endif

#include <httplib.h>
#include <indicators/block_progress_bar.hpp>
#include <indicators/dynamic_progress.hpp>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <openssl/opensslv.h>
#include <openssl/sha.h>

#include "exception.hpp"

using namespace std::chrono_literals;

namespace ailoy {

struct utsname {
  std::string sysname;
  std::string nodename;
  std::string release;
  std::string version;
  std::string machine;
};

utsname get_uname() {
  utsname uts;
#ifdef _WIN32
  OSVERSIONINFOEXW osInfo = {sizeof(OSVERSIONINFOEXW)};
  if (!GetVersionExW((LPOSVERSIONINFOW)&osInfo)) {
    throw ailoy::runtime_error("Failed to get OS info");
  }

  SYSTEM_INFO sysInfo;
  GetSystemInfo(&sysInfo);

  uts.sysname = "Windows";

  char hostname[256];
  DWORD hostnameSize = sizeof(hostname);
  if (GetComputerNameA(hostname, &hostnameSize)) {
    uts.nodename = hostname;
  } else {
    uts.nodename = "unknown";
  }

  uts.release = std::to_string(osInfo.dwMajorVersion) + "." +
                std::to_string(osInfo.dwMinorVersion);
  uts.version = std::to_string(osInfo.dwBuildNumber);

  switch (sysInfo.wProcessorArchitecture) {
  case PROCESSOR_ARCHITECTURE_AMD64:
    uts.machine = "x86_64";
    break;
  case PROCESSOR_ARCHITECTURE_ARM:
    uts.machine = "arm";
    break;
  case PROCESSOR_ARCHITECTURE_ARM64:
    uts.machine = "arm64";
    break;
  case PROCESSOR_ARCHITECTURE_IA64:
    uts.machine = "ia64";
    break;
  case PROCESSOR_ARCHITECTURE_INTEL:
    uts.machine = "x86";
    break;
  default:
    uts.machine = "unknown";
    break;
  }
#elif defined(__EMSCRIPTEN__)
  uts.sysname = "Emscripten";
  uts.nodename = "localhost"; // Browser has no traditional hostname
  uts.release = "1.0";        // Placeholder, no kernel version
  uts.version = EMSCRIPTEN_VERSION;
  uts.machine = "wasm32";
#else
  struct ::utsname posix_uts;
  if (uname(&posix_uts) != 0) {
    throw ailoy::runtime_error("Failed to get system info");
  }

  uts.sysname = posix_uts.sysname;
  uts.nodename = posix_uts.nodename;
  uts.release = posix_uts.release;
  uts.version = posix_uts.version;
  uts.machine = posix_uts.machine;
#endif
  return uts;
}

std::string sha1_checksum(const std::filesystem::path &filepath) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open file: " + filepath.string());
  }

  unsigned char hash[SHA_DIGEST_LENGTH];

#if OPENSSL_VERSION_NUMBER >= 0x30000000L
  EVP_MD_CTX *ctx = EVP_MD_CTX_new();
  if (!ctx)
    throw std::runtime_error("Failed to create EVP_MD_CTX");

  if (EVP_DigestInit_ex(ctx, EVP_sha1(), nullptr) != 1)
    throw std::runtime_error("EVP_DigestInit_ex failed");

  char buffer[8192];
  while (file.read(buffer, sizeof(buffer)) || file.gcount()) {
    if (EVP_DigestUpdate(ctx, buffer, file.gcount()) != 1) {
      EVP_MD_CTX_free(ctx);
      throw std::runtime_error("EVP_DigestUpdate failed");
    }
  }

  unsigned int len;
  if (EVP_DigestFinal_ex(ctx, hash, &len) != 1) {
    EVP_MD_CTX_free(ctx);
    throw std::runtime_error("EVP_DigestFinal_ex failed");
  }

  EVP_MD_CTX_free(ctx);
#else
  SHA_CTX sha1;
  SHA1_Init(&sha1);

  char buffer[8192];
  while (file.read(buffer, sizeof(buffer)) || file.gcount()) {
    SHA1_Update(&sha1, buffer, file.gcount());
  }

  SHA1_Final(hash, &sha1);
#endif

  std::ostringstream result;
  for (int i = 0; i < SHA_DIGEST_LENGTH; ++i) {
    result << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
  }

  return result.str();
}

class SigintGuard {
public:
  SigintGuard() {
    g_sigint = false;

#if defined(_WIN32)
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
#else
    struct sigaction new_action{};
    new_action.sa_handler = sigint_handler;
    sigemptyset(&new_action.sa_mask);

    // Save existing handler
    sigaction(SIGINT, nullptr, &old_action_);

    // Set our new handler
    sigaction(SIGINT, &new_action, nullptr);
#endif
  }

  ~SigintGuard() {
#if defined(_WIN32)
    SetConsoleCtrlHandler(console_ctrl_handler, FALSE);
#else
    sigaction(SIGINT, &old_action_, nullptr);
#endif
  }

  static bool interrupted() { return g_sigint; }

private:
#if !defined(_WIN32)
  static struct sigaction old_action_;
#endif

  static std::atomic<bool> g_sigint;

#if defined(_WIN32)
  static BOOL WINAPI console_ctrl_handler(DWORD signal) {
    if (signal == CTRL_C_EVENT || signal == CTRL_BREAK_EVENT) {
      g_sigint = true;
      return TRUE;
    }
    return FALSE;
  }
#else
  static void sigint_handler(int signum) {
    g_sigint = true;

    if (old_action_.sa_handler != nullptr) {
      old_action_.sa_handler(signum);
    }
  }
#endif
};

// Definition of static variable
#if !defined(_WIN32)
struct sigaction SigintGuard::old_action_;
#endif
std::atomic<bool> SigintGuard::g_sigint{false};

namespace fs = std::filesystem;
using json = nlohmann::json;

std::filesystem::path get_cache_root() {
  fs::path cache_root;
  if (std::getenv("AILOY_CACHE_ROOT")) {
    // Check environment variable
    cache_root = fs::path(std::getenv("AILOY_CACHE_ROOT"));
  } else {
    // Set to default cache root
#if defined(_WIN32)
    if (std::getenv("LOCALAPPDATA"))
      cache_root = fs::path(std::getenv("LOCALAPPDATA")) / "ailoy";
#else
    if (std::getenv("HOME"))
      cache_root = fs::path(std::getenv("HOME")) / ".cache" / "ailoy";
#endif
  }
  if (cache_root.empty()) {
    throw exception("Cannot get cache root");
  }

  try {
    // Create a directory
    // It returns false if already exists
    fs::create_directories(cache_root);
  } catch (const fs::filesystem_error &e) {
    throw exception("cache root directory creation failed");
  }

  return cache_root;
}

std::string get_models_url() {
  if (std::getenv("AILOY_MODELS_URL"))
    return std::getenv("AILOY_MODELS_URL");
  else
    return "https://models.download.ailoy.co";
}

std::pair<bool, std::string> download_file(httplib::SSLClient &client,
                                           const std::string &remote_path,
                                           const fs::path &local_path) {
  httplib::Result res = client.Get(("/" + remote_path).c_str());
  if (!res || (res->status != httplib::OK_200)) {
    return std::make_pair<bool, std::string>(
        false, "Failed to download " + remote_path + ": HTTP " +
                   (res ? std::to_string(res->status)
                        : httplib::to_string(res.error())));
  }

  std::ofstream ofs(local_path, std::ios::binary);
  ofs.write(res->body.c_str(), res->body.size());

  return std::make_pair<bool, std::string>(true, "");
}

std::pair<bool, std::string> download_file_with_progress(
    httplib::SSLClient &client, const std::string &remote_path,
    const fs::path &local_path,
    std::function<bool(uint64_t, uint64_t)> progress_callback) {
  SigintGuard sigint_guard;

  size_t existing_size = 0;
  httplib::Result res = client.Get(
      ("/" + remote_path).c_str(),
      [&](const char *data, size_t data_length) {
        // Stop on SIGINT
        if (sigint_guard.interrupted())
          return false;

        std::ofstream ofs(local_path, existing_size > 0
                                          ? std::ios::app | std::ios::binary
                                          : std::ios::binary);
        ofs.seekp(existing_size);
        ofs.write(data, data_length);
        existing_size += data_length;
        return ofs.good();
      },
      progress_callback);
  if (!res || (res->status != httplib::OK_200 &&
               res->status != httplib::PartialContent_206)) {
    // If SIGINT interrupted, return error message about interrupted
    if (sigint_guard.interrupted())
      return std::make_pair<bool, std::string>(
          false, "Interrupted while downloading the model");

    // Otherwise, return error message about HTTP error
    return std::make_pair<bool, std::string>(
        false, "Failed to download " + remote_path + ": HTTP " +
                   (res ? std::to_string(res->status)
                        : httplib::to_string(res.error())));
  }

  return std::make_pair<bool, std::string>(true, "");
}

fs::path get_model_base_path(const std::string &model_id) {
  std::string model_id_escaped =
      std::regex_replace(model_id, std::regex("/"), "--");
  fs::path model_base_path = fs::path("tvm-models") / model_id_escaped;

  return model_base_path;
}

std::vector<model_cache_list_result_t> list_local_models() {
  std::vector<model_cache_list_result_t> results;

  fs::path cache_base_path = get_cache_root();

  // TVM models
  fs::path tvm_models_path = cache_base_path / "tvm-models";
  if (!fs::exists(tvm_models_path))
    return results;

  // Directory structure example for TVM models:
  // BAAI--bge-m3 (model_id)
  // └── q4f16_1 (quantization)
  //     ├── manifest-arm64-Darwin-metal.json (manifest)
  //     └── ...files...
  for (const auto &model_entry : fs::directory_iterator{tvm_models_path}) {
    if (!model_entry.is_directory())
      continue;

    // Get model id and denormalize it
    // e.g. "BAAI--bge-m3" -> "BAAI/bge-m3"
    std::string model_id = model_entry.path().filename().string();
    model_id = std::regex_replace(model_id, std::regex("--"), "/");

    // Iterate over quantizations
    for (const auto &quant_entry : fs::directory_iterator(model_entry.path())) {
      if (!quant_entry.is_directory())
        continue;

      std::string quantization = quant_entry.path().filename().string();
      fs::path quant_dir = fs::absolute(quant_entry.path());

      // Iterate over files
      for (const auto &file_entry : fs::directory_iterator(quant_dir)) {
        if (!file_entry.is_regular_file())
          continue;

        // Find the manifest file
        std::string filename = file_entry.path().filename().string();
        if (filename.rfind("manifest-", 0) != 0 ||
            file_entry.path().extension() != ".json")
          continue;

        std::string manifest_stem = file_entry.path().stem().string();
        auto parts_start = manifest_stem.find("-");
        if (parts_start == std::string::npos)
          continue;

        // Get device name
        std::vector<std::string> parts;
        size_t start = parts_start + 1;
        size_t end;
        while ((end = manifest_stem.find("-", start)) != std::string::npos) {
          parts.push_back(manifest_stem.substr(start, end - start));
          start = end + 1;
        }
        parts.push_back(manifest_stem.substr(start));
        if (parts.size() != 3)
          continue;
        std::string device = parts[2];

        // Read manifest json
        std::ifstream manifest_in(file_entry.path());
        if (!manifest_in)
          continue;
        nlohmann::json manifest_json;
        try {
          manifest_in >> manifest_json;
        } catch (...) {
          continue;
        }

        // Calculate total bytes
        size_t total_bytes = 0;
        if (manifest_json.contains("files") &&
            manifest_json["files"].is_array()) {
          for (const auto &file_pair : manifest_json["files"]) {
            if (file_pair.is_array() && file_pair.size() >= 1) {
              fs::path file_path = quant_dir / file_pair[0].get<std::string>();
              if (fs::exists(file_path) && fs::is_regular_file(file_path)) {
                total_bytes += fs::file_size(file_path);
              }
            }
          }
        }

        // Fill the result
        model_cache_list_result_t result;
        result.model_type = "tvm";
        result.model_id = model_id;
        result.attributes = {
            {"quantization", quantization},
            {"device", device},
        };
        result.model_path = quant_dir;
        result.total_bytes = total_bytes;
        results.push_back(std::move(result));
      }
    }
  }

  return results;
}

model_cache_download_result_t
download_model(const std::string &model_id, const std::string &quantization,
               const std::string &target_device,
               std::optional<model_cache_callback_t> callback,
               bool print_progress_bar) {
  model_cache_download_result_t result{.success = false};

  auto client = httplib::SSLClient(
      std::regex_replace(get_models_url(), std::regex("^http(s)://"), ""));
  client.set_connection_timeout(10, 0);
  client.set_read_timeout(60, 0);
  client.enable_server_certificate_verification(false);
  client.enable_server_hostname_verification(false);

  // Create local cache directory
  fs::path model_base_path = get_model_base_path(model_id);
  fs::path model_cache_path = get_cache_root() / model_base_path / quantization;
  fs::create_directories(model_cache_path);

  // Assemble manifest filename based on arch, os and target device
  auto uname = get_uname();
  std::string target_lib =
      std::format("{}-{}-{}", uname.machine, uname.sysname, target_device);
  std::string manifest_filename = std::format("manifest-{}.json", target_lib);

  // Download manifest if not already present
  fs::path manifest_path = model_cache_path / manifest_filename;
  if (!fs::exists(manifest_path)) {
    auto [success, error_message] = download_file(
        client, (model_base_path / quantization / manifest_filename).string(),
        manifest_path);
    if (!success) {
      result.error_message = error_message;
      return result;
    }
  }

  // Read and parse manifest
  std::ifstream manifest_file(manifest_path);
  if (!manifest_file.is_open()) {
    result.error_message =
        "Failed to open manifest at " + manifest_path.string();
    return result;
  }

  json manifest;
  try {
    manifest_file >> manifest;
  } catch (const json::parse_error &e) {
    // Remove manifest if it's not a valid format
    manifest_file.close();
    fs::remove(manifest_path);

    result.error_message = "Failed to parse manifest: " + std::string(e.what());
    return result;
  }
  manifest_file.close();

  // Get files from "files" section
  if (!manifest.contains("files") || !manifest["files"].is_array()) {
    result.error_message = "Manifest is missing a valid 'files' array";
    return result;
  }

  std::vector<std::string> files_to_download;
  for (auto &[file, sha1] :
       manifest["files"]
           .get<std::vector<std::pair<std::string, std::string>>>()) {
    // Add to files_to_download if neither file exists nor sha1 checksum is same
    if (!(fs::exists(model_cache_path / file) &&
          sha1 == sha1_checksum(model_cache_path / file))) {
      files_to_download.emplace_back(file);
    }
  }

  indicators::DynamicProgress<indicators::BlockProgressBar> bars;
  bars.set_option(indicators::option::HideBarWhenComplete{true});

  size_t total_files = files_to_download.size();
  for (size_t i = 0; i < total_files; ++i) {
    const auto &file = files_to_download[i];
    fs::path local_path = model_cache_path / file;
    auto bar = std::make_unique<indicators::BlockProgressBar>(
        indicators::option::BarWidth{50},
        indicators::option::PrefixText{"Downloading " + file + " "},
        indicators::option::ShowPercentage{true},
        indicators::option::ShowElapsedTime{true});
    auto bar_idx = bars.push_back(std::move(bar));

    auto [download_success, download_error_message] =
        download_file_with_progress(
            client, (model_base_path / quantization / file).string(),
            local_path, [&](uint64_t current, uint64_t total) {
              float progress = static_cast<float>(current) / total * 100;
              if (callback.has_value())
                callback.value()(i, total_files, file, progress);
              if (print_progress_bar)
                bars[bar_idx].set_progress(progress);
              return true;
            });

    if (!download_success) {
      result.error_message = download_error_message;
      return result;
    }

    if (print_progress_bar)
      bars[bar_idx].mark_as_completed();
  }
  if (print_progress_bar) {
    // ensure final flush
    bars.print_progress();
  }

  // Get model lib file path
  std::string model_lib_file = manifest["lib"].get<std::string>();
  fs::path model_lib_path = model_cache_path / model_lib_file;

  result.success = true;
  result.model_path = model_cache_path;
  result.model_lib_path = model_lib_path;
  return result;
}

model_cache_remove_result_t remove_model(const std::string &model_id,
                                         bool ask_prompt) {
  model_cache_remove_result_t result{.success = false};

  fs::path model_path = get_cache_root() / get_model_base_path(model_id);
  if (!fs::exists(model_path)) {
    result.error_message = std::format(
        "The model id \"{}\" does not exist in local cache", model_id);
    return result;
  }

  if (ask_prompt) {
    std::string answer;
    do {
      std::cout << std::format(
          "Are you sure you want to remove model \"{}\"? (y/n)", model_id);
      std::cin >> answer;
      std::transform(answer.begin(), answer.end(), answer.begin(), ::tolower);
    } while (!std::cin.fail() && !(answer == "y" || answer == "n"));

    if (answer != "y") {
      result.success = true;
      result.skipped = true;
      result.model_path = model_path;
      return result;
    }
  }

  fs::remove_all(model_path);
  result.success = true;
  result.model_path = model_path;
  return result;
}

namespace operators {

value_or_error_t list_local_models(std::shared_ptr<const value_t> inputs) {
  auto models = ailoy::list_local_models();

  auto outputs = create<map_t>();
  auto results = create<array_t>();
  for (const auto &model : models) {
    auto item = create<map_t>();
    item->insert_or_assign("type", create<string_t>(model.model_type));
    item->insert_or_assign("model_id", create<string_t>(model.model_id));
    item->insert_or_assign("attributes", from_nlohmann_json(model.attributes));
    item->insert_or_assign("model_path",
                           create<string_t>(model.model_path.string()));
    item->insert_or_assign("total_bytes", create<uint_t>(model.total_bytes));
    results->push_back(item);
  }
  outputs->insert_or_assign("results", results);
  return outputs;
}

value_or_error_t download_model(std::shared_ptr<const value_t> inputs) {
  if (!inputs->is_type_of<map_t>())
    return error_output_t(
        type_error("download_model", "inputs", "map_t", inputs->get_type()));

  auto inputs_map = inputs->as<map_t>();

  // TODO: Currently there's no model type other than "tvm",
  // but we need to receive it after many model types are supported
  std::string model_type = "tvm";

  // Check model_id
  if (!inputs_map->contains("model_id"))
    return error_output_t(range_error("download_model", "model_id"));
  auto model_id_val = inputs_map->at("model_id");
  if (!model_id_val->is_type_of<string_t>())
    return error_output_t(type_error("download_model", "model_id", "string_t",
                                     model_id_val->get_type()));
  std::string model_id = *model_id_val->as<string_t>();

  if (model_type == "tvm") {
    // Check quantization
    if (!inputs_map->contains("quantization"))
      return error_output_t(range_error("download_model", "quantization"));
    auto quantization_val = inputs_map->at("quantization");
    if (!quantization_val->is_type_of<string_t>())
      return error_output_t(type_error("download_model", "quantization",
                                       "string_t",
                                       quantization_val->get_type()));
    std::string quantization = *quantization_val->as<string_t>();

    // Check device
    if (!inputs_map->contains("device"))
      return error_output_t(range_error("download_model", "device"));
    auto device_val = inputs_map->at("device");
    if (!device_val->is_type_of<string_t>())
      return error_output_t(type_error("download_model", "device", "string_t",
                                       device_val->get_type()));
    std::string device = *device_val->as<string_t>();

    // Download the model
    auto result = ailoy::download_model(model_id, quantization, device,
                                        std::nullopt, true);
    if (!result.success)
      return error_output_t(result.error_message.value());

    auto outputs = create<map_t>();
    outputs->insert_or_assign(
        "model_path", create<string_t>(result.model_path.value().string()));

    return outputs;
  } else {
    return error_output_t(
        std::format("Unsupported model type: {}", model_type));
  }
}

value_or_error_t remove_model(std::shared_ptr<const value_t> inputs) {
  if (!inputs->is_type_of<map_t>())
    return error_output_t(
        type_error("remove_model", "inputs", "map_t", inputs->get_type()));

  auto inputs_map = inputs->as<map_t>();

  // Check model_id
  if (!inputs_map->contains("model_id"))
    return error_output_t(range_error("download_model", "model_id"));
  auto model_id_val = inputs_map->at("model_id");
  if (!model_id_val->is_type_of<string_t>())
    return error_output_t(type_error("download_model", "model_id", "string_t",
                                     model_id_val->get_type()));
  std::string model_id = *model_id_val->as<string_t>();

  auto result = ailoy::remove_model(model_id, true);
  if (!result.success)
    return error_output_t(result.error_message.value());

  auto outputs = create<map_t>();
  outputs->insert_or_assign(
      "model_path", create<string_t>(result.model_path.value().string()));
  outputs->insert_or_assign("skipped", create<bool_t>(result.skipped));
  return outputs;
}

} // namespace operators

} // namespace ailoy
