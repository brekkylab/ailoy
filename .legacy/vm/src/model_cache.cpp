#include "model_cache.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <csignal>
#include <regex>
#include <string>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(EMSCRIPTEN)
#include <emscripten.h>
#else
#include <sys/utsname.h>
#endif

#include <indicators/block_progress_bar.hpp>
#include <indicators/dynamic_progress.hpp>
#include <nlohmann/json.hpp>

#include "exception.hpp"
#include "http.hpp"

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
  uts.version = "EMSCRIPTEN_VERSION";
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

class SHA1 {
private:
  // SHA-1 constants
  static const uint32_t K[4];

  // Initial hash values
  uint32_t h[5];

  // Message buffer
  std::vector<uint8_t> buffer;
  uint64_t totalLength;

  // Rotate left function
  uint32_t rotateLeft(uint32_t value, int shift) {
    return (value << shift) | (value >> (32 - shift));
  }

  // Process a 512-bit block
  void processBlock(const uint8_t *block) {
    uint32_t w[80];

    // Copy block into first 16 words of w array
    for (int i = 0; i < 16; i++) {
      w[i] = (block[i * 4] << 24) | (block[i * 4 + 1] << 16) |
             (block[i * 4 + 2] << 8) | block[i * 4 + 3];
    }

    // Extend the first 16 words into the remaining 64 words
    for (int i = 16; i < 80; i++) {
      w[i] = rotateLeft(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    }

    // Initialize hash value for this chunk
    uint32_t a = h[0];
    uint32_t b = h[1];
    uint32_t c = h[2];
    uint32_t d = h[3];
    uint32_t e = h[4];

    // Main loop
    for (int i = 0; i < 80; i++) {
      uint32_t f, k;

      if (i < 20) {
        f = (b & c) | (~b & d);
        k = K[0];
      } else if (i < 40) {
        f = b ^ c ^ d;
        k = K[1];
      } else if (i < 60) {
        f = (b & c) | (b & d) | (c & d);
        k = K[2];
      } else {
        f = b ^ c ^ d;
        k = K[3];
      }

      uint32_t temp = rotateLeft(a, 5) + f + e + k + w[i];
      e = d;
      d = c;
      c = rotateLeft(b, 30);
      b = a;
      a = temp;
    }

    // Add this chunk's hash to result so far
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
  }

public:
  SHA1() { reset(); }

  void reset() {
    // Initialize hash values (first 32 bits of the fractional parts
    // of the square roots of the first 5 primes 2..11)
    h[0] = 0x67452301;
    h[1] = 0xEFCDAB89;
    h[2] = 0x98BADCFE;
    h[3] = 0x10325476;
    h[4] = 0xC3D2E1F0;

    buffer.clear();
    totalLength = 0;
  }

  void update(const uint8_t *data, size_t length) {
    totalLength += length;

    for (size_t i = 0; i < length; i++) {
      buffer.push_back(data[i]);

      // Process complete 512-bit blocks
      if (buffer.size() == 64) {
        processBlock(buffer.data());
        buffer.clear();
      }
    }
  }

  void update(const std::string &data) {
    update(reinterpret_cast<const uint8_t *>(data.c_str()), data.length());
  }

  std::string finalize() {
    // Pre-processing: adding padding bits
    buffer.push_back(0x80); // APpend bit '1' to message

    // If buffer size if > 56 bytes, we need another block
    while (buffer.size() % 64 != 56) {
      buffer.push_back(0x00);
    }

    // Append original length in bits as 64-bit big-endian integer
    uint64_t bitLength = totalLength * 8;
    for (int i = 7; i >= 0; i--) {
      buffer.push_back((bitLength >> (i * 8)) & 0xFF);
    }

    // Process final block(s)
    for (size_t i = 0; i < buffer.size(); i += 64) {
      processBlock(buffer.data() + i);
    }

    // Produce the final hash value as a 160-bit number (20 bytes)
    std::stringstream ss;
    for (int i = 0; i < 5; i++) {
      ss << std::hex << std::setfill('0') << std::setw(8) << h[i];
    }

    return ss.str();
  }

  static std::string hash(const std::string &input) {
    SHA1 sha1;
    sha1.update(input);
    return sha1.finalize();
  }
};

// SHA-1 constants definition
const uint32_t SHA1::K[4] = {
    0x5A827999, // 0 <= t <= 19
    0x6ED9EBA1, // 20 <= t <= 39
    0x8F1BBCDC, // 40 <= t <= 59
    0xCA62C1D6  // 60 <= t <= 79
};

std::string sha1_checksum(const fs::path_t &filepath) {
  auto file = fs::ifstream(filepath);
  if (!file) {
    throw std::runtime_error("Cannot open file: " + filepath.string());
  }

  SHA1 sha1;
  std::vector<char> buffer(1048576);
  while (file->read(buffer.data(), buffer.size()) || file->gcount()) {
    sha1.update(reinterpret_cast<const uint8_t *>(buffer.data()),
                file->gcount());
  }

  return sha1.finalize();
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

using json = nlohmann::json;

fs::path_t get_cache_root() {
  fs::path_t cache_root;
  if (std::getenv("AILOY_CACHE_ROOT")) {
    // Check environment variable
    cache_root = fs::path_t(std::getenv("AILOY_CACHE_ROOT"));
  } else {
    // Set to default cache root
#if defined(_WIN32)
    if (std::getenv("LOCALAPPDATA"))
      cache_root = fs::path_t(std::getenv("LOCALAPPDATA")) / "ailoy";
#elif defined(EMSCRIPTEN)
    cache_root = "/ailoy";
#else
    if (std::getenv("HOME"))
      cache_root = fs::path_t(std::getenv("HOME")) / ".cache" / "ailoy";
#endif
  }
  if (cache_root.string().empty()) {
    throw exception("Cannot get cache root");
  }

  auto result = fs::create_directory(cache_root, true);
  if (!result.success() && result.code != fs::error_code_t::AlreadyExists) {
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

std::pair<bool, std::string> download_file(const std::string &remote_path,
                                           const fs::path_t &local_path) {
  auto res = ailoy::http::request({
      .url = std::format("{}/{}", get_models_url(), remote_path),
      .method = ailoy::http::method_t::GET,
  });
  if (res->status_code != ailoy::http::OK_200) {
    return std::make_pair<bool, std::string>(
        false,
        "Failed to download " + remote_path + ": " +
            (res ? "HTTP " + std::to_string(res->status_code) : res.error()));
  }

  // std::ofstream ofs(local_path, std::ios::binary);
  auto ofs = fs::ofstream(local_path);
  ofs->write(res->body.c_str(), res->body.size());

  return std::make_pair<bool, std::string>(true, "");
}

std::pair<bool, std::string> download_file_with_progress(
    const std::string &remote_path, const fs::path_t &local_path,
    std::function<bool(uint64_t, uint64_t)> progress_callback) {
  SigintGuard sigint_guard;

  size_t existing_size = 0;

  auto file_url = std::format("{}/{}", get_models_url(), remote_path);

  auto res = ailoy::http::request({
      .url = file_url,
      .method = ailoy::http::method_t::GET,
      .data_callback =
          [&](const char *data, size_t data_length) {
            // Stop on SIGINT
            if (sigint_guard.interrupted())
              return false;

            auto ofs = fs::ofstream(local_path, existing_size > 0);
            ofs->seekp(existing_size);
            ofs->write(data, data_length);
            existing_size += data_length;
            return ofs->good();
          },
      .progress_callback = progress_callback,
  });

  if (res->status_code != ailoy::http::OK_200 &&
      res->status_code != ailoy::http::PartialContent_206) {
    // If SIGINT interrupted, return error message about interrupted
    if (sigint_guard.interrupted())
      return std::make_pair<bool, std::string>(
          false, "Interrupted while downloading the model");

    // Otherwise, return error message about HTTP error
    return std::make_pair<bool, std::string>(
        false,
        "Failed to download " + remote_path + ": " +
            (res ? "HTTP " + std::to_string(res->status_code) : res.error()));
  }

  return std::make_pair<bool, std::string>(true, "");
}

fs::path_t get_model_base_path(const std::string &model_id) {
  std::string model_id_escaped =
      std::regex_replace(model_id, std::regex("/"), "--");
  fs::path_t model_base_path = fs::path_t("tvm-models") / model_id_escaped;

  return model_base_path;
}

std::vector<model_cache_list_result_t> list_local_models() {
  std::vector<model_cache_list_result_t> results;

  fs::path_t cache_base_path = get_cache_root();

  // TVM models
  fs::path_t tvm_models_path = cache_base_path / "tvm-models";
  if (!fs::directory_exists(tvm_models_path).unwrap())
    return results;

  // Directory structure example for TVM models:
  // BAAI--bge-m3 (model_id)
  // └── q4f16_1 (quantization)
  //     ├── manifest-arm64-Darwin-metal.json (manifest)
  //     └── ...files...
  auto model_entries = fs::list_directory(tvm_models_path).unwrap();
  for (const auto &model_entry : model_entries) {
    if (!model_entry.is_directory())
      continue;

    // Get model id and denormalize it
    // e.g. "BAAI--bge-m3" -> "BAAI/bge-m3"
    std::string model_id =
        std::regex_replace(model_entry.name, std::regex("--"), "/");

    // Iterate over quantizations
    auto quant_entries = fs::list_directory(model_entry.path).unwrap();
    for (const auto &quant_entry : quant_entries) {
      if (!quant_entry.is_directory())
        continue;

      std::string quantization = quant_entry.name;
      fs::path_t quant_dir = quant_entry.path;

      // Iterate over files
      auto file_entries = fs::list_directory(quant_dir).unwrap();
      for (const auto &file_entry : file_entries) {
        if (!file_entry.is_regular_file())
          continue;

        // Find the manifest file
        std::string filename = file_entry.name;
        if (filename.rfind("manifest-", 0) != 0 ||
            file_entry.path.extension() != ".json")
          continue;

        std::string manifest_stem = file_entry.path.stem();
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
        auto manifest_json =
            nlohmann::json::parse(fs::read_file_text(file_entry.path).unwrap());

        // Calculate total bytes
        size_t total_bytes = 0;
        if (manifest_json.contains("files") &&
            manifest_json["files"].is_array()) {
          for (const auto &file_pair : manifest_json["files"]) {
            if (file_pair.is_array() && file_pair.size() >= 1) {
              fs::path_t file_path =
                  quant_dir / file_pair[0].get<std::string>();
              if (fs::file_exists(file_path)) {
                total_bytes += fs::get_file_size(file_path);
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
               bool print_progress_bar, bool skip_integrity_check) {
  model_cache_download_result_t result{.success = false};

  // Create local cache directory
  fs::path_t model_base_path = get_model_base_path(model_id);
  fs::path_t model_cache_path =
      get_cache_root() / model_base_path / quantization;
  fs::create_directory(model_cache_path, true);

  // Assemble manifest filename based on arch, os and target device
  auto uname = get_uname();
  std::string target_lib =
      std::format("{}-{}-{}", uname.machine, uname.sysname, target_device);
  std::string manifest_filename = std::format("manifest-{}.json", target_lib);

  // Download manifest if not already present
  fs::path_t manifest_path = model_cache_path / manifest_filename;
  if (!fs::file_exists(manifest_path)) {
    auto [success, error_message] = download_file(
        (model_base_path / quantization / manifest_filename).string(),
        manifest_path);
    if (!success) {
      result.error_message = error_message;
      return result;
    }
  }

  // Read and parse manifest

  json manifest;
  try {
    manifest =
        nlohmann::json::parse(fs::read_file_text(manifest_path).unwrap());
  } catch (const json::parse_error &e) {
    // Remove manifest if it's not a valid format
    fs::delete_file(manifest_path);

    result.error_message = "Failed to parse manifest: " + std::string(e.what());
    return result;
  }

  // Get files from "files" section
  if (!manifest.contains("files") || !manifest["files"].is_array()) {
    result.error_message = "Manifest is missing a valid 'files' array";
    return result;
  }

  std::vector<std::string> files_to_download;
  for (auto &[file, sha1] :
       manifest["files"]
           .get<std::vector<std::pair<std::string, std::string>>>()) {
    // Skip downloading this file if
    // 1. The file exists and
    // 2. skip_integrity_check is enabled or
    // 3. sha1 checksum is same as expected
    if (fs::file_exists(model_cache_path / file).unwrap() &&
        (skip_integrity_check ||
         sha1 == sha1_checksum(model_cache_path / file)))
      continue;

    files_to_download.emplace_back(file);
  }

  size_t num_total_files = manifest["files"].size();
  size_t num_files_to_download = files_to_download.size();
  size_t num_files_downloaded = num_total_files - num_files_to_download;

#ifdef _WIN32
  SetConsoleOutputCP(CP_UTF8); // to print progress properly on Windows
#endif

  indicators::DynamicProgress<indicators::BlockProgressBar> bars;
  bars.set_option(indicators::option::HideBarWhenComplete{true});

  auto total_progress_bar = std::make_unique<indicators::BlockProgressBar>(
      indicators::option::BarWidth{0}, indicators::option::Start{""},
      indicators::option::End{""}, indicators::option::ShowPercentage{false},
      indicators::option::ShowElapsedTime{true});
  auto total_progress_bar_idx = bars.push_back(std::move(total_progress_bar));

  for (size_t i = 0; i < num_files_to_download; ++i) {
    const auto &file = files_to_download[i];
    fs::path_t local_path = model_cache_path / file;

    // Update total progress bar
    if (print_progress_bar) {
      bars[total_progress_bar_idx].set_option(indicators::option::PrefixText{
          std::format("Downloading model files ({}/{})",
                      num_files_downloaded + i + 1, num_total_files)});
      bars[total_progress_bar_idx].set_progress(
          static_cast<float>(num_files_downloaded + i + 1) * 100 /
          num_total_files);
    }

    // Create a new progress bar for the current file
    auto bar = std::make_unique<indicators::BlockProgressBar>(
        indicators::option::BarWidth{50},
        indicators::option::PrefixText{file + " "},
        indicators::option::ShowPercentage{true},
        indicators::option::ShowElapsedTime{true});
    auto bar_idx = bars.push_back(std::move(bar));

    auto [download_success, download_error_message] =
        download_file_with_progress(
            (model_base_path / quantization / file).string(), local_path,
            [&](uint64_t current, uint64_t total) {
              float progress = std::min(
                  static_cast<float>(current) /
                      (total + std::numeric_limits<float>::min()) * 100,
                  100.0f);
              if (callback.has_value())
                callback.value()(i + num_files_downloaded, num_total_files,
                                 file, progress);
              if (print_progress_bar)
                bars[bar_idx].set_progress(progress);
              return true;
            });

    if (!download_success) {
      result.error_message = download_error_message;
      return result;
    }

    if (print_progress_bar) {
      // Complete progress bar for the current file
      bars[bar_idx].mark_as_completed();
    }
  }
  if (print_progress_bar) {
    // Complete total progress bar
    bars[total_progress_bar_idx].mark_as_completed();
    // Ensure final flush
    bars.print_progress();
  }

  // Get model lib file path
  std::string model_lib_file = manifest["lib"].get<std::string>();
  fs::path_t model_lib_path = model_cache_path / model_lib_file;

  result.success = true;
  result.model_path = model_cache_path;
  result.model_lib_path = model_lib_path;
  return result;
}

model_cache_remove_result_t remove_model(const std::string &model_id,
                                         bool ask_prompt) {
  model_cache_remove_result_t result{.success = false};

  fs::path_t model_path = get_cache_root() / get_model_base_path(model_id);
  if (!fs::directory_exists(model_path)) {
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

  fs::delete_directory(model_path, true);
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

    // Check skip_integrity_check
    bool skip_integrity_check = false;
    if (inputs_map->contains("skip_integrity_check") &&
        inputs_map->at("skip_integrity_check")->is_type_of<bool_t>()) {
      skip_integrity_check = *inputs_map->at<bool_t>("skip_integrity_check");
    }

    // Download the model
    auto result =
        ailoy::download_model(model_id, quantization, device, std::nullopt,
                              true, skip_integrity_check);
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
