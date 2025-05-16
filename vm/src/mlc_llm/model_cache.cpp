#include "model_cache.hpp"

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

namespace fs = std::filesystem;
using json = nlohmann::json;

#if defined(_WIN32)
static const fs::path CACHE_ROOT =
    fs::path(std::getenv("LOCALAPPDATA")) / "ailoy";
#else
static const fs::path CACHE_ROOT =
    fs::path(std::getenv("HOME")) / ".cache" / "ailoy";
#endif

static const std::string R2_BASE_URL = "https://models.download.ailoy.co";

void download_file(httplib::Client &client, const std::string &remote_path,
                   const fs::path &local_path) {
  httplib::Result res = client.Get(("/" + remote_path).c_str());
  if (!res || (res->status != httplib::OK_200)) {
    throw ailoy::exception(
        "Failed to download " + remote_path + ": HTTP " +
        (res ? std::to_string(res->status) : httplib::to_string(res.error())));
  }

  std::ofstream ofs(local_path, std::ios::binary);
  ofs.write(res->body.c_str(), res->body.size());
}

void download_file_with_progress(
    httplib::Client &client, const std::string &remote_path,
    const fs::path &local_path,
    std::function<bool(uint64_t, uint64_t)> progress_callback) {
  size_t existing_size = 0;
  httplib::Result res = client.Get(
      ("/" + remote_path).c_str(),
      [&](const char *data, size_t data_length) {
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
    throw ailoy::exception(
        "Failed to download " + remote_path + ": HTTP " +
        (res ? std::to_string(res->status) : httplib::to_string(res.error())));
  }
}

fs::path get_model_base_path(const std::string &model_name,
                             const std::string &quantization) {
  std::string model_name_escaped =
      std::regex_replace(model_name, std::regex("/"), "--");
  fs::path model_base_path =
      fs::path("tvm-models") / model_name_escaped / quantization;

  return model_base_path;
}

void remove_model(const std::string &model_name,
                  const std::string &quantization) {
  fs::path model_cache_path =
      CACHE_ROOT / get_model_base_path(model_name, quantization);
  if (fs::exists(model_cache_path)) {
    fs::remove_all(model_cache_path);
  }
}

std::tuple<std::filesystem::path, std::filesystem::path>
get_model(const std::string &model_name, const std::string &quantization,
          const std::string &target_device,
          std::optional<model_cache_callback_t> callback,
          bool print_progress_bar) {
  auto client = httplib::Client(R2_BASE_URL);
  client.set_connection_timeout(10, 0);
  client.set_read_timeout(60, 0);

  // Create local cache directory
  fs::path model_base_path = get_model_base_path(model_name, quantization);
  fs::path model_cache_path = CACHE_ROOT / model_base_path;
  fs::create_directories(model_cache_path);

  // Assemble manifest filename based on arch, os and target device
  auto uname = get_uname();
  std::string target_lib =
      std::format("{}-{}-{}", uname.machine, uname.sysname, target_device);
  std::string manifest_filename = std::format("manifest-{}.json", target_lib);

  // Download manifest.json if not already present
  fs::path manifest_path = model_cache_path / manifest_filename;
  if (!fs::exists(manifest_path)) {
    download_file(client, (model_base_path / manifest_filename).string(),
                  manifest_path);
  }

  // Read and parse manifest.json
  std::ifstream manifest_file(manifest_path);
  if (!manifest_file.is_open()) {
    throw ailoy::exception("Failed to open manifest.json at " +
                           manifest_path.string());
  }

  json manifest;
  try {
    manifest_file >> manifest;
  } catch (const json::parse_error &e) {
    // Remove manifest.json if it's not a valid format
    manifest_file.close();
    fs::remove(manifest_path);
    throw ailoy::exception("Failed to parse manifest.json: " +
                           std::string(e.what()));
  }
  manifest_file.close();

  // Get files from "files" section
  if (!manifest.contains("files") || !manifest["files"].is_array()) {
    throw ailoy::exception("Manifest is missing a valid 'files' array");
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

    download_file_with_progress(
        client, (model_base_path / file).string(), local_path,
        [&](uint64_t current, uint64_t total) {
          float progress = static_cast<float>(current) / total * 100;
          if (callback.has_value())
            callback.value()(i, total_files, file, progress);
          if (print_progress_bar)
            bars[bar_idx].set_progress(progress);
          return true;
        });
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

  return std::make_tuple(model_cache_path, model_lib_path);
}

} // namespace ailoy
