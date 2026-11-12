#ifndef RIME_PREDICT_INIT_TRACE_H_
#define RIME_PREDICT_INIT_TRACE_H_

#include <glog/logging.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <exception>
#include <new>
#include <string>
#include <system_error>

#ifdef _WIN32
#include <windows.h>
#endif

namespace rime::predict {

struct InitTraceErrorState {
  int saved_errno = errno;
#ifdef _WIN32
  DWORD saved_windows_error = GetLastError();
#endif
  ~InitTraceErrorState() {
    errno = saved_errno;
#ifdef _WIN32
    SetLastError(saved_windows_error);
#endif
  }
};

inline thread_local const char* init_trace_stage = "query";

struct InitTraceScope {
  const char* previous_stage = init_trace_stage;
  ~InitTraceScope() noexcept { init_trace_stage = previous_stage; }
};
constexpr size_t kInitTraceMessageBytes = 384;

inline bool IsModelInitStage(const char* stage) noexcept {
  for (const auto* allowed : {"ensure_engine", "resolve_model_path", "path_cache_compare",
       "path_cache_store", "vocabulary_exists", "vocabulary_failure_log", "backend_config_path",
       "backend_shared", "prediction_engine", "ready_log", "ct2_decode_params",
       "ct2_compute_type", "ct2_construct_translator", "ct2_vocabulary", "ct2_tokenizer",
       "ct2_tokenizer_self_test"}) {
    if (std::strcmp(stage, allowed) == 0) return true;
  }
  return false;
}

inline const char* InitTraceDevice(const std::string& device) noexcept {
  if (device == "cpu") return "cpu";
  if (device.find("cuda") != std::string::npos || device.find("CUDA") != std::string::npos)
    return "cuda";
  return "other";
}

inline std::string SanitizeInitMessage(const char* message,
                                       const std::string& model_path) {
  constexpr size_t kReadLimit = 2048;
  size_t length = 0;
  if (message) while (length < kReadLimit && message[length]) ++length;
  std::string text(message ? message : "", length);
  const auto replace = [&](const std::string& path) {
    if (path.empty()) return;
    size_t position = 0;
    while ((position = text.find(path, position)) != std::string::npos) {
      text.replace(position, path.size(), "<model_path>");
      position += sizeof("<model_path>") - 1;
    }
  };
  replace(model_path);
  auto variant = model_path;
  std::replace(variant.begin(), variant.end(), '\\', '/');
  replace(variant);
  std::replace(variant.begin(), variant.end(), '/', '\\');
  replace(variant);
  // Unquoted paths may contain spaces; redact through the next delimiter.
  for (size_t i = 0; i < text.size();) {
    const bool drive = i + 2 < text.size() &&
        std::isalpha(static_cast<unsigned char>(text[i])) && text[i + 1] == ':' &&
        (text[i + 2] == '/' || text[i + 2] == '\\');
    const bool profile = text.compare(i, 7, "/Users/") == 0 ||
                         text.compare(i, 6, "/home/") == 0;
    const bool unc = i + 1 < text.size() &&
        ((text[i] == '\\' && text[i + 1] == '\\') || (text[i] == '/' && text[i + 1] == '/'));
    if (!drive && !profile && !unc) { ++i; continue; }
    const auto end = text.find_first_of("\"'\r\n])}", i);
    text.replace(i, (end == std::string::npos ? text.size() : end) - i, "<path>");
    i += sizeof("<path>") - 1;
  }
  for (char& ch : text) {
    if (static_cast<unsigned char>(ch) < 32 || ch == 127) ch = ' ';
    if (ch == '"') ch = '\'';
  }
  if (text.size() > kInitTraceMessageBytes) text.resize(kInitTraceMessageBytes);
  return text;
}

// Track phases internally; only failures emit diagnostics. Scopes preserve reentrant callers.
inline void TraceInitStage(const char* stage, const char*,
                           const std::string&,
                           const std::string&) noexcept {
  init_trace_stage = stage;
}

inline void TraceInitFailure(const char* stage, const char* kind,
                             const char* category, int code, int json_id,
                             const char* message, const std::string& model_path,
                             const std::string& device) noexcept {
  InitTraceErrorState preserve_errors;
  try {
    const bool non_ascii = std::any_of(model_path.begin(), model_path.end(),
        [](unsigned char ch) { return ch >= 128; });
    const char* safe_message = IsModelInitStage(stage) ? message : "<omitted outside initialization>";
#ifdef _WIN32
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    const BOOL memory_ok = GlobalMemoryStatusEx(&memory);
    const DWORD memory_error = memory_ok ? 0 : GetLastError();
#endif
    LOG(ERROR) << "ai_predict_init stage=" << stage << " phase=failed"
               << " device='" << InitTraceDevice(device) << "'"
               << " model_configured=" << !model_path.empty()
               << " model_nonascii=" << non_ascii << " model_bytes=" << model_path.size()
               << " exception_kind=" << kind
               << " system_category=" << category << " system_code=" << code
               << " json_id=" << json_id
               << " what=\"" << SanitizeInitMessage(safe_message, model_path) << "\""
#ifdef _WIN32
               << " mem_supported=1 mem_ok=" << !!memory_ok << " mem_error=" << memory_error
               << " total_phys=" << memory.ullTotalPhys << " avail_phys=" << memory.ullAvailPhys
               << " total_commit=" << memory.ullTotalPageFile << " avail_commit=" << memory.ullAvailPageFile
#else
               << " mem_supported=0"
#endif
               ;
  } catch (...) {
    std::fputs("ai_predict_init: exception diagnostics unavailable\n", stderr);
  }
}

inline void TraceInitException(const std::exception& error,
                               const std::string& model_path,
                               const std::string& device) noexcept {
  InitTraceErrorState preserve_errors;
  if (const auto* system = dynamic_cast<const std::system_error*>(&error)) {
    TraceInitFailure(init_trace_stage, "system_error", system->code().category().name(),
                     system->code().value(), 0, error.what(), model_path, device);
  } else if (const auto* json = dynamic_cast<const nlohmann::json::exception*>(&error)) {
    TraceInitFailure(init_trace_stage, "json_error", "none", 0, json->id,
                     error.what(), model_path, device);
  } else {
    TraceInitFailure(init_trace_stage,
                     dynamic_cast<const std::bad_alloc*>(&error) ? "bad_alloc" : "std_exception",
                     "none", 0, 0, error.what(), model_path, device);
  }
}

}  // namespace rime::predict
#endif  // RIME_PREDICT_INIT_TRACE_H_
