//
// Copyright RIME Developers
// Distributed under the BSD License
//

#include <glog/logging.h>
#include <rime_api.h>
#include <rime/common.h>
#include <rime/registry.h>
#include <rime/service.h>

#include <cstdlib>
#include <string>

#include "predict_filter.h"
#include "predict_translator.h"

namespace {

// Whether the plugin owns glog is a per-platform question, because how glog is
// linked differs.
//
// macOS/Windows: the plugin statically links its own copy, so the process holds
// two glog instances that can't see each other. This is by design -- the librime
// maintainers won't change the main project's linking (rime/librime#983, #984).
// An uninitialized instance would send our LOG calls to a dead sink and the
// logs would vanish, so the plugin must initialize its own.
//
// Linux distribution builds: glog is a shared library shared with the host, so
// there is only one instance and the host (fcitx5) has already initialized it.
// Initializing again is redundant, and the matching ShutdownGoogleLogging on
// finalize would tear down the host's logging with it. So the plugin touches
// neither.
//
// The guard is a static bool rather than google::IsGoogleLoggingInitialized(),
// which does not exist in glog 0.5 (Ubuntu 22.04).
//
// Naming convention (all platforms):
//   - log_dir comes from RIME_LOG_DIR / GOOGLE_LOG_DIR (set by the frontend);
//   - program name is the host's deployer.app_name + ".ai_predict", producing
//     "rime.<frontend>.ai_predict.*.log.INFO" -- physically separate from the
//     main process's "rime.<frontend>.*.log.INFO".
void EnsurePluginLoggingInitialized() {
  static bool initialized = false;
  if (initialized) {
    return;
  }
  initialized = true;

  if (const char* dir = std::getenv("RIME_LOG_DIR")) {
    if (dir[0] != '\0') {
      FLAGS_log_dir = dir;
    }
  } else if (const char* dir = std::getenv("GOOGLE_LOG_DIR")) {
    if (dir[0] != '\0') {
      FLAGS_log_dir = dir;
    }
  }

  // app_name must outlive this call (glog keeps it by pointer), hence static.
  static std::string app_name = [] {
    const auto& host = rime::Service::instance().deployer().app_name;
    return host.empty() ? std::string("rime.ai_predict")
                        : host + ".ai_predict";
  }();

  google::SetLogFilenameExtension(".log");
  google::SetLogSymlink(google::GLOG_INFO, app_name.c_str());
  google::SetLogSymlink(google::GLOG_WARNING, app_name.c_str());
  google::SetLogSymlink(google::GLOG_ERROR, app_name.c_str());
  FLAGS_logfile_mode = 0600;
#if defined(__APPLE__) || defined(_WIN32)
  google::InitGoogleLogging(app_name.c_str());
#endif
}

}  // namespace

static void rime_ai_predict_initialize() {
  EnsurePluginLoggingInitialized();
  LOG(INFO) << "registering components from module 'ai_predict'.";
  rime::Registry& r = rime::Registry::instance();
  r.Register("ai_predict_translator",
             new rime::Component<rime::predict::PredictTranslator>);
  r.Register("ai_predict_filter",
             new rime::Component<rime::predict::PredictFilter>);
}

static void rime_ai_predict_finalize() {
#if defined(__APPLE__) || defined(_WIN32)
  if (google::IsGoogleLoggingInitialized()) {
    google::ShutdownGoogleLogging();
  }
#endif
}

RIME_REGISTER_MODULE(ai_predict)
