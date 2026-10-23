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

#include "ct2_backend.h"
#include "predict_filter.h"
#include "predict_translator.h"

namespace {

// Static linking does not imply a separate glog instance: Windows links this
// plugin into rime.dll and resolves both users to the same symbols. Detect the
// actual instance and never reconfigure or shut down a logger owned by the host.
#if defined(__APPLE__) || defined(_WIN32)
bool plugin_owns_logging = false;
#endif

void EnsurePluginLoggingInitialized() {
#if defined(__APPLE__) || defined(_WIN32)
  if (google::IsGoogleLoggingInitialized()) return;
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
  google::InitGoogleLogging(app_name.c_str());
  plugin_owns_logging = true;
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
  rime::predict::CT2Backend::ReleaseShared();
#if defined(__APPLE__) || defined(_WIN32)
  if (plugin_owns_logging && google::IsGoogleLoggingInitialized()) {
    google::ShutdownGoogleLogging();
  }
  plugin_owns_logging = false;
#endif
}

RIME_REGISTER_MODULE(ai_predict)
