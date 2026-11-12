#include "predict_translator.h"
#include "ct2_backend.h"
#include "predict_filter.h"
#include "init_trace.h"

#include <rime_api.h>
#include <rime/component.h>
#include <rime/key_event.h>
#include <rime/menu.h>
#include <rime/registry.h>
#include <rime/config.h>
#include <rime/context.h>
#include <rime/engine.h>
#include <rime/schema.h>
#include <rime/service.h>
#include <glog/logging.h>

#include <chrono>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <typeinfo>
#include <vector>
#include <thread>

namespace fs = std::filesystem;
using namespace rime::predict;

namespace rime::predict {
struct PredictInitTest {
  static bool Ensure(PredictTranslator& translator) { return translator.EnsureEngine(); }
  static void SetPath(PredictTranslator& translator, const std::string& path) {
    translator.model_path_raw_ = path;
  }
  static bool PrimeCompletedInference(PredictTranslator& translator,
                                     std::shared_ptr<InferenceBackend> backend,
                                     const std::string& input) {
    translator.RefreshResourceObservation();
    translator.ctx_opt_.ai_only = true;
    translator.engine_->context()->set_input(input);
    const auto context = ContextBuilder::Build(translator.engine_, input,
                                               translator.ctx_opt_);
    if (!context) return false;
    translator.engine_path_ = translator.ResolveModelPath().u8string();
    translator.init_attempted_ = translator.init_ok_ = true;
    PredictionEngineOptions options;
    options.debounce_ms = 0;
    translator.prediction_ = std::make_unique<PredictionEngine>(
        translator.engine_, std::move(backend), options);
    translator.prediction_->Schedule(*context);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < deadline) {
      if (translator.prediction_->GetCachedResult(context->cache_key)) return true;
      std::this_thread::yield();
    }
    return false;
  }
};
}  // namespace rime::predict

class CompletedBackend : public InferenceBackend {
 public:
  bool Initialize(const InferenceBackendConfig&) override { return true; }
  void Shutdown() override {}
  bool IsInitialized() const override { return true; }
  std::optional<std::string> Predict(const std::string&, std::optional<int>) override {
    return "\xe4\xbd\xa0\xe5\xa5\xbd";
  }
};

static void Require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

class Capture : public google::LogSink {
 public:
  Capture() { google::AddLogSink(this); }
  ~Capture() override { Stop(); }
  void Stop() { if (active_) { google::RemoveLogSink(this); active_ = false; } }
  bool overwrite_error = false;
  struct Entry { google::LogSeverity severity; std::string message; };
  void send(google::LogSeverity severity, const char*, const char*, int,
            const google::LogMessageTime&, const char* message, size_t bytes) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (overwrite_error) {
      errno = EIO;
#ifdef _WIN32
      SetLastError(5678);
#endif
    }
    entries_.push_back({severity, std::string(message, bytes)});
  }
  bool Contains(google::LogSeverity severity, const std::string& text) {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::any_of(entries_.begin(), entries_.end(), [&](const Entry& entry) {
      return entry.severity == severity && entry.message.find(text) != std::string::npos;
    });
  }
  std::string LastError() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto i = entries_.rbegin(); i != entries_.rend(); ++i)
      if (i->severity == google::GLOG_ERROR && i->message.find("ai_predict_init ") == 0) return i->message;
    return {};
  }
 private:
  bool active_ = true;
  std::mutex mutex_;
  std::vector<Entry> entries_;
};

struct TempDirectory {
  fs::path path = fs::temp_directory_path() / fs::u8path(
      "bangke-ai-trace-中文 空格~" + std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()));
  TempDirectory() { fs::create_directories(path); }
  ~TempDirectory() {
    std::error_code ec;
    fs::remove_all(path, ec);
    if (ec) std::cerr << "cleanup failed code=" << ec.value() << '\n';
  }
};


static void Write(const fs::path& path, const std::string& content) {
  std::ofstream stream(path);
  stream << content;
  stream.close();
  Require(!stream.fail(), "write isolated fixture");
}

static rime::an<rime::ConfigList> List(std::initializer_list<const char*> values) {
  auto result = rime::New<rime::ConfigList>();
  for (auto value : values) result->Append(rime::New<rime::ConfigValue>(value));
  return result;
}

static rime::Config* Configuration(const fs::path& model, bool ai_only = false) {
  auto config = new rime::Config;
  config->SetString("schema/name", "init-test");
  config->SetItem("engine/processors", List({"speller"}));
  config->SetItem("engine/segmentors", List({"abc_segmentor"}));
  config->SetItem("engine/translators", List({"test_predict", "echo_translator"}));
  config->SetItem("engine/filters", List({"test_predict_filter"}));
  config->SetString("speller/alphabet", "abcdefghijklmnopqrstuvwxyz");
  config->SetString("ai_predict/model_path", model.u8string().c_str());
  config->SetString("ai_predict/enabled", "true");
  config->SetString("ai_predict/device", "cpu");
  config->SetString("ai_predict/ai_only", ai_only ? "true" : "false");
  return config;
}

static int RunTests(const fs::path& model_source) {
  try {
    google::InitGoogleLogging("predict_trace_test");
    FLAGS_logtostderr = true;
    Capture capture;
    TempDirectory temp;
    Write(temp.path / "default.yaml", "schema_list: []\n");
    Write(temp.path / "user.yaml", "{}\n");
    const auto root = temp.path.u8string();
    RimeTraits traits{};
    RIME_STRUCT_INIT(RimeTraits, traits);
    traits.shared_data_dir = root.c_str();
    traits.user_data_dir = root.c_str();
#ifdef _WIN32
    // The DLL's private glog keeps file handles until process exit.
    traits.log_dir = "";
#else
    const auto native_log_root = temp.path.string();
    traits.log_dir = native_log_root.c_str();
#endif
    traits.app_name = "rime.ai_trace_test";
    const char* modules[] = {"core", "dict", "gears", nullptr};
    traits.modules = modules;
    RimeApi* api = rime_get_api();
    api->setup(&traits);
    api->initialize(&traits);
    std::vector<std::string> resource_events;
    const auto resource_observer = [](void* context, RimeSessionId,
                                      const char* type, const char* value) {
      if (type && value && std::strcmp(type, "resource") == 0)
        static_cast<std::vector<std::string>*>(context)->emplace_back(value);
    };
    api->set_notification_handler(resource_observer, &resource_events);
    struct ClearResourceObserver {
      RimeApi* api;
      ~ClearResourceObserver() { api->set_notification_handler(nullptr, nullptr); }
    } clear_resource_observer{api};
    const auto resource_count = [&](const char* token) {
      return std::count(resource_events.begin(), resource_events.end(), token);
    };

    capture.overwrite_error = true;
    errno = EACCES;
#ifdef _WIN32
    SetLastError(1234);
#endif
    TraceInitStage("error_state_probe", "begin", "", "cpu");
    Require(errno == EACCES, "trace preserves caller errno");
#ifdef _WIN32
    Require(GetLastError() == 1234, "trace preserves caller Windows error");
#endif
    capture.overwrite_error = false;

    const std::string configured = "C:\\Users\\PrivateUser\\model path";
    const auto clean = SanitizeInitMessage(
        "failed C:/Users/PrivateUser/model path/file.bin] denied\r\n", configured);
    Require(clean.find("PrivateUser") == std::string::npos && clean.find("denied") != std::string::npos,
            "drive and slash variants are redacted without hiding trailing reason");
    const std::string relative = "predict_models/private-model";
    const auto exact = SanitizeInitMessage(("cannot open " + relative).c_str(), relative);
    Require(exact.find(relative) == std::string::npos && exact.find("cannot open") != std::string::npos,
            "configured relative model is redacted");
    const auto bounded = SanitizeInitMessage((std::string(3000, 'x') + "\n").c_str(), "");
    Require(bounded.size() <= kInitTraceMessageBytes, "what has a strict byte cap");
    Require(SanitizeInitMessage("bad\r\nvalue\t", "").find_first_of("\r\n\t") == std::string::npos,
            "what is a single control-free line");

    std::unique_ptr<rime::Engine> engine(rime::Engine::Create());
    engine->schema()->config()->SetString("ai_predict/enabled", "true");
    engine->schema()->config()->SetString("ai_predict/model_path", "missing-model");
    engine->schema()->config()->SetString("ai_predict/device", "cpu");
    auto* context = engine->context();
    rime::predict::PredictTranslator translator(rime::Ticket(engine.get(), "ai_predict"));
    const auto missing_before = resource_count("model.failed.vocabulary");
    Require(!PredictInitTest::Ensure(translator), "missing model produces one resource failure");
    Require(!PredictInitTest::Ensure(translator), "missing model remains fail-open on retry");
    Require(resource_count("model.failed.vocabulary") == missing_before + 1,
            "repeated missing-model query does not duplicate resource events");
    PredictInitTest::SetPath(translator, "");
    const auto empty_before = resource_count("model.failed.path");
    Require(!PredictInitTest::Ensure(translator), "empty model path is reported");
    Require(!PredictInitTest::Ensure(translator), "empty model path stays fail-open");
    Require(resource_count("model.failed.path") == empty_before + 1,
            "empty-model resource failure is emitted once");
    PredictInitTest::SetPath(translator, "missing-model");
    Require(!PredictInitTest::Ensure(translator), "changing models permits a new attempt");
    Require(resource_count("model.failed.vocabulary") == missing_before + 2,
            "model change resets resource failure deduplication");
    {
      PredictTranslator observed(rime::Ticket(engine.get(), "ai_predict"));
      const auto invoked_before = resource_count("model.invoked");
      Require(PredictInitTest::PrimeCompletedInference(
                  observed, std::make_shared<CompletedBackend>(), "nihao"),
              "a backend completes an asynchronous result in the real cache");
      Require(resource_count("model.invoked") == invoked_before,
              "worker completion itself never notifies Rime");
      Require(bool(observed.Query("nihao", rime::Segment(0, 5))),
              "client Query observes the completed result");
      Require(bool(observed.Query("nihao", rime::Segment(0, 5))),
              "cached candidate remains available");
      Require(resource_count("model.invoked") == invoked_before + 1,
              "only the first successful client observation emits invoked");
      PredictInitTest::SetPath(observed, "another-model");
      Require(PredictInitTest::PrimeCompletedInference(
                  observed, std::make_shared<CompletedBackend>(), "nihao"),
              "another model completes its own inference");
      Require(bool(observed.Query("nihao", rime::Segment(0, 5))),
              "client observes the new model's result");
      Require(resource_count("model.invoked") == invoked_before + 2,
              "model change resets successful invocation observation");
    }
    {
      PredictTranslator callback_failure(rime::Ticket(engine.get(), "ai_predict"));
      Require(PredictInitTest::PrimeCompletedInference(
                  callback_failure, std::make_shared<CompletedBackend>(), "nihao"),
              "prepare completed inference for notification failure");
      api->set_notification_handler([](void*, RimeSessionId, const char*, const char*) {
        throw std::runtime_error("synthetic resource observer failure");
      }, nullptr);
      Require(bool(callback_failure.Query("nihao", rime::Segment(0, 5))),
              "throwing resource observer cannot hide a completed candidate");
      api->set_notification_handler(resource_observer, &resource_events);
    }
    context->Clear();
    const auto original = std::make_exception_ptr(std::system_error(
        std::make_error_code(std::errc::permission_denied),
        "synthetic_document_secret: cannot open C:\\Users\\PrivateUser\\model.bin"));
    std::unique_ptr<rime::Engine> nested_engine(rime::Engine::Create());
    nested_engine->schema()->config()->SetString("ai_predict/enabled", "true");
    nested_engine->schema()->config()->SetString("ai_predict/model_path", "");
    rime::predict::PredictTranslator nested(rime::Ticket(nested_engine.get(), "ai_predict"));
    const auto inject = [&](rime::Context*, const std::string& property) {
      if (property == "ai_predict/pending") {
        Require(!nested.Query("n", rime::Segment(0, 1)), "nested missing-model Query returns null");
        std::rethrow_exception(original);
      }
    };
    for (unsigned attempt = 0; attempt < 2; ++attempt) {
      context->set_property("ai_predict/pending", "1");
      context->set_property("ai_predict/text", "stale");
      auto connection = context->property_update_notifier().connect(inject);
      const char* outer_stage = init_trace_stage;
      Require(!translator.Query("synthetic_input_secret", rime::Segment(0, 22)), "failed AI returns no translation");
      connection.disconnect();
      Require(init_trace_stage == outer_stage, "Query restores its caller stage after an exception");
      Require(context->get_property("ai_predict/pending").empty() &&
              context->get_property("ai_predict/text").empty(),
              "AI failure clears both properties despite a throwing notification");
      for (auto severity : {google::GLOG_INFO, google::GLOG_WARNING, google::GLOG_ERROR})
        Require(!capture.Contains(severity, "synthetic_input_secret"), "input body is absent from plugin logs");
      const auto error = capture.LastError();
      Require(error.find("stage=query phase=failed") != std::string::npos &&
              error.find("exception_kind=system_error") != std::string::npos &&
              error.find("system_category=generic") != std::string::npos &&
              error.find("system_code=" + std::to_string(std::make_error_code(std::errc::permission_denied).value())) != std::string::npos &&
              error.find("PrivateUser") == std::string::npos &&
              error.find("synthetic_document_secret") == std::string::npos &&
              error.find("omitted outside initialization") != std::string::npos,
              "runtime ERROR omits user content but retains stage and error metadata");
    }

    {
      context->set_property("ai_predict/pending", "1");
      const auto unknown = std::make_exception_ptr(42);
      auto connection = context->property_update_notifier().connect(
          [&](rime::Context*, const std::string& property) {
            if (property == "ai_predict/pending") std::rethrow_exception(unknown);
          });
      Require(!translator.Query("n", rime::Segment(0, 1)), "unknown AI exceptions fail open");
      connection.disconnect();
      Require(capture.LastError().find("exception_kind=unknown_exception") != std::string::npos,
              "unknown exceptions remain diagnosed");
    }
    TraceInitStage("resolve_model_path", "begin", "", "private device data");
    const std::system_error technical(std::make_error_code(std::errc::permission_denied),
                                       "cannot open model file [C:\\Users\\PrivateUser\\model.bin]");
    TraceInitException(technical, "", "private device data");
    Require(capture.LastError().find("cannot open model file") != std::string::npos &&
            capture.LastError().find("private device data") == std::string::npos &&
            capture.LastError().find("device='other'") != std::string::npos,
            "initialization retains technical what and emits only a device enum");
    TraceInitException(std::runtime_error(std::string(4096, 'x')), "", "cpu");
    const auto bounded_record = capture.LastError();
    const auto what_start = bounded_record.find(" what=\"");
    const auto what_end = bounded_record.find('"', what_start == std::string::npos ? what_start : what_start + 7);
    Require(what_start != std::string::npos && what_end != std::string::npos &&
            what_end - what_start - 7 <= kInitTraceMessageBytes,
            "actual ERROR sink has a bounded what field");
#ifdef _WIN32
    Require(bounded_record.find("mem_supported=1 mem_ok=") != std::string::npos &&
            bounded_record.find("avail_phys=") != std::string::npos &&
            bounded_record.find("avail_commit=") != std::string::npos,
            "failure records Windows memory availability at the exception");
#else
    Require(bounded_record.find("mem_supported=0") != std::string::npos &&
            bounded_record.find("avail_phys=") == std::string::npos,
            "non-Windows does not invent Windows memory values");
#endif

    TraceInitStage("query", "begin", "", "cpu");
    const char* backend_parent_stage = init_trace_stage;
    CT2Backend broken;
    InferenceBackendConfig broken_config;
    broken_config.model_path = temp.path.u8string();
    Require(!broken.Initialize(broken_config), "CT2 keeps its existing false return on a broken model");
    Require(init_trace_stage == backend_parent_stage, "independent CT2 initialization restores its parent stage");
    Require(capture.Contains(google::GLOG_ERROR, "stage=ct2_construct_translator phase=failed"),
            "backend failure has an actual constructor stage");
    Require(capture.LastError().find(temp.path.u8string()) == std::string::npos,
            "backend exception redacts the configured path");
    try {
      const auto parsed = nlohmann::json::parse("{");
      (void)parsed;
      Require(false, "invalid JSON throws before returning a value");
    } catch (const nlohmann::json::exception& error) {
      TraceInitStage("json_probe", "begin", "", "cpu");
      TraceInitException(error, "", "cpu");
      Require(capture.LastError().find("json_id=" + std::to_string(error.id)) != std::string::npos,
              "JSON exception retains its real error id");
    }
    TraceInitStage("allocation_probe", "begin", "", "cpu");
    TraceInitException(std::bad_alloc(), "", "cpu");
    Require(capture.LastError().find("exception_kind=bad_alloc") != std::string::npos,
            "allocation failures have a distinct kind");

    rime::Registry::instance().Register("test_predict", new rime::Component<PredictTranslator>);
    rime::Registry::instance().Register("test_predict_filter", new rime::Component<PredictFilter>);
    for (bool only : {false, true}) {
      std::unique_ptr<rime::Engine> fallback(rime::Engine::Create());
      fallback->ApplySchema(new rime::Schema("fail-open-test", Configuration(temp.path, only)));
      auto* ctx = fallback->context();
      ctx->set_property("ai_predict/pending", "1");
      ctx->set_property("ai_predict/text", "stale");
      auto connection = ctx->property_update_notifier().connect(
          [](rime::Context*, const std::string& property) {
            if (property == "ai_predict/pending" || property == "ai_predict/text")
              throw std::system_error(std::make_error_code(std::errc::illegal_byte_sequence));
          });
      Require(fallback->ProcessKey(rime::KeyEvent('n', 0)), "AI exception does not abort input");
      connection.disconnect();
      Require(ctx->get_property("ai_predict/pending").empty() && ctx->get_property("ai_predict/text").empty(),
              "AI exception clears both stale properties");
      auto menu = ctx->composition().back().menu;
      Require(menu && menu->Prepare(1) == (only ? 0u : 1u),
              "ordinary translator runs after AI exception while AI-only remains respected");
    }

    {
      std::unique_ptr<rime::Engine> fallback(rime::Engine::Create());
      fallback->ApplySchema(new rime::Schema("missing-model-test", Configuration(temp.path)));
      auto* ctx = fallback->context();
      ctx->set_property("ai_predict/pending", "1");
      ctx->set_property("ai_predict/text", "stale");
      Require(fallback->ProcessKey(rime::KeyEvent('n', 0)), "normal initialization failure does not abort input");
      Require(ctx->get_property("ai_predict/pending").empty() && ctx->get_property("ai_predict/text").empty(),
              "normal initialization failure clears stale AI state");
      auto menu = ctx->composition().back().menu;
      Require(menu && menu->Prepare(1) == 1u, "ordinary candidate survives initialization returning false");
    }

    if (!model_source.empty()) {
      CT2Backend real;
      InferenceBackendConfig config;
      const auto unicode_model = temp.path / fs::u8path("模型目录 空格~");
      fs::create_directories(unicode_model);
      for (const auto& entry : fs::directory_iterator(model_source)) {
        if (!entry.is_regular_file()) continue;
        const auto target = unicode_model / entry.path().filename();
        if (entry.path().filename() == "model.bin")
          fs::create_hard_link(entry.path(), target);
        else
          fs::copy_file(entry.path(), target);
      }
      config.model_path = unicode_model.u8string();
      engine->schema()->config()->SetString("ai_predict/model_path", config.model_path.c_str());
      PredictTranslator unicode_translator(rime::Ticket(engine.get(), "ai_predict"));
      const auto ready_before = resource_count("model.ready");
      Require(PredictInitTest::Ensure(unicode_translator), "translator resolves and loads the Unicode model path");
      Require(PredictInitTest::Ensure(unicode_translator), "ready model is reused");
      Require(resource_count("model.ready") == ready_before + 1,
              "real model readiness is reported once without raising severity");
      for (const char* corrupt : {"shared_vocabulary.json", "tokenizer.json"}) {
        const auto invalid_model = temp.path / fs::u8path(std::string("invalid-") + corrupt);
        fs::create_directories(invalid_model);
        for (const auto& entry : fs::directory_iterator(unicode_model)) {
          if (!entry.is_regular_file()) continue;
          const auto target = invalid_model / entry.path().filename();
          if (entry.path().filename() == "model.bin")
            fs::create_hard_link(entry.path(), target);
          else
            fs::copy_file(entry.path(), target);
        }
        Write(invalid_model / corrupt, "{invalid-json");
        auto corrupt_config = config;
        corrupt_config.model_path = invalid_model.u8string();
        Require(!CT2Backend::Shared(corrupt_config), "post-construction model metadata failure returns without deadlock");
      }
      Require(real.Initialize(config), "real CPU model still initializes");
      Require(real.Predict("nihao", 8).value_or("").size() > 0, "normal model prediction is unchanged");
      real.Shutdown();
      Require(!capture.Contains(google::GLOG_INFO, "phase=end"),
              "successful initialization does not emit per-stage INFO");
    }
    const std::vector<std::string> resource_tokens{
        "dictionary.ready", "dictionary.failed.no-table", "dictionary.failed.table",
        "dictionary.failed.prism", "model.ready", "model.failed.path",
        "model.failed.vocabulary", "model.failed.backend", "model.failed.exception",
        "model.invoked"};
    for (const auto& event : resource_events)
      Require(std::find(resource_tokens.begin(), resource_tokens.end(), event) != resource_tokens.end(),
              "resource notifications contain only fixed protocol tokens");
    Require(CT2Backend::ReleaseShared(), "explicit release drains background model teardown");
    if (!model_source.empty()) {
      InferenceBackendConfig config;
      config.model_path = (temp.path / fs::u8path("模型目录 空格~")).u8string();
      Require(!CT2Backend::Shared(config), "finalized lifecycle rejects new shared models");
      CT2Backend::StartShared();
      auto shared = CT2Backend::Shared(config);
      Require(shared && shared->Predict("nihao", 8).value_or("").size() > 0,
              "new lifecycle loads and predicts again");
      std::weak_ptr<CT2Backend> released = shared;
      shared.reset();
      Require(CT2Backend::ReleaseShared(), "second lifecycle drains model teardown");
      Require(released.expired(), "shared backend does not survive lifecycle finalization");
    }
    nested_engine.reset();
    engine.reset();
    api->finalize();
    api->initialize(&traits);
    CT2Backend::StartShared();
    Require(CT2Backend::ReleaseShared(), "empty reinitialized lifecycle also finalizes");
    api->finalize();
    fs::remove_all(temp.path);
    Require(!fs::exists(temp.path), "remove isolated fixture directory");
    capture.Stop();
    google::ShutdownGoogleLogging();
    std::cout << "PASS fail-open trace, bounded redaction and model behavior\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
  return RunTests(argc == 2 ? fs::path(argv[1]) : fs::path());
}
#else
int main(int argc, char** argv) {
  return RunTests(argc == 2 ? fs::u8path(argv[1]) : fs::path());
}
#endif
