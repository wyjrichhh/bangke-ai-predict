#include <glog/logging.h>
#include <rime/candidate.h>
#include <rime/config.h>
#include <rime/context.h>
#include <rime/engine.h>
#include <rime/schema.h>
#include <rime/translation.h>
#include <rime_api.h>

#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "context_builder.h"
#include "frontend_protocol.h"
#include "predict_filter.h"
#include "predict_translator.h"
#include "prediction_engine.h"

using namespace rime;
using namespace rime::predict;
using namespace std::chrono_literals;

static void Require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

class ControlledBackend : public InferenceBackend {
 public:
  bool Initialize(const InferenceBackendConfig&) override { return true; }
  void Shutdown() override {}
  bool IsInitialized() const override { return true; }
  std::optional<std::string> Predict(const std::string& prompt,
                                     std::optional<int>) override {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto index = calls_.size();
    calls_.push_back({prompt, false, "结果"});
    cv_.notify_all();
    cv_.wait(lock, [&] { return release_all_ || calls_[index].released; });
    return calls_[index].answer;
  }
  std::string WaitForCall(size_t count) {
    std::unique_lock<std::mutex> lock(mutex_);
    Require(cv_.wait_for(lock, 3s, [&] { return calls_.size() >= count; }),
            "backend request did not arrive");
    return calls_[count - 1].prompt;
  }
  void Release(size_t index, const std::string& answer) {
    std::lock_guard<std::mutex> lock(mutex_);
    calls_.at(index).answer = answer;
    calls_.at(index).released = true;
    cv_.notify_all();
  }
  void ReleaseAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    release_all_ = true;
    cv_.notify_all();
  }

 private:
  struct Call {
    std::string prompt;
    bool released;
    std::string answer;
  };
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<Call> calls_;
  bool release_all_ = false;
};

struct WorkerFixture {
  std::shared_ptr<ControlledBackend> backend =
      std::make_shared<ControlledBackend>();
  PredictionEngine worker{nullptr, backend, PredictionEngineOptions{0, {}}};
  ~WorkerFixture() { backend->ReleaseAll(); }
  static PredictionContext Request(const std::string& key) {
    PredictionContext ctx;
    ctx.effective_prompt = key;
    ctx.cache_key = key;
    ctx.ct2_input = key;
    return ctx;
  }
};

namespace rime::predict {
struct PredictInitTest {
  static void Seed(PredictTranslator& translator,
                   const std::shared_ptr<InferenceBackend>& backend,
                   const PredictionContext& context,
                   const std::string& result) {
    translator.engine_path_ = translator.ResolveModelPath().u8string();
    translator.init_attempted_ = translator.init_ok_ = true;
    translator.prediction_ = std::make_unique<PredictionEngine>(
        nullptr, backend, PredictionEngineOptions{0, {}});
    std::lock_guard<std::mutex> lock(translator.prediction_->result_mutex_);
    translator.prediction_->last_cache_key_ = context.cache_key;
    translator.prediction_->last_raw_result_ = result;
  }
};
}  // namespace rime::predict

struct PipelineFixture {
  std::unique_ptr<Engine> engine{Engine::Create()};
  std::shared_ptr<ControlledBackend> backend =
      std::make_shared<ControlledBackend>();
  std::unique_ptr<PredictTranslator> translator;
  std::unique_ptr<PredictFilter> filter;
  explicit PipelineFixture(bool ai_only = false, bool enabled = true) {
    auto* config = new Config;
    auto processors = New<ConfigList>();
    processors->Append(New<ConfigValue>("speller"));
    auto segmentors = New<ConfigList>();
    segmentors->Append(New<ConfigValue>("abc_segmentor"));
    auto translators = New<ConfigList>();
    translators->Append(New<ConfigValue>("echo_translator"));
    config->SetItem("engine/processors", processors);
    config->SetItem("engine/segmentors", segmentors);
    config->SetItem("engine/translators", translators);
    config->SetString("speller/alphabet",
                      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");
    config->SetString("ai_predict/model_path", "synthetic-model");
    config->SetString("ai_predict/enabled", enabled ? "true" : "false");
    config->SetString("ai_predict/ai_only", ai_only ? "true" : "false");
    config->SetInt("ai_predict/min_input_length", 12);
    config->SetInt("ai_predict/min_context_prompt_length", 2);
    engine->ApplySchema(new Schema("async-test", config));
    SetInput("gupiao");
    engine->context()->set_property(protocol::kClientContext,
                                    "这是供测试使用的正文上下文");
    translator =
        std::make_unique<PredictTranslator>(Ticket(engine.get(), "ai_predict"));
    filter =
        std::make_unique<PredictFilter>(Ticket(engine.get(), "ai_predict"));
  }
  ~PipelineFixture() {
    backend->ReleaseAll();
    translator.reset();
  }
  void SetInput(const std::string& input) {
    auto* context = engine->context();
    context->set_input(input);
    context->set_caret_pos(input.size());
    context->composition().clear();
    context->composition().push_back(
        Segment(0, static_cast<int>(input.size())));
  }
  an<Translation> Publish(const std::string& result, bool ai_only = false) {
    ContextBuilderOptions options;
    options.ai_only = ai_only;
    const auto built = ContextBuilder::Build(
        engine.get(), engine->context()->input(), options);
    Require(built.has_value(), "current test request must be eligible");
    PredictInitTest::Seed(*translator, backend, *built, result);
    return translator->Query(engine->context()->input(),
                             engine->context()->composition().back());
  }
  std::vector<std::string> Candidates(bool upstream_ai = false) {
    auto upstream = New<FifoTranslation>();
    if (upstream_ai)
      upstream->Append(
          New<SimpleCandidate>("ai_predict", 0, 2, "旧AI结果", "AI"));
    upstream->Append(New<SimpleCandidate>("phrase", 0, 2, "普通候选"));
    auto result = filter->Apply(upstream, nullptr);
    std::vector<std::string> values;
    for (int count = 0; result && !result->exhausted() && count < 8; ++count) {
      if (auto candidate = result->Peek()) values.push_back(candidate->text());
      result->Next();
    }
    return values;
  }
};

static bool Contains(const std::vector<std::string>& values,
                     const std::string& text) {
  return std::find(values.begin(), values.end(), text) != values.end();
}

int main(int argc, char** argv) {
  const auto isolated =
      std::filesystem::temp_directory_path() /
      ("bangke-ai-async-test-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(isolated);
  std::ofstream(isolated / "default.yaml") << "schema_list: []\n";
  std::ofstream(isolated / "user.yaml") << "{}\n";
  const auto isolated_utf8 = isolated.u8string();
#ifndef AI_TEST_RIME_OWNS_LOGGING
  google::InitGoogleLogging("predict_async_test");
#endif
  FLAGS_logtostderr = true;
  FLAGS_minloglevel = 2;
  RimeTraits traits{};
  RIME_STRUCT_INIT(RimeTraits, traits);
  traits.shared_data_dir = isolated_utf8.c_str();
  traits.user_data_dir = isolated_utf8.c_str();
  traits.log_dir = "";
  traits.app_name = "rime.async-test";
  const char* modules[] = {"core",     "dict",       "gears",
                           "deployer", "ai_predict", nullptr};
  traits.modules = modules;
  auto* api = rime_get_api();
  api->setup(&traits);
  api->initialize(&traits);
  int failures = 0;
  const auto test = [&](const char* name, const std::function<void()>& body) {
    try {
      body();
      std::cout << "PASS " << name << '\n';
    } catch (const std::exception& error) {
      ++failures;
      std::cout << "FAIL " << name << ": " << error.what() << '\n';
    }
  };
  test("superseded worker result is discarded", [] {
    WorkerFixture f;
    f.worker.Schedule(WorkerFixture::Request("A"));
    Require(f.backend->WaitForCall(1) == "A", "first request identity");
    f.worker.Schedule(WorkerFixture::Request("B"));
    f.backend->Release(0, "旧结果");
    Require(f.backend->WaitForCall(2) == "B", "replacement request identity");
    Require(!f.worker.GetCachedResult("A"),
            "obsolete worker output was published");
  });
  test("cache invalidation survives an in-flight completion", [] {
    WorkerFixture f;
    f.worker.Schedule(WorkerFixture::Request("A"));
    f.backend->WaitForCall(1);
    f.worker.ClearCache();
    f.worker.Schedule(WorkerFixture::Request("B"));
    f.backend->Release(0, "旧结果");
    f.backend->WaitForCall(2);
    Require(!f.worker.GetCachedResult("A"), "cleared result was resurrected");
  });
  test("A-B-A does not accept the first A completion", [] {
    WorkerFixture f;
    f.worker.Schedule(WorkerFixture::Request("A"));
    f.backend->WaitForCall(1);
    f.worker.Schedule(WorkerFixture::Request("B"));
    f.worker.Schedule(WorkerFixture::Request("A"));
    f.backend->Release(0, "第一次A");
    Require(f.backend->WaitForCall(2) == "A", "latest A must be executed");
    Require(!f.worker.GetCachedResult("A"),
            "first A result escaped into a later A request");
  });
  test("unbound AI context property cannot create a row", [] {
    PipelineFixture f;
    f.engine->context()->set_property("ai_predict/text",
                                      "属于先前输入的一大段旧文本");
    Require(f.Candidates() == std::vector<std::string>{"普通候选"},
            "unbound old property became a candidate");
  });
  test("valid matching prediction remains visible", [] {
    PipelineFixture f;
    Require(f.Publish("股票") != nullptr,
            "matching cached prediction was rejected");
    Require(Contains(f.Candidates(), "股票"), "valid current row is missing");
  });
  test("deleting input invalidates a published row without Query", [] {
    PipelineFixture f;
    f.Publish("股票");
    f.SetInput("gu");
    Require(!Contains(f.Candidates(), "股票"),
            "old row remained after deleting input");
  });
  test("clearing and retyping does not reuse a published row", [] {
    PipelineFixture f;
    f.Publish("股票");
    f.engine->context()->Clear();
    f.SetInput("gupiao");
    Require(!Contains(f.Candidates(), "股票"),
            "row survived composition cancellation");
  });
  test("caret movement invalidates a published row", [] {
    PipelineFixture f;
    f.Publish("股票");
    f.engine->context()->set_caret_pos(2);
    Require(!Contains(f.Candidates(), "股票"),
            "old row survived caret movement");
  });
  test("host document edits invalidate a published row", [] {
    PipelineFixture f;
    f.Publish("股票");
    f.engine->context()->set_property(protocol::kClientContext,
                                      "已删除原文并改成其他内容");
    Require(!Contains(f.Candidates(), "股票"),
            "old document context row survived");
  });
  test("disabled AI ignores a leftover property", [] {
    PipelineFixture f(false, false);
    f.engine->context()->set_property("ai_predict/text", "旧预测");
    Require(f.Candidates() == std::vector<std::string>{"普通候选"},
            "disabled filter inserted an AI row");
  });
  test("orphan upstream AI row is removed", [] {
    PipelineFixture f;
    Require(f.Candidates(true) == std::vector<std::string>{"普通候选"},
            "orphan translator AI row leaked through");
  });
  const std::string long_result =
      "市场运行情况需要关注稳市政策与充裕流动性以及投资者预期等多个方面";
  test("short ordinary input rejects a grossly overlong fresh prediction", [&] {
    PipelineFixture f;
    f.SetInput("gu");
    Require(!f.Publish(long_result),
            "fresh short request accepted an unrelated long continuation");
    Require(!Contains(f.Candidates(), long_result),
            "long continuation remained in the menu");
  });
  test("AI-only retains intentional free generation", [&] {
    PipelineFixture f(true);
    Require(f.Publish(long_result, true) != nullptr,
            "AI-only long completion was rejected");
    Require(Contains(f.Candidates(), long_result),
            "AI-only completion disappeared");
  });
  test("empty completed inference does not reschedule or keep polling", [] {
    PipelineFixture f;
    Require(!f.Publish(""), "empty result should not create a translation");
    Require(f.engine->context()->get_property("ai_predict/pending").empty(),
            "empty result restarted inference polling");
  });
  test("same-input refresh retains the current cached prediction", [] {
    PipelineFixture f;
    f.Publish("股票");
    f.engine->context()->set_option("soft_cursor", false);
    f.engine->context()->set_caret_pos(f.engine->context()->caret_pos());
    Require(f.translator->Query(
                "gupiao", f.engine->context()->composition().back()) != nullptr,
            "refresh canceled a matching cache entry");
    Require(Contains(f.Candidates(), "股票"),
            "refresh lost a valid prediction");
  });
  test("inactive segment cannot publish a cached whole-input result", [] {
    PipelineFixture f;
    f.Publish("股票");
    Require(!f.translator->Query("gu", Segment(0, 2)),
            "inactive segment reused the full-input AI row");
    Require(Contains(f.Candidates(), "股票"),
            "inactive lookup destroyed the active prediction");
  });
  test("ASCII mode invalidates the published prediction", [] {
    PipelineFixture f;
    f.Publish("股票");
    f.engine->context()->set_option("ascii_mode", true);
    Require(!Contains(f.Candidates(), "股票"), "ASCII mode retained an AI row");
  });
  test("AI-only uses current bound text instead of an orphan upstream row",
       [&] {
         PipelineFixture f(true);
         f.Publish(long_result, true);
         const auto values = f.Candidates(true);
         Require(values == std::vector<std::string>{long_result},
                 "AI-only reused a stale translator candidate");
       });
  test("ordinary long pinyin still permits the matching long sentence", [&] {
    PipelineFixture f;
    f.SetInput(std::string(100, 'a'));
    Require(f.Publish(long_result) != nullptr,
            "normal long input lost its prediction");
    Require(Contains(f.Candidates(), long_result),
            "normal long candidate disappeared");
  });
  test("malformed host context cannot block ordinary dictionary candidates",
       [] {
         PipelineFixture f;
         f.Publish("股票");
         f.translator.reset();
         f.engine->context()->set_property(
             protocol::kClientContext,
             std::string("private_context_sentinel") + "\xff");
         Require(f.Candidates() == std::vector<std::string>{"普通候选"},
                 "malformed context blocked dictionary fallback");
       });
  test("active segment changes cannot adopt a completed old cache entry", [] {
    PipelineFixture f;
    f.Publish("股票");
    f.engine->context()->composition().back().start = 2;
    Require(
        !f.translator->Query("piao", f.engine->context()->composition().back()),
        "a different segment adopted the previous completion");
  });
  test("ASCII-mode Query never starts inference", [] {
    PipelineFixture f;
    f.Publish("股票");
    f.engine->context()->set_option("ascii_mode", true);
    Require(!f.translator->Query("gupiao", Segment(0, 6)),
            "ASCII mode produced an AI translation");
    Require(f.engine->context()->get_property("ai_predict/pending").empty(),
            "ASCII mode scheduled inference");
  });
  if (argc == 2)
    test("real model through registered Rime CAPI lifecycle", [&] {
      std::string model = std::filesystem::u8path(argv[1]).generic_u8string();
      for (size_t at = 0; (at = model.find('\'', at)) != std::string::npos;
           at += 2)
        model.insert(at, "'");
      std::ofstream dictionary(isolated / "async_api.dict.yaml");
      dictionary
          << "---\nname: async_api\nversion: '1'\nsort: "
             "by_weight\nuse_preset_vocabulary: false\n...\n"
             "股\tgu\t100\n姑\tgu\t90\n股票\tgu piao\t100\nA股\tA gu\t100\n"
             "我\two\t100\n我想\two xiang\t100\n我知道了\two zhi dao le\t100\n"
             "我想知道现在输入法是不是正常\two xiang zhi dao xian zai shu ru "
             "fa shi bu shi zheng chang\t100\n";
      dictionary.close();
      std::ofstream schema(isolated / "async_api.schema.yaml");
      schema << "schema:\n  schema_id: async_api\n  name: async_api\n  "
                "version: '1'\n"
                "engine:\n  processors: [speller, selector, express_editor]\n"
                "  segmentors: [abc_segmentor]\n"
                "  translators: [ai_predict_translator, script_translator]\n"
                "  filters: [uniquifier, ai_predict_filter]\n"
                "speller:\n  alphabet: "
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\n"
                "  delimiter: \" '\"\n"
                "translator:\n  dictionary: async_api\n  enable_user_dict: "
                "false\n  enable_completion: false\n"
                "menu:\n  page_size: 5\n"
                "ai_predict:\n  model_path: '"
             << model
             << "'\n"
                "  enabled: true\n  debounce_ms: 20\n  max_tokens: 64\n"
                "  min_input_length: 12\n  min_context_prompt_length: 2\n";
      schema.close();
      Require(api->deploy_schema(
                  (isolated / "async_api.schema.yaml").u8string().c_str()),
              "real Chinese dictionary schema deployment");
      const auto session = api->create_session();
      Require(session != 0, "real CAPI session");
      struct SessionGuard {
        RimeApi* api;
        RimeSessionId id;
        ~SessionGuard() { api->destroy_session(id); }
      } guard{api, session};
      Require(api->select_schema(session, "async_api"),
              "real CAPI schema selection");
      api->set_option(session, "ascii_mode", false);
      const auto property = [&](const char* name) {
        char value[4096] = {};
        api->get_property(session, name, value, sizeof(value));
        return std::string(value);
      };
      const auto pump = [&] {
        const auto deadline = std::chrono::steady_clock::now() + 5s;
        while (true) {
          RimeContext context{};
          RIME_STRUCT_INIT(RimeContext, context);
          Require(api->get_context(session, &context), "real CAPI context");
          api->free_context(&context);
          if (property("ai_predict/pending").empty()) return;
          Require(std::chrono::steady_clock::now() < deadline,
                  "AI polling must settle");
          api->set_option(session, "soft_cursor",
                          api->get_option(session, "soft_cursor"));
          std::this_thread::yield();
        }
      };
      api->simulate_key_sequence(
          session, "woxiangzhidaoxianzaishurufashibushizhengchang");
      pump();
      Require(!property("ai_predict/text").empty(),
              "real long pinyin obtains a model prediction");
      Require(!property(protocol::kPredictionScope).empty(),
              "production translator publishes a bound scope");
      api->clear_composition(session);
      Require(property("ai_predict/text").empty() &&
                  property("ai_predict/pending").empty(),
              "real clear invalidates text and pending immediately");
      api->set_property(session, protocol::kClientContext,
                        "测试文档中的前一句已经输入完毕，接下来只输入一个很短的词。");
      api->simulate_key_sequence(session, "Agu");
      pump();
      const auto short_text = property("ai_predict/text");
      Require(short_text.empty() || CountHan(short_text) <= 3,
              "real short mixed input must not display a finance paragraph");
      RimeContext context{};
      RIME_STRUCT_INIT(RimeContext, context);
      Require(api->get_context(session, &context), "real short mixed menu");
      bool has_stock = false;
      for (int i = 0; i < context.menu.num_candidates; ++i)
        has_stock =
            has_stock || std::string(context.menu.candidates[i].text) == "A股";
      api->free_context(&context);
      Require(has_stock, "real dictionary keeps A股 available");
      api->clear_composition(session);
      api->set_property(session, protocol::kClientContext, "");
      api->simulate_key_sequence(
          session, "woxiangzhidaoxianzaishurufashibushizhengchang");
      pump();
      Require(!property("ai_predict/text").empty(),
              "long prediction still works after clear and retype");
      api->set_caret_pos(session, 2);
      Require(property("ai_predict/text").empty(),
              "real caret move removes the prior long row");
    });

  api->finalize();
  google::ShutdownGoogleLogging();
  std::error_code cleanup_error;
  std::filesystem::remove_all(isolated, cleanup_error);
  if (cleanup_error) {
    ++failures;
    std::cerr << "FAIL cleanup code=" << cleanup_error.value() << '\n';
  }
  std::cout << "RESULT failures=" << failures << '\n';
  return failures ? 1 : 0;
}
