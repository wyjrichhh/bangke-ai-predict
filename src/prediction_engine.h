// Async prediction with debounce and versioned cache, collected by the caller thread.
//
// Copyright RIME Developers
// Distributed under the BSD License

#ifndef RIME_PREDICT_PREDICTION_ENGINE_H_
#define RIME_PREDICT_PREDICTION_ENGINE_H_

#include "context_builder.h"
#include "inference_backend.h"

#include <rime/common.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace rime {

class Context;
class Engine;

namespace predict {

struct PredictionEngineOptions {
  int debounce_ms = 200;
  // nullopt = schema 没配，交给后端用模型包 generation_config.json 里的值
  std::optional<int> max_tokens;
};

class PredictionEngine {
 public:
  PredictionEngine(Engine* engine,
                   std::shared_ptr<InferenceBackend> backend,
                   const PredictionEngineOptions& opt);
  ~PredictionEngine();

  InferenceBackend* backend() { return backend_.get(); }

  /// Coalesce inference requests; the worker only publishes a cache result.
  void Schedule(const PredictionContext& ctx);

  /// Last completed raw model output keyed by `PredictionContext::cache_key`
  /// (which combines window_text and effective_prompt to avoid
  /// cross-context pollution). An empty string is a completed failure; nullopt
  /// means no completed result for this key.
  std::optional<std::string> GetCachedResult(const std::string& cache_key) const;

  /// Invalidate completed results and cancel pending/in-flight publication.
  void ClearCache();

 private:
  friend struct PredictInitTest;
  void WorkerLoop();

  // 共享后端（进程级单例）：会话销毁只减引用，绝不触发 CT2 线程池拆除
  std::shared_ptr<InferenceBackend> backend_;
  PredictionEngineOptions opt_;

  std::thread worker_;
  std::mutex worker_mutex_;
  std::condition_variable worker_cv_;
  bool shutdown_ = false;

  std::string pending_ct2_input_;
  std::string pending_cache_key_;
  // Changes on replacement or cancellation, including A -> B -> A.
  uint64_t request_generation_ = 0;

  mutable std::mutex result_mutex_;
  std::string last_cache_key_;
  std::optional<std::string> last_raw_result_;
};

}  // namespace predict
}  // namespace rime

#endif  // RIME_PREDICT_PREDICTION_ENGINE_H_
