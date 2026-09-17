// Copyright RIME Developers
// Distributed under the BSD License

#include "prediction_engine.h"

#include <glog/logging.h>

#include <algorithm>
#include <chrono>

#include <rime/context.h>
#include <rime/engine.h>

namespace rime {
namespace predict {

namespace {

string StripSpaces(const string& s) {
  string out = s;
  out.erase(std::remove(out.begin(), out.end(), ' '), out.end());
  return out;
}

}  // namespace

PredictionEngine::PredictionEngine(Engine* engine,
                                   std::unique_ptr<InferenceBackend> backend,
                                   const PredictionEngineOptions& opt)
    : engine_(engine), backend_(std::move(backend)), opt_(opt) {
  worker_ = std::thread(&PredictionEngine::WorkerLoop, this);
}

PredictionEngine::~PredictionEngine() {
  {
    std::lock_guard<std::mutex> lock(worker_mutex_);
    shutdown_ = true;
    worker_cv_.notify_one();
  }
  if (worker_.joinable()) {
    worker_.join();
  }
}

void PredictionEngine::Schedule(const PredictionContext& ctx) {
  std::lock_guard<std::mutex> lock(worker_mutex_);
  // Already in flight for this key -- nothing to do. The frontend re-composes
  // while collecting a result, which re-runs Query; without this guard each of
  // those refreshes would restart inference for a key already on its way.
  if (!pending_cache_key_.empty() && ctx.cache_key == pending_cache_key_) {
    return;
  }
  pending_ct2_input_ = ctx.ct2_input;
  pending_cache_key_ = ctx.cache_key;
  pending_prompt_ = ctx.effective_prompt;
  worker_cv_.notify_one();
}

std::optional<std::string> PredictionEngine::GetCachedResult(
    const std::string& cache_key) const {
  std::lock_guard<std::mutex> lock(result_mutex_);
  if (cache_key == last_cache_key_ && last_raw_result_) {
    return last_raw_result_;
  }
  return std::nullopt;
}

void PredictionEngine::ClearCache() {
  std::lock_guard<std::mutex> lock(result_mutex_);
  last_cache_key_.clear();
  last_raw_result_.reset();
}

void PredictionEngine::WorkerLoop() {
  while (true) {
    string ct2_input;
    string cache_key;
    string prompt_snap;
    {
      std::unique_lock<std::mutex> lock(worker_mutex_);
      worker_cv_.wait(lock, [this] {
        return shutdown_ || !pending_ct2_input_.empty();
      });
      if (shutdown_) {
        return;
      }
      ct2_input = pending_ct2_input_;
      pending_ct2_input_.clear();
      cache_key = pending_cache_key_;
      prompt_snap = pending_prompt_;

      // Debounce: wait for quiet period, coalescing rapid updates.
      while (true) {
        if (worker_cv_.wait_for(lock, std::chrono::milliseconds(opt_.debounce_ms)) ==
            std::cv_status::timeout) {
          break;
        }
        if (shutdown_)
          return;
        if (!pending_ct2_input_.empty()) {
          ct2_input = pending_ct2_input_;
          pending_ct2_input_.clear();
          cache_key = pending_cache_key_;
          prompt_snap = pending_prompt_;
        }
      }
    }

    if (ct2_input.empty() || !backend_ || !backend_->IsInitialized()) {
      continue;
    }

    LOG(INFO) << "PredictionEngine: invoking backend ct2_input='" << ct2_input
              << "' max_tokens=" << opt_.max_tokens;
    auto t0 = std::chrono::steady_clock::now();
    auto result = backend_->Predict(ct2_input, opt_.max_tokens);
    auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - t0)
                     .count();
    if (!result || result->empty()) {
      LOG(WARNING) << "PredictionEngine: inference failed or empty (took "
                   << dt_ms << " ms)";
      continue;
    }
    LOG(INFO) << "PredictionEngine: backend returned raw='" << *result
              << "' (took " << dt_ms << " ms)";

    {
      std::lock_guard<std::mutex> lock(result_mutex_);
      last_cache_key_ = cache_key;
      last_raw_result_ = result;
    }

    // Nothing else happens on this thread, on purpose.
    //
    // It used to call ctx->RefreshNonConfirmedComposition() here to surface the
    // candidate right away. That runs a full Compose() -- every translator and
    // filter, the Lua ones included -- on this thread, while the client thread
    // may be inside Compose() on the *same* Context because the user pressed a
    // key. librime's Context/Engine/translators are not thread-safe, so the two
    // raced: the crash report shows both threads inside LuaTranslator::Query
    // under concurrent Compose, one of them taking a bad access in luajit, and
    // the CPU profile shows this thread pinning a core in the same stack.
    //
    // So the worker now only fills the cache. Surfacing the result belongs to
    // the client thread: PredictTranslator::Query raises ai_predict/pending on
    // a cache miss and clears it on a hit, and the frontend forces a refresh
    // from its own thread while that flag is up.
    LOG(INFO) << "PredictionEngine: cached result for cache_key='" << cache_key
              << "'; frontend surfaces it";
    (void)prompt_snap;
  }
}

}  // namespace predict
}  // namespace rime
