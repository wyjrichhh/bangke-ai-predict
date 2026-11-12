// ct2_backend.cc — InferenceBackend implementation: CTranslate2 + 分词器。
// 分词本身在 tokenizer.{h,cc}（按 tokenizer.json 的管线选实现），本文件只转发。
//
// Copyright RIME Developers
// Distributed under the BSD License

#include "ct2_backend.h"
#include "init_trace.h"

#include <glog/logging.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <fstream>
#include <filesystem>
#include <optional>
#include <thread>
#include <condition_variable>
#include <new>
#include <cstdio>

#include <ctranslate2/translator.h>
#include <ctranslate2/translation.h>
#include <nlohmann/json.hpp>

namespace rime {
namespace predict {

namespace {

std::string UnkTokenForVocab(const std::unordered_map<std::string, int>& token_to_id) {
  static const char* kCandidates[] = {"<unk>", "<UNK>", "[UNK]", "<|endoftext|>"};
  for (const char* t : kCandidates) {
    if (token_to_id.count(t)) {
      return t;
    }
  }
  return "<unk>";
}

}  // namespace

namespace {

struct PendingRelease {
  std::unique_ptr<ctranslate2::Translator> translator;
  PendingRelease* next = nullptr;
  std::thread worker;
};

struct SharedBackendState {
  std::mutex mutex;
  std::condition_variable released;
  size_t releasing = 0;
  size_t allocation_failures = 0;
  PendingRelease* pending = nullptr;
  bool finalizing = false;
  std::string key;
  std::shared_ptr<CT2Backend> instance;
};

SharedBackendState& SharedState() {
  static SharedBackendState state;
  return state;
}

void ReleaseError(const char* operation, int code) noexcept {
  try {
    LOG(ERROR) << "CT2Backend release failure operation=" << operation << " code=" << code;
  } catch (...) {
    std::fputs("CT2Backend release diagnostics unavailable\n", stderr);
  }
}

// Keep failed thread launches for a retry outside the session destruction lock.
void LaunchRelease(PendingRelease* pending) noexcept {
  auto& state = SharedState();
  std::lock_guard<std::mutex> lock(state.mutex);
  pending->next = state.pending;
  state.pending = pending;
  ++state.releasing;
  try {
    pending->worker = std::thread([pending]() {
      pending->translator.reset();
      auto& state = SharedState();
      std::lock_guard<std::mutex> lock(state.mutex);
      --state.releasing;
      state.released.notify_all();
    });
  } catch (const std::system_error& error) {
    ReleaseError("thread", error.code().value());
    --state.releasing;
    state.released.notify_all();
  } catch (...) {
    ReleaseError("thread", -1);
    --state.releasing;
    state.released.notify_all();
  }
}

// Joining also waits for the release thread's CRT/TLS teardown, unlike detach.
bool DrainReleases(bool retry_failed) {
  auto& state = SharedState();
  PendingRelease* pending;
  {
    std::unique_lock<std::mutex> lock(state.mutex);
    state.released.wait(lock, [&] { return state.releasing == 0; });
    pending = state.pending;
    state.pending = nullptr;
  }
  bool complete = true;
  while (pending) {
    auto* next = pending->next;
    pending->next = nullptr;
    if (pending->worker.joinable()) {
      pending->worker.join();
      delete pending;
    } else if (retry_failed) {
      LaunchRelease(pending);
    } else {
      std::lock_guard<std::mutex> lock(state.mutex);
      pending->next = state.pending;
      state.pending = pending;
      complete = false;
    }
    pending = next;
  }
  return complete;
}

// 配置指纹：模型路径、device 与全部解码参数。任一不同就要换一个后端实例——
// 解码参数是在 Initialize 里解析并固化到实例上的，只换参数不换模型同样要重建。
std::string ConfigKey(const InferenceBackendConfig& c) {
  const char sep = '\x1f';
  auto num = [](const auto& o) {
    return o ? std::to_string(*o) : std::string("-");
  };
  std::string k = c.model_path;
  k += sep;
  k += c.device;
  k += sep;
  k += num(c.decode.beam_size);
  k += sep;
  k += num(c.decode.length_penalty);
  k += sep;
  k += num(c.decode.repetition_penalty);
  k += sep;
  k += num(c.decode.sampling_topk);
  k += sep;
  k += num(c.decode.sampling_temperature);
  k += sep;
  k += num(c.decode.no_repeat_ngram_size);
  k += sep;
  k += c.decode.compute_type ? *c.decode.compute_type : std::string("-");
  return k;
}

}  // namespace

CT2Backend::~CT2Backend() {
  if (!translator_) return;
  auto* pending = new (std::nothrow) PendingRelease;
  if (!pending) {
    // Do not tear down CT2 under a session lock, even under memory pressure.
    translator_.release();
    {
      auto& state = SharedState();
      std::lock_guard<std::mutex> lock(state.mutex);
      ++state.allocation_failures;
    }
    ReleaseError("allocate", -1);
    return;
  }
  pending->translator = std::move(translator_);
  LaunchRelease(pending);
}

void CT2Backend::StartShared() {
  auto& state = SharedState();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.finalizing = false;
}

bool CT2Backend::ReleaseShared() {
  auto& state = SharedState();
  std::shared_ptr<CT2Backend> retired;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    state.finalizing = true;
    state.key.clear();
    retired = std::move(state.instance);
  }
  // The host must have destroyed sessions and joined inference workers first.
  const bool exclusive = !retired || retired.use_count() == 1;
  if (!exclusive) ReleaseError("external_references", static_cast<int>(retired.use_count() - 1));
  retired.reset();
  DrainReleases(true);
  const bool drained = DrainReleases(false);
  std::lock_guard<std::mutex> lock(state.mutex);
  return exclusive && drained && state.allocation_failures == 0;
}

std::shared_ptr<CT2Backend> CT2Backend::Shared(
    const InferenceBackendConfig& config) {
  auto& state = SharedState();

  const std::string key = ConfigKey(config);

  std::shared_ptr<CT2Backend> result;
  std::shared_ptr<CT2Backend> retired;
  std::shared_ptr<CT2Backend> backend;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.finalizing) return nullptr;
    if (state.instance && state.key == key) {
      return state.instance;
    }

    backend = std::make_shared<CT2Backend>();
    if (!backend->Initialize(config)) {
      // 保留现有实例：新模型坏掉不该连带把能用的旧模型也弄没。
      return nullptr;
    }
    retired = std::move(state.instance);
    state.instance = std::move(backend);
    state.key = key;
    result = state.instance;
  }

  // 出锁再放：旧实例的拆除（后台线程）不该压在换代锁上。
  retired.reset();
  return result;
}

bool CT2Backend::Initialize(const InferenceBackendConfig& config) {
  InitTraceScope trace_scope;
  std::lock_guard<std::mutex> lock(mutex_);

  if (initialized_.load(std::memory_order_acquire)) {
    LOG(WARNING) << "CT2Backend already initialized";
    return false;
  }

  const std::string& model_path = config.model_path;
  const std::string& device = config.device;
  device_ = InitTraceDevice(device);

  LOG(INFO) << "CT2Backend initialize device=" << InitTraceDevice(device);

  try {
    // 必须在建 Translator 之前：compute_type 是构造参数
    TraceInitStage("ct2_decode_params", "begin", model_path, device);
    ResolveDecodeParams(model_path, config.decode);
    TraceInitStage("ct2_decode_params", "end", model_path, device);

    ctranslate2::Device ct2_device = ctranslate2::Device::CPU;
    if (device.find("cuda") != std::string::npos ||
        device.find("CUDA") != std::string::npos) {
      ct2_device = ctranslate2::Device::CUDA;
      LOG(INFO) << "CT2Backend using CUDA";
    }

    TraceInitStage("ct2_compute_type", "begin", model_path, device);
    ctranslate2::ComputeType compute_type =
        ctranslate2::str_to_compute_type(compute_type_);

    TraceInitStage("ct2_compute_type", "end", model_path, device);
    ctranslate2::ReplicaPoolConfig pool_config;
    pool_config.num_threads_per_replica =
        static_cast<size_t>(kCt2DefaultIntraThreads);

    TraceInitStage("ct2_construct_translator", "begin", model_path, device);
    translator_ = std::make_unique<ctranslate2::Translator>(
        model_path, ct2_device, compute_type, std::vector<int>{0}, false,
        pool_config);

    TraceInitStage("ct2_construct_translator", "end", model_path, device);
    if (!translator_) {
      LOG(ERROR) << "CT2Backend failed to create translator";
      return false;
    }

    const std::string vocab_path = model_path + "/shared_vocabulary.json";
    TraceInitStage("ct2_vocabulary", "begin", model_path, device);
    if (!LoadSharedVocabulary(vocab_path)) {
      TraceInitStage("ct2_vocabulary", "failed", model_path, device);
      LOG(ERROR) << "CT2Backend failed to load shared_vocabulary.json";
      translator_.reset();
      return false;
    }

    TraceInitStage("ct2_vocabulary", "end", model_path, device);
    unk_token_ = UnkTokenForVocab(token_to_id_);

    std::string tokenizer_error;
    TraceInitStage("ct2_tokenizer", "begin", model_path, device);
    tokenizer_ = Tokenizer::Create(model_path + "/tokenizer.json", token_to_id_,
                                   unk_token_, &tokenizer_error);
    if (!tokenizer_) {
      TraceInitStage("ct2_tokenizer", "failed", model_path, device);
      TraceInitFailure("ct2_tokenizer", "tokenizer_error", "none", 0, 0,
                       tokenizer_error.c_str(), model_path, device_);
      translator_.reset();
      return false;
    }

    TraceInitStage("ct2_tokenizer", "end", model_path, device);
    TraceInitStage("ct2_tokenizer_self_test", "begin", model_path, device);
    if (!VerifyTokenizerSelfTest()) {
      TraceInitStage("ct2_tokenizer_self_test", "failed", model_path, device);
      translator_.reset();
      return false;
    }

    TraceInitStage("ct2_tokenizer_self_test", "end", model_path, device);
    LOG(INFO) << "CT2Backend tokenizer: kind=" << tokenizer_->Kind()
              << " vocab=" << token_to_id_.size();

    initialized_.store(true, std::memory_order_release);
    return true;

  } catch (const std::exception& e) {
    TraceInitException(e, model_path, device);
    translator_.reset();
    return false;
  }
}

void CT2Backend::Shutdown() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_.load(std::memory_order_acquire)) {
    return;
  }

  LOG(INFO) << "CT2Backend shutting down";
  translator_.reset();
  tokenizer_.reset();
  token_to_id_.clear();
  id_to_token_.clear();
  unk_token_.clear();
  initialized_.store(false, std::memory_order_release);
}

std::optional<std::string> CT2Backend::Predict(
    const std::string& prompt, std::optional<int> max_tokens) {
  if (!initialized_.load(std::memory_order_acquire)) {
    LOG(ERROR) << "CT2Backend::Predict: service not initialized";
    return std::nullopt;
  }

  auto start_time = std::chrono::steady_clock::now();

  try {
    if (prompt.empty()) {
      return std::nullopt;
    }

    std::vector<std::string> source_tokens = tokenizer_->Encode(prompt);

    if (source_tokens.size() <= 2) {
      return std::nullopt;
    }

    ctranslate2::TranslationOptions options;
    options.beam_size = beam_size_;
    options.max_decoding_length = max_tokens.value_or(max_tokens_);
    options.length_penalty = length_penalty_;
    options.repetition_penalty = repetition_penalty_;
    options.sampling_topk = sampling_topk_;
    options.sampling_temperature = sampling_temperature_;
    options.no_repeat_ngram_size = no_repeat_ngram_size_;
    options.num_hypotheses = kCt2DefaultNumHypotheses;

    std::vector<std::string> end_tokens;
    if (token_to_id_.count("</hanzi_start>")) {
      end_tokens.push_back("</hanzi_start>");
    }
    if (token_to_id_.count("</s>")) {
      end_tokens.push_back("</s>");
    }
    if (!end_tokens.empty()) {
      options.end_token = end_tokens;
    }

    std::vector<std::vector<std::string>> batch = {source_tokens};
    auto results = translator_->translate_batch(batch, options);

    // 超时判定放在推理「之后」：放在之前只能量到分词（微秒级），永远不触发，
    // 真正需要保护的长时间推理反而完全没被覆盖。这里丢弃的是已经跑完的结果，
    // 省不下算力——目的是不让一次超长推理的结果再去驱动候选刷新。
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - start_time)
                          .count();

    if (elapsed_ms > kCt2DefaultTimeoutMs) {
      LOG(WARNING) << "CT2Backend::Predict: dropped result after " << elapsed_ms
                   << "ms (limit " << kCt2DefaultTimeoutMs << "ms)";
      return std::nullopt;
    }

    if (results.empty() || results[0].hypotheses.empty()) {
      return std::nullopt;
    }

    const auto& output_tokens = results[0].output();
    std::string result = tokenizer_->Decode(output_tokens);

    const std::string special_tags[] = {
        "<hanzi_start>", "</hanzi_start>", "<pinyin_start>", "</pinyin_start>",
        "<s>", "</s>", "<pad>"};
    for (const auto& tag : special_tags) {
      size_t pos;
      while ((pos = result.find(tag)) != std::string::npos) {
        result.replace(pos, tag.length(), "");
      }
    }

    result.erase(std::remove(result.begin(), result.end(), ' '), result.end());

    if (result.empty()) {
      return std::nullopt;
    }

    return result;

  } catch (const std::exception& e) {
    TraceInitScope trace_scope;
    TraceInitStage("prediction", "failed", "", device_);
    TraceInitException(e, "", device_);
    return std::nullopt;
  }
}

bool CT2Backend::LoadSharedVocabulary(const std::string& path) {
  std::ifstream file(std::filesystem::u8path(path));
  if (!file.is_open()) {
    LOG(ERROR) << "CT2Backend: cannot open shared_vocabulary.json";
    return false;
  }

  try {
    nlohmann::json j;
    file >> j;
    if (!j.is_array()) {
      LOG(ERROR) << "CT2Backend: shared_vocabulary.json must be a JSON array";
      return false;
    }
    int id = 0;
    for (const auto& item : j) {
      if (!item.is_string()) {
        continue;
      }
      const std::string token = item.get<std::string>();
      token_to_id_[token] = id;
      id_to_token_.push_back(token);
      ++id;
    }
    LOG(INFO) << "CT2Backend: loaded " << id
              << " tokens from shared_vocabulary.json";
    return id > 0;
  } catch (const std::exception& e) {
    TraceInitException(e, path, device_);
    return false;
  }
}

void CT2Backend::ResolveDecodeParams(const std::string& model_path,
                                     const DecodeParams& overrides) {
  beam_size_ = kCt2DefaultBeamSize;
  length_penalty_ = kCt2DefaultLengthPenalty;
  repetition_penalty_ = kCt2DefaultRepetitionPenalty;
  sampling_topk_ = kCt2DefaultSamplingTopk;
  sampling_temperature_ = kCt2DefaultSamplingTemperature;
  no_repeat_ngram_size_ = kCt2DefaultNoRepeatNgramSize;
  max_tokens_ = kCt2DefaultMaxTokens;
  compute_type_ = kCt2DefaultComputeType;

  // 模型包自带解码参数：训练侧调 beam 之类的约定只需重新出模型，不必我们发版。
  // 文件缺失是正常情况（旧模型），静默落到编译期默认值。
  const std::string path = model_path + "/generation_config.json";
  std::ifstream file(std::filesystem::u8path(path));
  if (file.is_open()) {
    try {
      nlohmann::json j;
      file >> j;
      auto take_int = [&](const char* key, int* out) {
        if (j.contains(key) && j[key].is_number_integer()) {
          *out = j[key].get<int>();
        }
      };
      auto take_float = [&](const char* key, float* out) {
        if (j.contains(key) && j[key].is_number()) {
          *out = j[key].get<float>();
        }
      };
      take_int("beam_size", &beam_size_);
      take_float("length_penalty", &length_penalty_);
      take_float("repetition_penalty", &repetition_penalty_);
      take_int("sampling_topk", &sampling_topk_);
      take_float("sampling_temperature", &sampling_temperature_);
      take_int("no_repeat_ngram_size", &no_repeat_ngram_size_);
      take_int("max_tokens", &max_tokens_);
      if (j.contains("compute_type") && j["compute_type"].is_string()) {
        compute_type_ = j["compute_type"].get<std::string>();
      }
    } catch (const std::exception& e) {
      TraceInitException(e, model_path, device_);
    }
  }

  // schema 显式配置的值优先级最高（optional 为空即表示「没配」）
  if (overrides.beam_size) beam_size_ = *overrides.beam_size;
  if (overrides.length_penalty) length_penalty_ = *overrides.length_penalty;
  if (overrides.repetition_penalty)
    repetition_penalty_ = *overrides.repetition_penalty;
  if (overrides.sampling_topk) sampling_topk_ = *overrides.sampling_topk;
  if (overrides.sampling_temperature)
    sampling_temperature_ = *overrides.sampling_temperature;
  if (overrides.no_repeat_ngram_size)
    no_repeat_ngram_size_ = *overrides.no_repeat_ngram_size;
  if (overrides.compute_type) compute_type_ = *overrides.compute_type;

  LOG(INFO) << "CT2Backend decode params: beam=" << beam_size_
            << " length_penalty=" << length_penalty_
            << " repetition_penalty=" << repetition_penalty_
            << " sampling_topk=" << sampling_topk_
            << " sampling_temperature=" << sampling_temperature_
            << " no_repeat_ngram=" << no_repeat_ngram_size_
            << " max_tokens=" << max_tokens_
            << " compute_type=" << compute_type_;
}

bool CT2Backend::VerifyTokenizerSelfTest() {
  // 声明的类型对了不代表编码得出来，只能靠实跑发现。探针覆盖两条路径——
  // 汉字（走 added tokens）与拼音（走 byte-level BPE）。
  static const char* kProbe = "今天天气<pinyin_start>zenmeyang</pinyin_start>";

  std::vector<std::string> probe_tokens = tokenizer_->Encode(kProbe);
  for (const auto& token : probe_tokens) {
    if (token == unk_token_) {
      LOG(ERROR) << "CT2Backend: tokenizer self-test produced an unknown token";
      return false;
    }
  }

  // 光比对字符串不够：词表与模型错配时，分词可能一个 unk 都不出（两边的词
  // 恰好都对得上），直到 CT2 按 ID 查表才炸。跑一次最小推理，把这类错配也
  // 拦在初始化——否则每敲一个键都要白付一次注定失败的推理。
  // 解码长度要用真实值：词表与模型错配时，越界的 ID 往往要到后面几步才吐出来，
  // 截得太短会漏。
  try {
    ctranslate2::TranslationOptions options;
    options.beam_size = 1;
    options.max_decoding_length = max_tokens_;
    translator_->translate_batch({probe_tokens}, options);
  } catch (const std::exception& e) {
    TraceInitException(e, "", device_);
    return false;
  }
  return true;
}

}  // namespace predict
}  // namespace rime
