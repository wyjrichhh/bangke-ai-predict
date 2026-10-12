// ct2_backend.cc — InferenceBackend implementation: CTranslate2 + 分词器。
// 分词本身在 tokenizer.{h,cc}（按 tokenizer.json 的管线选实现），本文件只转发。
//
// Copyright RIME Developers
// Distributed under the BSD License

#include "ct2_backend.h"

#include <glog/logging.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <fstream>
#include <optional>
#include <thread>

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
  if (!translator_) {
    return;
  }
  // 就地销毁 ctranslate2::Translator 要热拆线程池，而池线程退出时的 TLS 析构会
  // 反向 join 同一个池（环形互等）；若这发生在 librime 销毁会话的锁下——deploy
  // 就是——输入法会永久卡死。所以挪到独立线程去拆：最坏只赔一个线程，不卡界面。
  auto* pending =
      new std::unique_ptr<ctranslate2::Translator>(std::move(translator_));
  LOG(INFO) << "CT2Backend: releasing old model on a background thread";
  std::thread([pending]() {
    delete pending;
    LOG(INFO) << "CT2Backend: old model released";
  }).detach();
}

std::shared_ptr<CT2Backend> CT2Backend::Shared(
    const InferenceBackendConfig& config) {
  static std::mutex inst_mutex;
  static std::shared_ptr<CT2Backend> instance;
  static std::string instance_key;

  const std::string key = ConfigKey(config);

  std::shared_ptr<CT2Backend> result;
  std::shared_ptr<CT2Backend> retired;
  {
    std::lock_guard<std::mutex> lock(inst_mutex);
    if (instance && instance_key == key) {
      return instance;
    }

    auto backend = std::make_shared<CT2Backend>();
    if (!backend->Initialize(config)) {
      // 保留现有实例：新模型坏掉不该连带把能用的旧模型也弄没。
      return nullptr;
    }
    retired = std::move(instance);
    instance = std::move(backend);
    instance_key = key;
    result = instance;
  }

  // 出锁再放：旧实例的拆除（后台线程）不该压在换代锁上。
  retired.reset();
  return result;
}

bool CT2Backend::Initialize(const InferenceBackendConfig& config) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (initialized_.load(std::memory_order_acquire)) {
    LOG(WARNING) << "CT2Backend already initialized";
    return false;
  }

  const std::string& model_path = config.model_path;
  const std::string& device = config.device;

  LOG(INFO) << "CT2Backend initialize model_path=" << model_path
            << " device=" << device;

  try {
    // 必须在建 Translator 之前：compute_type 是构造参数
    ResolveDecodeParams(model_path, config.decode);

    ctranslate2::Device ct2_device = ctranslate2::Device::CPU;
    if (device.find("cuda") != std::string::npos ||
        device.find("CUDA") != std::string::npos) {
      ct2_device = ctranslate2::Device::CUDA;
      LOG(INFO) << "CT2Backend using CUDA";
    }

    ctranslate2::ComputeType compute_type =
        ctranslate2::str_to_compute_type(compute_type_);

    ctranslate2::ReplicaPoolConfig pool_config;
    pool_config.num_threads_per_replica =
        static_cast<size_t>(kCt2DefaultIntraThreads);

    translator_ = std::make_unique<ctranslate2::Translator>(
        model_path, ct2_device, compute_type, std::vector<int>{0}, false,
        pool_config);

    if (!translator_) {
      LOG(ERROR) << "CT2Backend failed to create translator";
      return false;
    }

    const std::string vocab_path = model_path + "/shared_vocabulary.json";
    if (!LoadSharedVocabulary(vocab_path)) {
      LOG(ERROR) << "CT2Backend failed to load shared_vocabulary.json";
      translator_.reset();
      return false;
    }

    unk_token_ = UnkTokenForVocab(token_to_id_);

    std::string tokenizer_error;
    tokenizer_ = Tokenizer::Create(model_path + "/tokenizer.json", token_to_id_,
                                   unk_token_, &tokenizer_error);
    if (!tokenizer_) {
      LOG(ERROR) << "CT2Backend: " << tokenizer_error
                 << "; AI prediction stays off instead of emitting [UNK].";
      translator_.reset();
      return false;
    }

    if (!VerifyTokenizerSelfTest()) {
      translator_.reset();
      return false;
    }

    LOG(INFO) << "CT2Backend tokenizer: kind=" << tokenizer_->Kind()
              << " vocab=" << token_to_id_.size();

    initialized_.store(true, std::memory_order_release);
    return true;

  } catch (const std::exception& e) {
    LOG(ERROR) << "CT2Backend initialize: " << e.what();
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

    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - start_time)
                           .count();
    DLOG(INFO) << "CT2Backend::Predict " << result.size() << " chars in "
               << duration_ms << "ms";

    return result;

  } catch (const std::exception& e) {
    LOG(ERROR) << "CT2Backend::Predict: " << e.what();
    return std::nullopt;
  }
}

bool CT2Backend::LoadSharedVocabulary(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    LOG(ERROR) << "CT2Backend: cannot open " << path;
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
    LOG(ERROR) << "CT2Backend: parse shared_vocabulary.json: " << e.what();
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
  std::ifstream file(path);
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
      LOG(WARNING) << "CT2Backend: parse generation_config.json: " << e.what();
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
      LOG(ERROR) << "CT2Backend: tokenizer self-test failed on \"" << kProbe
                 << "\" (produced " << unk_token_
                 << "); the model directory's tokenizer does not match this"
                 << " build. AI prediction stays off.";
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
    LOG(ERROR) << "CT2Backend: warmup inference failed (" << e.what()
               << "); the model directory is inconsistent — its vocabulary and"
               << " model.bin do not match. AI prediction stays off.";
    return false;
  }
  return true;
}

}  // namespace predict
}  // namespace rime
