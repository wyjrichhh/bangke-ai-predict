// Copyright RIME Developers
// Distributed under the BSD License
//
// Abstract inference backend for librime-ai-predict (CT2 now, llama.cpp later).

#ifndef RIME_PREDICT_INFERENCE_BACKEND_H_
#define RIME_PREDICT_INFERENCE_BACKEND_H_

#include <memory>
#include <optional>
#include <string>

namespace rime {
namespace predict {

// 解码参数。每项都是 optional，因为这里的实例只承载「调用方显式指定」的值：
// 未设 = nullopt，交给后端用模型包自带的 generation_config.json 或编译期默认值。
// 若用具体值 + 默认初值，schema 就再也无法表达「没配」，模型包的参数必被覆盖。
struct DecodeParams {
  std::optional<int> beam_size;
  std::optional<float> length_penalty;
  std::optional<float> repetition_penalty;
  std::optional<int> sampling_topk;
  std::optional<float> sampling_temperature;
  std::optional<int> no_repeat_ngram_size;
  std::optional<std::string> compute_type;
};

struct InferenceBackendConfig {
  std::string model_path;
  std::string device = "cpu";
  DecodeParams decode;
};

class InferenceBackend {
 public:
  virtual ~InferenceBackend() = default;

  virtual bool Initialize(const InferenceBackendConfig& config) = 0;
  virtual void Shutdown() = 0;
  virtual bool IsInitialized() const = 0;

  /// Run seq2seq inference. Returns nullopt on failure or empty output.
  /// `max_tokens` 为 nullopt 表示调用方没有配置，用后端自己解析出的上限。
  virtual std::optional<std::string> Predict(
      const std::string& prompt, std::optional<int> max_tokens) = 0;
};

}  // namespace predict
}  // namespace rime

#endif  // RIME_PREDICT_INFERENCE_BACKEND_H_
