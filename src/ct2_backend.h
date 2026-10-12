// CTranslate2 implementation of InferenceBackend (seq2seq / BART-style).
//
// Copyright RIME Developers
// Distributed under the BSD License

#ifndef RIME_PREDICT_CT2_BACKEND_H_
#define RIME_PREDICT_CT2_BACKEND_H_

#include "inference_backend.h"
#include "tokenizer.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <ctranslate2/translator.h>
#include <nlohmann/json.hpp>

namespace rime {
namespace predict {

// 解码参数的编译期兜底值。运行时按「这里 ← 模型包 generation_config.json ←
// schema ai_predict/*」逐级覆盖，见 ResolveDecodeParams。
constexpr int kCt2DefaultMaxTokens = 64;
constexpr int kCt2DefaultBeamSize = 1;
constexpr float kCt2DefaultLengthPenalty = 1.0f;
constexpr float kCt2DefaultRepetitionPenalty = 1.2f;
constexpr int kCt2DefaultSamplingTopk = 1;
constexpr float kCt2DefaultSamplingTemperature = 1.0f;
constexpr int kCt2DefaultNoRepeatNgramSize = 0;
constexpr int kCt2DefaultNumHypotheses = 1;
constexpr int64_t kCt2DefaultTimeoutMs = 5000;
constexpr int kCt2DefaultIntraThreads = 4;
constexpr const char* kCt2DefaultComputeType = "default";

class CT2Backend : public InferenceBackend {
 public:
  CT2Backend() = default;
  ~CT2Backend() override;

  /// 进程级共享实例，按配置指纹换代：模型路径、device 或任一解码参数变了就建
  /// 新的顶上，旧的交给后台线程拆除（见析构）。指纹相同则直接复用，所以每次
  /// 保存设置都走一遍这里也不会白白重载模型。
  ///
  /// 新配置初始化失败时返回 nullptr，**不动**现有实例——换了个坏模型不该把还能
  /// 用的那个也赔进去；AI 预测整体停用，日志有明确原因。
  static std::shared_ptr<CT2Backend> Shared(
      const InferenceBackendConfig& config);

  bool Initialize(const InferenceBackendConfig& config) override;
  void Shutdown() override;
  bool IsInitialized() const override {
    return initialized_.load(std::memory_order_acquire);
  }

  std::optional<std::string> Predict(const std::string& prompt,
                                     std::optional<int> max_tokens) override;

  /// 诊断用：只编码、不出结果，供离线工具把 token 序列与参考实现逐条比对。
  /// 只看最终输出会漏掉「分词错了但碰巧拼对」的情况。
  std::vector<std::string> DebugTokens(const std::string& text) const {
    return tokenizer_ ? tokenizer_->Encode(text) : std::vector<std::string>{};
  }

 private:
  bool LoadSharedVocabulary(const std::string& path);

  /// 按「编译期默认 ← 模型包 generation_config.json ← config.decode」求出最终
  /// 解码参数。模型包不存在该文件时静默跳过，兼容旧模型。
  void ResolveDecodeParams(const std::string& model_path,
                           const DecodeParams& overrides);

  /// 分词器自检：用固定串编码一遍，产出 unk 即判定分词器与模型不匹配。
  /// 光看 tokenizer.json 声明的管线不够——声明相符而实际编码不出词的模型同样要拦。
  bool VerifyTokenizerSelfTest();

  std::unique_ptr<ctranslate2::Translator> translator_;
  /// 分词行为全在 tokenizer 里（按 tokenizer.json 的管线选实现），本类只转发。
  std::unique_ptr<Tokenizer> tokenizer_;
  std::atomic<bool> initialized_{false};
  std::mutex mutex_;

  std::unordered_map<std::string, int> token_to_id_;
  std::vector<std::string> id_to_token_;
  std::string unk_token_;

  // 解析后的解码参数（进程级单例，只在首次 Initialize 求值一次）
  int beam_size_ = kCt2DefaultBeamSize;
  float length_penalty_ = kCt2DefaultLengthPenalty;
  float repetition_penalty_ = kCt2DefaultRepetitionPenalty;
  int sampling_topk_ = kCt2DefaultSamplingTopk;
  float sampling_temperature_ = kCt2DefaultSamplingTemperature;
  int no_repeat_ngram_size_ = kCt2DefaultNoRepeatNgramSize;
  int max_tokens_ = kCt2DefaultMaxTokens;
  std::string compute_type_ = kCt2DefaultComputeType;
};

}  // namespace predict
}  // namespace rime

#endif  // RIME_PREDICT_CT2_BACKEND_H_
