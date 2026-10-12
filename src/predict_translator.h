// Predict translator: CT2-backed async candidates (librime-ai-predict plugin).
//
// Copyright RIME Developers
// Distributed under the BSD License

#ifndef RIME_PREDICT_PREDICT_TRANSLATOR_H_
#define RIME_PREDICT_PREDICT_TRANSLATOR_H_

#include "context_builder.h"
#include "prediction_engine.h"

#include <rime/translator.h>

#include <filesystem>
#include <memory>

namespace rime {
namespace predict {

class PredictTranslator : public Translator {
 public:
  explicit PredictTranslator(const Ticket& ticket);
  ~PredictTranslator() override;

  an<Translation> Query(const string& input, const Segment& segment) override;

 private:
  bool EnsureEngine();
  std::filesystem::path ResolveModelPath() const;
  void LoadOptions();

  ContextBuilderOptions ctx_opt_;
  PredictionEngineOptions engine_opt_;
  std::string model_path_raw_;
  std::string device_;
  /// schema 里显式写了的解码参数；未写的项留空，让模型包的值生效
  DecodeParams decode_;
  /// Kill switch: false makes the translator inert without touching the
  /// schema's translators/filters lists (see ai_predict/enabled).
  bool enabled_ = true;
  /// Min Hanzi count for a prediction to be surfaced. Single-Hanzi homophones
  /// are the schema dictionary's strength and the model's weakness (no frequency
  /// signal), so we let the dictionary take those and only surface 2+ Hanzi.
  int min_hanzi_ = 2;
  /// Strip the punctuation the model emitted before the candidate menu sees it.
  /// Off keeps the model's own sentence punctuation ("，我们"), at the cost of
  /// letting its residue through again (see ExtractDisplayText).
  bool filter_punctuation_ = true;

  std::unique_ptr<PredictionEngine> prediction_;
  bool init_attempted_ = false;
  bool init_ok_ = false;
  /// 上次尝试初始化的模型目录。与当前配置不一致时说明换了模型，两个 latch
  /// 都要归零，否则旧引擎会一直被复用、失败状态也解不开。
  std::string engine_path_;
};

}  // namespace predict
}  // namespace rime

#endif  // RIME_PREDICT_PREDICT_TRANSLATOR_H_
