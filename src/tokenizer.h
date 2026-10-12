// tokenizer.json 管线的策略实现。
//
// 分词行为由四段共同决定（normalizer / pre_tokenizer / model / decoder），
// 不是只看 model.type：同样是 BPE，配 ByteLevel 前缀与配 Metaspace 前缀，输出
// 就不同。所以按四段选实现，认不出的组合明确拒绝，而不是猜一个近似的。
//
// special / added token 的切分各类型通用，留在基类；子类只负责
// 「一段普通文本 ↔ 若干 token」这一件事。
//
// Copyright RIME Developers
// Distributed under the BSD License

#ifndef RIME_PREDICT_TOKENIZER_H_
#define RIME_PREDICT_TOKENIZER_H_

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rime {
namespace predict {

/// Split 用的字符类，只覆盖 `[...]` 这一种形态（含范围 `a-z`、取反 `^`、
/// 反斜杠转义）。刻意不做通用正则引擎：Split 之外的正则形态由 Create 明确
/// 拒绝，而不是拿一个近似的匹配器去猜。
class CharClass {
 public:
  static bool Parse(const std::string& pattern, CharClass* out,
                    std::string* reason);

  bool Matches(uint32_t cp) const;

 private:
  std::vector<std::pair<uint32_t, uint32_t>> ranges_;
  bool negated_ = false;
};

class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  /// 编码。调用方不必自己包 <pinyin_start>：没有就补上，与模型约定一致。
  std::vector<std::string> Encode(const std::string& text) const;

  /// 解码。跳过 special token 后交给子类拼回。
  std::string Decode(const std::vector<std::string>& tokens) const;

  /// 按 tokenizer.json 声明选实现。不支持时返回 nullptr，并把原因写进 reason。
  static std::unique_ptr<Tokenizer> Create(
      const std::string& tokenizer_json_path,
      const std::unordered_map<std::string, int>& token_to_id,
      const std::string& unk_token,
      std::string* reason);

  bool has_token(const std::string& token) const {
    return token_to_id_.count(token) > 0;
  }

  /// 诊断用：这一份实现的名字，写进日志好判断实际加载的是哪种分词器。
  virtual const char* Kind() const = 0;

 protected:
  /// 把一段（不含 special / added token）编成 token。
  virtual std::vector<std::string> EncodePart(const std::string& part) const = 0;

  /// 把 token 拼回文本。收到的是已剔除 special token 的序列。
  virtual std::string DecodeParts(const std::vector<std::string>& tokens) const = 0;

  std::unordered_map<std::string, int> token_to_id_;
  std::string unk_token_;
  std::vector<std::string> special_tokens_sorted_;
  std::vector<std::string> added_tokens_sorted_;

  /// 按长度降序排，供最长匹配扫描用。
  static void SortByLengthDesc(std::vector<std::string>* tokens);

  static size_t Utf8CharLength(unsigned char c);
};

}  // namespace predict
}  // namespace rime

#endif  // RIME_PREDICT_TOKENIZER_H_
