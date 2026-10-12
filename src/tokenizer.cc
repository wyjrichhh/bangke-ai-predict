// Copyright RIME Developers
// Distributed under the BSD License

#include "tokenizer.h"

#include <glog/logging.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <fstream>

#include <nlohmann/json.hpp>

namespace rime {
namespace predict {

namespace {

size_t Utf8Length(unsigned char c) {
  if ((c & 0xF8) == 0xF0) return 4;
  if ((c & 0xF0) == 0xE0) return 3;
  if ((c & 0xE0) == 0xC0) return 2;
  return 1;
}

// 解出一个码点；越界或截断时按单字节处理，不抛异常。
size_t DecodeUtf8(const std::string& s, size_t pos, uint32_t* cp) {
  const unsigned char c = static_cast<unsigned char>(s[pos]);
  const size_t len = Utf8Length(c);
  if (len == 1 || pos + len > s.size()) {
    *cp = c;
    return 1;
  }
  uint32_t v = 0;
  switch (len) {
    case 2:
      v = (c & 0x1F) << 6;
      break;
    case 3:
      v = (c & 0x0F) << 12;
      break;
    default:
      v = (c & 0x07) << 18;
      break;
  }
  for (size_t i = 1; i < len; ++i) {
    v |= (static_cast<unsigned char>(s[pos + i]) & 0x3F) << (6 * (len - 1 - i));
  }
  *cp = v;
  return len;
}

std::string EncodeUtf8(uint32_t cp) {
  std::string r;
  if (cp < 0x80) {
    r += static_cast<char>(cp);
  } else if (cp < 0x800) {
    r += static_cast<char>(0xC0 | (cp >> 6));
    r += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    r += static_cast<char>(0xE0 | (cp >> 12));
    r += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    r += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    r += static_cast<char>(0xF0 | (cp >> 18));
    r += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    r += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    r += static_cast<char>(0x80 | (cp & 0x3F));
  }
  return r;
}

std::vector<std::string> SplitOnSpecialTokens(
    const std::string& text, const std::vector<std::string>& specials,
    std::vector<bool>* is_special) {
  std::vector<std::string> segments;
  std::string remaining = text;

  while (!remaining.empty()) {
    size_t earliest_pos = std::string::npos;
    std::string earliest_token;

    for (const auto& st : specials) {
      const size_t pos = remaining.find(st);
      if (pos != std::string::npos &&
          (pos < earliest_pos ||
           (pos == earliest_pos && st.length() > earliest_token.length()))) {
        earliest_pos = pos;
        earliest_token = st;
      }
    }

    if (earliest_pos == std::string::npos) {
      segments.push_back(remaining);
      is_special->push_back(false);
      break;
    }

    if (earliest_pos > 0) {
      segments.push_back(remaining.substr(0, earliest_pos));
      is_special->push_back(false);
    }

    segments.push_back(earliest_token);
    is_special->push_back(true);

    remaining = remaining.substr(earliest_pos + earliest_token.length());
  }

  return segments;
}

// 与 SplitOnSpecialTokens 同构，但切出的是 added token（非 special）。
std::vector<std::string> SplitOnAddedTokens(
    const std::string& text, const std::vector<std::string>& added,
    std::vector<bool>* is_added) {
  std::vector<std::string> parts;
  size_t pos = 0;

  while (pos < text.size()) {
    std::string best_match;
    for (const auto& token : added) {
      if (token.length() <= text.size() - pos &&
          text.compare(pos, token.length(), token) == 0) {
        best_match = token;
        break;
      }
    }

    if (!best_match.empty()) {
      parts.push_back(best_match);
      is_added->push_back(true);
      pos += best_match.length();
    } else {
      const size_t char_len = Utf8Length(static_cast<unsigned char>(text[pos]));
      if (pos + char_len > text.size()) break;

      if (!parts.empty() && !is_added->back()) {
        parts.back() += text.substr(pos, char_len);
      } else {
        parts.push_back(text.substr(pos, char_len));
        is_added->push_back(false);
      }
      pos += char_len;
    }
  }

  return parts;
}

}  // namespace

// ------------------------------------------------------------ 字符类 --

bool CharClass::Parse(const std::string& pattern, CharClass* out,
                      std::string* reason) {
  if (pattern.size() < 2 || pattern.front() != '[' || pattern.back() != ']') {
    *reason = "unsupported Split pattern (expected a single [...] character class)";
    return false;
  }

  size_t i = 1;
  const size_t end = pattern.size() - 1;
  if (i < end && pattern[i] == '^') {
    out->negated_ = true;
    ++i;
  }

  // 先解成「码点 + 是否来自转义」，再合并成区间——转义过的 '-' 是字面量。
  std::vector<uint32_t> items;
  std::vector<bool> escaped;
  while (i < end) {
    const bool esc = (pattern[i] == '\\' && i + 1 < end);
    const size_t at = esc ? i + 1 : i;
    uint32_t cp = 0;
    const size_t len = DecodeUtf8(pattern, at, &cp);
    items.push_back(cp);
    escaped.push_back(esc);
    i = at + len;
  }

  size_t k = 0;
  while (k < items.size()) {
    if (k + 2 < items.size() && !escaped[k + 1] && items[k + 1] == '-') {
      out->ranges_.emplace_back(items[k], items[k + 2]);
      k += 3;
    } else {
      out->ranges_.emplace_back(items[k], items[k]);
      ++k;
    }
  }
  return true;
}

bool CharClass::Matches(uint32_t cp) const {
  bool in = false;
  for (const auto& r : ranges_) {
    if (cp >= r.first && cp <= r.second) {
      in = true;
      break;
    }
  }
  return negated_ ? !in : in;
}

// ------------------------------------------------------------------ 基类 --

void Tokenizer::SortByLengthDesc(std::vector<std::string>* tokens) {
  std::sort(tokens->begin(), tokens->end(),
            [](const std::string& a, const std::string& b) {
              return a.length() > b.length();
            });
}

size_t Tokenizer::Utf8CharLength(unsigned char c) { return Utf8Length(c); }

std::vector<std::string> Tokenizer::Encode(const std::string& text) const {
  std::string input_text = text;
  if (input_text.find("<pinyin_start>") == std::string::npos) {
    input_text = "<pinyin_start>" + input_text + "</pinyin_start>";
  }

  std::vector<std::string> tokens;
  std::vector<bool> is_special;
  const auto segments =
      SplitOnSpecialTokens(input_text, special_tokens_sorted_, &is_special);

  for (size_t i = 0; i < segments.size(); ++i) {
    if (is_special[i]) {
      tokens.push_back(segments[i]);
      continue;
    }
    std::vector<bool> part_is_added;
    const auto parts =
        SplitOnAddedTokens(segments[i], added_tokens_sorted_, &part_is_added);
    for (size_t k = 0; k < parts.size(); ++k) {
      if (part_is_added[k]) {
        tokens.push_back(parts[k]);
      } else {
        const auto sub = EncodePart(parts[k]);
        tokens.insert(tokens.end(), sub.begin(), sub.end());
      }
    }
  }

  return tokens;
}

std::string Tokenizer::Decode(const std::vector<std::string>& tokens) const {
  static const char* kSkip[] = {
      "<hanzi_start>", "</hanzi_start>", "<s>",        "</s>",
      "<pad>",         "<|endoftext|>",   "<|im_start|>", "<|im_end|>",
      "<|PAD|>",       "<|EOS|>",         "<|BOS|>",   "[CLS]",
      "[SEP]"};

  std::vector<std::string> kept;
  for (const auto& token : tokens) {
    bool skip = false;
    for (const char* s : kSkip) {
      if (token == s) {
        skip = true;
        break;
      }
    }
    if (!skip) kept.push_back(token);
  }
  return DecodeParts(kept);
}

// ------------------------------------------------------- ByteLevel BPE --

namespace {

// GPT-2 bytes_to_unicode：直接映射的字节区间是固定的，其余按序挪到 256+。
std::unordered_map<uint8_t, uint32_t> BuildByteToUnicode() {
  std::vector<int> direct;
  for (int b = 33; b <= 126; ++b) direct.push_back(b);
  for (int b = 161; b <= 172; ++b) direct.push_back(b);
  for (int b = 174; b <= 255; ++b) direct.push_back(b);

  std::unordered_map<uint8_t, uint32_t> map;
  for (int b : direct) map[static_cast<uint8_t>(b)] = static_cast<uint32_t>(b);

  const std::set<int> direct_set(direct.begin(), direct.end());
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    if (direct_set.find(b) == direct_set.end()) {
      map[static_cast<uint8_t>(b)] = 256 + static_cast<uint32_t>(n++);
    }
  }
  return map;
}

class ByteLevelBpeTokenizer : public Tokenizer {
 public:
  explicit ByteLevelBpeTokenizer(
      std::vector<std::pair<std::string, std::string>> merges)
      : merges_(std::move(merges)), byte_to_unicode_(BuildByteToUnicode()) {
    for (const auto& m : merges_) {
      merge_ranks_[m.first + " " + m.second] = static_cast<int>(merge_ranks_.size());
    }
    for (const auto& kv : byte_to_unicode_) {
      unicode_to_byte_[kv.second] = kv.first;
    }
  }

  const char* Kind() const override { return "byte-level-bpe"; }

 protected:
  std::vector<std::string> EncodePart(const std::string& part) const override {
    if (part.empty()) return {};

    std::string encoded;
    for (unsigned char byte_val : part) {
      auto it = byte_to_unicode_.find(byte_val);
      encoded += (it != byte_to_unicode_.end()) ? EncodeUtf8(it->second)
                                                : std::string(1, static_cast<char>(byte_val));
    }

    std::vector<std::string> symbols;
    size_t pos = 0;
    while (pos < encoded.size()) {
      const size_t char_len = Utf8Length(static_cast<unsigned char>(encoded[pos]));
      if (pos + char_len > encoded.size()) break;
      symbols.push_back(encoded.substr(pos, char_len));
      pos += char_len;
    }

    if (symbols.size() <= 1) {
      if (!symbols.empty() && token_to_id_.count(symbols[0])) return symbols;
      return {unk_token_};
    }

    while (symbols.size() > 1) {
      int best_rank = INT_MAX;
      int best_idx = -1;
      for (size_t i = 0; i + 1 < symbols.size(); ++i) {
        auto it = merge_ranks_.find(symbols[i] + " " + symbols[i + 1]);
        if (it != merge_ranks_.end() && it->second < best_rank) {
          best_rank = it->second;
          best_idx = static_cast<int>(i);
        }
      }
      if (best_idx < 0) break;

      symbols[static_cast<size_t>(best_idx)] +=
          symbols[static_cast<size_t>(best_idx) + 1];
      symbols.erase(symbols.begin() + best_idx + 1);
    }

    std::vector<std::string> result;
    for (const auto& sym : symbols) {
      result.push_back(token_to_id_.count(sym) ? sym : unk_token_);
    }
    return result;
  }

  std::string DecodeParts(const std::vector<std::string>& tokens) const override {
    std::string raw;
    for (const auto& t : tokens) raw += t;

    std::vector<uint8_t> bytes;
    size_t pos = 0;
    while (pos < raw.size()) {
      uint32_t cp = 0;
      const size_t len = DecodeUtf8(raw, pos, &cp);
      auto it = unicode_to_byte_.find(cp);
      if (it != unicode_to_byte_.end()) {
        bytes.push_back(it->second);
      } else {
        for (size_t i = 0; i < len; ++i) {
          bytes.push_back(static_cast<uint8_t>(raw[pos + i]));
        }
      }
      pos += len;
    }

    std::string decoded(bytes.begin(), bytes.end());
    decoded.erase(std::remove(decoded.begin(), decoded.end(), ' '), decoded.end());
    return decoded;
  }

 private:
  std::vector<std::pair<std::string, std::string>> merges_;
  std::unordered_map<std::string, int> merge_ranks_;
  std::unordered_map<uint8_t, uint32_t> byte_to_unicode_;
  std::unordered_map<uint32_t, uint8_t> unicode_to_byte_;
};

// -------------------------------------------------------------- Unigram --

constexpr float kUnkCharScore = -100.0f;

class UnigramTokenizer : public Tokenizer {
 public:
  UnigramTokenizer(std::unordered_map<std::string, float> scores,
                   const CharClass& split, bool strip_whitespace)
      : scores_(std::move(scores)),
        split_(split),
        strip_whitespace_(strip_whitespace) {
    for (const auto& kv : scores_) {
      size_t chars = 0;
      for (size_t i = 0; i < kv.first.size();) {
        i += Utf8Length(static_cast<unsigned char>(kv.first[i]));
        ++chars;
      }
      max_chars_ = std::max(max_chars_, chars);
    }
  }

  const char* Kind() const override { return "unigram"; }

 protected:
  std::vector<std::string> EncodePart(const std::string& part) const override {
    std::string text = part;
    if (strip_whitespace_) {
      text.erase(std::remove_if(text.begin(), text.end(),
                                [](char c) {
                                  return c == ' ' || c == '\t' || c == '\n' ||
                                         c == '\r' || c == '\f' || c == '\v';
                                }),
                 text.end());
    }
    if (text.empty()) return {};

    std::vector<std::string> tokens;
    for (const auto& piece : SplitPieces(text)) {
      const auto sub = EncodePiece(piece);
      tokens.insert(tokens.end(), sub.begin(), sub.end());
    }
    return tokens;
  }

  std::string DecodeParts(const std::vector<std::string>& tokens) const override {
    std::string out;
    for (const auto& t : tokens) out += t;
    return out;
  }

 private:
  /// Split(behavior="Isolated")：命中字符各自成段，其余按连续非命中聚成一段。
  std::vector<std::string> SplitPieces(const std::string& text) const {
    std::vector<std::string> pieces;
    std::string run;
    size_t pos = 0;
    while (pos < text.size()) {
      uint32_t cp = 0;
      const size_t len = DecodeUtf8(text, pos, &cp);
      if (split_.Matches(cp)) {
        if (!run.empty()) {
          pieces.push_back(run);
          run.clear();
        }
        pieces.push_back(text.substr(pos, len));
      } else {
        run += text.substr(pos, len);
      }
      pos += len;
    }
    if (!run.empty()) pieces.push_back(run);
    return pieces;
  }

  /// Viterbi：在 vocab 里求总得分最大的切分。逐字符推进，不切在码点中间。
  std::vector<std::string> EncodePiece(const std::string& piece) const {
    const size_t n = piece.size();
    std::vector<float> best(n + 1, -1e30f);
    std::vector<size_t> prev(n + 1, 0);
    std::vector<std::string> chosen(n + 1);
    best[0] = 0.0f;

    size_t pos = 0;
    while (pos < n) {
      if (best[pos] > -1e29f) {
        bool matched = false;
        size_t p = pos;
        for (size_t chars = 1; chars <= max_chars_ && p < n; ++chars) {
          p += Utf8Length(static_cast<unsigned char>(piece[p]));
          if (p > n) break;
          const std::string cand = piece.substr(pos, p - pos);
          auto it = scores_.find(cand);
          if (it == scores_.end()) continue;
          matched = true;
          if (best[pos] + it->second > best[p]) {
            best[p] = best[pos] + it->second;
            prev[p] = pos;
            chosen[p] = cand;
          }
        }
        if (!matched) {
          // 词表里没有任何以此处开头的前缀：吃掉一个字符记 unk，保证路径能走完。
          size_t cl = Utf8Length(static_cast<unsigned char>(piece[pos]));
          if (pos + cl > n) cl = 1;
          const size_t next = pos + cl;
          if (best[pos] + kUnkCharScore > best[next]) {
            best[next] = best[pos] + kUnkCharScore;
            prev[next] = pos;
            chosen[next] = unk_token_;
          }
        }
      }
      pos += Utf8Length(static_cast<unsigned char>(piece[pos]));
    }

    if (best[n] <= -1e29f) return {unk_token_};

    std::vector<std::string> out;
    for (size_t at = n; at > 0; at = prev[at]) out.push_back(chosen[at]);
    std::reverse(out.begin(), out.end());
    return out;
  }

  std::unordered_map<std::string, float> scores_;
  CharClass split_;
  bool strip_whitespace_ = true;
  size_t max_chars_ = 1;
};

}  // namespace

// ------------------------------------------------------------- 工厂 --

std::unique_ptr<Tokenizer> Tokenizer::Create(
    const std::string& tokenizer_json_path,
    const std::unordered_map<std::string, int>& token_to_id,
    const std::string& unk_token, std::string* reason) {
  std::ifstream file(tokenizer_json_path);
  if (!file.is_open()) {
    *reason = "cannot open " + tokenizer_json_path;
    return nullptr;
  }

  nlohmann::json j;
  try {
    file >> j;
  } catch (const std::exception& e) {
    *reason = std::string("parse tokenizer.json: ") + e.what();
    return nullptr;
  }

  const std::string model_type =
      (j.contains("model") && j["model"].is_object())
          ? j["model"].value("type", "")
          : "";
  const std::string pre_type =
      (j.contains("pre_tokenizer") && j["pre_tokenizer"].is_object())
          ? j["pre_tokenizer"].value("type", "")
          : "";
  const std::string decoder_type =
      (j.contains("decoder") && j["decoder"].is_object())
          ? j["decoder"].value("type", "")
          : "";

  std::unique_ptr<Tokenizer> tok;

  if (model_type == "BPE" && pre_type == "ByteLevel" &&
      decoder_type == "ByteLevel") {
    std::vector<std::pair<std::string, std::string>> merges;
    const nlohmann::json* mp = nullptr;
    if (j.contains("model") && j["model"].is_object() &&
        j["model"].contains("merges")) {
      mp = &j["model"]["merges"];
    } else if (j.contains("merges")) {
      mp = &j["merges"];
    }
    if (mp && mp->is_array()) {
      for (const auto& item : *mp) {
        if (item.is_string()) {
          const std::string s = item.get<std::string>();
          const size_t sp = s.find(' ');
          if (sp != std::string::npos) {
            merges.push_back({s.substr(0, sp), s.substr(sp + 1)});
          }
        } else if (item.is_array() && item.size() >= 2 && item[0].is_string() &&
                   item[1].is_string()) {
          merges.push_back({item[0].get<std::string>(),
                            item[1].get<std::string>()});
        }
      }
    }
    tok = std::make_unique<ByteLevelBpeTokenizer>(std::move(merges));

  } else if (model_type == "Unigram" && decoder_type == "Fuse") {
    std::unordered_map<std::string, float> scores;
    if (j.contains("model") && j["model"].is_object() &&
        j["model"].contains("vocab") && j["model"]["vocab"].is_array()) {
      for (const auto& entry : j["model"]["vocab"]) {
        if (entry.is_array() && entry.size() >= 2 && entry[0].is_string()) {
          scores[entry[0].get<std::string>()] = entry[1].get<float>();
        }
      }
    }
    if (scores.empty()) {
      *reason = "Unigram model has no vocab";
      return nullptr;
    }

    // normalizer 只认「Sequence[Replace(Regex \\s+ -> \"\")]」这一种；
    // 别的形态要么改变文本语义、要么我们无法忠实复现，一律拒绝而不是近似。
    bool strip_whitespace = false;
    if (j.contains("normalizer") && j["normalizer"].is_object()) {
      const auto& norm = j["normalizer"];
      bool recognized = false;
      if (norm.value("type", "") == "Sequence" && norm.contains("normalizers") &&
          norm["normalizers"].is_array() && norm["normalizers"].size() == 1) {
        const auto& inner = norm["normalizers"][0];
        if (inner.value("type", "") == "Replace") {
          std::string pattern;
          if (inner.contains("pattern") && inner["pattern"].is_object()) {
            pattern = inner["pattern"].value("Regex", "");
          }
          if (pattern == "\\s+" && inner.value("content", "@@") == "") {
            strip_whitespace = true;
            recognized = true;
          }
        }
      }
      if (!recognized) {
        *reason =
            "unsupported normalizer (only Sequence[Replace(\\\\s+ -> \"\")] "
            "is implemented)";
        return nullptr;
      }
    }

    // pre_tokenizer：Sequence[Split(char-class, Isolated)]。字符类之外的一律拒绝。
    CharClass split;
    if (j.contains("pre_tokenizer") && j["pre_tokenizer"].is_object()) {
      const auto& pre = j["pre_tokenizer"];
      const nlohmann::json* split_node = nullptr;
      if (pre.value("type", "") == "Sequence" && pre.contains("pretokenizers") &&
          pre["pretokenizers"].is_array() && pre["pretokenizers"].size() == 1) {
        split_node = &pre["pretokenizers"][0];
      } else {
        split_node = &pre;
      }
      if (split_node->value("type", "") != "Split") {
        *reason = "unsupported pre_tokenizer (only a single Split is implemented)";
        return nullptr;
      }
      if (split_node->value("behavior", "") != "Isolated") {
        *reason = "unsupported Split behavior (only Isolated is implemented)";
        return nullptr;
      }
      if (split_node->value("invert", false)) {
        *reason = "unsupported Split invert=true";
        return nullptr;
      }
      std::string pattern;
      if (split_node->contains("pattern") && (*split_node)["pattern"].is_object()) {
        pattern = (*split_node)["pattern"].value("Regex", "");
      }
      if (!CharClass::Parse(pattern, &split, reason)) return nullptr;
    }

    tok = std::make_unique<UnigramTokenizer>(std::move(scores), split,
                                             strip_whitespace);

  } else {
    *reason = "unsupported tokenizer pipeline (model=" + model_type +
              " pre_tokenizer=" + pre_type + " decoder=" + decoder_type + ")";
    return nullptr;
  }

  Tokenizer* base = tok.get();
  base->token_to_id_ = token_to_id;
  base->unk_token_ = unk_token;

  std::set<std::string> specials;
  std::set<std::string> added;
  if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
    for (const auto& t : j["added_tokens"]) {
      if (!t.is_object()) continue;
      const std::string content = t.value("content", "");
      if (content.empty()) continue;
      (t.value("special", false) ? specials : added).insert(content);
    }
  }
  base->special_tokens_sorted_.assign(specials.begin(), specials.end());
  base->added_tokens_sorted_.assign(added.begin(), added.end());
  SortByLengthDesc(&base->special_tokens_sorted_);
  SortByLengthDesc(&base->added_tokens_sorted_);

  return tok;
}

}  // namespace predict
}  // namespace rime
