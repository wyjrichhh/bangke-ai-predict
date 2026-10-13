// Copyright RIME Developers
// Distributed under the BSD License

#include "context_builder.h"

#include "frontend_protocol.h"

#include <algorithm>
#include <vector>
#include <utf8.h>

#include <rime/commit_history.h>
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

// True iff `s` is a usable pinyin prompt: at least one lowercase letter and
// nothing but [a-z] plus the syllable-separator apostrophe (e.g. "xi'an").
// librime hands us a fresh segment for every keystroke, including punctuation
// (",", ".", "?"), ASCII symbols ("+", "/", "\\", "_"), and literal uppercase
// acronyms ("SSO"). None are pinyin; the CT2 model can only echo them back as
// garbage (window="...日志" + prompt="," -> candidate ","), so we reject them
// before they reach the trigger policy.
//
// `ai_only` widens what counts as input. That mode appends digits and a set of
// punctuation marks to the composition (and to the schema's speller alphabet so
// they stay inside the pinyin segment), handing us prompts like "zenmeyang1" or
// "woxiangqu,beijing"; rejecting those would silently stop the mode predicting
// at all. Ordinary mode keeps the narrow rule, so nothing there changes.
bool IsPinyinPrompt(const string& s, bool ai_only) {
  bool has_letter = false;
  for (char c : s) {
    if (c >= 'a' && c <= 'z') {
      has_letter = true;
    } else if (c >= 'A' && c <= 'Z') {
      // 英文缩写（如 APPLE/IBM）：不因此拒绝，prompt 保留大写交给模型处理。
    } else if (ai_only && (c >= '0' && c <= '9')) {
      // 数字：集合与前端 BangkeSettings.aiOnlyExtraInput 保持一致。
    } else if (ai_only &&
               (c == ',' || c == '.' || c == ';' || c == ':' || c == '!' ||
                c == '?')) {
      // 同上，标点也是该模式追加进 composition 的。
    } else if (c != '\'') {
      return false;
    }
  }
  return has_letter;
}

// True iff `cp` is punctuation/symbol the model may emit but that must never
// appear in a candidate. Matched by Unicode RANGE, not an enumerated denylist:
// the model can emit any CJK punctuation it saw in training, and a hand-listed
// set inevitably misses one (a fullwidth `）` once leaked into `编译）`).
bool IsPunctCodepoint(uint32_t cp) {
  // ASCII punctuation ([a-zA-Z0-9] and apostrophe fall in the gaps).
  if (cp == '.' || cp == ',' || cp == '!' || cp == '?' || cp == ';' ||
      cp == ':')
    return true;
  // CJK Symbols and Punctuation: 。、《》「」『』【】〈〉… etc.
  if (cp >= 0x3000 && cp <= 0x303F)
    return true;
  // Fullwidth ASCII forms: ！（），：；？ etc.
  if ((cp >= 0xFF01 && cp <= 0xFF0F) || (cp >= 0xFF1A && cp <= 0xFF20) ||
      (cp >= 0xFF3B && cp <= 0xFF40) || (cp >= 0xFF5B && cp <= 0xFF65))
    return true;
  // General Punctuation: “ ” ‘ ’ … — – etc.
  if (cp >= 0x2010 && cp <= 0x205E)
    return true;
  return false;
}

/// Keep the last `max_codepoints` codepoints of `text`.
///
/// The frontend reads a deliberately generous window and this is where it gets
/// trimmed, so how much context reaches the model is one schema knob on this
/// side rather than a constant each frontend has to agree on. Trimming by
/// codepoint (not byte) so a multi-byte character is never cut in half.
string TailCodepoints(const string& text, int max_codepoints) {
  if (max_codepoints <= 0) {
    return string();
  }
  auto it = text.end();
  for (int i = 0; i < max_codepoints && it != text.begin(); ++i) {
    utf8::prior(it, text.begin());
  }
  return string(it, text.end());
}

string StripAllPunctuation(const string& text) {
  string result;
  auto it = text.begin();
  while (it != text.end()) {
    auto start = it;
    uint32_t cp = utf8::next(it, text.end());
    if (!IsPunctCodepoint(cp))
      result.append(start, it);
  }
  return result;
}

/// Walk commit history (most recent first, capped at `max_records`) and
/// reconstruct the context window.
///
/// Every record is text that reached the screen, so every record belongs in the
/// window -- including the two types librime files for keys that never entered a
/// composition: "thru" (a key no processor claimed: digits, punctuation,
/// ascii-mode letters) and "raw" (input committed untranslated, e.g. the echo
/// candidate for "T3"). They sit in the same ordered list as Hanzi commits, so
/// concatenating records rebuilds the sentence in order. Dropping them turned
/// "我下午3点去T3航站楼" into "我下午点去航站楼".
///
/// The model takes mixed Hanzi/Latin/digit/punctuation context without replaying
/// it (measured against the shipped model). Residue it does emit is passed
/// through as-is -- IsDisplayableCandidate only asks for Hanzi now, so a stray
/// letter costs a candidate its polish, not its existence. Punctuation is the
/// exception -- ExtractDisplayText keeps it when ai_predict/filter_punctuation
/// is off, which is the point of that knob.
void BuildWindowContext(Context* ctx,
                        int max_records,
                        string* window_text_out) {
  if (!ctx || !window_text_out)
    return;
  const CommitHistory& history = ctx->commit_history();

  std::vector<string> recent_commits;
  size_t collected = 0;
  for (auto it = history.rbegin();
       it != history.rend() && collected < static_cast<size_t>(max_records);
       ++it) {
    if (it->text.empty())
      continue;
    recent_commits.push_back(it->text);
    ++collected;
  }
  std::reverse(recent_commits.begin(), recent_commits.end());

  string& out = *window_text_out;
  out.clear();
  for (const auto& text : recent_commits) {
    // No separator: committed punctuation already supplies sentence structure.
    out += text;
  }
}

}  // namespace

int CountHan(const string& text) {
  int n = 0;
  auto it = text.begin();
  while (it != text.end()) {
    uint32_t cp = utf8::next(it, text.end());
    if (cp >= 0x4E00 && cp <= 0x9FFF)
      ++n;
  }
  return n;
}

std::optional<PredictionContext> ContextBuilder::Build(
    Engine* engine,
    const string& raw_input,
    const ContextBuilderOptions& opt) {
  string prompt = StripSpaces(raw_input);
  if (prompt.empty()) {
    return std::nullopt;
  }
  if (!IsPinyinPrompt(prompt, opt.ai_only)) {
    return std::nullopt;
  }
  // prompt 保留大写（英文缩写如 APPLE），模型能直接处理并回显它。
  int threshold = opt.min_effective_length > 0 ? opt.min_effective_length : 12;

  // window_text has two sources, and the frontend's is strictly better when
  // available: it is what is actually on screen, so edits the engine never saw
  // (backspace on committed text, Cmd+A, paste) are already reflected. It is
  // absent whenever the host app declines to answer -- and an unhandled
  // BackSpace clears commit_history outright, so the fallback is not merely a
  // lesser source, it is the only one left. Empty means "use the fallback",
  // never "the document is empty".
  string window_text;
  string history_window_text;
  bool from_frontend = false;
  if (engine && engine->context()) {
    Context* context = engine->context();
    if (opt.use_frontend_context) {
      window_text = TailCodepoints(
          context->get_property(protocol::kClientContext),
          opt.context_window_chars);
      from_frontend = !window_text.empty();
    }
    if (from_frontend) {
      // Also build the fallback for the log, so the two can be compared
      // offline without a second deploy. Pure string work, no inference.
      BuildWindowContext(context, opt.context_window_size,
                         &history_window_text);
    } else {
      BuildWindowContext(context, opt.context_window_size, &window_text);
    }
  }

  // Context-first trigger: real Chinese context always fires (the committed
  // prefix is our strongest signal); cold start requires prompt >= threshold.
  // "Real context" means at least one Hanzi, not merely non-empty -- see
  // CountHan.
  bool has_context = CountHan(window_text) > 0;
  // Both length gates exist so the model does not guess where the dictionary
  // would answer better: a short fragment alone produces a wild continuation,
  // and the offer is never taken. AI-only mode has no dictionary to defer to,
  // so a gate there buys nothing but silence -- and since the mode is the only
  // thing on screen, silence reads as "broken". Both are skipped.
  if (!opt.ai_only) {
    if (!has_context && static_cast<int>(prompt.length()) < threshold) {
      return std::nullopt;
    }
    // Even with context, a one-letter prompt carries almost no signal: the model
    // just guesses the next character and the offer is never taken (log
    // analysis). Count letters only -- the syllable-separator apostrophe doesn't
    // add signal.
    if (has_context) {
      int letters = std::count_if(prompt.begin(), prompt.end(), [](char c) {
        return c >= 'a' && c <= 'z';
      });
      int min_letters = opt.min_context_prompt_length > 0
                            ? opt.min_context_prompt_length
                            : 2;
      if (letters < min_letters) {
        return std::nullopt;
      }
    }
  }

  PredictionContext ctx;
  ctx.effective_prompt = prompt;
  ctx.window_text = window_text;
  ctx.windowed = has_context;
  ctx.window_from_frontend = from_frontend;
  ctx.history_window_text = history_window_text;
  // Pipe is a safe separator: the prompt is [a-zA-Z'] only, so it never carries
  // one and the split point stays unambiguous no matter how much ASCII the
  // window picks up from thru/raw records.
  ctx.cache_key = window_text + "|" + prompt;
  ctx.ct2_input = window_text + "<pinyin_start>" + prompt + "</pinyin_start>";
  return ctx;
}

/// 去掉「整句中文里夹的那**一个**小写字母」。
///
/// 打错拼音时模型常给出几乎正确的整句，只把没消掉的那个字母原样留在中间
/// （日志："我想知道现在这个输入u法是不是正常的推理。"）。整句都是汉字、只多一个
/// 小写字母时，删掉它就是用户想要的结果。
///
/// **两个及以上一律不动**：那多半是模型有意吐出的英文/缩写（"google"、"AI" 的小写
/// 形态），删了才是真的错。除汉字外出现任何别的字符（数字、大写、符号）也不动 ——
/// 规则只认"纯中文 + 恰好一个小写字母"这一种形状。
string StripLoneLowercaseLetter(const string& text) {
  size_t lone_at = string::npos;
  int lowercase = 0;
  int hanzi = 0;
  auto it = text.begin();
  while (it != text.end()) {
    const char* pos = &*it;
    const uint32_t cp = utf8::next(it, text.end());
    if (cp >= 0x4E00 && cp <= 0x9FFF) {
      ++hanzi;
      continue;
    }
    if (cp >= 'a' && cp <= 'z') {
      if (++lowercase > 1) {
        return text;
      }
      lone_at = static_cast<size_t>(pos - text.data());
      continue;
    }
    return text;
  }
  if (lowercase != 1 || hanzi == 0) {
    return text;
  }
  string out = text;
  out.erase(lone_at, 1);  // ASCII 小写是单字节，直接抹掉不会切坏 UTF-8
  return out;
}

string ExtractDisplayText(const string& model_output, bool filter_punctuation) {
  return StripLoneLowercaseLetter(filter_punctuation
                                      ? StripAllPunctuation(model_output)
                                      : model_output);
}

bool IsDisplayableCandidate(const string& display) {
  // 只要含汉字就交出去 —— 不再因为夹着 ASCII 就把整条答案丢掉。
  //
  // 原来这里把「带 ASCII 的输出」判成模型瞎编（"qizh" → "期ZH"）一律丢弃，但真机
  // 日志里更常见的是模型**几乎答对**、只把没转换成功的字母原样留着：
  //   raw='我想知道现在这个输入ufa是不是正常的推理。'  → cache HIT suppressed
  // 唯一瑕疵是多打的 u，整条被判死之后用户在第 2 位什么都看不到，只能回头选词库里
  // 那半截候选。应用层不改模型输出的一个字符，只负责把它交给候选列表；
  // 「消掉多余字母」该由模型自己解决。
  return CountHan(display) > 0;
}

}  // namespace predict
}  // namespace rime
