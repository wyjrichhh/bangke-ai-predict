// Build CT2 prompt and sliding-window context from Rime composition / history.
//
// Copyright RIME Developers
// Distributed under the BSD License

#ifndef RIME_PREDICT_CONTEXT_BUILDER_H_
#define RIME_PREDICT_CONTEXT_BUILDER_H_

#include <rime/common.h>

#include <optional>
#include <string>

namespace rime {

class Context;
class Engine;

namespace predict {

struct PredictionContext {
  /// Pure pinyin being composed (spaces stripped). Doubles as the key for the
  /// engine's "does the composition still match the request we sent" check.
  string effective_prompt;
  /// Recent commit history: everything that reached the screen, Hanzi plus the
  /// digits/Latin/punctuation typed around it. Empty when there is no history.
  string window_text;
  /// Result cache key. Combines window_text + effective_prompt so the same
  /// pinyin under different contexts can't collide.
  string cache_key;
  /// Full line passed to the backend (Chinese prefix + pinyin tags).
  string ct2_input;
  /// True when issued with a non-empty window_text (context-aware mode); false
  /// for context-free cold start.
  bool windowed = false;
  /// Whether window_text came from the frontend's read of the host app (true)
  /// or from commit-history reconstruction (false).
  bool window_from_frontend = false;
  /// What the commit-history path would have produced. Only filled when
  /// window_from_frontend is true, and only for the log -- nothing downstream
  /// reads it. Exists so the two sources can be compared offline.
  string history_window_text;
};

struct ContextBuilderOptions {
  /// Min prompt length (bytes) to trigger prediction WITHOUT context. With
  /// context any non-empty prompt triggers; without it a short fragment alone
  /// just hallucinates, so we wait for more.
  int min_effective_length = 12;
  /// Min prompt length (letters) to trigger prediction WITH context. A single
  /// letter carries almost no signal -- the model can only guess the next
  /// character, and log analysis shows such offers are never selected. We still
  /// fire on >=2 letters since real context makes even a short fragment useful.
  int min_context_prompt_length = 2;
  /// Max history records to scan for the sliding window.
  int context_window_size = 10;
  /// Take window_text from the text the frontend read out of the host app
  /// instead of reconstructing it from commit history. The engine alone cannot
  /// see edits that never entered a composition (backspace on committed text,
  /// Cmd+A, paste); the frontend can. Falls back when the frontend has nothing.
  bool use_frontend_context = true;
  /// Cap on that text, counted in UTF-8 codepoints from the end. The frontend
  /// reads generously and lets us trim, so this is the single knob for how much
  /// context reaches the model.
  int context_window_chars = 24;
  /// AI-only mode (ai_predict/ai_only): the model is the only candidate source,
  /// so the two length gates above lose their reason to exist and are skipped.
  bool ai_only = false;
};

class ContextBuilder {
 public:
  /// Returns empty optional if prediction should be skipped (length gate).
  static std::optional<PredictionContext> Build(Engine* engine,
                                                const string& raw_input,
                                                const ContextBuilderOptions& opt);
};

/// Clean a raw model output for display in the candidate menu. `filter_punctuation`
/// strips any punctuation the model emitted (see BuildWindowContext for why
/// punctuation is tolerated on input); false hands it through untouched, so the
/// model's own sentence punctuation ("，我们") survives into the candidate.
string ExtractDisplayText(const string& model_output, bool filter_punctuation);

/// True iff `display` is fit to surface as a candidate. Rejects lowercase
/// Latin (a half-typed syllable echoed back: "qizh" -> "期ZH"), and rejects
/// uppercase Latin letters the prompt never contained — those are the model
/// inventing its own acronym. Uppercase the user's own input does carry, e.g.
/// "t3" answered with "T3", is allowed; so is any uppercase at all when the
/// prompt itself was an acronym (the user really typed APPLE/IBM). The
/// candidate must also contain Hanzi so a pure-English echo ("FEATURE") stays
/// suppressed.
bool IsDisplayableCandidate(const string& display,
                            const string& prompt,
                            bool ai_only);

/// Number of CJK Unified Ideographs in `text`. Count (not just a presence bool)
/// so callers can gate on "at least N Hanzi" — e.g. suppress single-Hanzi AI
/// candidates, which the schema dictionary already ranks by frequency far more
/// reliably than the model's context-free continuation.
int CountHan(const string& text);

}  // namespace predict
}  // namespace rime

#endif  // RIME_PREDICT_CONTEXT_BUILDER_H_
