// Test-only definitions for helpers absent from the matching Rime import library.
// Keep exported Translation methods in the DLL; do not compile translation.cc.
#include <rime/candidate.h>
#include <rime/config.h>
#include <rime/schema.h>
#include <rime/segmentation.h>
#include <rime/ticket.h>
#include <rime/translation.h>
#include <rime/gear/filter_commons.h>

namespace rime {

bool UniqueTranslation::Next() {
  if (exhausted()) return false;
  set_exhausted(true);
  return true;
}

an<Candidate> UniqueTranslation::Peek() {
  return exhausted() ? nullptr : candidate_;
}

TagMatching::TagMatching(const Ticket& ticket) {
  if (!ticket.schema) return;
  if (auto tags = ticket.schema->config()->GetList(ticket.name_space + "/tags")) {
    // Indexed access is exported; ConfigList's iterators are not.
    for (size_t i = 0; i < tags->size(); ++i) {
      if (auto value = tags->GetValueAt(i)) tags_.push_back(value->str());
    }
  }
}

bool TagMatching::TagsMatch(Segment* segment) {
  if (!segment) return false;
  if (tags_.empty()) return true;
  for (const string& tag : tags_) {
    if (segment->HasTag(tag)) return true;
  }
  return false;
}

}  // namespace rime
