// End-to-end: run the whole 28-layer stack on golden's input_ids and compare the
// logits. This is the first test in the project that checks the model rather
// than one op.
//
// When it fails it fails completely -- every logit wrong -- which is why
// forward_hidden_states_match_golden_layer_by_layer exists alongside it. golden
// captured each decoder layer's output, so that test names the first layer that
// diverges, turning a 28-layer search into a single layer.
//
// Loading the checkpoint and building the 2.4 GB f32 weight mirror takes
// seconds, so every test in this file shares one instance.

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "llmrt/common.h"
#include "llmrt/config.h"
#include "llmrt/forward.h"
#include "llmrt/model.h"
#include "llmrt/safetensors.h"

#include "golden.h"
#include "test_framework.h"

using namespace llmrt;

namespace {

std::string model_path(const char* file) {
  return golden::default_model_dir() + "/" + file;
}

bool checkpoint_exists() {
  return std::ifstream(model_path("model.safetensors")).good();
}

// Built once and shared. All three have to outlive the forward pass -- the
// SafeTensors mmap in particular, since the binding stores pointers into it --
// so they are static inside the initialiser rather than locals of it.
Qwen3Forward* shared_forward() {
  static Qwen3Forward* instance = []() -> Qwen3Forward* {
    golden::Store g;
    if (!g.available() || !checkpoint_exists()) return nullptr;
    static const SafeTensors st = SafeTensors::open(model_path("model.safetensors"));
    static const Qwen3Config cfg = Qwen3Config::from_model_dir(golden::default_model_dir());
    static const Qwen3Weights weights = Qwen3Weights::bind(st, cfg);
    return new Qwen3Forward(st, weights);
  }();
  return instance;
}

}  // namespace

// Bisect helper. golden's layer_hidden is [28, 1, 12, 1024] and covers prompt 0
// only, so this runs that one prompt and checks each layer's output as the
// observer hands it over.
LLMRT_TEST(forward_hidden_states_match_golden_layer_by_layer) {
  Qwen3Forward* fwd = shared_forward();
  if (fwd == nullptr) return;
  golden::Store g;

  size_t count = 0;
  const int32_t* ids = g.ids("input_ids", &count);
  const golden::Array* lh = g.get("layer_hidden");
  CHECK_TRUE(ids != nullptr && lh != nullptr);
  if (ids == nullptr || lh == nullptr) return;

  const int64_t layers = fwd->config().num_hidden_layers;
  // [layers, 1, seq, hidden] -- the 1 is the batch axis the hook kept.
  CHECK_EQ(lh->shape[0], layers);
  CHECK_EQ(lh->shape[1], int64_t{1});
  const int64_t seq = lh->shape[2];
  const int64_t hidden = lh->shape[3];
  CHECK_EQ(hidden, fwd->config().hidden_size);
  CHECK_EQ(count, size_t{60});

  const std::vector<int32_t> prompt0(ids, ids + seq);
  const size_t per_layer = static_cast<size_t>(seq * hidden);

  std::vector<std::vector<float>> states;
  states.reserve(static_cast<size_t>(layers));
  std::vector<float> logits;
  fwd->run(prompt0, logits, [&](int64_t layer, const std::vector<float>& h) {
    // The observer is asked to hand over each layer in order. Copying costs
    // 48 KB per layer for this prompt and keeps the comparison below off the
    // forward pass's scratch, which the next layer overwrites.
    CHECK_EQ(layer, static_cast<int64_t>(states.size()));
    CHECK_EQ(h.size(), per_layer);
    states.push_back(h);
  });
  CHECK_EQ(states.size(), static_cast<size_t>(layers));
  if (states.size() != static_cast<size_t>(layers)) return;

  int64_t first_bad = -1;
  for (int64_t l = 0; l < layers; ++l) {
    const float* want = lh->f() + static_cast<size_t>(l) * per_layer;
    const golden::Diff d =
        golden::compare(states[static_cast<size_t>(l)].data(), want, per_layer);
    std::printf("      layer %2lld  %s\n", static_cast<long long>(l),
                golden::diff_string(d).c_str());
    if (first_bad < 0 && !golden::within(d)) first_bad = l;
  }

  CHECK_MSG(first_bad < 0,
            "first layer whose hidden state disagrees with golden: " +
                std::to_string(first_bad) + " -- everything before it matched, so the "
                "problem is inside that layer, not in the stack");
}

// The gate. golden forwarded each prompt separately (batch 1, no padding) and
// concatenated the logits, so each prompt is run on its own here too.
LLMRT_TEST(forward_matches_golden_logits) {
  Qwen3Forward* fwd = shared_forward();
  if (fwd == nullptr) return;
  golden::Store g;

  size_t count = 0;
  const int32_t* ids = g.ids("input_ids", &count);
  const golden::Array* want = g.get("logits");
  CHECK_TRUE(ids != nullptr && want != nullptr);
  if (ids == nullptr || want == nullptr) return;

  const std::vector<int64_t> offsets = g.prompt_offsets();
  const std::vector<int64_t> lengths = g.prompt_lengths();
  CHECK_EQ(offsets.size(), lengths.size());
  CHECK_EQ(offsets.size(), size_t{7});
  if (offsets.size() != lengths.size()) return;

  const int64_t vocab = fwd->config().vocab_size;
  CHECK_EQ(want->shape[1], vocab);

  std::vector<float> logits;
  int64_t first_bad = -1;
  for (size_t i = 0; i < offsets.size(); ++i) {
    const int64_t begin = offsets[i];
    const int64_t len = lengths[i];
    const std::vector<int32_t> prompt(ids + begin, ids + begin + len);
    fwd->run(prompt, logits);

    const size_t n = static_cast<size_t>(len) * static_cast<size_t>(vocab);
    const float* want_rows = want->f() + static_cast<size_t>(begin) * static_cast<size_t>(vocab);
    const golden::Diff d = golden::compare(logits.data(), want_rows, n);
    std::printf("      prompt %zu (%2lld tokens)  %s\n", i, static_cast<long long>(len),
                golden::diff_string(d).c_str());
    if (first_bad < 0 && !golden::within(d)) first_bad = static_cast<int64_t>(i);
  }

  CHECK_MSG(first_bad < 0,
            "first prompt whose logits disagree with golden: " + std::to_string(first_bad));
}

LLMRT_TEST(forward_rejects_empty_prompt) {
  Qwen3Forward* fwd = shared_forward();
  if (fwd == nullptr) return;
  std::vector<float> logits;
  bool threw = false;
  try {
    fwd->run({}, logits);
  } catch (const Error&) {
    threw = true;
  }
  CHECK_TRUE(threw);
}

LLMRT_TEST(forward_rejects_prompt_longer_than_the_context) {
  Qwen3Forward* fwd = shared_forward();
  if (fwd == nullptr) return;
  // One past max_position_embeddings. The check runs before allocate(), so this
  // does not build a 40961-token activation set to find out.
  const std::vector<int32_t> too_long(
      static_cast<size_t>(fwd->config().max_position_embeddings + 1), 0);
  std::vector<float> logits;
  bool threw = false;
  try {
    fwd->run(too_long, logits);
  } catch (const Error&) {
    threw = true;
  }
  CHECK_TRUE(threw);
}

int main() { return llmrt_test::run_all("forward"); }
