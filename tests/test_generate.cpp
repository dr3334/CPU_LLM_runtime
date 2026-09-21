// Greedy generation: the loop, the two prompt modes, and end-of-sequence.
//
// Separate binary from test_forward because every generated token costs a full
// forward pass (~1.5 s here), so the default run of this suite is about a
// minute while test_forward is twenty seconds. The full 7x16 verification is
// opt-in:
//
//     LLMRT_GREEDY_STEPS=16 ./build/tests/test_generate
//
// No tokenizer is involved anywhere here. The chat wrapper is expressed as ids,
// which is what makes Chat mode testable at all without one.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "llmrt/common.h"
#include "llmrt/forward.h"
#include "llmrt/generate.h"

#include "golden.h"
#include "model_fixture.h"
#include "test_framework.h"

using namespace llmrt;
using llmrt_test::greedy_steps;
using llmrt_test::shared_forward;

// The chat wrapper, checked against what the reference tokenizer produces. This
// is the one claim in Chat mode that CAN be verified: golden has no
// chat-formatted prompts, so the generation itself has no reference, but the
// wrapper does.
//
// Read off apply_chat_template(msgs, add_generation_prompt=True,
// enable_thinking=False) for content 'The capital of France is'.
LLMRT_TEST(chat_wrapper_ids_match_the_reference_template) {
  const std::vector<int32_t> content = {785, 6722, 315, 9625, 374};
  const std::vector<int32_t> want = {
      151644, 872,  198,             // <|im_start|>user\n
      785,    6722, 315, 9625, 374,  // the content, untouched
      151645, 198,                   // <|im_end|>\n
      151644, 77091, 198,            // <|im_start|>assistant\n
      151667, 271,  151668, 271,     // <think>\n\n</think>\n\n
  };
  CHECK_EQ(wrap_prompt(content, PromptMode::Chat), want);
  // Raw is the identity, which is what makes it comparable with golden.
  CHECK_EQ(wrap_prompt(content, PromptMode::Raw), content);
}

// The loop: forward, argmax the last row, append, feed the result back.
//
// golden's greedy_ids was produced the same way -- an explicit re-forward per
// step, deliberately not model.generate(), which would use a KV cache and so
// differ in the last bits. See tools/gen_oracle.py.
LLMRT_TEST(greedy_generation_matches_golden) {
  Qwen3Forward* fwd = shared_forward();
  if (fwd == nullptr) return;
  golden::Store g;

  size_t count = 0;
  const int32_t* ids = g.ids("input_ids", &count);
  const golden::Array* gi = g.get("greedy_ids");
  CHECK_TRUE(ids != nullptr && gi != nullptr);
  if (ids == nullptr || gi == nullptr) return;

  const std::vector<int64_t> offs = g.prompt_offsets();
  const std::vector<int64_t> lens = g.prompt_lengths();
  const std::vector<int64_t> glens = g.greedy_lengths();
  CHECK_EQ(glens.size(), offs.size());
  if (glens.size() != offs.size()) return;

  const int64_t steps = greedy_steps();
  const int64_t row = gi->shape[1];  // greedy_ids is [prompts, max_new]

  int64_t checked = 0;
  int64_t mismatches = 0;
  for (size_t p = 0; p < offs.size(); ++p) {
    const int64_t want_new = std::min(glens[p], steps);
    const std::vector<int32_t> prompt(ids + offs[p], ids + offs[p] + lens[p]);

    GenerationParams params;
    params.mode = PromptMode::Raw;  // golden was tokenized with no template
    params.max_new_tokens = want_new;
    const GenerationResult r = generate(*fwd, prompt, params);

    CHECK_EQ(r.prompt_ids.size(), static_cast<size_t>(lens[p]));
    for (int64_t k = 0; k < want_new; ++k) {
      // A short result means the loop stopped on eos, which golden says never
      // happens inside 16 steps -- so a short result is itself a mismatch.
      const int32_t got =
          k < static_cast<int64_t>(r.tokens.size()) ? r.tokens[static_cast<size_t>(k)]
                                                    : -999;
      const int32_t want = gi->i()[p * row + k];
      ++checked;
      if (got != want) {
        ++mismatches;
        if (mismatches <= 5) {
          std::printf("      prompt %zu step %lld: got %d, golden %d\n", p,
                      static_cast<long long>(k), got, want);
        }
      }
    }
    if (r.stopped_on_eos) {
      std::printf("      prompt %zu stopped on eos, but golden generated %lld tokens\n", p,
                  static_cast<long long>(glens[p]));
    }
  }

  std::printf("      checked %lld greedy tokens across %zu prompts (%lld steps each)\n",
              static_cast<long long>(checked), offs.size(), static_cast<long long>(steps));
  if (steps < 16) {
    std::printf("      golden has 16; set LLMRT_GREEDY_STEPS=16 to check them all\n");
  }
  CHECK_EQ(mismatches, int64_t{0});
}

// The two modes must actually do different things. '中国的首都是' is the prompt
// where it matters: as raw text the model continues the fill-in-the-blank
// pattern ('____'), and as a user turn it should answer. Both ids are printed so
// a change in behaviour is visible rather than just "the test still passes".
LLMRT_TEST(chat_mode_answers_where_raw_mode_completes) {
  Qwen3Forward* fwd = shared_forward();
  if (fwd == nullptr) return;
  golden::Store g;

  size_t count = 0;
  const int32_t* ids = g.ids("input_ids", &count);
  const std::vector<int64_t> offs = g.prompt_offsets();
  const std::vector<int64_t> lens = g.prompt_lengths();
  CHECK_TRUE(ids != nullptr);
  if (ids == nullptr) return;

  // Prompt 4, '中国的首都是'.
  const std::vector<int32_t> prompt(ids + offs[4], ids + offs[4] + lens[4]);

  GenerationParams raw;
  raw.mode = PromptMode::Raw;
  raw.max_new_tokens = 1;
  GenerationParams chat;
  chat.mode = PromptMode::Chat;
  chat.max_new_tokens = 1;

  const GenerationResult r_raw = generate(*fwd, prompt, raw);
  const GenerationResult r_chat = generate(*fwd, prompt, chat);

  std::printf("      raw   first token %6d  (golden says 2130 = '____')\n",
              r_raw.tokens.at(0));
  std::printf("      chat  first token %6d   over %zu wrapped ids\n", r_chat.tokens.at(0),
              r_chat.prompt_ids.size());
  CHECK_TRUE(!r_raw.tokens.empty() && !r_chat.tokens.empty());
  CHECK_MSG(r_raw.tokens[0] != r_chat.tokens[0],
            "raw and chat modes produced the same first token, so the chat "
            "wrapper is not reaching the model");
}

// End-of-sequence handling, without needing a prompt that actually generates one
// (none of golden's do inside 16 steps). Telling the loop that the token it is
// about to produce is an eos must stop it after exactly one step.
LLMRT_TEST(generation_stops_when_it_produces_an_eos_token) {
  Qwen3Forward* fwd = shared_forward();
  if (fwd == nullptr) return;
  golden::Store g;

  size_t count = 0;
  const int32_t* ids = g.ids("input_ids", &count);
  const golden::Array* gi = g.get("greedy_ids");
  CHECK_TRUE(ids != nullptr && gi != nullptr);
  if (ids == nullptr || gi == nullptr) return;

  const std::vector<int64_t> offs = g.prompt_offsets();
  const std::vector<int64_t> lens = g.prompt_lengths();
  const int64_t row = gi->shape[1];

  // Prompt 1's first greedy token, which the test above proved the loop finds.
  const int32_t target = gi->i()[1 * row + 0];
  const std::vector<int32_t> prompt(ids + offs[1], ids + offs[1] + lens[1]);

  GenerationParams params;
  params.mode = PromptMode::Raw;
  params.max_new_tokens = 8;  // generous: only the eos rule may end this early
  params.eos_token_ids = {target};
  const GenerationResult r = generate(*fwd, prompt, params);

  CHECK_TRUE(r.stopped_on_eos);
  CHECK_EQ(r.tokens.size(), size_t{1});
  if (!r.tokens.empty()) CHECK_EQ(r.tokens[0], target);
  std::printf("      stopped after %zu token(s) on eos id %d\n", r.tokens.size(), target);
}

LLMRT_TEST(generate_rejects_an_empty_prompt) {
  Qwen3Forward* fwd = shared_forward();
  if (fwd == nullptr) return;
  bool threw = false;
  try {
    generate(*fwd, {});
  } catch (const Error&) {
    threw = true;
  }
  CHECK_TRUE(threw);
}

int main() { return llmrt_test::run_all("generate"); }
