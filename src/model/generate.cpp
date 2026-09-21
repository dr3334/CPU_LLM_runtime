// Greedy generation, and the two prompt wrappers.
//
// The wrapper is built from ids rather than from text, so nothing here needs a
// tokenizer. See generate.h for why the ids are hard-coded and where they came
// from.

#include "llmrt/generate.h"

#include <chrono>
#include <cmath>
#include <string>

#include "llmrt/common.h"
#include "llmrt/ops.h"
#include "llmrt/tensor.h"

namespace llmrt {

namespace {

using chat_ids::kAssistant;
using chat_ids::kBlankLine;
using chat_ids::kImEnd;
using chat_ids::kImStart;
using chat_ids::kNewline;
using chat_ids::kThinkEnd;
using chat_ids::kThinkStart;
using chat_ids::kUser;

bool is_eos(int32_t token, const std::vector<int32_t>& eos_ids) {
  for (const int32_t e : eos_ids) {
    if (token == e) return true;
  }
  return false;
}

double now_seconds() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

}  // namespace

std::vector<int32_t> wrap_prompt(const std::vector<int32_t>& content, PromptMode mode) {
  if (mode == PromptMode::Raw) return content;

  // <|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n
  //   <think>\n\n</think>\n\n
  std::vector<int32_t> out;
  out.reserve(content.size() + 16);
  out.insert(out.end(), {kImStart, kUser, kNewline});
  out.insert(out.end(), content.begin(), content.end());
  out.insert(out.end(), {kImEnd, kNewline, kImStart, kAssistant, kNewline});
  // The pre-filled empty reasoning block. Without it the model treats the
  // assistant turn as a place to think, and a 0.6B model's thinking is not
  // worth the tokens. These four ids are what enable_thinking=False produces.
  out.insert(out.end(), {kThinkStart, kBlankLine, kThinkEnd, kBlankLine});
  return out;
}

GenerationResult generate(Qwen3Forward& forward, const std::vector<int32_t>& content,
                          const GenerationParams& params) {
  LLMRT_CHECK(!content.empty(), "generate: empty prompt");
  LLMRT_CHECK(params.max_new_tokens >= 0, "generate: max_new_tokens must not be negative");

  GenerationResult result;
  result.prompt_ids = wrap_prompt(content, params.mode);

  // The sequence we feed grows one token per step. There is no KV cache, so
  // every step re-forwards the entire prefix -- correct, and deliberately the
  // thing Phase 4 will replace. Keeping the prompt in `ids` and passing it
  // whole is what makes the loop this short.
  std::vector<int32_t> ids = result.prompt_ids;
  std::vector<float> logits;

  for (int64_t step = 0; step < params.max_new_tokens; ++step) {
    const double t0 = now_seconds();
    forward.run(ids, logits);
    const double t1 = now_seconds();
    result.step_seconds.push_back(t1 - t0);

    // Only the last position's row is wanted, but run() produces the whole
    // [seq, vocab] because the reference does not slice either. That is 608 KB
    // of logits per step at this vocab size, most of it discarded -- exactly
    // the waste P3 records for the generating path.
    const int64_t seq = static_cast<int64_t>(ids.size());
    const int64_t vocab = forward.config().vocab_size;
    float* last_row = logits.data() + (seq - 1) * vocab;

    Tensor row = Tensor::contiguous(last_row, DType::F32, DeviceKind::CPU, {1, vocab});
    int32_t next_token = -1;
    Tensor next = Tensor::contiguous(&next_token, DType::I32, DeviceKind::CPU, {1});
    cpu::argmax(row, next);

    // argmax skips NaN rather than selecting it, so a model that has gone bad
    // still produces a plausible index. One finiteness check on the chosen
    // value catches the common case (every logit NaN) for the cost of a single
    // comparison, and it lives here rather than in the op because it is policy,
    // not reduction.
    LLMRT_CHECK(std::isfinite(last_row[next_token]),
                "generate: step " + std::to_string(step) +
                    ": the logit for the selected token is not finite -- the "
                    "weights have gone bad upstream");

    ids.push_back(next_token);
    result.tokens.push_back(next_token);

    if (is_eos(next_token, params.eos_token_ids)) {
      result.stopped_on_eos = true;
      break;
    }
  }

  return result;
}

}  // namespace llmrt
