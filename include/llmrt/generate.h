// Greedy generation on top of the forward pass, in two prompt modes.
//
// ---------------------------------------------------------------------------
// The two modes
// ---------------------------------------------------------------------------
//
//   Raw   the ids are fed exactly as given. No wrapper. This is what golden was
//         captured with -- gen_oracle.py calls
//         tokenizer(text, add_special_tokens=False) -- so this is the mode whose
//         output can be checked against the reference.
//
//   Chat  the ids are wrapped the way Qwen3's chat template wraps a user turn,
//         so the model behaves as an assistant answering a question instead of
//         as a text completer. Measured difference on a real prompt:
//
//           raw   '中国的首都是'      -> greedy picks '____' (a fill-in-the-blank
//                                        pattern from the training data); the
//                                        correct '北京' is second, 0.26 logits behind
//           chat  wrapped as a user turn -> the model answers
//
//         Not checkable against golden, because golden has no chat-formatted
//         prompts. What IS checkable is the wrapper itself: the ids below were
//         read off the reference tokenizer's apply_chat_template and the test
//         compares against them literally.
//
// No tokenizer is involved in either mode. The wrapper is expressed as ids, so
// the chat path costs no text handling at all.
//
// ---------------------------------------------------------------------------
// Why the wrapper is a table of ids rather than a string
// ---------------------------------------------------------------------------
//
// apply_chat_template produces this text for one user turn:
//
//   <|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n
//
// and with enable_thinking=False it appends <think>\n\n</think>\n\n, which
// pre-fills an empty reasoning block -- the model is told not to think and to
// answer directly. That is the sensible default for a 0.6B model, which is not
// good at reasoning and will spend its whole budget on it.
//
// Tokenizing that string would need a tokenizer. Hard-coding the ids avoids it,
// and they are stable: they are fixed special tokens plus three ordinary BPE
// tokens for 'user', 'assistant' and the newline. Every one of them was read
// off the reference tokenizer, and the test asserts the exact sequence.
#pragma once

#include <cstdint>
#include <vector>

#include "llmrt/forward.h"

namespace llmrt {

// Ids for the fixed pieces of the chat wrapper, read off
// apply_chat_template(..., add_generation_prompt=True) on the reference
// tokenizer. Kept as named constants because a bare 151644 in the middle of a
// concatenation is unreadable and one digit off is a very quiet bug.
namespace chat_ids {
inline constexpr int32_t kImStart = 151644;      // <|im_start|>
inline constexpr int32_t kImEnd = 151645;        // <|im_end|>
inline constexpr int32_t kThinkStart = 151667;   // <think>
inline constexpr int32_t kThinkEnd = 151668;     // </think>
inline constexpr int32_t kUser = 872;            // 'user'
inline constexpr int32_t kAssistant = 77091;     // 'assistant'
inline constexpr int32_t kNewline = 198;         // '\n'
inline constexpr int32_t kBlankLine = 271;       // '\n\n'
}  // namespace chat_ids

enum class PromptMode {
  // Feed the ids as given. Matches golden.
  Raw,
  // <|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
  Chat,
};

// Wraps `content` for the given mode. Raw returns a copy unchanged.
std::vector<int32_t> wrap_prompt(const std::vector<int32_t>& content, PromptMode mode);

struct GenerationParams {
  int64_t max_new_tokens = 32;
  // <|im_end|> and <|endoftext|>, from generation_config.json. Either ends the
  // sequence -- that is what Qwen3 was trained with.
  std::vector<int32_t> eos_token_ids = {151645, 151643};
  PromptMode mode = PromptMode::Raw;
};

struct GenerationResult {
  // What was actually fed to the model, after wrapping. Useful to print: in
  // Chat mode it is the only way to see what the model was asked.
  std::vector<int32_t> prompt_ids;
  // The newly generated tokens, not counting the prompt.
  std::vector<int32_t> tokens;
  bool stopped_on_eos = false;
  // Seconds per generated token: one full forward pass each. This is the
  // number Phase 4's KV cache has to beat, measured before it exists.
  std::vector<double> step_seconds;
};

// Greedy only: argmax at every step, no temperature, no top-k, no top-p. With
// no KV cache each step re-forwards the whole prefix, which is O(S^2) work over
// the sequence but costs the same weight traffic per step as any other forward
// -- see step_seconds.
GenerationResult generate(Qwen3Forward& forward, const std::vector<int32_t>& content,
                          const GenerationParams& params = {});

}  // namespace llmrt
