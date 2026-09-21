// `llmrt generate` -- greedy generation from token ids.
//
// No tokenizer, by design. The prompt is given as ids and the output is printed
// as ids; turning either into text is a table lookup done outside the runtime.
// Two prompt modes are available, and the difference between them is the point:
//
//   raw   the ids are fed as given, so the model completes text. Measured:
//         '中国的首都是' gets '____' -- the fill-in-the-blank pattern the
//         training data is full of -- with the correct '北京' second, 0.26
//         logits behind.
//   chat  the ids are wrapped as a user turn using the chat template's ids
//         directly (no text, no tokenizer). The model answers instead.
//
// Every generated token costs one full forward pass: there is no KV cache yet,
// so each step re-forwards the whole prefix. The per-step times printed here are
// the baseline Phase 4 has to beat.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "llmrt/common.h"
#include "llmrt/config.h"
#include "llmrt/forward.h"
#include "llmrt/generate.h"
#include "llmrt/model.h"
#include "llmrt/safetensors.h"

#include "commands.h"

namespace llmrt {
namespace cli {

namespace {

const char* kDefaultModelDir = "/home/dr/models/Qwen3-0.6B";

std::string default_model_dir() {
  if (const char* env = std::getenv("LLMRT_MODEL_DIR")) return env;
  return kDefaultModelDir;
}

// "785,6722,315" -> {785, 6722, 315}. Spaces and newlines are skipped so a
// wrapped list pasted from a terminal still parses.
std::vector<int32_t> parse_ids(const std::string& text) {
  std::vector<int32_t> out;
  size_t i = 0;
  while (i < text.size()) {
    while (i < text.size() &&
           (text[i] == ',' || text[i] == ' ' || text[i] == '\n' || text[i] == '\t' ||
            text[i] == '\r')) {
      ++i;
    }
    if (i >= text.size()) break;
    const size_t start = i;
    while (i < text.size() && text[i] >= '0' && text[i] <= '9') ++i;
    LLMRT_CHECK(i > start, "generate: --ids must be comma-separated integers, found '" +
                               text.substr(start, 16) + "'");
    out.push_back(static_cast<int32_t>(std::stol(text.substr(start, i - start))));
  }
  return out;
}

void print_ids(const char* label, const std::vector<int32_t>& ids) {
  std::printf("%-11s", label);
  for (size_t i = 0; i < ids.size(); ++i) {
    std::printf("%s%d", i == 0 ? "" : " ", ids[i]);
  }
  std::printf("   (%zu)\n", ids.size());
}

PromptMode parse_mode(const std::string& s) {
  if (s == "raw") return PromptMode::Raw;
  if (s == "chat") return PromptMode::Chat;
  LLMRT_CHECK(false, "generate: --mode must be 'raw' or 'chat', got '" + s + "'");
  return PromptMode::Raw;
}

}  // namespace

int cmd_generate(const Args& args) {
  const std::string model_dir = args.get("model", default_model_dir());
  const std::string ids_arg = args.get("ids");
  LLMRT_CHECK(!ids_arg.empty(),
              "generate: --ids is required (there is no tokenizer yet). "
              "Example: --ids 785,6722,315,9625,374");

  const std::vector<int32_t> content = parse_ids(ids_arg);
  const PromptMode mode = parse_mode(args.get("mode", "raw"));

  GenerationParams params;
  params.mode = mode;
  params.max_new_tokens = args.get_int("max-new", 16);

  std::printf("model        %s\n", model_dir.c_str());
  std::printf("mode         %s\n", mode == PromptMode::Raw
                                       ? "raw -- continue the text as given"
                                       : "chat -- wrap as a user turn, then answer");

  const SafeTensors st = SafeTensors::open(model_dir + "/model.safetensors");
  const Qwen3Config cfg = Qwen3Config::from_model_dir(model_dir);
  const Qwen3Weights weights = Qwen3Weights::bind(st, cfg);

  Qwen3Forward forward(st, weights);
  std::printf("weights      %.3f GB as f32\n",
              static_cast<double>(forward.weight_bytes()) / 1e9);

  const GenerationResult r = generate(forward, content, params);

  std::printf("\n");
  print_ids("input ids", content);
  if (mode == PromptMode::Chat) print_ids("wrapped", r.prompt_ids);
  print_ids("generated", r.tokens);
  std::printf("%-11s%s\n", "stop",
              r.stopped_on_eos ? "hit an end-of-sequence token" : "hit --max-new");

  double total = 0.0;
  for (const double s : r.step_seconds) total += s;
  if (!r.step_seconds.empty()) {
    std::printf("timing       %.2f s total, %.2f s/token over %zu tokens\n", total,
                total / static_cast<double>(r.step_seconds.size()), r.step_seconds.size());
    std::printf("per step     ");
    for (const double s : r.step_seconds) std::printf(" %.2f", s);
    std::printf("  (s)\n");
    std::printf("             no KV cache: every step re-forwards the whole prefix, and\n");
    std::printf("             the weight traffic is the same whatever the length -- which\n");
    std::printf("             is exactly why this number is Phase 4's target\n");
  }

  std::printf("\nTo read the output, look the ids up in the model's vocab.json.\n");
  return 0;
}

}  // namespace cli
}  // namespace llmrt
