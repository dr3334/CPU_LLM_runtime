// Subcommand entry points. Each is implemented in its own cmd_*.cpp so that
// the CLI layer stays thin and the runtime library stays UI-agnostic.
#pragma once

#include "args.h"

namespace llmrt {
namespace cli {

// Dumps model config, tensor table and memory footprint.
int cmd_inspect(const Args& args);

// Tokenizer round-trip: text -> ids -> text.
int cmd_tok(const Args& args);

// Prompt + autoregressive decode, with optional sampling knobs.
int cmd_generate(const Args& args);

// Benchmark / profiling report (TTFT, prefill and decode throughput).
int cmd_bench(const Args& args);

}  // namespace cli
}  // namespace llmrt
