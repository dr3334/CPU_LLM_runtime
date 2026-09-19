// llmrt CLI entry point.
//
// Subcommands are deliberately separate so that the runtime library never
// depends on the CLI layer.
#include <cstdio>
#include <exception>
#include <string>

#include "llmrt/version.h"

#include "args.h"
#include "commands.h"

namespace {

void print_usage(FILE* out, const char* prog) {
  std::fprintf(out,
      "%s\n"
      "\n"
      "Usage: %s <command> [options]\n"
      "\n"
      "Commands:\n"
      "  inspect    Dump model config, tensor table and memory footprint\n"
      "  tok        Tokenizer round-trip: text -> ids -> text\n"
      "  generate   Run the model on a prompt and print generated text\n"
      "  bench      Measure TTFT / prefill / decode throughput with profiling\n"
      "\n"
      "Common options:\n"
      "  --model <dir>     Path to a safetensors model directory\n"
      "                    (default: /home/dr/models/Qwen3-0.6B)\n"
      "  --split <spec>    Layer placement across devices, e.g.\n"
      "                    '0-13:cpu,14-27:opencl'  (default: all cpu)\n"
      "  --help            Show this message\n"
      "  --version         Show version\n",
      llmrt::build_description(), prog);
}

}  // namespace

int main(int argc, char** argv) {
  using namespace llmrt::cli;

  if (argc < 2) {
    print_usage(stderr, argv[0]);
    return 2;
  }

  const std::string cmd = argv[1];
  if (cmd == "-h" || cmd == "--help" || cmd == "help") {
    print_usage(stdout, argv[0]);
    return 0;
  }
  if (cmd == "--version" || cmd == "-v") {
    std::printf("%s\n", llmrt::build_description());
    return 0;
  }

  const Args args = parse_args(argc, argv, 2);

  try {
    if (cmd == "inspect") return cmd_inspect(args);
    if (cmd == "tok") return cmd_tok(args);
    if (cmd == "generate") return cmd_generate(args);
    if (cmd == "bench") return cmd_bench(args);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }

  std::fprintf(stderr, "error: unknown command '%s'\n\n", cmd.c_str());
  print_usage(stderr, argv[0]);
  return 2;
}
