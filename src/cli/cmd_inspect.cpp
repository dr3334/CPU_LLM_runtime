// `llmrt inspect` -- model directory and checkpoint report.
//
// Prints the parsed config, the full tensor table, and the resulting memory
// footprint (on-disk vs. deduplicated vs. f32-compute). This is the fastest
// way to confirm that a checkpoint is what the runtime expects before running
// any inference.
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "llmrt/config.h"
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

std::string human_bytes(size_t n) {
  static const char* units[] = {"B", "KiB", "MiB", "GiB"};
  double v = static_cast<double>(n);
  int u = 0;
  while (v >= 1024.0 && u < 3) {
    v /= 1024.0;
    ++u;
  }
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.1f %s", v, units[u]);
  return buf;
}

// Lexicographic, except digit runs compare numerically, so
// model.layers.2 sorts before model.layers.10.
bool natural_less(const std::string& a, const std::string& b) {
  size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    const bool digit_a = std::isdigit(static_cast<unsigned char>(a[i])) != 0;
    const bool digit_b = std::isdigit(static_cast<unsigned char>(b[j])) != 0;
    if (digit_a && digit_b) {
      size_t ia = i, jb = j;
      while (ia < a.size() && std::isdigit(static_cast<unsigned char>(a[ia]))) ++ia;
      while (jb < b.size() && std::isdigit(static_cast<unsigned char>(b[jb]))) ++jb;
      const size_t len_a = ia - i, len_b = jb - j;
      // More digits means a larger number (leading zeros are not used here).
      if (len_a != len_b) return len_a < len_b;
      const int c = a.compare(i, len_a, b, j, len_b);
      if (c != 0) return c < 0;
      i = ia;
      j = jb;
    } else {
      if (a[i] != b[j]) return a[i] < b[j];
      ++i;
      ++j;
    }
  }
  return (a.size() - i) < (b.size() - j);
}

}  // namespace

int cmd_inspect(const Args& args) {
  const std::string dir = args.get("model", default_model_dir());
  const bool summary_only = args.has_flag("summary");
  const bool show_tied = !args.has_flag("no-tied");

  std::printf("model            %s\n", dir.c_str());

  const Qwen3Config cfg = Qwen3Config::from_model_dir(dir);
  const std::string ckpt = dir + "/model.safetensors";
  const SafeTensors st = SafeTensors::open(ckpt);

  std::printf("checkpoint       %s\n", ckpt.c_str());
  std::printf("file size        %zu bytes (%s)\n", st.file_size(),
              human_bytes(st.file_size()).c_str());
  std::printf("data section     %zu bytes at offset %zu\n", st.data_bytes(), st.data_offset());
  std::printf("tensors          %zu\n", st.size());
  std::printf("\n");

  std::printf("%s\n", cfg.describe().c_str());

  // ---- tensor table -----------------------------------------------------
  std::vector<const TensorInfo*> table;
  table.reserve(st.size());
  for (const TensorInfo& t : st.tensors()) table.push_back(&t);
  std::sort(table.begin(), table.end(), [](const TensorInfo* a, const TensorInfo* b) {
    return natural_less(a->name, b->name);
  });

  if (summary_only) {
    std::printf("tensors (%zu, table suppressed by --summary)\n\n", table.size());
  } else {
    std::printf("tensors (%zu)\n", table.size());
    std::printf("  %-58s %-6s %-22s %12s\n", "name", "dtype", "shape", "bytes");
    for (const TensorInfo* t : table) {
      std::printf("  %-58s %-6s %-22s %12zu\n", t->name.c_str(), t->dtype_str.c_str(),
                  t->shape_string().c_str(), t->nbytes());
    }
    std::printf("\n");
  }

  // ---- dtype histogram --------------------------------------------------
  std::printf("dtype histogram\n");
  for (const auto& e : st.dtype_histogram()) {
    size_t bytes = 0;
    for (const TensorInfo* t : table)
      if (t->dtype_str == e.first) bytes += t->nbytes();
    std::printf("  %-6s %4zu tensors  %12zu bytes (%s)\n", e.first.c_str(), e.second, bytes,
                human_bytes(bytes).c_str());
  }
  std::printf("\n");

  // ---- binding + footprint ---------------------------------------------
  const Qwen3Weights w = Qwen3Weights::bind(st, cfg);

  const size_t resident = w.resident_bytes();
  const size_t slots = w.reference_slots();
  const size_t not_loaded = st.data_bytes() > resident ? st.data_bytes() - resident : 0;

  std::printf("footprint\n");
  std::printf("  %-34s %6zu\n", "referenced tensor slots", slots);
  std::printf("  %-34s %6zu\n", "distinct tensors", w.all_tensors().size());
  std::printf("  %-34s %12zu bytes (%s)\n", "checkpoint data section", st.data_bytes(),
              human_bytes(st.data_bytes()).c_str());
  std::printf("  %-34s %12zu bytes (%s)\n", "resident weights (mixed dtype)", resident,
              human_bytes(resident).c_str());
  if (not_loaded > 0) {
    std::printf("  %-34s %12zu bytes (%s)\n", "  not loaded (deduplicated)", not_loaded,
                human_bytes(not_loaded).c_str());
  }
  std::printf("  %-34s %12zu bytes (%s)\n", "resident weights as f32", resident * 2,
              human_bytes(resident * 2).c_str());

  if (show_tied) {
    std::printf("\nweight tying\n");
    std::printf("  tie_word_embeddings              %s\n",
                cfg.tie_word_embeddings ? "true" : "false");
    std::printf("  lm_head stored on disk           %s\n", w.lm_head_on_disk() ? "yes" : "no");
    std::printf("  lm_head bound to                 %s\n",
                w.lm_head_is_tied() ? "model.embed_tokens.weight (aliased)"
                                    : "lm_head.weight (separate)");
    if (w.lm_head_is_tied() && w.lm_head_on_disk()) {
      std::printf("                                   the on-disk copy is ignored\n");
    }
  }
  return 0;
}

}  // namespace cli
}  // namespace llmrt
