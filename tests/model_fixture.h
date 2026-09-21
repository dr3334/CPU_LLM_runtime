// Shared setup for the tests that need a real model loaded.
//
// Loading the checkpoint and building the 2.4 GB f32 weight mirror costs about
// six seconds, and every full forward after that costs another second or two, so
// these suites are the slow ones. The fixture is shared between test_forward and
// test_generate rather than duplicated.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "llmrt/config.h"
#include "llmrt/forward.h"
#include "llmrt/model.h"
#include "llmrt/safetensors.h"

#include "golden.h"

namespace llmrt_test {

inline std::string model_path(const char* file) {
  return golden::default_model_dir() + "/" + file;
}

inline bool checkpoint_exists() {
  return std::ifstream(model_path("model.safetensors")).good();
}

// Built once per test binary and shared by every test in it.
//
// All three objects have to outlive the forward pass -- the SafeTensors mmap in
// particular, since the binding stores pointers into it -- so they are static
// inside the initialiser rather than locals of it.
inline llmrt::Qwen3Forward* shared_forward() {
  static llmrt::Qwen3Forward* instance = []() -> llmrt::Qwen3Forward* {
    golden::Store g;
    if (!g.available() || !checkpoint_exists()) return nullptr;
    static const llmrt::SafeTensors st =
        llmrt::SafeTensors::open(model_path("model.safetensors"));
    static const llmrt::Qwen3Config cfg =
        llmrt::Qwen3Config::from_model_dir(golden::default_model_dir());
    static const llmrt::Qwen3Weights weights = llmrt::Qwen3Weights::bind(st, cfg);
    return new llmrt::Qwen3Forward(st, weights);
  }();
  return instance;
}

// How many greedy steps to verify. golden has 16; each is a full forward pass,
// so the default keeps the suite usable and the full 7x16 = 112-pass check is
// opt-in:
//
//     LLMRT_GREEDY_STEPS=16 ./build/tests/test_generate
//
inline int64_t greedy_steps() {
  if (const char* e = std::getenv("LLMRT_GREEDY_STEPS")) {
    const int v = std::atoi(e);
    if (v > 0) return v;
  }
  return 3;
}

}  // namespace llmrt_test
