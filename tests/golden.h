// Loader for the fixtures under data/golden, shared by the op/model tests.
//
// The manifest lists every array's dtype, shape and file; arrays are read
// lazily on first use and cached. Tests that need the fixtures skip cleanly
// when they are absent (run tools/gen_oracle.py to create them).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "llmrt/common.h"
#include "llmrt/json.h"

namespace golden {

inline std::string default_dir() {
  if (const char* env = std::getenv("LLMRT_GOLDEN_DIR")) return env;
  return std::string(LLMRT_SOURCE_DIR) + "/data/golden";
}

inline std::string default_model_dir() {
  if (const char* env = std::getenv("LLMRT_MODEL_DIR")) return env;
  return "/home/dr/models/Qwen3-0.6B";
}

struct Array {
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<float> f32;
  std::vector<int32_t> i32;

  size_t numel() const {
    size_t n = 1;
    for (const int64_t d : shape) n *= static_cast<size_t>(d);
    return n;
  }
  const float* f() const { return f32.data(); }
  const int32_t* i() const { return i32.data(); }
};

class Store {
 public:
  explicit Store(std::string dir = default_dir()) : dir_(std::move(dir)) {
    const std::string manifest_path = dir_ + "/manifest.json";
    std::ifstream probe(manifest_path);
    available_ = probe.good();
    probe.close();
    if (available_) manifest_ = llmrt::json::parse_file(manifest_path);
  }

  bool available() const { return available_; }
  const std::string& dir() const { return dir_; }
  const llmrt::json::Value& manifest() const { return manifest_; }

  // Returns nullptr when the array or the fixture set is absent.
  const Array* get(const std::string& name) const {
    const auto it = cache_.find(name);
    if (it != cache_.end()) return it->second.get();

    std::shared_ptr<Array> arr;
    const llmrt::json::Value* info = manifest_.find("arrays");
    if (info != nullptr) {
      if (const llmrt::json::Value* e = info->find(name)) {
        arr = std::make_shared<Array>();
        arr->dtype = e->get_string("dtype");
        if (const llmrt::json::Value::Array* shape = e->get_array("shape")) {
          for (const llmrt::json::Value& d : *shape) arr->shape.push_back(d.as_int());
        }
        const std::string path = dir_ + "/" + e->get_string("file");
        std::ifstream in(path, std::ios::binary);
        LLMRT_CHECK(in.good(), "golden: cannot open " + path);
        const size_t n = arr->numel();
        if (arr->dtype == "f32") {
          arr->f32.resize(n);
          in.read(reinterpret_cast<char*>(arr->f32.data()),
                  static_cast<std::streamsize>(n * sizeof(float)));
        } else {
          arr->i32.resize(n);
          in.read(reinterpret_cast<char*>(arr->i32.data()),
                  static_cast<std::streamsize>(n * sizeof(int32_t)));
        }
      }
    }
    cache_[name] = arr;
    return arr.get();
  }

  // Read a 1-D int32 array (e.g. input_ids).
  const int32_t* ids(const std::string& name, size_t* count) const {
    const Array* a = get(name);
    if (a == nullptr) return nullptr;
    *count = a->numel();
    return a->i();
  }

  // Per-prompt token offsets, as recorded when the fixtures were generated.
  std::vector<int64_t> prompt_offsets() const {
    std::vector<int64_t> out;
    if (const llmrt::json::Value::Array* v = manifest_.get_array("prompt_offsets")) {
      for (const llmrt::json::Value& e : *v) out.push_back(e.as_int());
    }
    return out;
  }

  std::vector<int64_t> prompt_lengths() const {
    std::vector<int64_t> out;
    if (const llmrt::json::Value::Array* v = manifest_.get_array("prompt_lengths")) {
      for (const llmrt::json::Value& e : *v) out.push_back(e.as_int());
    }
    return out;
  }

  // How many of each greedy_ids row are real. The array is right-padded with -1
  // so that it can be rectangular; these lengths say where each sequence stops.
  std::vector<int64_t> greedy_lengths() const {
    std::vector<int64_t> out;
    if (const llmrt::json::Value::Array* v = manifest_.get_array("greedy_lengths")) {
      for (const llmrt::json::Value& e : *v) out.push_back(e.as_int());
    }
    return out;
  }

 private:
  std::string dir_;
  bool available_ = false;
  llmrt::json::Value manifest_;
  mutable std::unordered_map<std::string, std::shared_ptr<Array>> cache_;
};

// ---------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------

struct Diff {
  double cosine = 0.0;
  double max_abs = 0.0;
  double rel = 0.0;  // max_abs / max|expected|
  size_t n = 0;
};

inline Diff compare(const float* got, const float* want, size_t n) {
  Diff d;
  d.n = n;
  double dot = 0.0, na = 0.0, nb = 0.0, max_abs = 0.0, max_ref = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double a = got[i];
    const double b = want[i];
    dot += a * b;
    na += a * a;
    nb += b * b;
    max_abs = std::max(max_abs, std::fabs(a - b));
    max_ref = std::max(max_ref, std::fabs(b));
  }
  const double denom = std::sqrt(na) * std::sqrt(nb);
  d.cosine = denom > 0.0 ? dot / denom : 1.0;
  d.max_abs = max_abs;
  d.rel = max_abs / (max_ref + 1e-12);
  return d;
}

// Tolerance tiers, calibrated from the observed fp32 round-off of an
// independent re-derivation of the forward pass (see tools/check_oracle.py):
// intermediates land at rel ~5e-7, full-stack logits at rel ~2e-6.
struct Tolerance {
  double cosine = 0.99999;
  double rel = 1e-4;
};

inline bool within(const Diff& d, const Tolerance& tol = {}) {
  return d.cosine > tol.cosine && d.rel < tol.rel;
}

inline std::string diff_string(const Diff& d) {
  char buf[160];
  std::snprintf(buf, sizeof(buf), "cos=%.8f maxabs=%.3e rel=%.3e (n=%zu)", d.cosine,
                d.max_abs, d.rel, d.n);
  return buf;
}

}  // namespace golden
