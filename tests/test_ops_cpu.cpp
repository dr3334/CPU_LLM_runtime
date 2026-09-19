#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "llmrt/common.h"
#include "llmrt/ops.h"
#include "llmrt/safetensors.h"
#include "llmrt/tensor.h"

#include "golden.h"
#include "test_framework.h"

using namespace llmrt;

namespace {

// A Tensor is only ever a view, so binding one to a const buffer means casting
// away const. Safe here: the ops under test never write through their inputs,
// and the destination vector is genuinely non-const at the call site.
Tensor view_of(const std::vector<float>& v, std::vector<int64_t> shape) {
  return Tensor::contiguous(const_cast<float*>(v.data()), DType::F32, DeviceKind::CPU,
                            std::move(shape));
}

// Runs one rmsnorm and returns the output.
//
// The op takes its destination by non-const reference so that the write is
// explicit at the call site (and so the signature already matches the planned
// IBackend method). That means the three views have to be named locals -- an
// rvalue Tensor cannot bind to Tensor&.
std::vector<float> run_rmsnorm(const std::vector<float>& x,
                               const std::vector<int64_t>& x_shape,
                               const std::vector<float>& w, float eps = 1e-6f) {
  std::vector<float> out(x.size());
  Tensor xv = view_of(x, x_shape);
  Tensor wv = view_of(w, {static_cast<int64_t>(w.size())});
  Tensor ov = view_of(out, x_shape);
  cpu::rmsnorm(xv, wv, ov, eps);
  return out;
}

// True when rmsnorm rejects the arguments instead of computing something wrong.
bool rmsnorm_rejects(const std::vector<float>& x, const std::vector<int64_t>& x_shape,
                     const std::vector<float>& w, const std::vector<int64_t>& out_shape,
                     float eps = 1e-6f) {
  std::vector<float> out(x.size(), 0.0f);
  Tensor xv = view_of(x, x_shape);
  Tensor wv = view_of(w, {static_cast<int64_t>(w.size())});
  Tensor ov = view_of(out, out_shape);
  try {
    cpu::rmsnorm(xv, wv, ov, eps);
  } catch (const Error&) {
    return true;
  }
  return false;
}

std::string model_path(const char* file) {
  return golden::default_model_dir() + "/" + file;
}

bool checkpoint_exists() { return std::ifstream(model_path("model.safetensors")).good(); }

}  // namespace

// ---------------------------------------------------------------------------
// Golden comparison: the real bar
// ---------------------------------------------------------------------------

// input_layernorm: normalises over hidden_size = 1024.
LLMRT_TEST(rmsnorm_matches_golden_input_layernorm) {
  golden::Store g;
  if (!g.available() || !checkpoint_exists()) {
    std::printf("      (skipped: fixtures or checkpoint missing)\n");
    return;
  }
  const golden::Array* in = g.get("input_layernorm_in");
  const golden::Array* want = g.get("input_layernorm_out");
  CHECK_TRUE(in != nullptr && want != nullptr);
  CHECK_TRUE(in->shape == want->shape);

  const SafeTensors st = SafeTensors::open(model_path("model.safetensors"));
  const std::vector<float> w = st.to_f32("model.layers.0.input_layernorm.weight");
  CHECK_EQ(w.size(), size_t{1024});

  const std::vector<float> out = run_rmsnorm(in->f32, in->shape, w);

  const golden::Diff d = golden::compare(out.data(), want->f(), want->numel());
  std::printf("      input_layernorm  %s\n", golden::diff_string(d).c_str());
  CHECK_MSG(golden::within(d), "input_layernorm: " + golden::diff_string(d));
}

// QK-Norm: the SAME kernel, but normalising over head_dim = 128 while the row
// is 2048 wide. This is the test that catches normalising over the wrong axis,
// the mistake that makes Qwen3 output plausible-looking garbage.
LLMRT_TEST(rmsnorm_matches_golden_q_norm_over_head_dim) {
  golden::Store g;
  if (!g.available() || !checkpoint_exists()) return;
  const golden::Array* proj = g.get("q_proj_out");  // [1, 12, 2048]
  const golden::Array* want = g.get("q_norm_out");  // [1, 12, 16, 128]
  CHECK_TRUE(proj != nullptr && want != nullptr);

  const SafeTensors st = SafeTensors::open(model_path("model.safetensors"));
  const std::vector<float> w = st.to_f32("model.layers.0.self_attn.q_norm.weight");
  CHECK_EQ(w.size(), size_t{128});

  // [1, 12, 2048] -> [1, 12, 16, 128] is a free reshape (contiguous).
  const std::vector<float> out = run_rmsnorm(proj->f32, {1, 12, 16, 128}, w);

  const golden::Diff d = golden::compare(out.data(), want->f(), want->numel());
  std::printf("      q_norm (head_dim) %s\n", golden::diff_string(d).c_str());
  CHECK_MSG(golden::within(d), "q_norm: " + golden::diff_string(d));
}

// Negative control: prove the test above can actually fail. Normalising the
// same data over 2048 instead of per-128-head must not reproduce the golden.
LLMRT_TEST(rmsnorm_over_the_wrong_axis_does_not_match) {
  golden::Store g;
  if (!g.available() || !checkpoint_exists()) return;
  const golden::Array* proj = g.get("q_proj_out");
  const golden::Array* want = g.get("q_norm_out");
  CHECK_TRUE(proj != nullptr && want != nullptr);

  // The plausible-but-wrong implementation: normalise the whole 2048-wide row.
  const std::vector<float> w(2048, 1.0f);
  const std::vector<float> out = run_rmsnorm(proj->f32, proj->shape, w);

  const golden::Diff d = golden::compare(out.data(), want->f(), want->numel());
  CHECK_MSG(!golden::within(d),
            "normalising over 2048 unexpectedly matched the per-head golden, so the "
            "q_norm test would have no discriminating power: " + golden::diff_string(d));
}

LLMRT_TEST(rmsnorm_matches_golden_k_norm) {
  golden::Store g;
  if (!g.available() || !checkpoint_exists()) return;
  const golden::Array* proj = g.get("k_proj_out");  // [1, 12, 1024]
  const golden::Array* want = g.get("k_norm_out");  // [1, 12, 8, 128]
  CHECK_TRUE(proj != nullptr && want != nullptr);

  const SafeTensors st = SafeTensors::open(model_path("model.safetensors"));
  const std::vector<float> w = st.to_f32("model.layers.0.self_attn.k_norm.weight");
  CHECK_EQ(w.size(), size_t{128});

  const std::vector<float> out = run_rmsnorm(proj->f32, {1, 12, 8, 128}, w);

  const golden::Diff d = golden::compare(out.data(), want->f(), want->numel());
  std::printf("      k_norm (head_dim) %s\n", golden::diff_string(d).c_str());
  CHECK_MSG(golden::within(d), "k_norm: " + golden::diff_string(d));
}

// ---------------------------------------------------------------------------
// Properties that hold regardless of the reference
// ---------------------------------------------------------------------------

LLMRT_TEST(rmsnorm_normalises_to_unit_rms_when_weight_is_ones) {
  // out_j = x_j / sqrt(mean(x^2) + eps)  =>  mean(out^2) = m / (m + eps) ~ 1
  std::vector<float> x(64);
  for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(static_cast<float>(i) * 1.7f) * 3.0f;
  const std::vector<float> w(64, 1.0f);

  const std::vector<float> out = run_rmsnorm(x, {1, 64}, w);

  double mean_sq = 0.0;
  for (const float v : out) mean_sq += static_cast<double>(v) * v;
  mean_sq /= static_cast<double>(out.size());
  CHECK_NEAR(mean_sq, 1.0, 1e-5);
}

LLMRT_TEST(rmsnorm_is_linear_in_the_weight) {
  std::vector<float> x(32), w1(32), w2(32);
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = static_cast<float>(i) - 16.0f;
    w1[i] = 0.5f + 0.01f * static_cast<float>(i);
    w2[i] = 2.0f * w1[i];
  }
  const std::vector<float> o1 = run_rmsnorm(x, {1, 32}, w1);
  const std::vector<float> o2 = run_rmsnorm(x, {1, 32}, w2);
  for (size_t i = 0; i < o1.size(); ++i) CHECK_NEAR(o2[i], 2.0 * o1[i], 1e-6);
}

LLMRT_TEST(rmsnorm_treats_leading_axes_as_independent_rows) {
  // Two rows with very different magnitudes must not influence each other: if
  // the reduction accidentally spanned all 8 elements, both rows would come
  // out at ~0.28 instead of ~1.
  const std::vector<float> x = {1.0f, 1.0f, 1.0f, 1.0f,      // rms 1
                                10.0f, 10.0f, 10.0f, 10.0f};  // rms 10
  const std::vector<float> w(4, 1.0f);

  const std::vector<float> out = run_rmsnorm(x, {2, 4}, w);

  for (int r = 0; r < 2; ++r) {
    for (int j = 0; j < 4; ++j) {
      CHECK_NEAR(out[static_cast<size_t>(r * 4 + j)], 1.0, 1e-5);
    }
  }
}

LLMRT_TEST(rmsnorm_handles_rank_1_input) {
  const std::vector<float> x = {3.0f, 4.0f};  // mean(x^2) = 12.5
  const std::vector<float> w(2, 1.0f);
  const std::vector<float> out = run_rmsnorm(x, {2}, w);
  const double scale = 1.0 / std::sqrt(12.5 + 1e-6);
  CHECK_NEAR(out[0], 3.0 * scale, 1e-6);
  CHECK_NEAR(out[1], 4.0 * scale, 1e-6);
}

// ---------------------------------------------------------------------------
// Argument validation: wrong input must fail loudly
// ---------------------------------------------------------------------------

LLMRT_TEST(rmsnorm_rejects_weight_length_that_does_not_match_the_axis) {
  const std::vector<float> x(8, 1.0f);
  const std::vector<float> short_w(3, 1.0f);
  CHECK_TRUE(rmsnorm_rejects(x, {2, 4}, short_w, {2, 4}));
}

LLMRT_TEST(rmsnorm_rejects_output_shape_mismatch) {
  const std::vector<float> x(8, 1.0f);
  const std::vector<float> w(4, 1.0f);
  // Same element count, different shape: must still be refused.
  CHECK_TRUE(rmsnorm_rejects(x, {2, 4}, w, {4, 2}));
}

LLMRT_TEST(rmsnorm_rejects_non_positive_eps) {
  const std::vector<float> x(8, 1.0f);
  const std::vector<float> w(4, 1.0f);
  CHECK_TRUE(rmsnorm_rejects(x, {2, 4}, w, {2, 4}, 0.0f));
}

LLMRT_TEST(rmsnorm_rejects_rank_0_input) {
  // A scalar has no axis to normalise over.
  const std::vector<float> scalar(1, 1.0f);
  const std::vector<float> w(1, 1.0f);
  CHECK_TRUE(rmsnorm_rejects(scalar, {}, w, {}));
}

LLMRT_TEST(rmsnorm_rejects_strided_input) {
  // A transposed view claims a dense shape it does not have. Reading it as
  // dense would silently normalise the wrong elements.
  std::vector<float> x(8, 1.0f);
  std::vector<float> out(8, 0.0f);
  const std::vector<float> w(4, 1.0f);
  Tensor xv = view_of(x, {2, 4}).transpose(0, 1);
  Tensor wv = view_of(w, {4});
  Tensor ov = view_of(out, {4, 2});
  bool threw = false;
  try {
    cpu::rmsnorm(xv, wv, ov, 1e-6f);
  } catch (const Error& e) {
    threw = true;
    CHECK_MSG(std::string(e.what()).find("rmsnorm") != std::string::npos, e.what());
  }
  CHECK_TRUE(threw);
}

LLMRT_TEST(rmsnorm_rejects_wrong_dtype) {
  std::vector<float> x(8, 1.0f);
  std::vector<float> out(8, 0.0f);
  const std::vector<float> w(4, 1.0f);
  Tensor bf16 = Tensor::contiguous(x.data(), DType::BF16, DeviceKind::CPU, {2, 4});
  Tensor wv = view_of(w, {4});
  Tensor ov = view_of(out, {2, 4});
  bool threw = false;
  try {
    cpu::rmsnorm(bf16, wv, ov, 1e-6f);
  } catch (const Error&) {
    threw = true;
  }
  CHECK_TRUE(threw);
}

int main() { return llmrt_test::run_all("ops_cpu"); }
