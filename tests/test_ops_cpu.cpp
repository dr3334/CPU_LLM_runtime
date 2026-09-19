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

// Independent reimplementation that accumulates the sum of squares in double,
// mirroring the alternative implementation under discussion. Used only to
// measure the accumulator-width choice against the golden.
std::vector<float> rmsnorm_double_accumulator(const std::vector<float>& x,
                                              const std::vector<int64_t>& x_shape,
                                              const std::vector<float>& w, float eps) {
  const int64_t n = x_shape.back();
  const int64_t rows = static_cast<int64_t>(x.size()) / n;
  std::vector<float> out(x.size());
  for (int64_t r = 0; r < rows; ++r) {
    const float* row = x.data() + r * n;
    double sum = 0.0;
    for (int64_t j = 0; j < n; ++j) {
      const double v = row[j];
      sum += v * v;
    }
    const float mean = static_cast<float>(sum / static_cast<double>(n));
    const float scale = 1.0f / std::sqrt(mean + eps);
    for (int64_t j = 0; j < n; ++j) out[r * n + j] = w[j] * (row[j] * scale);
  }
  return out;
}

// Runs one matmul and returns the output [M, N].
std::vector<float> run_matmul(const std::vector<float>& a, const std::vector<int64_t>& a_shape,
                              const std::vector<float>& b, const std::vector<int64_t>& b_shape,
                              bool transpose_b) {
  const int64_t M = a_shape[0];
  const int64_t N = transpose_b ? b_shape[0] : b_shape[1];
  std::vector<float> out(static_cast<size_t>(M * N));
  Tensor at = view_of(a, a_shape);
  Tensor bt = view_of(b, b_shape);
  Tensor ot = view_of(out, {M, N});
  cpu::matmul(at, bt, ot, transpose_b);
  return out;
}

// Checks one Qwen3 projection against its captured golden output. `in_key` is a
// [1, 12, K] activation, `weight_name` a [N, K] nn.Linear weight, `want_key` the
// [1, 12, N] result.
void check_projection(const golden::Store& g, const SafeTensors& st,
                      const std::string& in_key, const std::string& weight_name,
                      const std::string& want_key, int64_t K, int64_t N,
                      const char* label) {
  const golden::Array* in = g.get(in_key);
  const golden::Array* want = g.get(want_key);
  CHECK_TRUE(in != nullptr && want != nullptr);
  if (in == nullptr || want == nullptr) return;

  const std::vector<float> w = st.to_f32(weight_name);
  CHECK_MSG(w.size() == static_cast<size_t>(K) * static_cast<size_t>(N),
            std::string(label) + ": weight has unexpected size");

  // [1, 12, K] -> [12, K] is a free reshape (contiguous).
  const std::vector<float> out = run_matmul(in->f32, {12, K}, w, {N, K}, true);

  const golden::Diff d = golden::compare(out.data(), want->f(), want->numel());
  std::printf("      %-10s M=12 K=%-5lld N=%-5lld  %s\n", label, (long long)K, (long long)N,
              golden::diff_string(d).c_str());
  CHECK_MSG(golden::within(d), std::string(label) + ": " + golden::diff_string(d));
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

// The accumulator width is a design choice worth measuring rather than arguing
// about. A double accumulator is more accurate in the mathematical sense, but
// the golden was itself produced with fp32 reductions -- so "more accurate than
// the reference" is not the same as "closer to the reference". Both are checked
// so the numbers, not the intuition, decide.
LLMRT_TEST(rmsnorm_accumulator_width_float_vs_double) {
  golden::Store g;
  if (!g.available() || !checkpoint_exists()) return;
  const golden::Array* in = g.get("input_layernorm_in");
  const golden::Array* want = g.get("input_layernorm_out");
  CHECK_TRUE(in != nullptr && want != nullptr);

  const SafeTensors st = SafeTensors::open(model_path("model.safetensors"));
  const std::vector<float> w = st.to_f32("model.layers.0.input_layernorm.weight");

  const std::vector<float> f = run_rmsnorm(in->f32, in->shape, w);
  const std::vector<float> d = rmsnorm_double_accumulator(in->f32, in->shape, w, 1e-6f);

  const golden::Diff df = golden::compare(f.data(), want->f(), want->numel());
  const golden::Diff dd = golden::compare(d.data(), want->f(), want->numel());
  std::printf("      float  accumulator  %s\n", golden::diff_string(df).c_str());
  std::printf("      double accumulator  %s\n", golden::diff_string(dd).c_str());

  // Only the accumulator differs, so the two must agree to within its error.
  const golden::Diff dff = golden::compare(d.data(), f.data(), f.size());
  std::printf("      float vs double     %s\n", golden::diff_string(dff).c_str());

  CHECK_MSG(golden::within(df), "float accumulator: " + golden::diff_string(df));
  CHECK_MSG(golden::within(dd), "double accumulator: " + golden::diff_string(dd));
}

// ---------------------------------------------------------------------------
// matmul -- the Qwen3 projection path, out = a @ b^T
// ---------------------------------------------------------------------------

// All four distinct projection shapes are checked, because between them they
// exercise K = 1024/2048/3072 and N = 1024/2048/3072. If the indexing ever
// confuses M, N and K, at least one of these disagrees.
LLMRT_TEST(matmul_matches_golden_q_proj) {
  golden::Store g;
  if (!g.available() || !checkpoint_exists()) return;
  const SafeTensors st = SafeTensors::open(model_path("model.safetensors"));
  // q_proj makes the head dimension visible: 1024 in, 2048 = 16 heads x 128 out.
  check_projection(g, st, "input_layernorm_out", "model.layers.0.self_attn.q_proj.weight",
                   "q_proj_out", 1024, 2048, "q_proj");
}

LLMRT_TEST(matmul_matches_golden_o_proj) {
  // o_proj projects the other way: 2048 (merged heads) back down to 1024.
  golden::Store g;
  if (!g.available() || !checkpoint_exists()) return;
  const SafeTensors st = SafeTensors::open(model_path("model.safetensors"));
  check_projection(g, st, "attn_out", "model.layers.0.self_attn.o_proj.weight", "o_proj_out",
                   2048, 1024, "o_proj");
}

LLMRT_TEST(matmul_matches_golden_gate_proj) {
  golden::Store g;
  if (!g.available() || !checkpoint_exists()) return;
  const SafeTensors st = SafeTensors::open(model_path("model.safetensors"));
  check_projection(g, st, "post_attention_layernorm_out",
                   "model.layers.0.mlp.gate_proj.weight", "gate_proj_out", 1024, 3072,
                   "gate_proj");
}

LLMRT_TEST(matmul_matches_golden_down_proj) {
  // The widest K in the model.
  golden::Store g;
  if (!g.available() || !checkpoint_exists()) return;
  const SafeTensors st = SafeTensors::open(model_path("model.safetensors"));
  check_projection(g, st, "swiglu_out", "model.layers.0.mlp.down_proj.weight", "down_proj_out",
                   3072, 1024, "down_proj");
}

// ---------------------------------------------------------------------------
// matmul properties
// ---------------------------------------------------------------------------

LLMRT_TEST(matmul_produces_hand_computed_values) {
  // a = [[1,2],[3,4]]   b = [[5,6],[7,8]] stored as [N=2, K=2]
  // a @ b^T = [[1*5+2*6, 1*7+2*8], [3*5+4*6, 3*7+4*8]] = [[17,23],[39,53]]
  const std::vector<float> a = {1, 2, 3, 4};
  const std::vector<float> b = {5, 6, 7, 8};
  const std::vector<float> out = run_matmul(a, {2, 2}, b, {2, 2}, true);
  CHECK_NEAR(out[0], 17.0, 1e-6);
  CHECK_NEAR(out[1], 23.0, 1e-6);
  CHECK_NEAR(out[2], 39.0, 1e-6);
  CHECK_NEAR(out[3], 53.0, 1e-6);
}

LLMRT_TEST(matmul_by_identity_returns_the_other_operand) {
  const std::vector<float> a = {1, 2, 3, 4, 5, 6};            // [2, 3]
  const std::vector<float> eye = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // [3, 3]
  const std::vector<float> out = run_matmul(a, {2, 3}, eye, {3, 3}, true);
  for (size_t i = 0; i < a.size(); ++i) CHECK_NEAR(out[i], a[i], 1e-6);
}

// Cross-checks the two branches against each other. A bug in either indexing
// scheme shows up as a disagreement, and the two use completely different loop
// orders (dot-product vs outer-product) so they are unlikely to be wrong in
// the same way.
LLMRT_TEST(matmul_both_transpose_modes_agree) {
  const int64_t M = 3, K = 4, N = 5;
  std::vector<float> a(static_cast<size_t>(M * K));
  std::vector<float> b(static_cast<size_t>(N * K));
  for (size_t i = 0; i < a.size(); ++i) a[i] = 0.5f * static_cast<float>(i) - 1.0f;
  for (size_t i = 0; i < b.size(); ++i) b[i] = 1.0f - 0.25f * static_cast<float>(i);

  // Materialise b^T as [K, N] so the same product can be computed both ways.
  std::vector<float> b_t(static_cast<size_t>(K * N));
  for (int64_t n = 0; n < N; ++n)
    for (int64_t k = 0; k < K; ++k)
      b_t[static_cast<size_t>(k * N + n)] = b[static_cast<size_t>(n * K + k)];

  const std::vector<float> o_tb = run_matmul(a, {M, K}, b, {N, K}, true);
  const std::vector<float> o_tf = run_matmul(a, {M, K}, b_t, {K, N}, false);

  const golden::Diff d = golden::compare(o_tf.data(), o_tb.data(), o_tb.size());
  CHECK_MSG(golden::within(d), "transpose_b branches disagree: " + golden::diff_string(d));
}

LLMRT_TEST(matmul_supports_a_single_row_decode) {
  // M=1 is the decode shape: one token at a time. This is the hot path, and it
  // is memory bound on streaming the weight.
  const int64_t K = 8, N = 4;
  std::vector<float> a(K);
  std::vector<float> b(static_cast<size_t>(N * K));
  for (size_t i = 0; i < a.size(); ++i) a[i] = 1.0f;  // all ones: out[j] = sum of b row j
  for (size_t i = 0; i < b.size(); ++i) b[i] = static_cast<float>(i);

  const std::vector<float> out = run_matmul(a, {1, K}, b, {N, K}, true);
  CHECK_EQ(out.size(), size_t{4});
  for (int64_t j = 0; j < N; ++j) {
    float row_sum = 0.0f;
    for (int64_t k = 0; k < K; ++k) row_sum += b[static_cast<size_t>(j * K + k)];
    CHECK_NEAR(out[static_cast<size_t>(j)], row_sum, 1e-5);
  }
}

// ---------------------------------------------------------------------------
// matmul argument validation
// ---------------------------------------------------------------------------

LLMRT_TEST(matmul_rejects_disagreeing_inner_dimensions) {
  const std::vector<float> a = {1, 2, 3, 4, 5, 6};  // [2, 3], so K=3
  const std::vector<float> b(20, 0.0f);             // [4, 5], inner dim 5 != 3
  std::vector<float> out(8, 0.0f);                  // [2, 4]
  Tensor at = view_of(a, {2, 3});
  Tensor bt = view_of(b, {4, 5});
  Tensor ot = view_of(out, {2, 4});
  bool threw = false;
  try {
    cpu::matmul(at, bt, ot, true);
  } catch (const Error& e) {
    threw = true;
    CHECK_MSG(std::string(e.what()).find("inner dimensions") != std::string::npos, e.what());
  }
  CHECK_TRUE(threw);
}

LLMRT_TEST(matmul_rejects_output_shape_mismatch) {
  const std::vector<float> a = {1, 2, 3, 4, 5, 6};  // [2, 3]
  const std::vector<float> b = {1, 0, 0, 1, 0, 0};  // [2, 3] -> out should be [2, 2]
  std::vector<float> out(6, 0.0f);                  // [3, 2]: wrong
  Tensor at = view_of(a, {2, 3});
  Tensor bt = view_of(b, {2, 3});
  Tensor ot = view_of(out, {3, 2});
  bool threw = false;
  try {
    cpu::matmul(at, bt, ot, true);
  } catch (const Error& e) {
    threw = true;
    CHECK_MSG(std::string(e.what()).find("out should be") != std::string::npos, e.what());
  }
  CHECK_TRUE(threw);
}

LLMRT_TEST(matmul_rejects_non_2d_operands) {
  // Callers reshape higher-rank activations, so a 3-D input is a caller bug.
  std::vector<float> a(24, 1.0f), b(24, 1.0f), out(8, 0.0f);
  Tensor at = view_of(a, {2, 3, 4});
  Tensor bt = view_of(b, {4, 6});
  Tensor ot = view_of(out, {2, 4});
  bool threw = false;
  try {
    cpu::matmul(at, bt, ot, true);
  } catch (const Error&) {
    threw = true;
  }
  CHECK_TRUE(threw);
}

LLMRT_TEST(matmul_rejects_strided_input) {
  std::vector<float> a(6, 1.0f), b(6, 1.0f), out(4, 0.0f);
  Tensor at = view_of(a, {2, 3}).transpose(0, 1);  // strided, [3, 2]
  Tensor bt = view_of(b, {2, 3});
  Tensor ot = view_of(out, {2, 2});
  bool threw = false;
  try {
    cpu::matmul(at, bt, ot, true);
  } catch (const Error& e) {
    threw = true;
    CHECK_MSG(std::string(e.what()).find("matmul") != std::string::npos, e.what());
  }
  CHECK_TRUE(threw);
}

int main() { return llmrt_test::run_all("ops_cpu"); }
