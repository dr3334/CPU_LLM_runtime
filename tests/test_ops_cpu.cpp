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

// ---------------------------------------------------------------------------
// swiglu -- out[i] = silu(gate[i]) * up[i]
//
// NOTE: these fail against the unimplemented stub in src/ops/cpu/swiglu.cpp.
// They are the target to code against.
// ---------------------------------------------------------------------------

LLMRT_TEST(swiglu_matches_golden) {
  golden::Store g;
  if (!g.available()) {
    std::printf("      (skipped: fixtures missing)\n");
    return;
  }
  const golden::Array* gate = g.get("gate_proj_out");  // [1, 12, 3072]
  const golden::Array* up = g.get("up_proj_out");      // [1, 12, 3072]
  const golden::Array* want = g.get("swiglu_out");     // [1, 12, 3072]
  CHECK_TRUE(gate != nullptr && up != nullptr && want != nullptr);
  if (gate == nullptr || up == nullptr || want == nullptr) return;
  CHECK_TRUE(gate->shape == up->shape);
  CHECK_TRUE(gate->shape == want->shape);

  std::vector<float> out(gate->numel());
  Tensor gt = view_of(gate->f32, gate->shape);
  Tensor ut = view_of(up->f32, up->shape);
  Tensor ot = view_of(out, want->shape);
  cpu::swiglu(gt, ut, ot);

  const golden::Diff d = golden::compare(out.data(), want->f(), want->numel());
  std::printf("      swiglu  %s\n", golden::diff_string(d).c_str());
  CHECK_MSG(golden::within(d), "swiglu: " + golden::diff_string(d));
}

// This one does not call the op at all: swiglu_out and down_proj_in were
// captured from two different hooks and must be the same tensor, so a
// disagreement would mean the fixtures themselves are broken.
LLMRT_TEST(swiglu_fixtures_swiglu_out_and_down_proj_in_agree) {
  golden::Store g;
  if (!g.available()) return;
  const golden::Array* a = g.get("swiglu_out");
  const golden::Array* b = g.get("down_proj_in");
  CHECK_TRUE(a != nullptr && b != nullptr);
  if (a == nullptr || b == nullptr) return;

  CHECK_EQ(a->numel(), b->numel());
  CHECK_EQ(std::memcmp(a->f32.data(), b->f32.data(), a->numel() * sizeof(float)), 0);
}

LLMRT_TEST(swiglu_hand_computed_values) {
  // silu(0) = 0, silu(1) = 1/(1+e^-1), silu(-1) = -1/(1+e^1)
  const std::vector<float> gate = {0.0f, 1.0f, -1.0f};
  const std::vector<float> up = {1.0f, 1.0f, 1.0f};
  std::vector<float> out(3);
  Tensor gt = view_of(gate, {3});
  Tensor ut = view_of(up, {3});
  Tensor ot = view_of(out, {3});
  cpu::swiglu(gt, ut, ot);

  CHECK_NEAR(out[0], 0.0, 1e-6);
  CHECK_NEAR(out[1], 0.7310586, 1e-6);
  CHECK_NEAR(out[2], -0.2689414, 1e-6);
}

LLMRT_TEST(swiglu_handles_negative_gate_without_overflow) {
  // Large negative gate: exp(-x) overflows to +inf, and x/(1+inf) must give a
  // clean 0 rather than NaN. Large positive must saturate to x itself.
  const std::vector<float> gate = {-100.0f, 100.0f};
  const std::vector<float> up = {1.0f, 1.0f};
  std::vector<float> out(2);
  Tensor gt = view_of(gate, {2});
  Tensor ut = view_of(up, {2});
  Tensor ot = view_of(out, {2});
  cpu::swiglu(gt, ut, ot);

  CHECK_TRUE(out[0] == out[0]);  // not NaN
  CHECK_NEAR(out[0], 0.0, 1e-6);
  CHECK_NEAR(out[1], 100.0, 1e-3);
}

LLMRT_TEST(swiglu_is_linear_in_up) {
  // out = silu(gate) * up, so scaling up scales out by the same factor.
  std::vector<float> gate(16), up(16), up2(16);
  for (size_t i = 0; i < gate.size(); ++i) {
    gate[i] = static_cast<float>(i) - 8.0f;
    up[i] = 0.5f + 0.1f * static_cast<float>(i);
    up2[i] = 3.0f * up[i];
  }
  std::vector<float> o1(16), o2(16);
  Tensor gt = view_of(gate, {16});
  Tensor ut1 = view_of(up, {16});
  Tensor ut2 = view_of(up2, {16});
  Tensor ot1 = view_of(o1, {16});
  Tensor ot2 = view_of(o2, {16});
  cpu::swiglu(gt, ut1, ot1);
  cpu::swiglu(gt, ut2, ot2);

  // Without this, a no-op implementation would "pass": it writes zeros both
  // times, and 0 == 3*0.
  bool any_nonzero = false;
  for (const float v : o1) {
    if (v != 0.0f) any_nonzero = true;
  }
  CHECK_MSG(any_nonzero, "output is all zeros -- the implementation is a no-op");

  for (size_t i = 0; i < o1.size(); ++i) CHECK_NEAR(o2[i], 3.0 * o1[i], 1e-5);
}

LLMRT_TEST(swiglu_rejects_shape_mismatch) {
  std::vector<float> gate(6, 1.0f), up(4, 1.0f), out(6, 0.0f);
  Tensor gt = view_of(gate, {2, 3});
  Tensor ut = view_of(up, {2, 2});
  Tensor ot = view_of(out, {2, 3});
  bool threw = false;
  try {
    cpu::swiglu(gt, ut, ot);
  } catch (const Error&) {
    threw = true;
  }
  CHECK_TRUE(threw);
}

LLMRT_TEST(swiglu_rejects_non_contiguous) {
  std::vector<float> gate(6, 1.0f), up(6, 1.0f), out(6, 0.0f);
  Tensor gt = view_of(gate, {2, 3}).transpose(0, 1);  // strided, [3, 2]
  Tensor ut = view_of(up, {2, 3});
  Tensor ot = view_of(out, {2, 3});
  bool threw = false;
  try {
    cpu::swiglu(gt, ut, ot);
  } catch (const Error& e) {
    threw = true;
    CHECK_MSG(std::string(e.what()).find("swiglu") != std::string::npos, e.what());
  }
  CHECK_TRUE(threw);
}

LLMRT_TEST(swiglu_rejects_wrong_dtype) {
  std::vector<float> gate(6, 1.0f), up(6, 1.0f), out(6, 0.0f);
  Tensor gt = Tensor::contiguous(gate.data(), DType::BF16, DeviceKind::CPU, {2, 3});
  Tensor ut = view_of(up, {2, 3});
  Tensor ot = view_of(out, {2, 3});
  bool threw = false;
  try {
    cpu::swiglu(gt, ut, ot);
  } catch (const Error&) {
    threw = true;
  }
  CHECK_TRUE(threw);
}

// ---------------------------------------------------------------------------
// rope -- half-split rotary position embedding
//
// NOTE: these fail against the unimplemented stubs in src/ops/cpu/rope.cpp.
// ---------------------------------------------------------------------------

namespace {

// The golden tables, flattened to [seq, head_dim].
bool load_rope_tables(const golden::Store& g, std::vector<float>* cos, std::vector<float>* sin,
                      int64_t* seq, int64_t* head_dim) {
  const golden::Array* c = g.get("rope_cos");
  const golden::Array* s = g.get("rope_sin");
  if (c == nullptr || s == nullptr) return false;
  *seq = c->shape[1];
  *head_dim = c->shape[2];
  *cos = c->f32;
  *sin = s->f32;
  return true;
}

}  // namespace

LLMRT_TEST(rope_frequencies_match_golden) {
  golden::Store g;
  if (!g.available()) {
    std::printf("      (skipped: fixtures missing)\n");
    return;
  }
  std::vector<float> want_cos, want_sin;
  int64_t seq = 0, head_dim = 0;
  if (!load_rope_tables(g, &want_cos, &want_sin, &seq, &head_dim)) return;

  std::vector<float> cos(want_cos.size()), sin(want_sin.size());
  Tensor ct = view_of(cos, {seq, head_dim});
  Tensor st = view_of(sin, {seq, head_dim});
  cpu::rope_frequencies(seq, head_dim, 1e6f, ct, st);

  const golden::Diff dc = golden::compare(cos.data(), want_cos.data(), cos.size());
  const golden::Diff ds = golden::compare(sin.data(), want_sin.data(), sin.size());
  std::printf("      rope_cos  %s\n", golden::diff_string(dc).c_str());
  std::printf("      rope_sin  %s\n", golden::diff_string(ds).c_str());
  CHECK_MSG(golden::within(dc), "rope cos: " + golden::diff_string(dc));
  CHECK_MSG(golden::within(ds), "rope sin: " + golden::diff_string(ds));
}

// q has 16 heads and k has 8 (GQA), and both are rotated with the same table --
// exactly the reference's apply_rotary_pos_emb(q, k, cos, sin). Rotating them in
// one call is what makes a broadcast bug visible.
LLMRT_TEST(rope_apply_matches_golden_q_and_k) {
  golden::Store g;
  if (!g.available()) return;
  std::vector<float> cos, sin;
  int64_t seq = 0, head_dim = 0;
  if (!load_rope_tables(g, &cos, &sin, &seq, &head_dim)) return;

  const golden::Array* q_in = g.get("rope_q_in");    // [1, 16, 12, 128]
  const golden::Array* q_out = g.get("rope_q_out");
  const golden::Array* k_in = g.get("rope_k_in");    // [1,  8, 12, 128]
  const golden::Array* k_out = g.get("rope_k_out");
  CHECK_TRUE(q_in != nullptr && q_out != nullptr && k_in != nullptr && k_out != nullptr);
  if (q_in == nullptr || q_out == nullptr || k_in == nullptr || k_out == nullptr) return;

  const int64_t q_heads = q_in->shape[1];
  const int64_t k_heads = k_in->shape[1];

  std::vector<float> q(q_in->f32), k(k_in->f32);  // in place, so work on copies
  Tensor qt = view_of(q, {q_heads, seq, head_dim});
  Tensor kt = view_of(k, {k_heads, seq, head_dim});
  Tensor ct = view_of(cos, {seq, head_dim});
  Tensor st = view_of(sin, {seq, head_dim});
  cpu::rope_apply(qt, kt, ct, st);

  const golden::Diff dq = golden::compare(q.data(), q_out->f(), q_out->numel());
  const golden::Diff dk = golden::compare(k.data(), k_out->f(), k_out->numel());
  std::printf("      rope q (16 heads)  %s\n", golden::diff_string(dq).c_str());
  std::printf("      rope k ( 8 heads)  %s\n", golden::diff_string(dk).c_str());
  CHECK_MSG(golden::within(dq), "rope q: " + golden::diff_string(dq));
  CHECK_MSG(golden::within(dk), "rope k: " + golden::diff_string(dk));
}

// Isolates rope_apply from the table generator. Two cases, because the first
// alone is passed by a no-op implementation:
//   cos =  1, sin = 0  ->  out = x    (identity)
//   cos = -1, sin = 0  ->  out = -x   (angle pi)
LLMRT_TEST(rope_apply_identity_and_pi_rotations) {
  const int64_t seq = 3, head_dim = 4;
  const size_t n = static_cast<size_t>(seq * head_dim);

  {
    std::vector<float> x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const std::vector<float> before = x;
    std::vector<float> k(n, 0.0f);
    const std::vector<float> cos(n, 1.0f), sin(n, 0.0f);
    Tensor xt = view_of(x, {1, seq, head_dim});
    Tensor kt = view_of(k, {1, seq, head_dim});
    Tensor ct = view_of(cos, {seq, head_dim});
    Tensor st = view_of(sin, {seq, head_dim});
    cpu::rope_apply(xt, kt, ct, st);
    for (size_t i = 0; i < n; ++i) CHECK_NEAR(x[i], before[i], 1e-6);
  }

  {
    std::vector<float> x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const std::vector<float> before = x;
    std::vector<float> k(n, 0.0f);
    const std::vector<float> cos(n, -1.0f), sin(n, 0.0f);
    Tensor xt = view_of(x, {1, seq, head_dim});
    Tensor kt = view_of(k, {1, seq, head_dim});
    Tensor ct = view_of(cos, {seq, head_dim});
    Tensor st = view_of(sin, {seq, head_dim});
    cpu::rope_apply(xt, kt, ct, st);
    for (size_t i = 0; i < n; ++i) CHECK_NEAR(x[i], -before[i], 1e-6);
  }
}

// Pins the half-split convention independently of the table generator.
//
// With cos = 0 and sin = 1, out[j] = rotate_half(x)[j], i.e.
//     j <  h :  out[j] = -x[j+h]
//     j >= h :  out[j] =  x[j-h]
// An interleaved implementation (pairing element 2j with 2j+1) applies a
// different permutation and fails here.
LLMRT_TEST(rope_apply_with_cosine_zero_reveals_the_pairing) {
  const int64_t head_dim = 8, seq = 1;
  std::vector<float> x = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> k(8, 0.0f);
  const std::vector<float> cos(8, 0.0f), sin(8, 1.0f);

  Tensor xt = view_of(x, {1, seq, head_dim});
  Tensor kt = view_of(k, {1, seq, head_dim});
  Tensor ct = view_of(cos, {seq, head_dim});
  Tensor st = view_of(sin, {seq, head_dim});
  cpu::rope_apply(xt, kt, ct, st);

  // First half becomes the negated second half, and vice versa.
  CHECK_NEAR(x[0], -5.0, 1e-6);
  CHECK_NEAR(x[1], -6.0, 1e-6);
  CHECK_NEAR(x[2], -7.0, 1e-6);
  CHECK_NEAR(x[3], -8.0, 1e-6);
  CHECK_NEAR(x[4], 1.0, 1e-6);
  CHECK_NEAR(x[5], 2.0, 1e-6);
  CHECK_NEAR(x[6], 3.0, 1e-6);
  CHECK_NEAR(x[7], 4.0, 1e-6);
}

// Each (j, j+h) pair undergoes a 2-D rotation, so the pair's squared norm is
// invariant. True for any table, so it catches sign and pairing errors even
// when a golden comparison would absorb them.
LLMRT_TEST(rope_apply_preserves_pair_norms) {
  const int64_t rows = 2, seq = 3, head_dim = 8, half = head_dim / 2;
  std::vector<float> x(static_cast<size_t>(rows * seq * head_dim));
  for (size_t i = 0; i < x.size(); ++i) x[i] = 0.5f * static_cast<float>(i) - 5.0f;

  // A real (non-degenerate) table, built here so this test does not depend on
  // rope_frequencies being implemented.
  std::vector<float> cos(static_cast<size_t>(seq * head_dim)), sin(cos.size());
  for (int64_t p = 0; p < seq; ++p) {
    for (int64_t j = 0; j < head_dim; ++j) {
      const float angle = 0.7f * static_cast<float>(p) + 1.1f * static_cast<float>(j);
      cos[static_cast<size_t>(p * head_dim + j)] = std::cos(angle);
      sin[static_cast<size_t>(p * head_dim + j)] = std::sin(angle);
    }
  }

  std::vector<float> k(x.size(), 0.0f);
  Tensor xt = view_of(x, {rows, seq, head_dim});
  Tensor kt = view_of(k, {rows, seq, head_dim});
  Tensor ct = view_of(cos, {seq, head_dim});
  Tensor st = view_of(sin, {seq, head_dim});

  std::vector<float> before = x;
  cpu::rope_apply(xt, kt, ct, st);

  // A no-op implementation preserves norms trivially, so require that the data
  // actually changed before trusting the invariant.
  bool changed = false;
  for (size_t i = 0; i < x.size(); ++i) {
    if (x[i] != before[i]) changed = true;
  }
  CHECK_MSG(changed, "output equals input -- the implementation is a no-op");

  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t p = 0; p < seq; ++p) {
      for (int64_t j = 0; j < half; ++j) {
        const size_t a = static_cast<size_t>((r * seq + p) * head_dim + j);
        const size_t b = static_cast<size_t>((r * seq + p) * head_dim + j + half);
        const double n_before = static_cast<double>(before[a]) * before[a] +
                                static_cast<double>(before[b]) * before[b];
        const double n_after =
            static_cast<double>(x[a]) * x[a] + static_cast<double>(x[b]) * x[b];
        CHECK_NEAR(n_after, n_before, 1e-4);
      }
    }
  }
}

LLMRT_TEST(rope_apply_rejects_head_dim_mismatch) {
  std::vector<float> q(2 * 3 * 4, 1.0f), k(2 * 3 * 4, 1.0f);
  const std::vector<float> cos(2 * 8, 1.0f), sin(2 * 8, 0.0f);  // head_dim 8 != 4
  Tensor qt = view_of(q, {2, 3, 4});
  Tensor kt = view_of(k, {2, 3, 4});
  Tensor ct = view_of(cos, {2, 8});
  Tensor st = view_of(sin, {2, 8});
  bool threw = false;
  try {
    cpu::rope_apply(qt, kt, ct, st);
  } catch (const Error&) {
    threw = true;
  }
  CHECK_TRUE(threw);
}

LLMRT_TEST(rope_apply_rejects_seq_mismatch) {
  std::vector<float> q(2 * 3 * 4, 1.0f), k(2 * 3 * 4, 1.0f);
  const std::vector<float> cos(5 * 4, 1.0f), sin(5 * 4, 0.0f);  // seq 5 != 3
  Tensor qt = view_of(q, {2, 3, 4});
  Tensor kt = view_of(k, {2, 3, 4});
  Tensor ct = view_of(cos, {5, 4});
  Tensor st = view_of(sin, {5, 4});
  bool threw = false;
  try {
    cpu::rope_apply(qt, kt, ct, st);
  } catch (const Error&) {
    threw = true;
  }
  CHECK_TRUE(threw);
}

LLMRT_TEST(rope_frequencies_rejects_odd_head_dim) {
  std::vector<float> cos(12, 0.0f), sin(12, 0.0f);
  Tensor ct = view_of(cos, {2, 6});
  Tensor st = view_of(sin, {2, 6});
  bool threw = false;
  try {
    cpu::rope_frequencies(2, 6, 1e6f, ct, st);  // half-split needs an even head_dim
  } catch (const Error&) {
    threw = true;
  }
  CHECK_TRUE(threw);
}

int main() { return llmrt_test::run_all("ops_cpu"); }
