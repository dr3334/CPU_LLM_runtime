#include "llmrt/ops.h"

#include <cmath>

#include "llmrt/common.h"

namespace llmrt {
namespace cpu {

namespace {

// Rejects anything the kernel would otherwise misread. Kept separate so the
// checks read as one contract rather than being scattered through the maths.
void check_rmsnorm_args(const Tensor& x, const Tensor& weight, const Tensor& out, float eps) {
  x.require_contiguous("rmsnorm");
  weight.require_contiguous("rmsnorm");
  out.require_contiguous("rmsnorm");

  LLMRT_CHECK(x.is_cpu() && weight.is_cpu() && out.is_cpu(),
              "rmsnorm: all tensors must be host (CPU) tensors");
  LLMRT_CHECK(x.dtype == DType::F32 && weight.dtype == DType::F32 && out.dtype == DType::F32,
              "rmsnorm: all tensors must be F32");
  LLMRT_CHECK(x.rank() >= 1, "rmsnorm: input must have rank >= 1");
  LLMRT_CHECK(weight.rank() == 1, "rmsnorm: weight must be 1-D");
  LLMRT_CHECK(x.shape == out.shape,
              "rmsnorm: output shape " + out.shape_string() + " must match input " +
                  x.shape_string());

  const int64_t n = x.dim(x.rank() - 1);
  LLMRT_CHECK(n > 0, "rmsnorm: the normalised axis must be non-empty");
  LLMRT_CHECK(weight.dim(0) == n, "rmsnorm: weight has " + std::to_string(weight.dim(0)) +
                                      " elements but the last axis is " + std::to_string(n));
  LLMRT_CHECK(eps > 0.0f, "rmsnorm: eps must be positive");
}

}  // namespace

void rmsnorm(const Tensor& x, const Tensor& weight, Tensor& out, float eps) {
  check_rmsnorm_args(x, weight, out, eps);

  const int64_t n = x.dim(x.rank() - 1);
  const int64_t rows = static_cast<int64_t>(x.numel() / static_cast<size_t>(n));

  const float* xp = x.f32();
  const float* wp = weight.f32();
  float* op = out.f32();

  for (int64_t r = 0; r < rows; ++r) {
    const float* row = xp + r * n;
    float* dst = op + r * n;

    // Pass 1: sum of squares. Must finish before any output is written,
    // because the scale factor depends on the entire row.
    float sum_sq = 0.0f;
    for (int64_t j = 0; j < n; ++j) {
      sum_sq += row[j] * row[j];
    }

    // eps goes AFTER the mean and INSIDE the sqrt: rsqrt(mean + eps), not
    // rsqrt(mean(x^2 + eps)). 1/sqrt is the exact form; a hardware rsqrt
    // approximation differs in the last bits and fails the golden comparison.
    const float mean_sq = sum_sq / static_cast<float>(n);
    const float scale = 1.0f / std::sqrt(mean_sq + eps);

    // Pass 2: scale, then apply the weight. The row was read moments ago and
    // is still resident in L1 at the sizes we care about (n = 1024 -> 4 KiB),
    // so this second pass does not go back to memory.
    for (int64_t j = 0; j < n; ++j) {
      dst[j] = wp[j] * (row[j] * scale);
    }
  }
}

}  // namespace cpu
}  // namespace llmrt
