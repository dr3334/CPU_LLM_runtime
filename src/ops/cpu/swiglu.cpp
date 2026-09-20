// SwiGLU -- the gated MLP activation for Qwen3.
//
//   out[i] = silu(gate[i]) * up[i],   where silu(x) = x / (1 + exp(-x))
//
// Qwen3MLP computes down_proj(silu(gate_proj(x)) * up_proj(x)), so this is where
// the two parallel projections are combined. The op is purely elementwise --
// no reduction, no axis -- so the tensors are walked as flat arrays.
//
// Fixtures (data/golden/ops/), all [1, 12, 3072]:
//   gate_proj_out.f32.bin   gate_proj(x)
//   up_proj_out.f32.bin     up_proj(x)
//   swiglu_out.f32.bin      this op's expected output
//   down_proj_in.f32.bin    the same tensor, captured from a different hook
//
// Numerical note: the division form is not interchangeable with
// x * sigmoid(x). It is bit-identical to torch's SiLU, whereas the
// multiplication form diverges by ~2.4e-7. See "Floating-point rules" in
// CODEBUDDY.md.

#include "llmrt/ops.h"

#include <cmath>

#include "llmrt/common.h"

namespace llmrt {
namespace cpu {

namespace {

// gate, up and out are combined elementwise, so a shape disagreement would read
// past the end of a buffer with no complaint from anything. Checked once here
// rather than on every iteration of the loop below.
void check_swiglu_args(const Tensor& gate, const Tensor& up, const Tensor& out) {
  gate.require_contiguous("swiglu");
  up.require_contiguous("swiglu");
  out.require_contiguous("swiglu");

  LLMRT_CHECK(gate.is_cpu() && up.is_cpu() && out.is_cpu(),
              "swiglu: all tensors must be host (CPU) tensors");
  LLMRT_CHECK(gate.dtype == DType::F32 && up.dtype == DType::F32 && out.dtype == DType::F32,
              "swiglu: all tensors must be F32");
  LLMRT_CHECK(gate.shape == up.shape && gate.shape == out.shape,
              "swiglu: gate, up and out must have identical shapes, got " +
                  gate.shape_string() + ", " + up.shape_string() + " and " +
                  out.shape_string());
}

}  // namespace

void swiglu(const Tensor& gate, const Tensor& up, Tensor& out) {
  check_swiglu_args(gate, up, out);

  const int64_t n = static_cast<int64_t>(gate.numel());

  const float* gate_base = gate.f32();
  const float* up_base = up.f32();
  float* out_base = out.f32();

  for (int64_t i = 0; i < n; ++i) {
    const float g = gate_base[i];
    const float u = up_base[i];
    // The parentheses around g/d are load-bearing: without them C groups u*g
    // first, and the extra rounding costs one ulp against the reference.
    out_base[i] = u * (g / (1.0f + std::exp(-g)));
  }
}

}  // namespace cpu
}  // namespace llmrt
