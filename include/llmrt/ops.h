// Hand-written CPU kernels.
//
// Conventions shared by every op declared here
// -------------------------------------------
//  * Inputs and outputs are contiguous f32 tensors on DeviceKind::CPU. Anything
//    else throws -- a strided or bf16 tensor must fail loudly rather than be
//    read as if it were dense f32.
//  * Ops never allocate. Scratch space is passed in by the caller, which keeps
//    the memory manager (Phase 4) the single owner of bytes.
//  * Every op is checked against data/golden within the tolerances in
//    tests/golden.h. "It runs" is not the bar; matching the reference is.
//
// Namespaces mirror the device: llmrt::cpu now, llmrt::opencl later. When
// IBackend lands (Phase 5) each backend method is a thin call into one of
// these, so the kernels stay testable without going through the interface.
#pragma once

#include "llmrt/tensor.h"

namespace llmrt {
namespace cpu {

// Root-mean-square layer normalisation over the LAST axis:
//
//   out[i][j] = weight[j] * x[i][j] * rsqrt(mean_k(x[i][k]^2) + eps)
//
// There is no mean subtraction -- that is what makes it RMSNorm rather than
// LayerNorm. Both x and out have rank >= 1 and identical shape; the leading
// dimensions are treated as independent rows. `weight` is 1-D and its length
// equals the last axis: hidden_size for the block norms, head_dim for QK-Norm.
void rmsnorm(const Tensor& x, const Tensor& weight, Tensor& out, float eps);

}  // namespace cpu
}  // namespace llmrt
