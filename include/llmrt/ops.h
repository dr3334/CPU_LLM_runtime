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

// Matrix multiply with an optional transposed right-hand operand.
//
//   transpose_b == true :  a [M, K] @ b [N, K]  ->  out [M, N]
//                          out[i][j] = sum_k a[i][k] * b[j][k]
//
//   transpose_b == false:  a [M, K] @ b [K, N]  ->  out [M, N]
//                          out[i][j] = sum_k a[i][k] * b[k][j]
//
// All three tensors must be 2-D, contiguous, f32 and on the host. Callers with
// higher-rank activations reshape first ([1, 12, 1024] -> [12, 1024]); that is
// free for contiguous data.
//
// Qwen3 uses the transposed form everywhere: every projection weight is stored
// as [out_features, in_features] (nn.Linear's layout), so a projection is
// matmul(x, W, y, /*transpose_b=*/true). Materialising W^T instead would copy
// 1.2 GB of weights, and this layout is in fact the better one -- the inner
// loop becomes a dot product between two contiguous rows.
void matmul(const Tensor& a, const Tensor& b, Tensor& out, bool transpose_b);

// SwiGLU: the gated MLP activation, applied elementwise.
//
//   out[i] = silu(gate[i]) * up[i],   where silu(x) = x / (1 + exp(-x))
//
// Qwen3MLP computes down_proj(silu(gate_proj(x)) * up_proj(x)), so this op is
// where the two parallel projections are combined. `gate`, `up` and `out` must
// have identical shapes; the op is purely elementwise, so there is no axis or
// reduction parameter.
//
// The 1/(1+exp(-x)) form is what torch's SiLU uses -- verified bit-exact
// against ACT2FN["silu"]. Writing x * sigmoid(x) instead differs in the last
// bits, so it is not used here.
void swiglu(const Tensor& gate, const Tensor& up, Tensor& out);

}  // namespace cpu
}  // namespace llmrt
