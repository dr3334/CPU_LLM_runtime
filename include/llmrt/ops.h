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

// Rotary position embedding, split into the two halves that are worth testing
// separately (and that have different lifetimes -- the tables are built once
// per forward and reused by all 28 layers).
//
// Qwen3 uses the half-split (GPT-NeoX) convention, not the interleaved one:
// rotate_half pairs element j with element j + head_dim/2 and negates the
// front half. Getting this wrong still produces finite, plausible activations.
//
// The tables have the shape [seq_len, head_dim]:
//
//   inv_freq[i] = 1 / theta^(2i / head_dim)       i in [0, head_dim/2)
//   angle[p][j] = p * inv_freq[j % (head_dim/2)]
//   cos[p][j]   = cos(angle[p][j]),  sin likewise
//
// The second half of each row repeats the first -- that repetition is what
// makes the rotation below a plain elementwise multiply.
void rope_frequencies(int64_t seq_len, int64_t head_dim, float theta, Tensor& cos,
                      Tensor& sin);

// Applies the rotation in place to q and k:
//
//   out = x * cos + rotate_half(x) * sin
//
// q and k are [rows, seq, head_dim] with head_dim last and contiguous; `rows`
// is batch*heads, and every row is rotated with the same table, which is what
// the reference's `cos.unsqueeze(1)` broadcasts over. cos/sin are
// [seq, head_dim] from rope_frequencies.
//
// NOTE on layout: the golden fixtures capture q/k as [1, heads, seq, head_dim],
// so this contract matches the reference exactly. Callers holding
// [seq, heads, head_dim] must transpose first -- a view, but this op (like every
// other) requires contiguous input, so it is a real copy. At 98 KiB per tensor
// per layer that is under 1% of a forward pass's memory traffic; fusing the
// transpose into the rotation is a Phase 8 optimisation.
void rope_apply(Tensor& q, Tensor& k, const Tensor& cos, const Tensor& sin);

// Numerically stable softmax over the LAST axis:
//
//   m       = max_j x[i][j]              (per row, for stability)
//   e[j]    = exp(x[i][j] - m)
//   out[i][j] = e[j] / sum_k e[k]
//
// The max subtraction is not decorative. Without it exp() overflows for scores
// above ~88 and the row comes out as [0, nan, nan]; measured against torch on
// [88, 89, 90]. torch's softmax also subtracts the max -- verified bit-exact --
// so doing the same is both correct and closer to the reference.
//
// Row semantics match rmsnorm: `x` and `out` share a shape of rank >= 1, the
// leading axes are independent rows flattened, and the last axis is the one
// normalised. -inf entries are supported and come out as exactly 0, which is
// how the causal mask is applied (attention adds -inf above the diagonal).
// A row that is entirely -inf yields NaN, so callers must guarantee at least one
// unmasked entry -- true for causal attention, where row 0 sees position 0.
//
// Note this op takes an explicit output, like the others; attention later fuses
// it into a single kernel (Phase 8), at which point it stops being called
// standalone.
void softmax(const Tensor& x, Tensor& out);

}  // namespace cpu
}  // namespace llmrt
