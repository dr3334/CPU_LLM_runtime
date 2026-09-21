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

// Everything attention needs that cannot be read off the tensors.
//
// Lives in llmrt rather than llmrt::cpu because it describes the *operation*,
// not an implementation of it: the OpenCL backend (Phase 5) and the IBackend
// interface will use this same struct.
//
// There is deliberately no `scale` field: Qwen3 defines it as head_dim ** -0.5,
// and deriving it here means a caller cannot pass a value that disagrees with
// what the reference used. Measured that (float)(1.0 / sqrt((double)d)) and
// 1.0f / sqrtf((float)d) are the same bits for d = 128 (0x3db504f3), so the
// exact spelling does not matter.
struct AttentionParams {
  int64_t num_heads = 0;
  int64_t num_kv_heads = 0;
  int64_t head_dim = 0;
  bool causal = true;

  // GQA: kv head g serves query heads [g*k, (g+1)*k). Matches the reference's
  // repeat_kv, which expands along an inserted middle axis and then reshapes.
  int64_t num_kv_groups() const {
    return num_kv_heads > 0 ? num_heads / num_kv_heads : 0;
  }
};

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
// normalised. Deeply negative entries come out as exactly 0, which is how the
// causal mask takes effect: attention adds a very negative constant above the
// diagonal and exp() underflows it to 0.
//
// A row that is *entirely* that constant is the caller's choice: -inf gives
// NaN, a finite value such as -FLT_MAX gives a uniform distribution. Attention
// passes -FLT_MAX to match the reference; see the note there.
//
// Note this op takes an explicit output, like the others; attention later fuses
// it into a single kernel (Phase 8), at which point it stops being called
// standalone.
void softmax(const Tensor& x, Tensor& out);

// Scaled dot-product attention with grouped-query attention and an optional
// causal mask.
//
//   q   [B, H,  S, D]    after QK-Norm and RoPE
//   k   [B, Kv, S, D]    after QK-Norm and RoPE
//   v   [B, Kv, S, D]    straight from v_proj -- v gets no RoPE
//   out [B, S, H*D]      the o_proj input
//
// `out` is NOT [B, H, S, D]. The reference does
// `attn_output.transpose(1, 2).reshape(B, S, -1)`, folding the head axis into
// the last one in head-major order: out[b][s][h*D + d]. Since H*D == 2048 ==
// the flattened size either way, a transposed layout is invisible to a shape
// check and costs rel ~1.0 against the golden.
//
// Per (batch, head h), with g = h / num_kv_groups and scale = 1/sqrt(head_dim):
//
//   scores[j] = (sum_d q[h][i][d] * k[g][j][d]) * scale
//   scores[j] += mask[i][j]                          -FLT_MAX where masked
//   probs     = softmax(scores)                      fp32, over j
//   out[i][h*D + d] = sum_j probs[j] * v[g][j][d]
//
// The mask value is -FLT_MAX (torch.finfo(float32).min), not -inf. For a row
// with any unmasked entry both give the same result, but a fully masked row
// becomes uniform with -FLT_MAX (m == -FLT_MAX, exp(0) == 1) and NaN with -inf.
// The reference uses -FLT_MAX, so this does too.
void attention(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& out,
               const AttentionParams& params);

}  // namespace cpu
}  // namespace llmrt
