// Scaled dot-product attention with grouped-query attention and a causal mask.
//
// ---------------------------------------------------------------------------
// 张量布局 —— 这一节是重点，写错任何一个都很难查
// ---------------------------------------------------------------------------
//
//   q   [B, H,  S, D]     QK-Norm + RoPE 之后      H  = 16
//   k   [B, Kv, S, D]     QK-Norm + RoPE 之后      Kv = 8
//   v   [B, Kv, S, D]     v_proj 直接出来的，**v 不过 RoPE**
//   out [B, S, H*D]       o_proj 的输入            H*D = 2048
//
// 注意 q/k/v 的轴序是 [batch, head, seq, dim] —— 头在 seq **前面**。
// 而 out 是 [batch, seq, H*D]，因为参考实现做了
//
//   attn_output.transpose(1, 2).reshape(B, S, -1)
//
// 把 head 轴折进最后一维，按 head 优先排布：
//
//   out[b][s][h*D + d]
//
// ⚠️ 这个布局写错**在形状上看不出来**（H*D == 2048，和反过来展平一样大），
//    但结果全错。实测按 head-major 展平 rel = 1.008e+00。
//
// ---------------------------------------------------------------------------
// 逐 (batch, head) 的计算
// ---------------------------------------------------------------------------
//
//   g = h / num_kv_groups                    GQA：n_rep 个 query 头共用一个 kv 头
//
//   对每个 query 位置 i：
//     scores[j] = (sum_d q[b][h][i][d] * k[b][g][j][d]) * scale      j 遍历所有 key
//     scores[j] += mask[i][j]                                        被遮的位置置 -FLT_MAX
//     probs     = softmax(scores)                                    fp32，沿 j
//     out[b][i][h*D + d] = sum_j probs[j] * v[b][g][j][d]
//
// ---------------------------------------------------------------------------
// 三个已实测确认的细节
// ---------------------------------------------------------------------------
//
// ① GQA 映射是 h / num_kv_groups，不是 h % num_kv_groups
//
//    参考实现的 repeat_kv：把 (B, Kv, S, D) 沿**中间插入**一维再 expand(n_rep)，
//    然后 reshape 成 (B, Kv*n_rep, S, D)。所以 kv 头 g 服务的是**连续**的
//    n_rep 个 query 头：
//
//      heads 0,1 -> kv 0     heads 2,3 -> kv 1     ...     heads 14,15 -> kv 7
//
//    实测：用 h % 2 rel = 9.558e-01（全错）。
//
// ② mask 的值是 -FLT_MAX（= torch.finfo(float32).min = -3.4028235e38），不是 -inf
//
//    对有可见项的行，两者结果一样（softmax 后都被压成 0）。差别在**整行全遮**
//    时：
//      -FLT_MAX : m 也是 -FLT_MAX，于是 exp(0) = 1，得到**均匀分布**
//      -inf     : -inf - (-inf) = nan
//    参考实现用的是 -FLT_MAX（`min_dtype = torch.finfo(dtype).min`），所以照抄。
//    causal 下第 0 行至少能看见自己，用不到这个差异，但照抄就不会在别处栽。
//
//    另外注意参考里的顺序是 **先乘 scale，再加 mask**：
//      attn_weights = matmul(q, k^T) * scaling
//      attn_weights = attn_weights + causal_mask
//    两者都在 softmax 之前，顺序其实无关紧要（mask 是常量），但照抄省得想。
//
// ③ scale = 1/sqrt(head_dim)
//
//    实测两种写法逐位相同（d = 128 时都是 0x3db504f3）：
//      (float)(1.0 / sqrt((double)d))      // Python 的 d ** -0.5
//      1.0f / sqrtf((float)d)
//    所以不纠结，用 float 那个。
//
// ---------------------------------------------------------------------------
// 实现建议
// ---------------------------------------------------------------------------
//
// **不需要 [H, S, S] 的完整 scores 矩阵。** 逐 (head, query 位置) 处理，
// 只需要一个长度为 S 的 scratch buffer 放当前这一行 scores：
//
//   prefill S=512   -> scratch 2 KiB，L1 里躺着
//   最长 S=40960    -> scratch 160 KiB，仍在 L2（1.25 MB）内
//
// 反过来，如果真去落地 [16, 512, 512] 的 probs 矩阵，那是 16 MB —— 白花内存
// 和带宽。这也是后面 flash-attention 的思路（online softmax），只是这里因为
// 逐行处理，天然就不需要落地。
//
// **复用 cpu::softmax。** 把 scratch 包成形状 [S] 的 Tensor 传进去即可，
// softmax 的契约就是"沿最后一维归一化"，rank=1 时 n=S、rows=1，正好。
// （x 和 out 可以传同一个 Tensor —— 实现里是先读 x[j] 再写 out[j]，原地安全。）
//
// 加权求和的循环顺序要注意：写成 **j 外层、d 内层** 的外积形式
//
//   for j:  p = probs[j]
//           for d: out_row[d] += p * v_row_j[d]
//
// 这样 v 和 out 两路都是连续访问。反过来写成 d 外层、j 内层的话，
// v[b][g][j][d] 会按 D 跨步读 —— 每读一个数跳一整个 cache line。
// （这和 matmul 里 transpose_b 那个分支是同一个道理。）
//
// 建议顺序：先写最直白的版本把对拍跑通，再谈优化。

#include "llmrt/ops.h"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "llmrt/common.h"

namespace llmrt {
namespace cpu {

namespace {

// Four tensors and a params struct, so this is the widest argument check in the
// project. Order matches the other ops and goes from "can this memory be read
// at all" to "does it describe the operation we think it does": contiguity,
// device, dtype, rank, shapes, then the params the shapes are compared against.
//
// Every message prints the actual shapes. Without them a mismatch sends you
// back to the call site to work out which tensor was wrong.
void check_attention_args(const Tensor& q, const Tensor& k, const Tensor& v,
                          const Tensor& out, const AttentionParams& params) {
  q.require_contiguous("attention");
  k.require_contiguous("attention");
  v.require_contiguous("attention");
  out.require_contiguous("attention");

  LLMRT_CHECK(q.is_cpu() && k.is_cpu() && v.is_cpu() && out.is_cpu(),
              "attention: all tensors must be host (CPU) tensors");
  LLMRT_CHECK(q.dtype == DType::F32 && k.dtype == DType::F32 && v.dtype == DType::F32 &&
                  out.dtype == DType::F32,
              "attention: all tensors must be F32");

  // Positivity first, so the modulo below cannot divide by zero.
  LLMRT_CHECK(params.num_heads > 0 && params.num_kv_heads > 0 && params.head_dim > 0,
              "attention: num_heads, num_kv_heads and head_dim must all be positive, got " +
                  std::to_string(params.num_heads) + ", " +
                  std::to_string(params.num_kv_heads) + " and " +
                  std::to_string(params.head_dim));
  LLMRT_CHECK(params.num_heads % params.num_kv_heads == 0,
              "attention: num_heads (" + std::to_string(params.num_heads) +
                  ") must be a multiple of num_kv_heads (" +
                  std::to_string(params.num_kv_heads) +
                  "); GQA repeats each kv head an equal number of times");

  // The rank checks come before any dim(i) call: every shape assertion below
  // assumes these axis counts exist.
  LLMRT_CHECK(q.rank() == 4 && k.rank() == 4 && v.rank() == 4,
              "attention: q, k and v must be rank 4 [batch, head, seq, dim], got " +
                  std::to_string(q.rank()) + ", " + std::to_string(k.rank()) + " and " +
                  std::to_string(v.rank()));
  LLMRT_CHECK(out.rank() == 3,
              "attention: out must be rank 3 [batch, seq, num_heads * head_dim], got rank " +
                  std::to_string(out.rank()) + " shape " + out.shape_string());

  LLMRT_CHECK(k.shape == v.shape,
              "attention: k and v must have identical shapes, got " + k.shape_string() +
                  " and " + v.shape_string());
  LLMRT_CHECK(q.dim(0) == k.dim(0),
              "attention: batch mismatch, q is " + q.shape_string() + " and k is " +
                  k.shape_string());
  LLMRT_CHECK(q.dim(2) == k.dim(2),
              "attention: sequence length mismatch, q is " + q.shape_string() +
                  " and k is " + k.shape_string());
  LLMRT_CHECK(q.dim(3) == k.dim(3) && q.dim(3) == params.head_dim,
              "attention: head_dim mismatch, q is " + q.shape_string() + ", k is " +
                  k.shape_string() + " and params.head_dim is " +
                  std::to_string(params.head_dim));
  LLMRT_CHECK(q.dim(1) == params.num_heads,
              "attention: q has " + std::to_string(q.dim(1)) +
                  " heads but params.num_heads is " + std::to_string(params.num_heads));
  LLMRT_CHECK(k.dim(1) == params.num_kv_heads,
              "attention: k has " + std::to_string(k.dim(1)) +
                  " kv heads but params.num_kv_heads is " +
                  std::to_string(params.num_kv_heads));

  // out's width is H*D, not D, and not H (a [B,S,H,D] tensor has the same
  // element count as [B,S,H*D] and would otherwise slip through).
  const int64_t expected_width = params.num_heads * params.head_dim;
  LLMRT_CHECK(out.dim(0) == q.dim(0) && out.dim(1) == q.dim(2) &&
                  out.dim(2) == expected_width,
              "attention: out must be [batch, seq, num_heads * head_dim] = [" +
                  std::to_string(q.dim(0)) + ", " + std::to_string(q.dim(2)) + ", " +
                  std::to_string(expected_width) + "] but is " + out.shape_string());
}

}  // namespace

void attention(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& out,
               const AttentionParams& params) {
  check_attention_args(q, k, v, out, params);

  const int64_t B = q.dim(0);
  const int64_t H = params.num_heads;
  const int64_t Kv = params.num_kv_heads;
  const int64_t S = q.dim(2);
  const int64_t D = params.head_dim;
  const int64_t groups = params.num_kv_groups();
  const int64_t width = H * D;  // out's last axis: heads flattened, head-major

  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const float masked = -std::numeric_limits<float>::max();  // -FLT_MAX, see the header

  const float* q_base = q.f32();
  const float* k_base = k.f32();
  const float* v_base = v.f32();
  float* out_base = out.f32();

  // One row of scores, reused by every (batch, head, query position). This is
  // why the [H, S, S] score matrix is never materialised: at S = 512 that would
  // be 16 MB of write-then-read, while this is 2 KiB and stays in L1 (and even
  // the longest Qwen3 context, 40960, only reaches 160 KiB, inside L2).
  //
  // The Tensor wrapper is built once, outside the loops. Building it inside
  // would allocate two std::vectors per iteration, H*S of them, for nothing --
  // `scores` is never resized, so its address stays valid.
  std::vector<float> scores(static_cast<size_t>(S));
  Tensor scores_view = Tensor::contiguous(scores.data(), DType::F32, DeviceKind::CPU, {S});

  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      // GQA: query head h reads kv head h / groups. NOT h % groups -- see the
      // header, where the wrong mapping measures rel ~0.96.
      const int64_t g = h / groups;

      // Row bases, hoisted out of the inner loops. Written inline these would be
      // ((b*H + h)*S + i)*D and friends, repeated a dozen times, one parenthesis
      // away from a silent offset error.
      const float* q_head = q_base + ((b * H + h) * S) * D;
      const float* k_head = k_base + ((b * Kv + g) * S) * D;
      const float* v_head = v_base + ((b * Kv + g) * S) * D;

      for (int64_t i = 0; i < S; ++i) {
        const float* q_row = q_head + i * D;

        // scores[j] = dot(q[i], k[j]) * scale. d innermost, so both rows are
        // walked contiguously -- the same layout argument as matmul's
        // transpose_b branch. Striding k by D instead would touch a fresh cache
        // line for every element.
        for (int64_t j = 0; j < S; ++j) {
          const float* k_row = k_head + j * D;
          float dot = 0.0f;
          for (int64_t d = 0; d < D; ++d) dot += q_row[d] * k_row[d];
          scores[static_cast<size_t>(j)] = dot * scale;
        }

        // Query position i may attend to key positions j <= i only. The
        // reference's mask is `arange(target_length) > cache_position`, i.e.
        // j > i is masked.
        //
        // Assigning `masked` rather than adding the mask is exact: a finite
        // score plus -FLT_MAX rounds to -FLT_MAX anyway, which is what the
        // reference computes.
        if (params.causal) {
          for (int64_t j = i + 1; j < S; ++j) scores[static_cast<size_t>(j)] = masked;
        }

        // Delegated to the softmax op rather than open-coded: same last-axis
        // contract, so all 100+ softmax call sites in a forward pass share one
        // implementation. In place is safe -- softmax reads x_row[j] before it
        // writes out_row[j], and here those are the same address.
        cpu::softmax(scores_view, scores_view);

        // out[i] = sum_j probs[j] * v[j], in outer-product order: j outer, d
        // inner, so v and out are both walked contiguously. This is the mirror
        // image of the scores loop above, and the reason no transposed copy of
        // v is ever needed.
        //
        // Seeded with the j = 0 term instead of zeroing first, saving a pass
        // over out_row. 0.0f + x == x bit-for-bit for every x except x == -0.0f,
        // and -0.0f compares equal to 0.0f regardless.
        float* out_row = out_base + (b * S + i) * width + h * D;
        {
          const float p = scores[0];
          const float* v_row = v_head;
          for (int64_t d = 0; d < D; ++d) out_row[d] = p * v_row[d];
        }
        for (int64_t j = 1; j < S; ++j) {
          const float p = scores[static_cast<size_t>(j)];
          const float* v_row = v_head + j * D;
          for (int64_t d = 0; d < D; ++d) out_row[d] += p * v_row[d];
        }
      }
    }
  }
}

}  // namespace cpu
}  // namespace llmrt
