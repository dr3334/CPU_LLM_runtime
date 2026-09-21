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
#include <vector>

#include "llmrt/common.h"

namespace llmrt {
namespace cpu {

namespace {

// TODO(你): 仿照 check_softmax_args 实现参数校验。
//
// 需要检查的（不检查的后果都写在括号里）：
//
//   连续性     q / k / v / out 四个都要 require_contiguous
//              （跨步视图被当连续读，读到别的元素）
//   设备       q / k / v / out 都在 CPU 上
//   类型       都是 F32
//   维度       q / k / v 是 rank 4，out 是 rank 3
//              （下面所有 dim(i) 都假设了轴数）
//   形状一致性
//                k.shape == v.shape                        k 和 v 必须同形
//                q.dim(0) == k.dim(0)                      同一个 batch
//                q.dim(2) == k.dim(2)                      S 必须一致
//                q.dim(3) == k.dim(3) == params.head_dim   D 必须一致
//                q.dim(1) == params.num_heads
//                k.dim(1) == params.num_kv_heads
//                out.shape == {B, S, num_heads * head_dim}  ← 注意是 H*D，不是 D
//   参数       num_heads > 0、num_kv_heads > 0、head_dim > 0
//              num_heads % num_kv_heads == 0             GQA 必须整除
//
// 错误信息里把实际形状打出来（用 shape_string()），否则修的时候还得自己翻。
void check_attention_args(const Tensor& q, const Tensor& k, const Tensor& v,
                          const Tensor& out, const AttentionParams& params) {
  (void)q;
  (void)k;
  (void)v;
  (void)out;
  (void)params;
}

}  // namespace

void attention(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& out,
               const AttentionParams& params) {
  check_attention_args(q, k, v, out, params);

  // TODO(你): 实现。骨架大致是：
  //
  //   const int64_t B  = q.dim(0);
  //   const int64_t H  = params.num_heads;
  //   const int64_t Kv = params.num_kv_heads;
  //   const int64_t S  = q.dim(2);
  //   const int64_t D  = params.head_dim;
  //   const int64_t groups = params.num_kv_groups();
  //   const float   scale  = 1.0f / std::sqrt(static_cast<float>(D));
  //   const float   neg    = -std::numeric_limits<float>::max();   // -FLT_MAX
  //
  //   const float* q_base = q.f32();
  //   const float* k_base = k.f32();
  //   const float* v_base = v.f32();
  //   float*       o_base = out.f32();
  //
  //   std::vector<float> scores(static_cast<size_t>(S));   // 一行，摊到所有 head/i 复用
  //
  //   for (b ...) {
  //     for (h ...) {
  //       const int64_t g = h / groups;
  //       // q[b][h] 的基址、k[b][g] 的基址、v[b][g] 的基址
  //       for (i ...) {
  //         // 1) scores[j] = dot(q[b][h][i], k[b][g][j]) * scale
  //         // 2) causal 时 j > i 的置 neg
  //         // 3) 包成 [S] 的 Tensor 交给 cpu::softmax（原地）
  //         // 4) out[b][i][h*D + d] = Σ_j probs[j] * v[b][g][j][d]
  //         //    注意 j 外层 d 内层
  //       }
  //     }
  //   }
  //
  // 几个具体提示：
  //
  //   - q[b][h][i] 的起点 = q_base + ((b*H + h)*S + i) * D
  //     k[b][g][j] 的起点 = k_base + ((b*Kv + g)*S + j) * D
  //     v 同 k 的算法；out[b][i] 的起点 = o_base + (b*S + i) * (H*D)
  //     推荐照 rope.cpp 的做法：每个循环开头把行基址提出一个局部变量，
  //     别在表达式里反复写 ((b*H+h)*S+i)*D 这种 —— 错一个括号就悄悄偏了。
  //
  //   - 第 4 步的 out_row 要先清零（j 从 0 累加），或者用 j = 0 的那一项
  //     直接赋值、再从 j = 1 累加，省一趟清零。
  //
  //   - causal 判断：参考实现是 `arange(target) > cache_position`，
  //     即 mask[i][j] 在 j > i 时为真。所以 i 行只允许 j <= i。
  //
  //   - softmax 原地调用：
  //       Tensor sc = Tensor::contiguous(scores.data(), DType::F32,
  //                                      DeviceKind::CPU, {S});
  //       cpu::softmax(sc, sc);
  //     注意 Tensor 是视图，这个 sc 指向 scores 这块内存，不会拷贝。
  (void)q;
  (void)k;
  (void)v;
  (void)out;
  (void)params;
}

}  // namespace cpu
}  // namespace llmrt
