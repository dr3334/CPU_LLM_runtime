// Rotary position embedding (RoPE) for Qwen3.
//
// Qwen3 uses the HALF-SPLIT (GPT-NeoX) convention. `rotate_half` pairs element
// j with element j + head_dim/2 and negates the front half:
//
//   rotate_half([a0 a1 ... a63 | b0 b1 ... b63])
//     = [-b0 -b1 ... -b63 | a0 a1 ... a63]
//
// The interleaved convention (pairing 2j with 2j+1) is a different function
// that still produces finite, plausible-looking activations -- which is why it
// is dangerous. The fixtures below pin it down.
//
// The rotation itself, elementwise, with d = head_dim and h = d/2:
//
//   j <  h :  out[j] = x[j] * cos[j] - x[j+h] * sin[j]
//   j >= h :  out[j] = x[j] * cos[j] + x[j-h] * sin[j]
//
// This is the expansion of the reference's
//     q_embed = (q * cos) + (rotate_half(q) * sin)
//
// Fixtures (data/golden/ops/):
//   rope_q_in.f32.bin   [1, 16, 12, 128]   q after QK-Norm, before RoPE
//   rope_q_out.f32.bin  [1, 16, 12, 128]   expected result
//   rope_k_in.f32.bin   [1,  8, 12,  128]
//   rope_k_out.f32.bin  [1,  8, 12,  128]
//   rope_cos.f32.bin    [1, 12, 128]       the table, positions 0..11
//   rope_sin.f32.bin    [1, 12, 128]
//
// Note q and k have DIFFERENT head counts (16 vs 8 -- GQA) but the same table:
// cos/sin broadcast over the head axis.

#include "llmrt/ops.h"

#include <cmath>

#include "llmrt/common.h"

namespace llmrt {
namespace cpu {

namespace {

// TODO(你): 参数校验。
//
// 需要检查：
//   seq_len > 0
//   head_dim > 0 且 head_dim % 2 == 0      （半拆分要求能对半分）
//   theta > 0
//   cos / sin：连续、f32、在 CPU、形状都是 [seq_len, head_dim]
//
// 可以仿照前面两个算子（rmsnorm.cpp / swiglu.cpp）的 check_*_args 写法。
void check_rope_frequencies_args(int64_t seq_len, int64_t head_dim, float theta,
                                 const Tensor& cos, const Tensor& sin) {
  (void)seq_len;
  (void)head_dim;
  (void)theta;
  (void)cos;
  (void)sin;
}

// TODO(你): 参数校验。
//
// 需要检查：
//   q / k：连续、f32、在 CPU、rank >= 2
//   cos / sin：连续、f32、在 CPU、形状相同且为 [seq, head_dim]
//   q 和 k 的最后一维 == cos 的最后一维 (head_dim)
//   q 和 k 的倒数第二维 == cos 的倒数第二维 (seq)
//     ← 注意 q/k 是 [rows, seq, head_dim]，seq 在 **倒数第二维**
//   q 和 k 的 seq/head_dim 必须一致（虽然 heads 数可以不同：GQA 16 vs 8）
void check_rope_apply_args(const Tensor& q, const Tensor& k, const Tensor& cos,
                           const Tensor& sin) {
  (void)q;
  (void)k;
  (void)cos;
  (void)sin;
}

}  // namespace

void rope_frequencies(int64_t seq_len, int64_t head_dim, float theta, Tensor& cos,
                      Tensor& sin) {
  check_rope_frequencies_args(seq_len, head_dim, theta, cos, sin);

  // TODO(你): 生成 cos/sin 表 [seq_len, head_dim]
  //
  // 公式（已用 check_oracle.py 对过 golden，rel ~5e-7）：
  //
  //   half    = head_dim / 2
  //   inv_freq[i] = 1 / theta^( 2i / head_dim )        i 取 [0, half)
  //   angle[p][j] = p * inv_freq[j % half]             ← j % half：前后半段同频
  //   cos[p][j]   = cos( angle[p][j] )
  //   sin[p][j]   = sin( angle[p][j] )
  //
  // 提示：
  //   - inv_freq 只有 half 个值，**先算一遍存下来**再用。
  //     直接在 j 循环里算 powf 会把同一个值重算 head_dim 倍。
  //   - 用 float 全程：1.0f / std::pow(theta, ...)、std::cos(...)、std::sin(...)
  //   - 行主序：[p][j] 的线性下标是 p * head_dim + j
  //   - 这是整个模型里唯一会调用超越函数的地方之一，prefill 时 seq 可能到几百，
  //     而这个表每个 forward 只算一次、28 层共用 —— 这正是把它拆成独立函数的原因
  const int64_t half =head_dim/2;
  std::vector<float> inv_freq(static_cast<size_t>(half));
  
  for(int64_t i=0;i<half;i++){
    inv_freq[i] = 1.0f / std::pow(theta, 2.0f * i / head_dim);
  }
  for(int64_t p=0;p<seq_len;p++){
    for(int64_t j=0;j<head_dim;j++){
      float angle = p * inv_freq[j % half];
      cos[p * head_dim + j] = std::cos(angle);
      sin[p * head_dim + j] = std::sin(angle);
    }
  }
  
}

void rope_apply(Tensor& q, Tensor& k, const Tensor& cos, const Tensor& sin) {
  check_rope_apply_args(q, k, cos, sin);

  // TODO(你): 对 q 和 k 施加旋转（原地修改）
  //
  // q / k 形状 [rows, seq, head_dim]，cos/sin 形状 [seq, head_dim]。
  // 每一行（row）独立旋转，但**所有 row 共用同一张 cos/sin 表** ——
  // 这正是参考实现里 cos.unsqueeze(1) 沿 heads 维广播的含义。
  //
  // 元素级公式（d = head_dim，h = d/2，p = 位置，j = 维内下标）：
  //
  //   out[j]   = x[j] * cos[j] - x[j+h] * sin[j]        j <  h
  //   out[j+h] = x[j+h] * cos[j+h] + x[j] * sin[j+h]    后半段同理
  //
  // ⚠️ 原地修改有别名陷阱：out[j] 和 out[j+h] **互相需要对方的原值**。
  //    如果按 j = 0,1,2,... 顺序逐个写：
  //      写到 j 时，x[j] 被覆盖
  //      写到 j+h 时，公式需要 x[j] —— 但它已经被覆盖成 out[j] 了 ❌
  //
  //    正确做法：**每次成对处理** (j, j+h)，先把两个原值读进局部变量，
  //    再一次性写回两个位置。一趟循环，无别名问题。
  //
  //    即：for (j = 0; j < h; ++j) { a = x[j]; b = x[j+h]; ...; x[j] = ...; x[j+h] = ...; }
  //
  // 提示：
  //   - 三层循环：row → p → j，其中 j 只走到 half 就够（一次处理一对）
  //   - q 和 k 的行数不同（16 vs 8），所以 q、k 要各自循环
  //   - 建议先写一个只处理单个张量的内部辅助函数，q 和 k 各调一次
  float q_base=q.f32();
  float k_base=k.f32();
  int64_t rows_q=q.dim(0);
  int64_t rows_k=k.dim(0);
  int64_t seq=q.dim(1);
  int64_t head_dim=q.dim(2);
  int64_t half=head_dim/2;
  for(int64_t r=0;r<rows_q;r++){
    for(int64_t p=0;p<seq;p++){
      for(int64_t j=0;j<half;j++){
        float q_j=q_base[r*seq*head_dim+p*head_dim+j];
        float q_jh=q_base[r*seq*head_dim+p*head_dim+j+
half];
        float k_j=k_base[r*seq*head_dim+p*head_dim+j];
        float k_jh=k_base[r*seq*head_dim+p*head_dim+j+half];
        q_base[r*seq*head_dim+p*head_dim+j]=q_j*cos[p*head_dim+j]-q_jh*sin[p*head_dim+j];
        q_base[r*seq*head_dim+p*head_dim+j+half]=q_jh*cos[p*head
_dim+j+half]+q_j*sin[p*head_dim+j+half];
        k_base[r*seq*head_dim+p*head_dim+j]=k_j*cos[p*head_dim+j]-k_jh*sin[p*head_dim+j];
        k_base[r*seq*head_dim+p*head_dim+j+half]=k_jh*cos[p*head_dim+j+half]+k_j*sin[p*head_dim+j+half];
      }
  }
}
  (void)k;
  (void)cos;
  (void)sin;
}

}  // namespace cpu
}  // namespace llmrt
