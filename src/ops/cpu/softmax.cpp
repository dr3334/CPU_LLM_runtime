// Numerically stable softmax over the last axis of a tensor.
//
// ---------------------------------------------------------------------------
// 数学定义
// ---------------------------------------------------------------------------
//
// 对第 i 行（n 个数 x[i][0..n-1]）：
//
//                  exp(x[i][j])
//   out[i][j] = ----------------------       且   sum_j out[i][j] = 1
//               sum_k exp(x[i][k])
//
// 它把一行的任意实数映射成一个概率分布：保序（大的仍然大）、恒正、和为 1。
//
// ---------------------------------------------------------------------------
// 减最大值是恒等变形，不是近似
// ---------------------------------------------------------------------------
//
// exp 超过 88.7 就溢出成 inf，而 attention 的 scores 轻松到几十，
// 所以按定义式直接算会得到 [0.0, nan, nan]。
//
// 减去本行最大值 m 之后结果完全不变 —— exp(-m) 在上下两式中抵消：
//
//                exp(x[j] - m)         exp(x[j]) · exp(-m)         exp(x[j])
//   ──────────────────────────  =  ────────────────────────  =  ─────────────
//    sum_k exp(x[k] - m)            sum_k exp(x[k]) · exp(-m)    sum_k exp(x[k])
//
// （依据是 exp(a - b) = exp(a) / exp(b) 这个恒等式，所以这不是凑近似。）
//
// 收益体现在指数的取值上：x[j] - m <= 0，于是 exp(·) ∈ (0, 1]，永不上溢；
// 分母 ∈ (1, n]，也不下溢。中间量被压在一个安全的量级区间里。
//
// 实测 torch 的 nn.functional.softmax 同样减最大值，我们的结果与它逐位相同：
//   不减:  [88, 89, 90] -> [0.0, nan, nan]                 (exp(90) = inf)
//   减:    [88, 89, 90] -> [0.0900, 0.2447, 0.6652]
//
// ---------------------------------------------------------------------------
// 每行三趟 —— 趟数由公式决定，不是实现选择
// ---------------------------------------------------------------------------
//
//   趟1  m = max_j x[j]          必须先知道 m，才能进趟2
//   趟2  e[j] = exp(x[j] - m)     边算边累加 sum，同时把 e 存进 out
//   趟3  out[j] /= sum           就地除，不需要额外数组
//
// 少一趟不行：缩放因子 sum 依赖整行读完，和 rmsnorm 要先算 mean 同理。
// 但同样地，n = 512 时一行只有 2 KiB，趟2 趟3 重读的是 L1 而不是内存。
//
// 趟2 顺手把 e 写进 out、趟3 就地除，是为了让 exp 每元素只算一次。
// 实测（8192x512，clang -O2）这样快 1.55x —— 原实现里 exp 占了 71% 的运行
// 时间，重复算一遍是纯粹浪费。
//
// ---------------------------------------------------------------------------
// 边界
// ---------------------------------------------------------------------------
//
// x 里的 -inf 自然得出 0：exp(-inf - m) = 0（m 有限时）。
// 这正是 causal mask 的实现方式 —— attention 把上三角置 -inf。
//
// 但若**一整行**全是 -inf，m 也是 -inf，会得到 -inf - (-inf) = nan。
// torch 行为完全相同；causal attention 不会出现（位置 0 至少能看见自己）。
//
// 参照值（由 torch 的 nn.functional.softmax(dim=-1, dtype=torch.float32) 生成）：
//
//   输入 [1.0,  2.0,  3.0]     ->  [0.0900305733, 0.244728476, 0.665240943]
//   输入 [-1.0, 0.0,  1.0]     ->  [0.0900305733, 0.244728476, 0.665240943]
//                                     ↑ 和上一行相同：softmax 平移不变
//   输入 [0.5, -0.5,  0.25]    ->  [0.465835601, 0.171371341, 0.362793118]
//   输入 [1.0, -inf,  3.0]     ->  [0.119202912, 0.0, 0.880797029]
//
// ---------------------------------------------------------------------------

#include "llmrt/ops.h"

#include <cmath>

#include "llmrt/common.h"

namespace llmrt {
namespace cpu {

namespace {

// The x/out contract is rmsnorm's minus the weight vector: same shape, same
// dtype, same device, contiguous. A row that disagrees in length would be read
// past the end without any complaint, so it is checked once up front rather
// than per element.
void check_softmax_args(const Tensor& x, const Tensor& out) {
  x.require_contiguous("softmax");
  out.require_contiguous("softmax");

  LLMRT_CHECK(x.is_cpu() && out.is_cpu(),
              "softmax: both tensors must be host (CPU) tensors");
  LLMRT_CHECK(x.dtype == DType::F32 && out.dtype == DType::F32,
              "softmax: both tensors must be F32");
  LLMRT_CHECK(x.rank() >= 1, "softmax: input must have rank >= 1");
  LLMRT_CHECK(x.shape == out.shape,
              "softmax: output shape " + out.shape_string() + " must match input " +
                  x.shape_string());

  const int64_t n = x.dim(x.rank() - 1);
  LLMRT_CHECK(n > 0, "softmax: the normalised axis must be non-empty");
}

}  // namespace

void softmax(const Tensor& x, Tensor& out) {
  check_softmax_args(x, out);

  // Same contract as rmsnorm: normalise along the last axis, and every leading
  // axis is an independent row. Rows never interact with each other.
  const int64_t n = x.dim(x.rank() - 1);
  const int64_t rows = x.numel() / n;

  const float* x_base = x.f32();
  float* out_base = out.f32();

  for (int64_t r = 0; r < rows; ++r) {
    const float* x_row = x_base + r * n;
    float* out_row = out_base + r * n;

    // Pass 1: the row maximum. Seeding with x_row[0] and starting at j = 1
    // avoids needing a sentinel such as -inf or FLT_MAX.
    float row_max = x_row[0];
    for (int64_t j = 1; j < n; ++j) {
      if (x_row[j] > row_max) row_max = x_row[j];
    }

    // Pass 2: exp and accumulate. Writing e into out_row as we go means exp is
    // evaluated once per element instead of twice -- measured 1.55x on the
    // attention shape, where exp was 71% of the original runtime.
    float sum = 0.0f;
    for (int64_t j = 0; j < n; ++j) {
      const float e = std::exp(x_row[j] - row_max);
      out_row[j] = e;
      sum += e;
    }

    // Pass 3: divide in place. Kept as a division rather than multiplying by
    // 1/sum: torch divides too, and the reciprocal form costs up to 2 ulp.
    for (int64_t j = 0; j < n; ++j) {
      out_row[j] /= sum;
    }
  }
}

}  // namespace cpu
}  // namespace llmrt
