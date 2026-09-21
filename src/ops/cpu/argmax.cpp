// Index of the largest element along the last axis, per row.
//
//   x   [rows, n]   f32
//   out [rows]      i32
//
//   out[i] = the smallest j that maximises x[i][j]
//
// ---------------------------------------------------------------------------
// 为什么是算子而不是调用方的一个循环
// ---------------------------------------------------------------------------
//
// ops.h 不是"模型数学"的清单 —— embedding 是 gather，也不是数学。它真正的
// 定义是"这个运行时按后端逐个实现的那些原语"。按这个定义 argmax 是算子：
//
//   GPU 后端会想原生实现它。单个 token 的 logits 行是 151936 个 float
//   （608 KB），让 kernel 直接返回一个下标，比"算完 608 KB 再拷回主机求 max"
//   干净得多。
//
// 而且它的语义需要钉死（平局、NaN），不是"显然对" —— 见下面的契约测试。
//
// ---------------------------------------------------------------------------
// 它刻意不做的事
// ---------------------------------------------------------------------------
//
// temperature、top-k、top-p、eos 判断、"只取最后一个位置" —— 那些是**解码策略**，
// 住 generate()。让它们渗进算子，GPU 内核就没法做成纯粹的归约了。
//
// ---------------------------------------------------------------------------
// 两个契约细节（都有测试）
// ---------------------------------------------------------------------------
//
//  ① 平局取**最小下标**，和 numpy.argmax / torch.argmax 一致。
//     不定死的话，两个后端可能给出不同的 token —— "任意 --split 结果与纯 CPU
//     完全一致"（Phase 6 的门）就破了。
//
//  ② NaN **永远不会赢**。循环以 -infinity 做种子，而 `NaN > v` 对任何 v 都是假，
//     所以 NaN 拿不到"更大"。这是有意和 numpy 不同的：numpy 会返回 NaN 的下标，
//     而我们宁愿不选它 —— NaN 是权重坏掉的产物，这里不是诊断它的地方。整行都是
//     NaN（或都是 -inf）时返回下标 0，调用方检查那一个值就能发现。
//
//     （最初写成以 row[0] 做种子，那样下标 0 处的 NaN 会变成无法超越的，等于凭空
//     多一条"第 0 位例外"。是契约测试把这个特例逼出来的。）

#include "llmrt/ops.h"

#include <limits>

#include "llmrt/common.h"

namespace llmrt {
namespace cpu {

namespace {

void check_argmax_args(const Tensor& x, const Tensor& out) {
  x.require_contiguous("argmax");
  out.require_contiguous("argmax");

  LLMRT_CHECK(x.is_cpu() && out.is_cpu(),
              "argmax: both tensors must be host (CPU) tensors");
  // One reduction means one dtype for the values and one for the indices, so
  // this checks the two separately rather than with a single "all tensors" test.
  LLMRT_CHECK(x.dtype == DType::F32,
              std::string("argmax: x must be F32, got ") + dtype_name(x.dtype));
  LLMRT_CHECK(out.dtype == DType::I32,
              std::string("argmax: out must be I32 (indices, not values), got ") +
                  dtype_name(out.dtype));

  LLMRT_CHECK(x.rank() >= 1, "argmax: x must have rank >= 1");
  LLMRT_CHECK(out.rank() == 1,
              "argmax: out must be 1-D [rows], got " + out.shape_string());

  const int64_t n = x.dim(x.rank() - 1);
  LLMRT_CHECK(n > 0, "argmax: the reduced axis must be non-empty");
  const int64_t rows = static_cast<int64_t>(x.numel()) / n;
  // out is I32, so its element count is what matters, not its bytes.
  LLMRT_CHECK(out.numel() == static_cast<size_t>(rows),
              "argmax: out must hold one index per row -- x is " + x.shape_string() +
                  " so it needs " + std::to_string(rows) + " but got " +
                  out.shape_string());
}

}  // namespace

void argmax(const Tensor& x, Tensor& out) {
  check_argmax_args(x, out);

  const int64_t n = x.dim(x.rank() - 1);
  const int64_t rows = static_cast<int64_t>(x.numel()) / n;

  const float* x_base = x.f32();
  int32_t* out_base = out.i32();

  for (int64_t r = 0; r < rows; ++r) {
    const float* row = x_base + r * n;

    // Seeded with -infinity rather than row[0], then take strictly greater
    // values. Three consequences, all intended:
    //
    //   * ties keep the lowest index, because the first of two equal values is
    //     the one that displaces the sentinel;
    //   * NaN never wins anywhere -- `NaN > v` is false for every v, including
    //     -inf. Seeding from row[0] would have made a NaN in position 0
    //     unbeatable, which is a special case nobody wants to remember;
    //   * a row that is entirely NaN or entirely -inf returns index 0, which
    //     the caller can detect by looking at that one value.
    float best = -std::numeric_limits<float>::infinity();
    int64_t best_j = 0;
    for (int64_t j = 0; j < n; ++j) {
      if (row[j] > best) {
        best = row[j];
        best_j = j;
      }
    }
    out_base[r] = static_cast<int32_t>(best_j);
  }
}

}  // namespace cpu
}  // namespace llmrt
