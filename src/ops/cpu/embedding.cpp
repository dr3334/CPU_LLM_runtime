// Gathers rows of an embedding table by token id -- the first step of the
// forward pass, and the only op here that does no arithmetic.
//
//   table [vocab, hidden]   f32
//   ids   [S]               i32, every value in [0, vocab)
//   out   [S, hidden]       f32
//
//   out[s][:] = table[ids[s]][:]
//
// ---------------------------------------------------------------------------
// 三个"没有"
// ---------------------------------------------------------------------------
//
// 每一条在别的模型里都有对应做法，顺手加上就全错：
//
//  ① 没有 padding 行
//     config.json 里**没有 pad_token_id**，所以参考实现构造的是
//       nn.Embedding(vocab_size, hidden_size, padding_idx=None)
//     每个 id 都指向真行。实测 checkpoint 里全零行数为 0，确认没有预留的
//     padding 行。有些实现会特判 padding_idx 输出零向量 —— 这里不该有。
//
//  ② 没有缩放
//     Gemma 系会乘 sqrt(hidden_size)。modeling_qwen3.py 里没有这个因子
//     （grep embed_scale 无结果）。乘了就全错。
//
//  ③ 没有重归一化 / 没有类型转换
//
// ---------------------------------------------------------------------------
// 为什么"逐位相同"是可以断言的
// ---------------------------------------------------------------------------
//
// 别的算子的判据是"误差够小"（rel < 1e-4），因为它们要做浮点运算，累加顺序
// 不同结果就不同。这个算子**只搬字节**，没有任何舍入点，所以
//   table[input_ids] 复现 embed_out
// 是构造上成立的，不是靠容差兜住的。实测最大绝对差 0.000e+00。
//
// 实现上用 memcpy 而不是逐元素循环，不只是为了快：
//   - 它准确表达了"搬一段连续字节"，而不是"处理一串数值"
//   - 循环容易被后来的人改成"顺便做点什么"，memcpy 不会
//
// ---------------------------------------------------------------------------
// ids 的检查和其他张量不一样
// ---------------------------------------------------------------------------
//
// ids 是运行时里唯一用整数 dtype 的张量，所以：
//   - 查 I32 而不是 F32
//   - 多一项**范围检查** ids[s] ∈ [0, vocab)
//
// 越界 id 会读到表外，属于"静默损坏"那一类。代价是 O(S) 次比较 —— S=512 时
// 512 次，而 gather 本身要搬 512*1024 个 float，可以忽略。
//
// ⚠️ 下界和上界一样重要：int32_t 是**有符号**的，负 id 得到的是负字节偏移
//    （指向表前面），而不是一个很大的偏移。只写 `id < vocab` 是不够的。

#include "llmrt/ops.h"

#include <cstring>

#include "llmrt/common.h"

namespace llmrt {
namespace cpu {

namespace {

void check_embedding_args(const Tensor& table, const Tensor& ids, const Tensor& out) {
  table.require_contiguous("embedding");
  ids.require_contiguous("embedding");
  out.require_contiguous("embedding");

  LLMRT_CHECK(table.is_cpu() && ids.is_cpu() && out.is_cpu(),
              "embedding: all tensors must be host (CPU) tensors");
  // Split rather than combined: ids is I32 while the other two are F32, so a
  // single message would have to say "these are not all F32", which is true but
  // unhelpful when the problem is specifically that ids arrived as floats.
  LLMRT_CHECK(table.dtype == DType::F32 && out.dtype == DType::F32,
              "embedding: table and out must be F32, got " + table.shape_string() + " / " +
                  out.shape_string());
  LLMRT_CHECK(ids.dtype == DType::I32,
              std::string("embedding: ids must be I32, got ") + dtype_name(ids.dtype) +
                  " -- token ids are indices, not quantities");

  LLMRT_CHECK(table.rank() == 2,
              "embedding: table must be 2-D [vocab, hidden], got " + table.shape_string());
  LLMRT_CHECK(ids.rank() == 1,
              "embedding: ids must be 1-D [S], got " + ids.shape_string());
  LLMRT_CHECK(out.rank() == 2,
              "embedding: out must be 2-D [S, hidden], got " + out.shape_string());

  const int64_t vocab = table.dim(0);
  const int64_t hidden = table.dim(1);
  const int64_t num_ids = ids.dim(0);

  LLMRT_CHECK(out.dim(0) == num_ids && out.dim(1) == hidden,
              "embedding: out should be [" + std::to_string(num_ids) + ", " +
                  std::to_string(hidden) + "] to match ids and table, but is " +
                  out.shape_string());

  // The range scan. O(S), and the reason this op can promise not to read past
  // the table.
  const int32_t* id = ids.i32();
  for (int64_t s = 0; s < num_ids; ++s) {
    LLMRT_CHECK(id[s] >= 0 && id[s] < vocab,
                "embedding: id at position " + std::to_string(s) + " is " +
                    std::to_string(id[s]) + ", outside [0, " + std::to_string(vocab) +
                    ") -- reading it would leave the table");
  }
}

}  // namespace

void embedding(const Tensor& table, const Tensor& ids, Tensor& out) {
  check_embedding_args(table, ids, out);

  const int64_t num_ids = ids.dim(0);
  const int64_t hidden = table.dim(1);
  const size_t row_bytes = static_cast<size_t>(hidden) * sizeof(float);

  const int32_t* id_base = ids.i32();
  const float* table_base = table.f32();
  float* out_base = out.f32();

  // One memcpy per id. The source row is found by scaling the id by the row
  // length -- the same r*n idea as rmsnorm, with n = hidden and r = id.
  for (int64_t s = 0; s < num_ids; ++s) {
    const float* row = table_base + static_cast<int64_t>(id_base[s]) * hidden;
    std::memcpy(out_base + s * hidden, row, row_bytes);
  }
}

}  // namespace cpu
}  // namespace llmrt
