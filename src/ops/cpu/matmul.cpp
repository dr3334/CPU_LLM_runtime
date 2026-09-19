#include "llmrt/ops.h"

#include "llmrt/common.h"

namespace llmrt {
namespace cpu {

namespace {

void check_matmul_args(const Tensor& a, const Tensor& b, const Tensor& out, bool transpose_b) {
  a.require_contiguous("matmul");
  b.require_contiguous("matmul");
  out.require_contiguous("matmul");

  LLMRT_CHECK(a.is_cpu() && b.is_cpu() && out.is_cpu(),
              "matmul: all tensors must be host (CPU) tensors");
  LLMRT_CHECK(a.dtype == DType::F32 && b.dtype == DType::F32 && out.dtype == DType::F32,
              "matmul: all tensors must be F32");

  // Higher-rank activations are reshaped by the caller, which is free for
  // contiguous data and keeps this kernel's indexing two-dimensional.
  LLMRT_CHECK(a.rank() == 2, "matmul: a must be 2-D [M, K], got " + a.shape_string());
  LLMRT_CHECK(b.rank() == 2, "matmul: b must be 2-D, got " + b.shape_string());
  LLMRT_CHECK(out.rank() == 2, "matmul: out must be 2-D [M, N], got " + out.shape_string());

  const int64_t M = a.dim(0);
  const int64_t K = a.dim(1);
  const int64_t N = transpose_b ? b.dim(0) : b.dim(1);
  const int64_t b_inner = transpose_b ? b.dim(1) : b.dim(0);

  LLMRT_CHECK(b_inner == K,
              std::string("matmul: inner dimensions disagree -- a is [") + std::to_string(M) +
                  ", " + std::to_string(K) + "] so b must be [" +
                  (transpose_b ? std::to_string(N) + ", " + std::to_string(K)
                               : std::to_string(K) + ", " + std::to_string(N)) +
                  "] with transpose_b=" + (transpose_b ? "true" : "false") + ", but b is " +
                  b.shape_string());
  LLMRT_CHECK(out.dim(0) == M && out.dim(1) == N,
              "matmul: out should be [" + std::to_string(M) + ", " + std::to_string(N) +
                  "] but is " + out.shape_string());
}

}  // namespace

void matmul(const Tensor& a, const Tensor& b, Tensor& out, bool transpose_b) {
  check_matmul_args(a, b, out, transpose_b);

  const int64_t M = a.dim(0);
  const int64_t K = a.dim(1);
  const int64_t N = transpose_b ? b.dim(0) : b.dim(1);

  const float* a_base = a.f32();
  const float* b_base = b.f32();
  float* out_base = out.f32();

  if (transpose_b) {
    // The Qwen3 path. Every projection weight is [out_features, in_features],
    // so both operands are consumed row-wise and the inner loop is a dot
    // product between two contiguous rows -- two sequential streams, no
    // strided access, which is what the hardware wants.
    for (int64_t i = 0; i < M; ++i) {
      const float* a_row = a_base + i * K;
      float* out_row = out_base + i * N;

      for (int64_t j = 0; j < N; ++j) {
        const float* b_row = b_base + j * K;

        float acc = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          acc += a_row[k] * b_row[k];
        }
        out_row[j] = acc;
      }
    }
  } else {
    // b is [K, N] here, so indexing b[k][j] inside a k-innermost loop would
    // stride by N -- a new cache line per element. Iterating k outside j
    // instead makes b's row k contiguous and lets out_row stay in cache while
    // it is accumulated.
    for (int64_t i = 0; i < M; ++i) {
      const float* a_row = a_base + i * K;
      float* out_row = out_base + i * N;

      for (int64_t j = 0; j < N; ++j) out_row[j] = 0.0f;

      for (int64_t k = 0; k < K; ++k) {
        const float a_ik = a_row[k];
        const float* b_row = b_base + k * N;
        for (int64_t j = 0; j < N; ++j) {
          out_row[j] += a_ik * b_row[j];
        }
      }
    }
  }
}

}  // namespace cpu
}  // namespace llmrt
