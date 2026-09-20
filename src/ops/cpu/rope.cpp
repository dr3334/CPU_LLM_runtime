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
#include <vector>

#include "llmrt/common.h"

namespace llmrt {
namespace cpu {

namespace {

// ---------------------------------------------------------------------------
// Argument checks
// ---------------------------------------------------------------------------

void check_rope_frequencies_args(int64_t seq_len, int64_t head_dim, float theta,
                                 const Tensor& cos, const Tensor& sin) {
  LLMRT_CHECK(seq_len > 0, "rope_frequencies: seq_len must be positive");
  LLMRT_CHECK(head_dim > 0, "rope_frequencies: head_dim must be positive");
  // The half-split convention folds the axis in two, so an odd head_dim has no
  // valid rotation.
  LLMRT_CHECK(head_dim % 2 == 0, "rope_frequencies: head_dim must be even, got " +
                                     std::to_string(head_dim));
  LLMRT_CHECK(theta > 0.0f, "rope_frequencies: theta must be positive");

  cos.require_contiguous("rope_frequencies");
  sin.require_contiguous("rope_frequencies");
  LLMRT_CHECK(cos.is_cpu() && sin.is_cpu(),
              "rope_frequencies: cos and sin must be host (CPU) tensors");
  LLMRT_CHECK(cos.dtype == DType::F32 && sin.dtype == DType::F32,
              "rope_frequencies: cos and sin must be F32");

  const std::vector<int64_t> want = {seq_len, head_dim};
  LLMRT_CHECK(cos.shape == want && sin.shape == want,
              "rope_frequencies: cos and sin must both be [seq_len, head_dim] = [" +
                  std::to_string(seq_len) + ", " + std::to_string(head_dim) + "], got " +
                  cos.shape_string() + " and " + sin.shape_string());
}

void check_rope_apply_args(const Tensor& q, const Tensor& k, const Tensor& cos,
                           const Tensor& sin) {
  q.require_contiguous("rope_apply");
  k.require_contiguous("rope_apply");
  cos.require_contiguous("rope_apply");
  sin.require_contiguous("rope_apply");

  LLMRT_CHECK(q.is_cpu() && k.is_cpu() && cos.is_cpu() && sin.is_cpu(),
              "rope_apply: all tensors must be host (CPU) tensors");
  LLMRT_CHECK(q.dtype == DType::F32 && k.dtype == DType::F32 && cos.dtype == DType::F32 &&
                  sin.dtype == DType::F32,
              "rope_apply: all tensors must be F32");

  LLMRT_CHECK(q.rank() >= 2 && k.rank() >= 2,
              "rope_apply: q and k must be at least 2-D [.., seq, head_dim], got " +
                  q.shape_string() + " and " + k.shape_string());
  LLMRT_CHECK(cos.rank() == 2,
              "rope_apply: cos/sin must be 2-D [seq, head_dim], got " + cos.shape_string());
  LLMRT_CHECK(cos.shape == sin.shape,
              "rope_apply: cos and sin must have the same shape, got " + cos.shape_string() +
                  " and " + sin.shape_string());

  // Only the last two axes of q/k must agree with the table. The leading axes
  // are batch and head, and q and k are allowed to differ there: GQA gives q 16
  // heads against k's 8. That is why they are rotated by separate calls below.
  const int64_t seq = cos.dim(0);
  const int64_t head_dim = cos.dim(1);
  LLMRT_CHECK(head_dim % 2 == 0,
              "rope_apply: head_dim must be even, got " + std::to_string(head_dim));
  LLMRT_CHECK(q.dim(q.rank() - 2) == seq && q.dim(q.rank() - 1) == head_dim,
              "rope_apply: q is " + q.shape_string() +
                  " but must end in [seq, head_dim] = [" + std::to_string(seq) + ", " +
                  std::to_string(head_dim) + "]");
  LLMRT_CHECK(k.dim(k.rank() - 2) == seq && k.dim(k.rank() - 1) == head_dim,
              "rope_apply: k is " + k.shape_string() +
                  " but must end in [seq, head_dim] = [" + std::to_string(seq) + ", " +
                  std::to_string(head_dim) + "]");
}

// ---------------------------------------------------------------------------
// Rotation
// ---------------------------------------------------------------------------

// Rotates one tensor in place. `x` ends in [seq, head_dim], contiguous.
//
// A separate function rather than one loop covering both q and k, because under
// GQA they have different row counts. A single loop driven by q's row count
// would write past the end of k's buffer -- silently, since nothing checks.
//
// cos_base / sin_base are [seq, head_dim] and are shared by every row, which is
// exactly the reference's `cos.unsqueeze(1)` broadcast over the head axis.
void rope_apply_one(Tensor& x, const float* cos_base, const float* sin_base) {
  // Taken from the END of the shape, so any number of leading batch/head axes
  // works: they all just become "rows".
  const int64_t head_dim = x.dim(x.rank() - 1);
  const int64_t seq = x.dim(x.rank() - 2);
  const int64_t rows =
      static_cast<int64_t>(x.numel() / static_cast<size_t>(seq) / static_cast<size_t>(head_dim));
  const int64_t half = head_dim / 2;

  float* base = x.f32();

  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t p = 0; p < seq; ++p) {
      // Computed once and reused four times below. Spelling the full index
      // expression out at each use is how off-by-one slips stay invisible.
      const int64_t x_row = (r * seq + p) * head_dim;
      // The table has no row axis, so its offset stops at the position.
      const int64_t angle_row = p * head_dim;

      for (int64_t j = 0; j < half; ++j) {
        // Either output element needs BOTH inputs, so read both before writing
        // either. Writing element by element in increasing j order would
        // clobber x[j] before the second assignment reads it.
        const float lo = base[x_row + j];
        const float hi = base[x_row + j + half];

        // The four products are:
        //   lo * cos[j]      the element's own cosine term
        //   hi * sin[j]      the partner's sine term, negated (front half)
        //   hi * cos[j+h]    the partner's own cosine term
        //   lo * sin[j+h]    the element's sine term (back half)
        base[x_row + j] = lo * cos_base[angle_row + j] - hi * sin_base[angle_row + j];
        base[x_row + j + half] =
            hi * cos_base[angle_row + j + half] + lo * sin_base[angle_row + j + half];
      }
    }
  }
}

}  // namespace

void rope_frequencies(int64_t seq_len, int64_t head_dim, float theta, Tensor& cos,
                      Tensor& sin) {
  check_rope_frequencies_args(seq_len, head_dim, theta, cos, sin);

  const int64_t half = head_dim / 2;

  // f32() hands back the data pointer and validates dtype/device on the way, so
  // a wrong tensor is rejected before a single element is written.
  float* cos_base = cos.f32();
  float* sin_base = sin.f32();

  // Only `half` distinct frequencies exist and each is reused by every
  // position, so hoist the pow() out of the element loop -- otherwise it is
  // recomputed head_dim/2 extra times per position.
  std::vector<float> inv_freq(static_cast<size_t>(half));
  for (int64_t i = 0; i < half; ++i) {
    // Matches the reference's `1 / base ** (2i / dim)` exactly. The f-suffixed
    // literals keep the expression in float; a bare 2.0 would promote it to
    // double and round differently from the reference.
    inv_freq[static_cast<size_t>(i)] = 1.0f / std::pow(theta, 2.0f * i / head_dim);
  }

  for (int64_t p = 0; p < seq_len; ++p) {
    for (int64_t j = 0; j < head_dim; ++j) {
      // `j % half` is what makes the second half of a row repeat the first.
      // That repetition is why the half-split rotation collapses into a plain
      // elementwise multiply: element j and element j+half carry the same
      // angle, so a single cos/sin pair serves both.
      const float angle = static_cast<float>(p) * inv_freq[static_cast<size_t>(j % half)];
      cos_base[p * head_dim + j] = std::cos(angle);
      sin_base[p * head_dim + j] = std::sin(angle);
    }
  }
}

void rope_apply(Tensor& q, Tensor& k, const Tensor& cos, const Tensor& sin) {
  check_rope_apply_args(q, k, cos, sin);

  const float* cos_base = cos.f32();
  const float* sin_base = sin.f32();

  // Two calls rather than one shared loop: q and k have different head counts
  // under GQA (16 vs 8 here) and each must index only its own rows.
  rope_apply_one(q, cos_base, sin_base);
  rope_apply_one(k, cos_base, sin_base);
}

}  // namespace cpu
}  // namespace llmrt
