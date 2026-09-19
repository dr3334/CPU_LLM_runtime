// Element-wise conversions between on-disk dtypes and the runtime compute
// dtype (f32).
//
// Qwen3 checkpoints ship as BF16 on disk. BF16 -> F32 is exact (bf16 is simply
// the upper 16 bits of an f32), so upcasting introduces no error at all: any
// numerical difference against the reference comes from the maths, never from
// the loader. F16 is supported as well since quantised / half-precision
// variants are a later optimisation target.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "llmrt/common.h"

namespace llmrt {

// Exact: bfloat16 is the truncated upper half of a float32.
inline float bf16_to_f32(uint16_t v) {
  const uint32_t u = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

// IEEE 754 binary16 -> binary32, exact (including subnormals and Inf/NaN).
float f16_to_f32(uint16_t v);

// Converts `count` elements of `src_dtype` at `src` into f32 at `dst`.
// `src` and `dst` may not overlap. I4 is intentionally unsupported here:
// nibble-packed weights are unpacked by the quantisation layer instead.
void convert_to_f32(const void* src, DType src_dtype, float* dst, size_t count);

// Converts a single element.
float to_f32(const void* p, DType dtype);

// True when convert_to_f32() accepts this dtype.
inline bool is_dtype_convertible(DType t) {
  return t == DType::F32 || t == DType::BF16 || t == DType::F16 || t == DType::I8;
}

}  // namespace llmrt
