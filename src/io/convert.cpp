#include "llmrt/convert.h"

namespace llmrt {

namespace {

inline float bitcast_f32(uint32_t u) {
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

}  // namespace

float f16_to_f32(uint16_t v) {
  const uint32_t sign = static_cast<uint32_t>(v & 0x8000u) << 16;
  uint32_t exp = (v >> 10) & 0x1Fu;
  uint32_t mant = v & 0x3FFu;

  if (exp == 0) {
    if (mant == 0) return bitcast_f32(sign);  // +/- 0
    // Subnormal: renormalise into the f32 exponent range.
    exp = 127 - 15 + 1;
    while ((mant & 0x400u) == 0) {
      mant <<= 1;
      --exp;
    }
    mant &= 0x3FFu;
    return bitcast_f32(sign | (exp << 23) | (mant << 13));
  }
  if (exp == 0x1Fu) {
    // Inf or NaN: keep the payload, force the f32 exponent to all ones.
    return bitcast_f32(sign | 0x7F800000u | (mant << 13));
  }
  return bitcast_f32(sign | ((exp - 15 + 127) << 23) | (mant << 13));
}

void convert_to_f32(const void* src, DType src_dtype, float* dst, size_t count) {
  switch (src_dtype) {
    case DType::F32:
      if (src != dst) std::memcpy(dst, src, count * sizeof(float));
      break;
    case DType::BF16: {
      const uint16_t* p = static_cast<const uint16_t*>(src);
      for (size_t i = 0; i < count; ++i) dst[i] = bf16_to_f32(p[i]);
      break;
    }
    case DType::F16: {
      const uint16_t* p = static_cast<const uint16_t*>(src);
      for (size_t i = 0; i < count; ++i) dst[i] = f16_to_f32(p[i]);
      break;
    }
    case DType::I8: {
      const int8_t* p = static_cast<const int8_t*>(src);
      for (size_t i = 0; i < count; ++i) dst[i] = static_cast<float>(p[i]);
      break;
    }
    case DType::I4:
      LLMRT_CHECK(false, "convert_to_f32: I4 is nibble-packed, not elementwise");
  }
}

float to_f32(const void* p, DType dtype) {
  float out = 0.0f;
  convert_to_f32(p, dtype, &out, 1);
  return out;
}

}  // namespace llmrt
