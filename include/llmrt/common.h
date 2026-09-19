// Core scalar type and error-handling definitions shared by the whole runtime.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace llmrt {

// ---------------------------------------------------------------------------
// DType / DeviceKind
// ---------------------------------------------------------------------------

// Storage element type. F32 is the compute dtype for the whole runtime in the
// first milestones; BF16 is what Qwen3 checkpoints ship with on disk.
enum class DType : uint8_t {
  F32 = 0,
  BF16,
  F16,
  I8,
  I4,
};

// Where a buffer physically lives. This is the axis that heterogeneous
// scheduling splits Transformer layers along.
enum class DeviceKind : uint8_t {
  CPU = 0,
  OpenCL,
};

inline const char* dtype_name(DType t) {
  switch (t) {
    case DType::F32: return "F32";
    case DType::BF16: return "BF16";
    case DType::F16: return "F16";
    case DType::I8: return "I8";
    case DType::I4: return "I4";
  }
  return "?";
}

inline const char* device_name(DeviceKind d) {
  switch (d) {
    case DeviceKind::CPU: return "cpu";
    case DeviceKind::OpenCL: return "opencl";
  }
  return "?";
}

// Size in bytes of a single element of `t`. Only meaningful for byte-
// addressable dtypes -- check dtype_is_packed() first. For I4 the two nibbles
// share a byte, so this returns 1 while the element size is really 4 bits.
inline size_t dtype_size(DType t) {
  switch (t) {
    case DType::F32: return 4;
    case DType::BF16: return 2;
    case DType::F16: return 2;
    case DType::I8: return 1;
    case DType::I4: return 1;
  }
  return 0;
}

// True when elements are not byte-addressable and share storage. Callers that
// need a byte count must round up (see Tensor::nbytes).
inline bool dtype_is_packed(DType t) { return t == DType::I4; }

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

class Error : public std::runtime_error {
 public:
  explicit Error(const std::string& what) : std::runtime_error(what) {}
};

#define LLMRT_CHECK(cond, msg)                                             \
  do {                                                                     \
    if (!(cond)) {                                                         \
      throw ::llmrt::Error(std::string(__FILE__) + ":" +                   \
                           std::to_string(__LINE__) + ": " + (msg));       \
    }                                                                      \
  } while (0)

#define LLMRT_UNIMPLEMENTED(msg) LLMRT_CHECK(false, std::string("unimplemented: ") + (msg))

}  // namespace llmrt
