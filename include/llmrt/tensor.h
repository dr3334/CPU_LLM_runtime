// The Tensor descriptor: what shape of data lives at which pointer, on which device.
//
// Design decisions worth knowing before using it
// ----------------------------------------------
// 1. A Tensor is a VIEW, not an owner. It holds a pointer, an element offset,
//    a shape and strides -- nothing else. Copying one is cheap. This keeps
//    ownership in exactly one place (the memory manager, or a std::vector in a
//    test) instead of scattering "does this tensor own its bytes?" checks
//    through every call site.
//
// 2. `data` is deliberately untyped. For DeviceKind::CPU it is a host pointer;
//    for DeviceKind::OpenCL it holds a cl_mem. One descriptor therefore
//    describes work on either backend, and only the backend knows how to
//    interpret the pointer. That is what lets a layer be moved between devices
//    without changing the code that builds it.
//
// 3. shape and strides are counted in ELEMENTS, not bytes, so that a view can
//    be expressed without knowing the element size. Strides exist from the
//    start because attention needs the same buffer read as [seq, heads, dim]
//    and as [heads, seq, dim], and because decode is memory-bandwidth bound --
//    being able to express a transposed view without copying the bytes is the
//    difference between one pass over memory and two.
//
// 4. The hand-written CPU kernels currently REQUIRE contiguous inputs and say
//    so explicitly (see Tensor::require_contiguous). Strided kernels are a
//    later optimisation; the views below let us describe the layouts now, and
//    materialising a contiguous copy at these sizes is cheap.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "llmrt/common.h"

namespace llmrt {

// Row-major dense strides for `shape`. Empty shape (a scalar) yields empty
// strides and a numel of 1.
std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape);

struct Tensor {
  void* data = nullptr;      // base pointer of the buffer (host ptr or cl_mem)
  int64_t offset = 0;        // element offset from `data`
  DType dtype = DType::F32;
  DeviceKind device = DeviceKind::CPU;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;  // in elements

  // ---- construction ------------------------------------------------------
  // Describes row-major contiguous data.
  static Tensor contiguous(void* data, DType dtype, DeviceKind device,
                           std::vector<int64_t> shape);

  // ---- introspection -----------------------------------------------------
  int rank() const { return static_cast<int>(shape.size()); }
  int64_t dim(int i) const;
  size_t numel() const;
  // Bytes actually occupied. Nibble-packed dtypes (I4) round up.
  size_t nbytes() const;
  // Bytes per element. Throws for packed dtypes, whose elements are not
  // byte-addressable.
  size_t element_size() const;

  bool is_contiguous() const;
  bool is_cpu() const { return device == DeviceKind::CPU; }
  bool empty() const { return numel() == 0; }
  bool same_shape(const Tensor& other) const { return shape == other.shape; }

  std::string shape_string() const;
  std::string describe() const;

  // ---- data access -------------------------------------------------------
  // Typed accessors. Only meaningful on CPU tensors: a cl_mem is not
  // dereferenceable from the host. Both throw on a device or dtype mismatch.
  float* f32();
  const float* f32() const;

  // Untyped escape hatch, for handing the raw pointer to a backend. Returns
  // data + offset * element_size, so views resolve correctly.
  void* raw();
  const void* raw() const;

  // ---- views (no copy) ---------------------------------------------------
  // Reinterprets the same bytes under a new shape. Requires contiguity.
  Tensor reshape(const std::vector<int64_t>& new_shape) const;
  // Swaps two axes by swapping their shape and stride entries.
  Tensor transpose(int axis_a, int axis_b) const;
  // Half-open slice along `axis`: [begin, end).
  Tensor slice(int axis, int64_t begin, int64_t end) const;

  // Throws a descriptive error unless the tensor is contiguous. The CPU ops
  // call this first so a strided input fails loudly rather than silently
  // producing wrong numbers.
  void require_contiguous(const char* who) const;
};

}  // namespace llmrt
