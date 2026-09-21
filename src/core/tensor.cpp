#include "llmrt/tensor.h"

#include <sstream>
#include <utility>

namespace llmrt {

std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size());
  int64_t acc = 1;
  for (size_t i = shape.size(); i-- > 0;) {
    strides[i] = acc;
    acc *= shape[i];
  }
  return strides;
}

Tensor Tensor::contiguous(void* data, DType dtype, DeviceKind device,
                          std::vector<int64_t> shape) {
  Tensor t;
  t.data = data;
  t.offset = 0;
  t.dtype = dtype;
  t.device = device;
  t.shape = std::move(shape);
  t.strides = contiguous_strides(t.shape);
  return t;
}

int64_t Tensor::dim(int i) const {
  LLMRT_CHECK(i >= 0 && static_cast<size_t>(i) < shape.size(),
              "Tensor::dim: axis " + std::to_string(i) + " is out of range for shape " +
                  shape_string());
  return shape[static_cast<size_t>(i)];
}

size_t Tensor::numel() const {
  size_t n = 1;
  for (const int64_t d : shape) {
    LLMRT_CHECK(d >= 0, "Tensor::numel: negative dimension in " + shape_string());
    n *= static_cast<size_t>(d);
  }
  return n;
}

size_t Tensor::nbytes() const {
  const size_t n = numel();
  if (dtype_is_packed(dtype)) return (n + 1) / 2;  // two elements per byte
  return n * dtype_size(dtype);
}

size_t Tensor::element_size() const {
  LLMRT_CHECK(!dtype_is_packed(dtype), std::string("Tensor::element_size: ") +
                                           dtype_name(dtype) +
                                           " is nibble-packed and has no element size");
  return dtype_size(dtype);
}

bool Tensor::is_contiguous() const {
  // Walks back to front accumulating the densely-packed stride: each axis must
  // match, except axes of extent 1, whose stride is irrelevant.
  int64_t acc = 1;
  for (size_t i = shape.size(); i-- > 0;) {
    if (shape[i] == 1) continue;
    if (strides[i] != acc) return false;
    acc *= shape[i];
  }
  return true;
}

std::string Tensor::shape_string() const {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) s += ", ";
    s += std::to_string(shape[i]);
  }
  s += "]";
  return s;
}

std::string Tensor::describe() const {
  std::ostringstream os;
  os << dtype_name(dtype) << ' ' << shape_string() << " on " << device_name(device);
  if (offset != 0) os << " offset=" << offset;
  if (!is_contiguous()) {
    os << " strides=[";
    for (size_t i = 0; i < strides.size(); ++i) {
      if (i) os << ", ";
      os << strides[i];
    }
    os << ']';
  }
  return os.str();
}

void* Tensor::raw() {
  const int64_t bits_per_element =
      dtype_is_packed(dtype) ? 4 : static_cast<int64_t>(dtype_size(dtype)) * 8;
  const int64_t bit_offset = offset * bits_per_element;
  LLMRT_CHECK(bit_offset % 8 == 0,
              std::string("Tensor::raw: element offset ") + std::to_string(offset) +
                  " is not byte-aligned for " + dtype_name(dtype));
  return static_cast<char*>(data) + bit_offset / 8;
}

const void* Tensor::raw() const { return const_cast<Tensor*>(this)->raw(); }

namespace {
// Shared by the typed accessors: "hand me a host pointer of exactly this
// dtype, or tell me why you cannot". Keeping both checks in one place means a
// new accessor cannot accidentally skip the device test and end up
// dereferencing a cl_mem.
void* checked_host_ptr(Tensor& t, DType want, const char* who) {
  LLMRT_CHECK(t.device == DeviceKind::CPU,
              std::string(who) + ": tensor lives on " + device_name(t.device) +
                  ", not on the host");
  LLMRT_CHECK(t.dtype == want, std::string(who) + ": dtype is " + dtype_name(t.dtype) +
                                   ", not " + dtype_name(want));
  return t.raw();
}
}  // namespace

float* Tensor::f32() {
  return static_cast<float*>(checked_host_ptr(*this, DType::F32, "Tensor::f32"));
}

const float* Tensor::f32() const { return const_cast<Tensor*>(this)->f32(); }

int32_t* Tensor::i32() {
  return static_cast<int32_t*>(checked_host_ptr(*this, DType::I32, "Tensor::i32"));
}

const int32_t* Tensor::i32() const { return const_cast<Tensor*>(this)->i32(); }

void Tensor::require_contiguous(const char* who) const {
  LLMRT_CHECK(is_contiguous(), std::string(who) +
                                   ": expected a contiguous tensor, got " + describe());
}

Tensor Tensor::reshape(const std::vector<int64_t>& new_shape) const {
  require_contiguous("reshape");
  Tensor t = *this;
  t.shape = new_shape;
  t.strides = contiguous_strides(new_shape);
  // Evaluated after the swap so the error can report both shapes.
  LLMRT_CHECK(t.numel() == numel(), "reshape: cannot reshape " + shape_string() + " (" +
                                        std::to_string(numel()) + " elements) into " +
                                        t.shape_string() + " (" + std::to_string(t.numel()) +
                                        " elements)");
  return t;
}

Tensor Tensor::transpose(int axis_a, int axis_b) const {
  const int r = rank();
  LLMRT_CHECK(axis_a >= 0 && axis_a < r,
              "transpose: axis " + std::to_string(axis_a) + " is out of range for shape " +
                  shape_string());
  LLMRT_CHECK(axis_b >= 0 && axis_b < r,
              "transpose: axis " + std::to_string(axis_b) + " is out of range for shape " +
                  shape_string());
  Tensor t = *this;
  std::swap(t.shape[axis_a], t.shape[axis_b]);
  std::swap(t.strides[axis_a], t.strides[axis_b]);
  return t;
}

Tensor Tensor::slice(int axis, int64_t begin, int64_t end) const {
  const int r = rank();
  LLMRT_CHECK(axis >= 0 && axis < r,
              "slice: axis " + std::to_string(axis) + " is out of range for shape " +
                  shape_string());
  LLMRT_CHECK(begin >= 0 && begin <= end && end <= shape[static_cast<size_t>(axis)],
              "slice: [" + std::to_string(begin) + ", " + std::to_string(end) +
                  ") is out of range for axis " + std::to_string(axis) + " of extent " +
                  std::to_string(shape[static_cast<size_t>(axis)]));
  Tensor t = *this;
  t.offset += begin * strides[static_cast<size_t>(axis)];
  t.shape[static_cast<size_t>(axis)] = end - begin;
  return t;
}

}  // namespace llmrt
