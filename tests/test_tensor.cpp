#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "llmrt/common.h"
#include "llmrt/tensor.h"

#include "test_framework.h"

using namespace llmrt;

namespace {

bool throws(const std::function<void()>& fn) {
  try {
    fn();
  } catch (const Error&) {
    return true;
  }
  return false;
}

std::string vec_str(const std::vector<int64_t>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) s += ", ";
    s += std::to_string(v[i]);
  }
  return s + "]";
}

void expect_strides(const std::vector<int64_t>& shape, const std::vector<int64_t>& want) {
  const std::vector<int64_t> got = contiguous_strides(shape);
  CHECK_MSG(got == want, "strides for " + vec_str(shape) + ": got " + vec_str(got) +
                             ", want " + vec_str(want));
}

}  // namespace

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

LLMRT_TEST(computes_row_major_strides) {
  expect_strides({}, {});
  expect_strides({5}, {1});
  expect_strides({3, 4}, {4, 1});
  expect_strides({2, 3, 4}, {12, 4, 1});
  // The Qwen3 attention shapes.
  expect_strides({12, 16, 128}, {2048, 128, 1});
  expect_strides({16, 12, 128}, {1536, 128, 1});
}

LLMRT_TEST(numel_and_nbytes_follow_shape_and_dtype) {
  std::vector<float> buf(24);
  const Tensor t = Tensor::contiguous(buf.data(), DType::F32, DeviceKind::CPU, {2, 3, 4});
  CHECK_EQ(t.rank(), 3);
  CHECK_EQ(t.numel(), size_t{24});
  CHECK_EQ(t.nbytes(), size_t{96});
  CHECK_EQ(t.element_size(), size_t{4});
  CHECK_EQ(t.dim(0), int64_t{2});
  CHECK_EQ(t.dim(2), int64_t{4});

  CHECK_EQ(Tensor::contiguous(nullptr, DType::BF16, DeviceKind::CPU, {2, 3, 4}).nbytes(),
           size_t{48});
  CHECK_EQ(Tensor::contiguous(nullptr, DType::I8, DeviceKind::CPU, {7}).nbytes(), size_t{7});
}

LLMRT_TEST(i4_is_nibble_packed_and_rounds_up) {
  // 4 elements occupy 2 bytes, 5 elements occupy 3.
  CHECK_EQ(Tensor::contiguous(nullptr, DType::I4, DeviceKind::CPU, {4}).nbytes(), size_t{2});
  CHECK_EQ(Tensor::contiguous(nullptr, DType::I4, DeviceKind::CPU, {5}).nbytes(), size_t{3});
  CHECK_EQ(Tensor::contiguous(nullptr, DType::I4, DeviceKind::CPU, {2, 3}).nbytes(),
           size_t{3});  // 6 elements -> 3 bytes
  // A packed dtype has no byte-addressable element size.
  CHECK_TRUE(throws([] {
    (void)Tensor::contiguous(nullptr, DType::I4, DeviceKind::CPU, {4}).element_size();
  }));
}

LLMRT_TEST(scalar_shape_yields_one_element) {
  const Tensor t = Tensor::contiguous(nullptr, DType::F32, DeviceKind::CPU, {});
  CHECK_EQ(t.rank(), 0);
  CHECK_EQ(t.numel(), size_t{1});
  CHECK_EQ(t.nbytes(), size_t{4});
  CHECK_TRUE(t.is_contiguous());
}

LLMRT_TEST(out_of_range_axis_is_reported) {
  const Tensor t = Tensor::contiguous(nullptr, DType::F32, DeviceKind::CPU, {2, 3});
  CHECK_TRUE(throws([&] { (void)t.dim(2); }));
  CHECK_TRUE(throws([&] { (void)t.dim(-1); }));
  CHECK_TRUE(throws([&] { (void)t.transpose(0, 5); }));
}

// ---------------------------------------------------------------------------
// Contiguity
// ---------------------------------------------------------------------------

LLMRT_TEST(contiguous_tensors_report_contiguous) {
  for (const std::vector<int64_t>& shape :
       std::vector<std::vector<int64_t>>{{}, {1}, {5}, {3, 4}, {2, 3, 4}, {12, 16, 128}}) {
    CHECK_TRUE(Tensor::contiguous(nullptr, DType::F32, DeviceKind::CPU, shape).is_contiguous());
  }
}

LLMRT_TEST(transposed_2d_view_is_not_contiguous) {
  const Tensor t = Tensor::contiguous(nullptr, DType::F32, DeviceKind::CPU, {3, 4});
  const Tensor tt = t.transpose(0, 1);
  CHECK_TRUE(tt.shape == std::vector<int64_t>({4, 3}));
  CHECK_TRUE(tt.strides == std::vector<int64_t>({1, 4}));
  CHECK_FALSE(tt.is_contiguous());
  // Transposing twice returns to the original layout.
  CHECK_TRUE(tt.transpose(0, 1).is_contiguous());
}

LLMRT_TEST(size_one_axes_do_not_affect_contiguity) {
  // A 5-element run with a stride that only "looks wrong" on a size-1 axis is
  // still contiguous; a naive strides == dense_strides check gets this wrong.
  Tensor t = Tensor::contiguous(nullptr, DType::F32, DeviceKind::CPU, {1, 8});
  t.strides = {999, 1};
  CHECK_TRUE(t.is_contiguous());

  // Transposing a size-1 axis keeps it contiguous.
  const Tensor a = Tensor::contiguous(nullptr, DType::F32, DeviceKind::CPU, {5, 1});
  CHECK_TRUE(a.transpose(0, 1).is_contiguous());
}

LLMRT_TEST(slicing_axis_0_stays_contiguous_but_axis_1_does_not) {
  const Tensor t = Tensor::contiguous(nullptr, DType::F32, DeviceKind::CPU, {3, 4});
  const Tensor s = t.slice(0, 1, 3);
  CHECK_TRUE(s.shape == std::vector<int64_t>({2, 4}));
  CHECK_TRUE(s.strides == std::vector<int64_t>({4, 1}));
  CHECK_TRUE(s.is_contiguous());

  // Slicing the innermost axis of a 2-D tensor makes it strided, because rows
  // are no longer back to back.
  const Tensor inner = t.slice(1, 1, 3);
  CHECK_TRUE(inner.shape == std::vector<int64_t>({3, 2}));
  CHECK_TRUE(inner.strides == std::vector<int64_t>({4, 1}));
  CHECK_FALSE(inner.is_contiguous());
}

// ---------------------------------------------------------------------------
// Views carry an offset into the same storage
// ---------------------------------------------------------------------------

LLMRT_TEST(views_share_storage_and_offset_into_it) {
  std::vector<float> buf(12);
  for (size_t i = 0; i < buf.size(); ++i) buf[i] = static_cast<float>(i);

  const Tensor t = Tensor::contiguous(buf.data(), DType::F32, DeviceKind::CPU, {3, 4});
  const Tensor row1 = t.slice(0, 1, 2);

  CHECK_EQ(row1.offset, int64_t{4});
  CHECK_EQ(row1.numel(), size_t{4});
  // raw() resolves the offset, so the view really points at element 4.
  CHECK_TRUE(static_cast<const float*>(row1.raw()) == buf.data() + 4);
  CHECK_EQ(*static_cast<const float*>(row1.raw()), 4.0);
  // The base pointer is unchanged: this is a view, not a copy.
  CHECK_TRUE(t.data == buf.data());
  CHECK_TRUE(row1.data == buf.data());
}

LLMRT_TEST(reshape_is_free_for_contiguous_data) {
  std::vector<float> buf(24);
  buf[7] = 42.0f;
  const Tensor t = Tensor::contiguous(buf.data(), DType::F32, DeviceKind::CPU, {2, 3, 4});
  const Tensor r = t.reshape({6, 4});

  CHECK_TRUE(r.shape == std::vector<int64_t>({6, 4}));
  CHECK_EQ(r.numel(), size_t{24});
  CHECK_TRUE(r.data == t.data);
  CHECK_EQ(r.offset, int64_t{0});
  // Element 7 of the original is element 7 of the reshape: same bytes.
  CHECK_EQ(static_cast<const float*>(r.raw())[7], 42.0f);
}

LLMRT_TEST(reshape_rejects_element_count_mismatch) {
  const Tensor t = Tensor::contiguous(nullptr, DType::F32, DeviceKind::CPU, {2, 3, 4});
  CHECK_TRUE(throws([&] { (void)t.reshape({5, 5}); }));
  CHECK_TRUE(throws([&] { (void)t.reshape({24, 2}); }));
}

LLMRT_TEST(reshape_rejects_strided_input) {
  const Tensor t = Tensor::contiguous(nullptr, DType::F32, DeviceKind::CPU, {3, 4});
  const Tensor strided = t.transpose(0, 1);
  // Rearranging a transposed view by claiming a new dense shape would silently
  // read the wrong elements, so it must fail loudly.
  CHECK_TRUE(throws([&] { (void)strided.reshape({12}); }));

  bool named = false;
  try {
    strided.require_contiguous("matmul");
  } catch (const Error& e) {
    named = std::string(e.what()).find("matmul") != std::string::npos;
  }
  CHECK_TRUE(named);
}

// ---------------------------------------------------------------------------
// Typed access
// ---------------------------------------------------------------------------

LLMRT_TEST(f32_accessor_checks_dtype_and_device) {
  std::vector<float> buf(4);
  Tensor t = Tensor::contiguous(buf.data(), DType::F32, DeviceKind::CPU, {4});
  CHECK_TRUE(t.f32() == buf.data());
  CHECK_TRUE(static_cast<const Tensor&>(t).f32() == buf.data());

  // Wrong dtype.
  Tensor bf16 = Tensor::contiguous(buf.data(), DType::BF16, DeviceKind::CPU, {4});
  CHECK_TRUE(throws([&] { (void)bf16.f32(); }));

  // Not on the host: `data` would be a cl_mem and must not be dereferenced.
  Tensor gpu = Tensor::contiguous(reinterpret_cast<void*>(0x10), DType::F32, DeviceKind::OpenCL,
                                  {4});
  CHECK_TRUE(throws([&] { (void)gpu.f32(); }));
  // raw() is still usable to hand the handle to a backend.
  CHECK_TRUE(gpu.raw() == reinterpret_cast<void*>(0x10));
}

LLMRT_TEST(descriptions_name_dtype_shape_device_and_layout) {
  std::vector<float> buf(12);
  const Tensor t = Tensor::contiguous(buf.data(), DType::F32, DeviceKind::CPU, {3, 4});
  CHECK_EQ(t.shape_string(), std::string("[3, 4]"));
  CHECK_EQ(t.describe(), std::string("F32 [3, 4] on cpu"));

  const Tensor g = Tensor::contiguous(nullptr, DType::BF16, DeviceKind::OpenCL, {2048, 1024});
  CHECK_EQ(g.shape_string(), std::string("[2048, 1024]"));
  CHECK_EQ(g.describe(), std::string("BF16 [2048, 1024] on opencl"));

  // A strided view spells out its strides, which is what makes failures legible.
  const Tensor s = t.transpose(0, 1);
  CHECK_MSG(s.describe().find("strides=[1, 4]") != std::string::npos, s.describe());
}

// ---------------------------------------------------------------------------
// The shapes the Qwen3 forward pass actually uses
// ---------------------------------------------------------------------------

LLMRT_TEST(qwen3_head_shapes_behave_as_expected) {
  // q_proj: [seq, 1024] -> [seq, 16, 128] -> transposed [16, seq, 128]
  std::vector<float> q(12 * 2048);
  const Tensor proj = Tensor::contiguous(q.data(), DType::F32, DeviceKind::CPU, {12, 2048});
  const Tensor heads = proj.reshape({12, 16, 128});
  CHECK_TRUE(heads.is_contiguous());
  CHECK_EQ(heads.numel(), size_t{12} * 2048);

  const Tensor transposed = heads.transpose(0, 1);
  CHECK_TRUE(transposed.shape == std::vector<int64_t>({16, 12, 128}));
  CHECK_TRUE(transposed.strides == std::vector<int64_t>({128, 2048, 1}));
  CHECK_FALSE(transposed.is_contiguous());

  // A KV-cache-style view: one head's slot at a single position.
  const Tensor cache =
      Tensor::contiguous(q.data(), DType::F32, DeviceKind::CPU, {8, 4096, 128});
  const Tensor head3 = cache.slice(0, 3, 4);
  const Tensor pos7 = head3.slice(1, 7, 8);
  CHECK_TRUE(pos7.shape == std::vector<int64_t>({1, 1, 128}));
  CHECK_EQ(pos7.offset, int64_t{3} * 4096 * 128 + 7 * 128);
  CHECK_TRUE(pos7.is_contiguous());
}

int main() { return llmrt_test::run_all("tensor"); }
