#include <cstring>
#include <string>

#include "llmrt/common.h"
#include "llmrt/version.h"

#include "test_framework.h"

using namespace llmrt;

LLMRT_TEST(dtype_sizes_are_exact) {
  CHECK_EQ(dtype_size(DType::F32), size_t{4});
  CHECK_EQ(dtype_size(DType::BF16), size_t{2});
  CHECK_EQ(dtype_size(DType::F16), size_t{2});
  CHECK_EQ(dtype_size(DType::I8), size_t{1});
}

LLMRT_TEST(dtype_and_device_names) {
  CHECK_EQ(std::string(dtype_name(DType::BF16)), std::string("BF16"));
  CHECK_EQ(std::string(device_name(DeviceKind::CPU)), std::string("cpu"));
  CHECK_EQ(std::string(device_name(DeviceKind::OpenCL)), std::string("opencl"));
}

LLMRT_TEST(version_is_reported) {
  CHECK_TRUE(std::strlen(version()) > 0);
  CHECK_TRUE(std::strlen(build_description()) > 0);
}

LLMRT_TEST(llmrt_check_throws_error_with_location) {
  bool threw = false;
  try {
    LLMRT_CHECK(false, "boom");
  } catch (const Error& e) {
    threw = true;
    const std::string what = e.what();
    CHECK_MSG(what.find("boom") != std::string::npos, "message lost: " + what);
  }
  CHECK_TRUE(threw);
}

LLMRT_TEST(llmrt_check_passes_when_true) {
  bool threw = false;
  try {
    LLMRT_CHECK(true, "should not fire");
  } catch (...) {
    threw = true;
  }
  CHECK_FALSE(threw);
}

int main() { return llmrt_test::run_all("smoke"); }
