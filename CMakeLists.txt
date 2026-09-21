cmake_minimum_required(VERSION 3.16)
project(llmrt LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(LLMRT_BUILD_TESTS "Build unit tests" ON)
option(LLMRT_ENABLE_OPENCL "Enable the hand-written OpenCL backend" OFF)
option(LLMRT_ENABLE_AVX2 "Enable AVX2/FMA CPU kernels" OFF)

# ---------------------------------------------------------------------------
# llmrt_core -- the runtime library.
# No third-party dependencies by design: the inference path is hand-written.
# ---------------------------------------------------------------------------
add_library(llmrt_core STATIC
  src/core/json.cpp
  src/core/tensor.cpp
  src/core/version.cpp
  src/io/convert.cpp
  src/io/safetensors.cpp
  src/model/config.cpp
  src/model/weights.cpp
  src/ops/cpu/attention.cpp
  src/ops/cpu/matmul.cpp
  src/ops/cpu/rmsnorm.cpp
  src/ops/cpu/rope.cpp
  src/ops/cpu/softmax.cpp
  src/ops/cpu/swiglu.cpp
)
target_include_directories(llmrt_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(NOT MSVC)
  target_compile_options(llmrt_core PRIVATE -Wall -Wextra)
endif()

if(LLMRT_ENABLE_AVX2 AND NOT ANDROID)
  target_compile_options(llmrt_core PRIVATE -mavx2 -mfma)
endif()

# ---------------------------------------------------------------------------
# llmrt -- command line front end
# ---------------------------------------------------------------------------
add_executable(llmrt
  src/cli/main.cpp
  src/cli/args.cpp
  src/cli/cmd_inspect.cpp
  src/cli/cmd_tok.cpp
  src/cli/cmd_generate.cpp
  src/cli/cmd_bench.cpp
)
target_link_libraries(llmrt PRIVATE llmrt_core)
target_include_directories(llmrt PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/cli")

if(NOT MSVC)
  target_compile_options(llmrt PRIVATE -Wall -Wextra)
endif()

# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
if(LLMRT_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
