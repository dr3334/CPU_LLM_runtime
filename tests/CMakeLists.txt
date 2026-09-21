# Registers one CTest test per added name. Run everything with:
#   ctest --test-dir build --output-on-failure
# Run a single test with:
#   ctest --test-dir build -R test_json --output-on-failure
function(llmrt_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE llmrt_core)
  target_include_directories(${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_definitions(${name} PRIVATE
    LLMRT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  if(NOT MSVC)
    target_compile_options(${name} PRIVATE -Wall -Wextra)
  endif()
  add_test(NAME ${name} COMMAND ${name})
endfunction()

llmrt_add_test(test_smoke test_smoke.cpp)
llmrt_add_test(test_tensor test_tensor.cpp)
llmrt_add_test(test_json test_json.cpp)
llmrt_add_test(test_safetensors test_safetensors.cpp)
llmrt_add_test(test_config test_config.cpp)
llmrt_add_test(test_ops_cpu test_ops_cpu.cpp)
llmrt_add_test(test_forward test_forward.cpp)
