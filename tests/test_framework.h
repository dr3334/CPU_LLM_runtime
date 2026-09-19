// Tiny dependency-free test harness.
//
// Rationale: the project avoids third-party libraries, and the op tests only
// need registration, assertions and a pass/fail exit code.
//
// Usage:
//   #include "test_framework.h"
//   LLMRT_TEST(my_case) {
//     CHECK_TRUE(1 + 1 == 2);
//     CHECK_NEAR(0.1f + 0.2f, 0.3f, 1e-6f);
//   }
//   int main() { return llmrt_test::run_all("my_suite"); }
#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

namespace llmrt_test {

struct TestCase {
  const char* name;
  std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> r;
  return r;
}

inline int& failure_count() {
  static int n = 0;
  return n;
}

inline void report_failure(const char* file, int line, const std::string& msg) {
  ++failure_count();
  std::fprintf(stderr, "    FAIL %s:%d\n      %s\n", file, line, msg.c_str());
}

// Formats a value for assertion messages when an arithmetic type is available.
template <typename T>
std::string to_str(const T& v) {
  if constexpr (std::is_same_v<std::decay_t<T>, std::string>) {
    return v;
  } else if constexpr (std::is_same_v<std::decay_t<T>, const char*> ||
                       std::is_same_v<std::decay_t<T>, char*>) {
    return std::string(v);
  } else if constexpr (std::is_same_v<std::decay_t<T>, bool>) {
    return v ? "true" : "false";
  } else if constexpr (std::is_arithmetic_v<std::decay_t<T>>) {
    return std::to_string(v);
  } else {
    return "<value>";
  }
}

struct Registrar {
  Registrar(const char* name, std::function<void()> fn) {
    registry().push_back({name, std::move(fn)});
  }
};

inline int run_all(const char* suite) {
  int failed_tests = 0;
  std::printf("== %s: %zu test(s)\n", suite, registry().size());
  for (auto& t : registry()) {
    failure_count() = 0;
    std::printf("  [ RUN  ] %s\n", t.name);
    std::fflush(stdout);
    // A test that throws is a failing test, not a reason to abort the binary:
    // otherwise one bad case hides the results of every remaining case.
    try {
      t.fn();
    } catch (const std::exception& e) {
      report_failure("<exception>", 0, std::string("threw: ") + e.what());
    } catch (...) {
      report_failure("<exception>", 0, "threw a non-std::exception");
    }
    if (failure_count() == 0) {
      std::printf("  [ OK   ] %s\n", t.name);
    } else {
      std::printf("  [ FAIL ] %s (%d failed check(s))\n", t.name, failure_count());
      ++failed_tests;
    }
    std::fflush(stdout);
  }
  std::printf("== %s: %zu test(s), %d failed\n", suite, registry().size(), failed_tests);
  return failed_tests == 0 ? 0 : 1;
}

}  // namespace llmrt_test

#define LLMRT_TEST(test_name)                                               \
  static void test_name();                                                  \
  static const ::llmrt_test::Registrar llmrt_registrar_##test_name(         \
      #test_name, test_name);                                               \
  static void test_name()

#define CHECK_MSG(cond, msg)                                                \
  do {                                                                      \
    if (!(cond)) ::llmrt_test::report_failure(__FILE__, __LINE__, (msg));   \
  } while (0)

#define CHECK_TRUE(cond)                                                    \
  CHECK_MSG(cond, std::string("expected true: ") + #cond)

#define CHECK_FALSE(cond)                                                   \
  CHECK_MSG(!(cond), std::string("expected false: ") + #cond)

#define CHECK_EQ(a, b)                                                      \
  do {                                                                      \
    const auto& llmrt_a = (a);                                              \
    const auto& llmrt_b = (b);                                              \
    if (!(llmrt_a == llmrt_b)) {                                            \
      ::llmrt_test::report_failure(                                         \
          __FILE__, __LINE__,                                               \
          std::string(#a " == " #b " -- got ") +                            \
              ::llmrt_test::to_str(llmrt_a) + " vs " +                      \
              ::llmrt_test::to_str(llmrt_b));                               \
    }                                                                       \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                               \
  do {                                                                      \
    const double llmrt_a = static_cast<double>(a);                          \
    const double llmrt_b = static_cast<double>(b);                          \
    const double llmrt_t = static_cast<double>(tol);                        \
    if (!(std::fabs(llmrt_a - llmrt_b) <= llmrt_t)) {                       \
      ::llmrt_test::report_failure(                                         \
          __FILE__, __LINE__,                                               \
          std::string(#a " ~= " #b " -- got ") +                            \
              ::llmrt_test::to_str(llmrt_a) + " vs " +                      \
              ::llmrt_test::to_str(llmrt_b) + " (tol " +                    \
              ::llmrt_test::to_str(llmrt_t) + ")");                         \
    }                                                                       \
  } while (0)
