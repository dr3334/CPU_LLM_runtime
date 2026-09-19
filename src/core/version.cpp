#include "llmrt/version.h"

#define LLMRT_VERSION_STRING "0.1.0"

namespace llmrt {

const char* version() { return LLMRT_VERSION_STRING; }

const char* build_description() {
#if defined(__ANDROID__)
  return "llmrt " LLMRT_VERSION_STRING " (android)";
#else
  return "llmrt " LLMRT_VERSION_STRING " (desktop)";
#endif
}

}  // namespace llmrt
