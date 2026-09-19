#pragma once

namespace llmrt {

// Returns the runtime version string, e.g. "0.1.0".
const char* version();

// Returns the git-ish build description ("llmrt 0.1.0 <build_type>").
const char* build_description();

}  // namespace llmrt
