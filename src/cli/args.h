// Minimal hand-written argument parser for the CLI.
//
// Deliberately dependency-free: the whole project avoids third-party libraries
// in the inference path, and the CLI is built from the same sources.
#pragma once

#include <string>
#include <vector>

namespace llmrt {
namespace cli {

struct Args {
  std::vector<std::string> positional;               // bare arguments
  std::vector<std::pair<std::string, std::string>> options;  // --key value
  std::vector<std::string> flags;                    // --key (no value)

  bool has_flag(const std::string& key) const;
  bool has_option(const std::string& key) const;

  // Returns the value for `key`, or `fallback` if absent.
  std::string get(const std::string& key, const std::string& fallback = "") const;

  // Returns the value for `key`, or throws if absent.
  std::string require(const std::string& key) const;

  // Returns the value for `key` parsed as int, or `fallback` if absent.
  int get_int(const std::string& key, int fallback) const;

  float get_float(const std::string& key, float fallback) const;
};

// Parses argv[start..argc). `--key=value` and `--key value` are both accepted.
Args parse_args(int argc, char** argv, int start);

}  // namespace cli
}  // namespace llmrt
