#include "args.h"

#include <cstdlib>
#include <stdexcept>

namespace llmrt {
namespace cli {

bool Args::has_flag(const std::string& key) const {
  for (const auto& f : flags)
    if (f == key) return true;
  return false;
}

bool Args::has_option(const std::string& key) const {
  for (const auto& kv : options)
    if (kv.first == key) return true;
  return false;
}

std::string Args::get(const std::string& key, const std::string& fallback) const {
  for (const auto& kv : options)
    if (kv.first == key) return kv.second;
  return fallback;
}

std::string Args::require(const std::string& key) const {
  if (!has_option(key)) throw std::runtime_error("missing required option --" + key);
  return get(key);
}

int Args::get_int(const std::string& key, int fallback) const {
  if (!has_option(key)) return fallback;
  return std::atoi(get(key).c_str());
}

float Args::get_float(const std::string& key, float fallback) const {
  if (!has_option(key)) return fallback;
  return std::strtof(get(key).c_str(), nullptr);
}

Args parse_args(int argc, char** argv, int start) {
  Args out;
  for (int i = start; i < argc; ++i) {
    std::string a = argv[i];
    if (a.size() >= 2 && a[0] == '-' && a[1] == '-') {
      std::string key = a.substr(2);
      auto eq = key.find('=');
      if (eq != std::string::npos) {
        out.options.emplace_back(key.substr(0, eq), key.substr(eq + 1));
      } else if (i + 1 < argc && argv[i + 1][0] != '-') {
        out.options.emplace_back(key, argv[++i]);
      } else {
        out.flags.push_back(key);
      }
    } else {
      out.positional.push_back(a);
    }
  }
  return out;
}

}  // namespace cli
}  // namespace llmrt
