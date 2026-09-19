// Minimal safetensors *writer*, used only to build synthetic fixtures for the
// negative tests (missing tensor, wrong shape, unsupported dtype, truncated
// file). Those guards exist to turn silent numerical corruption into an
// immediate error, so they need to be exercised.
#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "llmrt/common.h"

namespace fixture {

struct Entry {
  std::string dtype;                  // "BF16", "F32", ...
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;         // raw payload, size must match dtype*numel
};

inline size_t entry_numel(const Entry& e) {
  size_t n = 1;
  for (const int64_t d : e.shape) n *= static_cast<size_t>(d);
  return n;
}

// Writes a syntactically valid safetensors file. Shape/dtype consistency is
// deliberately NOT enforced: the tests need to produce malformed files.
inline std::string write_safetensors(
    const std::string& path,
    const std::vector<std::pair<std::string, Entry>>& tensors) {
  std::string json = "{";
  size_t offset = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    const std::string& name = tensors[i].first;
    const Entry& e = tensors[i].second;
    if (i) json += ',';
    json += "\"" + name + "\":{\"dtype\":\"" + e.dtype + "\",\"shape\":[";
    for (size_t j = 0; j < e.shape.size(); ++j) {
      if (j) json += ',';
      json += std::to_string(e.shape[j]);
    }
    json += "],\"data_offsets\":[" + std::to_string(offset) + "," +
            std::to_string(offset + e.bytes.size()) + "]}";
    offset += e.bytes.size();
  }
  json += "}";

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  LLMRT_CHECK(out.good(), "fixture: cannot write " + path);
  const uint64_t header_len = json.size();
  out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
  out.write(json.data(), static_cast<std::streamsize>(json.size()));
  for (const auto& t : tensors) {
    out.write(reinterpret_cast<const char*>(t.second.bytes.data()),
              static_cast<std::streamsize>(t.second.bytes.size()));
  }
  out.close();
  return path;
}

// All-zero bf16 payload for a shape.
inline Entry bf16_zeros(std::vector<int64_t> shape) {
  Entry e;
  e.dtype = "BF16";
  e.shape = std::move(shape);
  e.bytes.assign(entry_numel(e) * 2, 0);
  return e;
}

// bf16 payload filled with a constant value.
inline Entry bf16_filled(std::vector<int64_t> shape, uint16_t bits) {
  Entry e = bf16_zeros(std::move(shape));
  for (size_t i = 0; i + 1 < e.bytes.size(); i += 2) {
    e.bytes[i] = static_cast<uint8_t>(bits & 0xFF);
    e.bytes[i + 1] = static_cast<uint8_t>(bits >> 8);
  }
  return e;
}

}  // namespace fixture
