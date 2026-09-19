// Reader for the safetensors format (https://github.com/huggingface/safetensors).
//
// Layout: [u64 header_len][header_len bytes of JSON][tensor data]
// `data_offsets` in the header are relative to the start of the data section.
//
// The file is memory-mapped read-only and never copied: tensor lookups hand
// back pointers straight into the mapping. Converting a tensor to the compute
// dtype is an explicit opt-in via to_f32(), so a caller can decide to keep
// weights in bf16 and convert lazily (which is what the scheduler will do when
// placing weights on a device).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "llmrt/common.h"
#include "llmrt/convert.h"
#include "llmrt/json.h"

namespace llmrt {

struct TensorInfo {
  std::string name;
  DType dtype = DType::F32;
  std::string dtype_str;  // exactly as written in the header ("BF16", ...)
  std::vector<int64_t> shape;
  size_t begin = 0;  // offset into the data section
  size_t end = 0;

  size_t nbytes() const { return end - begin; }
  size_t numel() const;
  int64_t dim(int i) const;
  std::string shape_string() const;
};

class SafeTensors {
 public:
  // Maps `path` and parses the header. Throws on malformed files.
  static SafeTensors open(const std::string& path);

  ~SafeTensors();
  SafeTensors(SafeTensors&& other) noexcept;
  SafeTensors& operator=(SafeTensors&& other) noexcept;
  SafeTensors(const SafeTensors&) = delete;
  SafeTensors& operator=(const SafeTensors&) = delete;

  const std::vector<TensorInfo>& tensors() const { return tensors_; }
  size_t size() const { return tensors_.size(); }

  // Returns nullptr when absent.
  const TensorInfo* find(std::string_view name) const;
  // Throws when absent.
  const TensorInfo& at(std::string_view name) const;

  // Pointer to the tensor's raw on-disk bytes (still in `dtype`).
  const void* raw(const TensorInfo& t) const {
    return static_cast<const char*>(map_) + data_offset_ + t.begin;
  }

  // Typed view of the raw bytes; the caller must know the dtype.
  template <typename T>
  const T* ptr(const TensorInfo& t) const {
    return static_cast<const T*>(raw(t));
  }

  // Copies and converts the whole tensor to f32.
  std::vector<float> to_f32(const TensorInfo& t) const;
  std::vector<float> to_f32(std::string_view name) const;

  size_t file_size() const { return file_size_; }
  size_t data_offset() const { return data_offset_; }
  // Sum of all tensor payloads (excludes the header).
  size_t data_bytes() const { return data_bytes_; }
  const std::string& path() const { return path_; }
  const json::Value& header() const { return header_; }

  // {dtype string -> number of tensors}, in first-seen order.
  std::vector<std::pair<std::string, size_t>> dtype_histogram() const;

 private:
  SafeTensors() = default;
  void release() noexcept;

  std::string path_;
  void* map_ = nullptr;
  size_t file_size_ = 0;
  size_t data_offset_ = 0;
  size_t data_bytes_ = 0;
  std::vector<TensorInfo> tensors_;
  json::Value header_;
};

// Maps a safetensors dtype string to DType. Throws for unsupported dtypes.
DType dtype_from_string(std::string_view s);

}  // namespace llmrt
