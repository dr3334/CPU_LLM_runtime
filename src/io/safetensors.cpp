#include "llmrt/safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>

namespace llmrt {

namespace {

// safetensors is strictly little-endian; every target we care about
// (x86-64, aarch64) is little-endian too, so a direct load is correct.
uint64_t load_le64(const void* p) {
  uint64_t v = 0;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

const char* kMetadataKey = "__metadata__";

}  // namespace

DType dtype_from_string(std::string_view s) {
  if (s == "F32") return DType::F32;
  if (s == "F16") return DType::F16;
  if (s == "BF16") return DType::BF16;
  if (s == "I8") return DType::I8;
  LLMRT_CHECK(false,
              "safetensors: unsupported dtype '" + std::string(s) +
                  "' (supported: F32, F16, BF16, I8)");
}

size_t TensorInfo::numel() const {
  size_t n = 1;
  for (const int64_t d : shape) n *= static_cast<size_t>(d);
  return n;
}

int64_t TensorInfo::dim(int i) const {
  LLMRT_CHECK(i >= 0 && static_cast<size_t>(i) < shape.size(),
              "TensorInfo::dim: index " + std::to_string(i) + " out of range for '" + name + "'");
  return shape[static_cast<size_t>(i)];
}

std::string TensorInfo::shape_string() const {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) s += ", ";
    s += std::to_string(shape[i]);
  }
  s += "]";
  return s;
}

SafeTensors::SafeTensors(SafeTensors&& other) noexcept { *this = std::move(other); }

SafeTensors& SafeTensors::operator=(SafeTensors&& other) noexcept {
  if (this != &other) {
    release();
    path_ = std::move(other.path_);
    map_ = other.map_;
    file_size_ = other.file_size_;
    data_offset_ = other.data_offset_;
    data_bytes_ = other.data_bytes_;
    tensors_ = std::move(other.tensors_);
    header_ = std::move(other.header_);
    other.map_ = nullptr;
    other.file_size_ = 0;
    other.data_offset_ = 0;
    other.data_bytes_ = 0;
  }
  return *this;
}

void SafeTensors::release() noexcept {
  if (map_ != nullptr) {
    ::munmap(map_, file_size_);
    map_ = nullptr;
  }
}

SafeTensors::~SafeTensors() { release(); }

SafeTensors SafeTensors::open(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  LLMRT_CHECK(fd >= 0, "safetensors: cannot open '" + path + "'");

  struct stat st {};
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    LLMRT_CHECK(false, "safetensors: fstat failed for '" + path + "'");
  }
  const size_t file_size = static_cast<size_t>(st.st_size);
  LLMRT_CHECK(file_size >= 8, "safetensors: '" + path + "' is too small to be valid");

  void* map = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  LLMRT_CHECK(map != MAP_FAILED, "safetensors: mmap failed for '" + path + "'");

  SafeTensors out;
  out.path_ = path;
  out.map_ = map;
  out.file_size_ = file_size;

  const char* base = static_cast<const char*>(map);
  const uint64_t header_len = load_le64(base);
  LLMRT_CHECK(header_len <= file_size - 8,
              "safetensors: header length " + std::to_string(header_len) +
                  " exceeds file size " + std::to_string(file_size) + " for '" + path + "'");

  out.data_offset_ = 8 + static_cast<size_t>(header_len);
  out.header_ = json::parse(std::string_view(base + 8, static_cast<size_t>(header_len)));
  LLMRT_CHECK(out.header_.is_object(), "safetensors: header is not a JSON object in '" + path + "'");

  size_t total = 0;
  const auto& obj = out.header_.as_object();
  out.tensors_.reserve(obj.size());
  for (const auto& kv : obj) {
    if (kv.first == kMetadataKey) continue;
    const json::Value& e = kv.second;

    TensorInfo t;
    t.name = kv.first;
    t.dtype_str = e.get_string("dtype");
    LLMRT_CHECK(!t.dtype_str.empty(),
                "safetensors: tensor '" + t.name + "' has no dtype");
    t.dtype = dtype_from_string(t.dtype_str);

    if (const json::Value::Array* shape = e.get_array("shape")) {
      t.shape.reserve(shape->size());
      for (const json::Value& d : *shape) t.shape.push_back(d.as_int());
    }

    const json::Value::Array* offs = e.get_array("data_offsets");
    LLMRT_CHECK(offs != nullptr && offs->size() == 2,
                "safetensors: tensor '" + t.name + "' has malformed data_offsets");
    t.begin = static_cast<size_t>((*offs)[0].as_int());
    t.end = static_cast<size_t>((*offs)[1].as_int());
    LLMRT_CHECK(t.end >= t.begin, "safetensors: tensor '" + t.name + "' has end < begin");

    // Guard against a header/data mismatch, which is the failure mode that
    // silently produces garbage activations later.
    if (is_dtype_convertible(t.dtype)) {
      const size_t expected = t.numel() * dtype_size(t.dtype);
      LLMRT_CHECK(t.nbytes() == expected,
                  "safetensors: tensor '" + t.name + "' declares " + t.shape_string() +
                      " (" + std::to_string(expected) + " bytes) but occupies " +
                      std::to_string(t.nbytes()) + " bytes");
    }

    total += t.nbytes();
    out.tensors_.push_back(std::move(t));
  }

  LLMRT_CHECK(out.data_offset_ + total <= file_size,
              "safetensors: tensor payload (" + std::to_string(total) +
                  " bytes) overruns file '" + path + "'");
  out.data_bytes_ = total;
  return out;
}

const TensorInfo* SafeTensors::find(std::string_view name) const {
  for (const auto& t : tensors_) {
    if (t.name == name) return &t;
  }
  return nullptr;
}

const TensorInfo& SafeTensors::at(std::string_view name) const {
  const TensorInfo* t = find(name);
  LLMRT_CHECK(t != nullptr, "safetensors: no tensor named '" + std::string(name) + "'");
  return *t;
}

std::vector<float> SafeTensors::to_f32(const TensorInfo& t) const {
  std::vector<float> out(t.numel());
  convert_to_f32(raw(t), t.dtype, out.data(), out.size());
  return out;
}

std::vector<float> SafeTensors::to_f32(std::string_view name) const {
  return to_f32(at(name));
}

std::vector<std::pair<std::string, size_t>> SafeTensors::dtype_histogram() const {
  std::vector<std::pair<std::string, size_t>> out;
  for (const auto& t : tensors_) {
    bool seen = false;
    for (auto& e : out) {
      if (e.first == t.dtype_str) {
        ++e.second;
        seen = true;
        break;
      }
    }
    if (!seen) out.emplace_back(t.dtype_str, 1);
  }
  return out;
}

}  // namespace llmrt
