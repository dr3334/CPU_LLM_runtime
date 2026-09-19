// Weight binding: resolves every tensor the forward pass reads by name and
// checks its shape against the config.
//
// Binding is intentional about tying. Qwen3-0.6B sets tie_word_embeddings, and
// the checkpoint nonetheless stores lm_head.weight as a full second copy of
// embed_tokens (verified byte-identical). Binding therefore aliases lm_head to
// embed_tokens when tied, so the runtime allocates one copy instead of two --
// 311 MB of the 1.5 GB checkpoint.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "llmrt/config.h"
#include "llmrt/safetensors.h"

namespace llmrt {

// Pointers into a SafeTensors table for a single decoder layer.
struct LayerWeights {
  const TensorInfo* input_layernorm = nullptr;
  const TensorInfo* q_proj = nullptr;
  const TensorInfo* k_proj = nullptr;
  const TensorInfo* v_proj = nullptr;
  const TensorInfo* q_norm = nullptr;
  const TensorInfo* k_norm = nullptr;
  const TensorInfo* o_proj = nullptr;
  const TensorInfo* post_attention_layernorm = nullptr;
  const TensorInfo* gate_proj = nullptr;
  const TensorInfo* up_proj = nullptr;
  const TensorInfo* down_proj = nullptr;
};

class Qwen3Weights {
 public:
  // Resolves all tensors. Throws if any is missing.
  static Qwen3Weights bind(const SafeTensors& st, const Qwen3Config& cfg);

  const Qwen3Config& config() const { return config_; }
  int64_t num_layers() const { return static_cast<int64_t>(layers_.size()); }

  const TensorInfo& embed_tokens() const { return *embed_tokens_; }
  const TensorInfo& final_norm() const { return *final_norm_; }
  // Aliases embed_tokens when the config ties word embeddings.
  const TensorInfo& lm_head() const { return *lm_head_; }

  bool lm_head_is_tied() const { return lm_head_tied_; }
  // True when the checkpoint also stored a separate lm_head tensor (the usual
  // case for a tied model, and pure dead weight for us).
  bool lm_head_on_disk() const { return lm_head_on_disk_; }

  const LayerWeights& layer(int64_t i) const;

  // Checks every bound tensor's shape against the config. Throws on mismatch.
  void validate_shapes() const;

  // Payload bytes that must actually be resident: each distinct tensor once,
  // so a tied lm_head is counted a single time. This is what a loader
  // allocates, and it is what `llmrt inspect` reports as the weight footprint.
  size_t resident_bytes() const;
  // Number of tensor slots the forward pass references (3 global + 11*L).
  // Exceeds the distinct count when lm_head is tied.
  size_t reference_slots() const;

  // Every distinct tensor, in a stable order.
  std::vector<const TensorInfo*> all_tensors() const;

 private:
  Qwen3Config config_;
  std::vector<LayerWeights> layers_;
  const TensorInfo* embed_tokens_ = nullptr;
  const TensorInfo* final_norm_ = nullptr;
  const TensorInfo* lm_head_ = nullptr;
  bool lm_head_tied_ = false;
  bool lm_head_on_disk_ = false;
};

}  // namespace llmrt
