// Qwen3 hyper-parameters.
//
// Read from the model directory's config.json. The fields here are the ones
// the forward pass actually needs; nothing is inferred that the checkpoint
// states explicitly.
//
// Qwen3 detail that must not be guessed at: `head_dim` is an explicit field
// and is NOT hidden_size / num_attention_heads. Qwen3-0.6B has head_dim=128
// while hidden_size/heads = 1024/16 = 64, so q_proj projects to
// num_attention_heads * head_dim = 2048, which is twice hidden_size.
#pragma once

#include <cstdint>
#include <string>

#include "llmrt/json.h"

namespace llmrt {

struct Qwen3Config {
  std::string model_type = "qwen3";
  std::string hidden_act = "silu";

  int64_t num_hidden_layers = 0;
  int64_t hidden_size = 0;
  int64_t num_attention_heads = 0;
  int64_t num_key_value_heads = 0;
  int64_t head_dim = 0;
  int64_t intermediate_size = 0;
  int64_t vocab_size = 0;
  int64_t max_position_embeddings = 0;

  double rope_theta = 0.0;
  double rms_norm_eps = 0.0;

  bool tie_word_embeddings = false;
  bool attention_bias = false;

  // ---- derived -----------------------------------------------------------
  // Output width of q_proj / o_proj input: num_attention_heads * head_dim.
  int64_t q_proj_dim() const { return num_attention_heads * head_dim; }
  // Output width of k_proj and v_proj: num_key_value_heads * head_dim.
  int64_t kv_proj_dim() const { return num_key_value_heads * head_dim; }
  // GQA repeat factor: how many query heads share each KV head.
  int64_t num_key_value_groups() const;
  // 1 / sqrt(head_dim).
  double attention_scale() const;
  // True when head_dim differs from hidden_size / num_attention_heads.
  bool head_dim_is_explicit() const;

  // Throws llmrt::Error describing the first inconsistency found.
  void validate() const;

  static Qwen3Config from_json(const json::Value& v);
  // Reads <dir>/config.json.
  static Qwen3Config from_model_dir(const std::string& dir);

  // Human-readable multi-line summary, used by `llmrt inspect`.
  std::string describe() const;
};

}  // namespace llmrt
