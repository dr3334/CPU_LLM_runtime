// The Qwen3 forward pass: embedding, 28 decoder layers, final norm, lm_head.
//
// This is the first part of the runtime that is not an op, and the first that
// cannot be checked op-by-op. Everything in ops.h is verified in isolation
// against a golden intermediate; a mistake here shows up as "every logit is
// wrong" rather than "this op is wrong". Two things compensate:
//
//   * every step is a call into an op that is already verified on its own, so
//     the forward pass contributes layout and sequencing, not arithmetic;
//   * run() takes an optional per-layer observer, so a test can compare each
//     layer against golden's `layer_hidden [28, 1, 12, 1024]` and bisect to the
//     layer where divergence starts instead of staring at a bad logits array.
//
// Scope for now: batch 1, prefill only, no KV cache. The sequence is a flat
// vector of token ids, and every position produces logits -- the reference does
// not slice to the last position, and neither does the golden capture.
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "llmrt/config.h"
#include "llmrt/model.h"
#include "llmrt/safetensors.h"
#include "llmrt/tensor.h"

namespace llmrt {

// A flat f32 mirror of every weight the forward pass reads.
//
// The kernels take f32 and bf16 -> f32 is a lossless widening, so converting
// once at load means bf16 never appears past this point. The cost is 2x the
// checkpoint resident -- 2.384 GB for Qwen3-0.6B, against 1.192 GB in bf16 --
// and, because matmul is memory-bound at 0.25 FLOP/byte, roughly 2x the weight
// traffic on every forward. Teaching the kernels to read bf16 is the real fix
// (docs/DECISIONS.md, P1) and it belongs with the quantisation work, not here.
//
// lm_head is tied to embed_tokens, so it is stored once; `make()` allocates
// embed_tokens once and nothing else refers to a second copy.
//
// Not copyable or movable, deliberately. Every Tensor below points into
// `arena`; a copy would duplicate the arena and leave the pointers in the copy
// aimed at the original's memory, which is freed when the original dies. The
// move case is no better -- the pointers would still aim at the old buffer.
class F32Weights {
 public:
  struct Layer {
    Tensor input_layernorm;           // [hidden]
    Tensor q_proj;                    // [heads * head_dim, hidden]
    Tensor k_proj;                    // [kv_heads * head_dim, hidden]
    Tensor v_proj;                    // [kv_heads * head_dim, hidden]
    Tensor q_norm;                    // [head_dim]
    Tensor k_norm;                    // [head_dim]
    Tensor o_proj;                    // [hidden, heads * head_dim]
    Tensor post_attention_layernorm;  // [hidden]
    Tensor gate_proj;                 // [intermediate, hidden]
    Tensor up_proj;                   // [intermediate, hidden]
    Tensor down_proj;                 // [hidden, intermediate]
  };

  // Copies and converts every tensor. arena is sized once, before any view
  // exists -- see the note on copying above.
  F32Weights(const SafeTensors& st, const Qwen3Weights& w);

  F32Weights(const F32Weights&) = delete;
  F32Weights& operator=(const F32Weights&) = delete;
  F32Weights(F32Weights&&) = delete;
  F32Weights& operator=(F32Weights&&) = delete;

  std::vector<float> arena;
  std::vector<Layer> layers;
  Tensor embed_tokens;  // [vocab, hidden], also serves as lm_head
  Tensor final_norm;    // [hidden]
};

// Runs the stack once for one prompt.
class Qwen3Forward {
 public:
  // Called after each decoder layer with the hidden state that layer produced,
  // as [seq, hidden_size] row-major. golden's layer_hidden is the same thing
  // for prompt 0, which is what makes a per-layer bisect possible. The
  // reference is only valid for the duration of the call -- it points straight
  // at the forward pass's scratch, and the next layer overwrites it.
  using LayerObserver = std::function<void(int64_t layer, const std::vector<float>&)>;

  Qwen3Forward(const SafeTensors& st, const Qwen3Weights& w);

  // `tokens` are the prompt's ids; `logits` is resized to [seq, vocab_size] and
  // filled. Every position gets logits.
  void run(const std::vector<int32_t>& tokens, std::vector<float>& logits,
           const LayerObserver& observe = {});

  const Qwen3Config& config() const { return config_; }
  // Resident weight bytes: the f32 mirror's arena.
  size_t weight_bytes() const { return weights_.arena.size() * sizeof(float); }

 private:
  // Resizes every activation buffer for `seq`. Called at the top of run().
  void allocate(int64_t seq);
  void forward_layer(int64_t layer, int64_t seq);

  Qwen3Config config_;
  F32Weights weights_;

  // Activations, sized for the current sequence length. Ops never allocate;
  // this is the one place that does. Resizing may reallocate, which is exactly
  // why no Tensor view is cached across calls -- views are built where they are
  // used, and the cost of that is a few hundred small vector allocations per
  // forward against 2.4 GB of weight reads.
  std::vector<float> hidden_;   // [seq, hidden]      residual stream
  std::vector<float> normed_;   // [seq, hidden]      rmsnorm output
  std::vector<float> q_;        // [seq, heads * head_dim]
  std::vector<float> k_;        // [seq, kv_heads * head_dim]
  std::vector<float> v_;        // [seq, kv_heads * head_dim]
  std::vector<float> q_heads_;  // [heads, seq, head_dim]     after the transpose
  std::vector<float> k_heads_;  // [kv_heads, seq, head_dim]
  std::vector<float> v_heads_;  // [kv_heads, seq, head_dim]
  std::vector<float> attn_;     // [seq, heads * head_dim]    o_proj input
  std::vector<float> proj_;     // [seq, hidden]      o_proj / down_proj output
  std::vector<float> gate_;     // [seq, intermediate]
  std::vector<float> up_;       // [seq, intermediate]
  std::vector<float> act_;      // [seq, intermediate]        swiglu output
  std::vector<float> cos_;      // [seq, head_dim]    built once, shared by all layers
  std::vector<float> sin_;      // [seq, head_dim]
};

}  // namespace llmrt
