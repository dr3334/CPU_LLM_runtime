#include "llmrt/model.h"

#include <algorithm>

#include "llmrt/common.h"

namespace llmrt {

namespace {

std::string layer_tensor_name(int64_t i, const char* suffix) {
  return "model.layers." + std::to_string(i) + "." + suffix;
}

// Asserts a tensor has exactly `rank` dimensions equal to `want`.
void expect_shape(const TensorInfo& t, std::initializer_list<int64_t> want) {
  LLMRT_CHECK(t.shape.size() == want.size(),
              "weight '" + t.name + "' has rank " + std::to_string(t.shape.size()) +
                  " but " + std::to_string(want.size()) + " was expected (got " +
                  t.shape_string() + ")");
  size_t i = 0;
  for (const int64_t w : want) {
    LLMRT_CHECK(t.shape[i] == w,
                "weight '" + t.name + "' has shape " + t.shape_string() +
                    " but " + std::to_string(w) + " was expected at axis " +
                    std::to_string(i));
    ++i;
  }
}

void expect_vector(const TensorInfo& t, int64_t n) {
  LLMRT_CHECK(t.shape.size() == 1 && t.shape[0] == n,
              "weight '" + t.name + "' should be [" + std::to_string(n) +
                  "] but is " + t.shape_string());
}

}  // namespace

Qwen3Weights Qwen3Weights::bind(const SafeTensors& st, const Qwen3Config& cfg) {
  cfg.validate();

  Qwen3Weights w;
  w.config_ = cfg;

  w.embed_tokens_ = &st.at("model.embed_tokens.weight");
  w.final_norm_ = &st.at("model.norm.weight");
  w.lm_head_on_disk_ = st.find("lm_head.weight") != nullptr;

  if (cfg.tie_word_embeddings) {
    // One physical copy serves both the embedding lookup and the output
    // projection. Deliberately ignore the on-disk lm_head duplicate.
    w.lm_head_ = w.embed_tokens_;
    w.lm_head_tied_ = true;
  } else {
    LLMRT_CHECK(w.lm_head_on_disk_,
                "weights: config.tie_word_embeddings is false but the checkpoint has no "
                "'lm_head.weight'");
    w.lm_head_ = &st.at("lm_head.weight");
    w.lm_head_tied_ = false;
  }

  w.layers_.resize(static_cast<size_t>(cfg.num_hidden_layers));
  for (int64_t i = 0; i < cfg.num_hidden_layers; ++i) {
    LayerWeights& l = w.layers_[static_cast<size_t>(i)];
    l.input_layernorm = &st.at(layer_tensor_name(i, "input_layernorm.weight"));
    l.q_proj = &st.at(layer_tensor_name(i, "self_attn.q_proj.weight"));
    l.k_proj = &st.at(layer_tensor_name(i, "self_attn.k_proj.weight"));
    l.v_proj = &st.at(layer_tensor_name(i, "self_attn.v_proj.weight"));
    l.q_norm = &st.at(layer_tensor_name(i, "self_attn.q_norm.weight"));
    l.k_norm = &st.at(layer_tensor_name(i, "self_attn.k_norm.weight"));
    l.o_proj = &st.at(layer_tensor_name(i, "self_attn.o_proj.weight"));
    l.post_attention_layernorm =
        &st.at(layer_tensor_name(i, "post_attention_layernorm.weight"));
    l.gate_proj = &st.at(layer_tensor_name(i, "mlp.gate_proj.weight"));
    l.up_proj = &st.at(layer_tensor_name(i, "mlp.up_proj.weight"));
    l.down_proj = &st.at(layer_tensor_name(i, "mlp.down_proj.weight"));
  }

  w.validate_shapes();
  return w;
}

const LayerWeights& Qwen3Weights::layer(int64_t i) const {
  LLMRT_CHECK(i >= 0 && i < num_layers(),
              "weights: layer index " + std::to_string(i) + " out of range [0, " +
                  std::to_string(num_layers()) + ")");
  return layers_[static_cast<size_t>(i)];
}

void Qwen3Weights::validate_shapes() const {
  const Qwen3Config& c = config_;
  const int64_t h = c.hidden_size;
  const int64_t qd = c.q_proj_dim();
  const int64_t kvd = c.kv_proj_dim();
  const int64_t inter = c.intermediate_size;

  expect_shape(*embed_tokens_, {c.vocab_size, h});
  if (lm_head_ != embed_tokens_) expect_shape(*lm_head_, {c.vocab_size, h});
  expect_vector(*final_norm_, h);

  for (int64_t i = 0; i < num_layers(); ++i) {
    const LayerWeights& l = layers_[static_cast<size_t>(i)];
    expect_vector(*l.input_layernorm, h);
    expect_vector(*l.post_attention_layernorm, h);
    expect_shape(*l.q_proj, {qd, h});
    expect_shape(*l.k_proj, {kvd, h});
    expect_shape(*l.v_proj, {kvd, h});
    expect_shape(*l.o_proj, {h, qd});
    // QK-Norm is applied per head over head_dim, so these are [head_dim].
    expect_vector(*l.q_norm, c.head_dim);
    expect_vector(*l.k_norm, c.head_dim);
    expect_shape(*l.gate_proj, {inter, h});
    expect_shape(*l.up_proj, {inter, h});
    expect_shape(*l.down_proj, {h, inter});
  }
}

std::vector<const TensorInfo*> Qwen3Weights::all_tensors() const {
  std::vector<const TensorInfo*> v;
  auto push_unique = [&v](const TensorInfo* t) {
    if (std::find(v.begin(), v.end(), t) == v.end()) v.push_back(t);
  };
  push_unique(embed_tokens_);
  push_unique(final_norm_);
  push_unique(lm_head_);
  for (const LayerWeights& l : layers_) {
    push_unique(l.input_layernorm);
    push_unique(l.q_proj);
    push_unique(l.k_proj);
    push_unique(l.v_proj);
    push_unique(l.q_norm);
    push_unique(l.k_norm);
    push_unique(l.o_proj);
    push_unique(l.post_attention_layernorm);
    push_unique(l.gate_proj);
    push_unique(l.up_proj);
    push_unique(l.down_proj);
  }
  return v;
}

size_t Qwen3Weights::resident_bytes() const {
  size_t total = 0;
  for (const TensorInfo* t : all_tensors()) total += t->nbytes();
  return total;
}

size_t Qwen3Weights::reference_slots() const {
  // embed_tokens + final_norm + lm_head, then 11 tensors per layer.
  return 3 + 11 * static_cast<size_t>(num_layers());
}

}  // namespace llmrt
