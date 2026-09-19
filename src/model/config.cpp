#include "llmrt/config.h"

#include <cmath>
#include <sstream>

#include "llmrt/common.h"

namespace llmrt {

int64_t Qwen3Config::num_key_value_groups() const {
  LLMRT_CHECK(num_key_value_heads > 0, "config: num_key_value_heads must be positive");
  return num_attention_heads / num_key_value_heads;
}

double Qwen3Config::attention_scale() const {
  LLMRT_CHECK(head_dim > 0, "config: head_dim must be positive");
  return 1.0 / std::sqrt(static_cast<double>(head_dim));
}

bool Qwen3Config::head_dim_is_explicit() const {
  if (num_attention_heads <= 0) return false;
  return head_dim != hidden_size / num_attention_heads;
}

void Qwen3Config::validate() const {
  LLMRT_CHECK(num_hidden_layers > 0, "config: num_hidden_layers must be positive");
  LLMRT_CHECK(hidden_size > 0, "config: hidden_size must be positive");
  LLMRT_CHECK(num_attention_heads > 0, "config: num_attention_heads must be positive");
  LLMRT_CHECK(num_key_value_heads > 0, "config: num_key_value_heads must be positive");
  LLMRT_CHECK(head_dim > 0, "config: head_dim must be positive");
  LLMRT_CHECK(intermediate_size > 0, "config: intermediate_size must be positive");
  LLMRT_CHECK(vocab_size > 0, "config: vocab_size must be positive");
  LLMRT_CHECK(max_position_embeddings > 0, "config: max_position_embeddings must be positive");
  LLMRT_CHECK(rope_theta > 0.0, "config: rope_theta must be positive");
  LLMRT_CHECK(rms_norm_eps > 0.0, "config: rms_norm_eps must be positive");
  LLMRT_CHECK(num_attention_heads % num_key_value_heads == 0,
              "config: num_attention_heads (" + std::to_string(num_attention_heads) +
                  ") must be a multiple of num_key_value_heads (" +
                  std::to_string(num_key_value_heads) + ") for GQA");
  LLMRT_CHECK(!attention_bias,
              "config: attention_bias=true is not supported by the hand-written kernels");
  LLMRT_CHECK(hidden_act == "silu",
              "config: hidden_act '" + hidden_act + "' is not supported (only 'silu')");
}

Qwen3Config Qwen3Config::from_json(const json::Value& v) {
  Qwen3Config c;
  c.model_type = v.get_string("model_type", c.model_type);
  c.hidden_act = v.get_string("hidden_act", c.hidden_act);

  c.num_hidden_layers = v.get_int("num_hidden_layers", 0);
  c.hidden_size = v.get_int("hidden_size", 0);
  c.num_attention_heads = v.get_int("num_attention_heads", 0);
  c.num_key_value_heads = v.get_int("num_key_value_heads", c.num_attention_heads);
  c.intermediate_size = v.get_int("intermediate_size", 0);
  c.vocab_size = v.get_int("vocab_size", 0);
  c.max_position_embeddings = v.get_int("max_position_embeddings", 0);

  c.rope_theta = v.get_double("rope_theta", 0.0);
  c.rms_norm_eps = v.get_double("rms_norm_eps", 0.0);

  c.tie_word_embeddings = v.get_bool("tie_word_embeddings", false);
  c.attention_bias = v.get_bool("attention_bias", false);

  // head_dim is explicit in Qwen3. The fallback mirrors HuggingFace's
  // `getattr(config, "head_dim", hidden_size // num_attention_heads)` so that
  // a config without the field still loads correctly.
  if (const json::Value* hd = v.find("head_dim")) {
    c.head_dim = hd->as_int();
  } else if (c.num_attention_heads > 0) {
    c.head_dim = c.hidden_size / c.num_attention_heads;
  }

  c.validate();
  return c;
}

Qwen3Config Qwen3Config::from_model_dir(const std::string& dir) {
  std::string path = dir;
  if (path.empty() || path.back() != '/') path += '/';
  path += "config.json";
  const json::Value v = json::parse_file(path);
  LLMRT_CHECK(v.is_object(), "config: '" + path + "' does not contain a JSON object");
  return from_json(v);
}

std::string Qwen3Config::describe() const {
  std::ostringstream os;
  auto row = [&os](const char* k, const std::string& v) {
    os << "  " << std::left;
    os.width(36);
    os << k << v << '\n';
  };
  auto rowi = [&row](const char* k, int64_t v) { row(k, std::to_string(v)); };
  auto rowd = [&row](const char* k, double v) {
    std::ostringstream t;
    t << v;
    row(k, t.str());
  };
  auto rowb = [&row](const char* k, bool v) { row(k, v ? "true" : "false"); };

  os << "config\n";
  row("model_type", model_type);
  rowi("num_hidden_layers", num_hidden_layers);
  rowi("hidden_size", hidden_size);
  rowi("num_attention_heads", num_attention_heads);
  rowi("num_key_value_heads", num_key_value_heads);
  rowi("head_dim", head_dim);
  rowi("intermediate_size", intermediate_size);
  rowi("vocab_size", vocab_size);
  rowi("max_position_embeddings", max_position_embeddings);
  row("hidden_act", hidden_act);
  rowd("rope_theta", rope_theta);
  rowd("rms_norm_eps", rms_norm_eps);
  rowb("tie_word_embeddings", tie_word_embeddings);
  rowb("attention_bias", attention_bias);

  os << "  derived\n";
  rowi("q_proj_dim (heads*head_dim)", q_proj_dim());
  rowi("kv_proj_dim (kv_heads*head_dim)", kv_proj_dim());
  rowi("num_key_value_groups (GQA)", num_key_value_groups());
  rowd("attention_scale (1/sqrt(head_dim))", attention_scale());
  if (head_dim_is_explicit()) {
    row("head_dim vs hidden/heads",
        std::to_string(head_dim) + " vs " +
            std::to_string(hidden_size / num_attention_heads) +
            "   <- Qwen3 uses an explicit head_dim");
  }
  return os.str();
}

}  // namespace llmrt
