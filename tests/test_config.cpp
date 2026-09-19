#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "llmrt/common.h"
#include "llmrt/config.h"
#include "llmrt/json.h"
#include "llmrt/model.h"
#include "llmrt/safetensors.h"

#include "golden.h"
#include "safetensors_fixture.h"
#include "test_framework.h"

using namespace llmrt;

namespace {

// A 1-layer, 8-wide Qwen3-shaped model. Small enough to build inline, complete
// enough that binding and every shape guard can be exercised.
//
// Built programmatically rather than by patching a literal, so that the
// negative tests below cannot accidentally produce malformed JSON and pass
// for the wrong reason.
std::string tiny_config_json(int heads = 2, int kv_heads = 1,
                             const std::string& act = "silu",
                             bool attention_bias = false, int head_dim = 4,
                             bool include_head_dim = true,
                             bool include_kv_heads = true) {
  std::ostringstream os;
  os << "{"
     << "\"model_type\":\"qwen3\","
     << "\"hidden_act\":\"" << act << "\","
     << "\"num_hidden_layers\":1,"
     << "\"hidden_size\":8,"
     << "\"num_attention_heads\":" << heads << ",";
  if (include_kv_heads) os << "\"num_key_value_heads\":" << kv_heads << ",";
  if (include_head_dim) os << "\"head_dim\":" << head_dim << ",";
  os << "\"intermediate_size\":16,"
     << "\"vocab_size\":32,"
     << "\"max_position_embeddings\":64,"
     << "\"rope_theta\":1000000,"
     << "\"rms_norm_eps\":1e-06,"
     << "\"tie_word_embeddings\":true,"
     << "\"attention_bias\":" << (attention_bias ? "true" : "false") << "}";
  return os.str();
}

std::string layer_name(int i, const char* suffix) {
  return "model.layers." + std::to_string(i) + "." + suffix;
}

// All tensors the binder requires, with shapes valid for kTinyConfig.
std::vector<std::pair<std::string, fixture::Entry>> tiny_tensors() {
  std::vector<std::pair<std::string, fixture::Entry>> t;
  t.emplace_back("model.embed_tokens.weight", fixture::bf16_zeros({32, 8}));
  t.emplace_back("model.norm.weight", fixture::bf16_filled({8}, 0x3F80));  // 1.0
  t.emplace_back(layer_name(0, "input_layernorm.weight"), fixture::bf16_filled({8}, 0x3F80));
  t.emplace_back(layer_name(0, "self_attn.q_proj.weight"), fixture::bf16_zeros({8, 8}));
  t.emplace_back(layer_name(0, "self_attn.k_proj.weight"), fixture::bf16_zeros({4, 8}));
  t.emplace_back(layer_name(0, "self_attn.v_proj.weight"), fixture::bf16_zeros({4, 8}));
  t.emplace_back(layer_name(0, "self_attn.q_norm.weight"), fixture::bf16_filled({4}, 0x3F80));
  t.emplace_back(layer_name(0, "self_attn.k_norm.weight"), fixture::bf16_filled({4}, 0x3F80));
  t.emplace_back(layer_name(0, "self_attn.o_proj.weight"), fixture::bf16_zeros({8, 8}));
  t.emplace_back(layer_name(0, "post_attention_layernorm.weight"),
                 fixture::bf16_filled({8}, 0x3F80));
  t.emplace_back(layer_name(0, "mlp.gate_proj.weight"), fixture::bf16_zeros({16, 8}));
  t.emplace_back(layer_name(0, "mlp.up_proj.weight"), fixture::bf16_zeros({16, 8}));
  t.emplace_back(layer_name(0, "mlp.down_proj.weight"), fixture::bf16_zeros({8, 16}));
  return t;
}

const char* kTinyPath = "/tmp/llmrt_test_tiny.safetensors";

// Writes the tiny model, optionally mutating it first.
std::string write_tiny(void (*mutate)(std::vector<std::pair<std::string, fixture::Entry>>&) = nullptr) {
  auto tensors = tiny_tensors();
  if (mutate != nullptr) mutate(tensors);
  return fixture::write_safetensors(kTinyPath, tensors);
}

Qwen3Config tiny_config() { return Qwen3Config::from_json(json::parse(tiny_config_json())); }

// Finds a fixture entry by name fragment; name-based so the tests do not
// silently target the wrong tensor if the payload list is reordered.
fixture::Entry& entry_named(std::vector<std::pair<std::string, fixture::Entry>>& t,
                            const std::string& needle) {
  for (auto& e : t) {
    if (e.first.find(needle) != std::string::npos) return e.second;
  }
  throw Error("fixture: no tensor matching '" + needle + "'");
}

// Returns the error message thrown by parsing `text`, or "" if it did not throw.
std::string parse_error(const std::string& text) {
  try {
    Qwen3Config::from_json(json::parse(text));
  } catch (const Error& e) {
    return e.what();
  }
  return "";
}

}  // namespace

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

LLMRT_TEST(parses_real_qwen3_config_into_struct) {
  const std::string dir = golden::default_model_dir();
  if (!std::ifstream(dir + "/config.json").good()) {
    std::printf("      (skipped: model not found)\n");
    return;
  }
  const Qwen3Config c = Qwen3Config::from_model_dir(dir);
  CHECK_EQ(c.model_type, std::string("qwen3"));
  CHECK_EQ(c.num_hidden_layers, int64_t{28});
  CHECK_EQ(c.hidden_size, int64_t{1024});
  CHECK_EQ(c.num_attention_heads, int64_t{16});
  CHECK_EQ(c.num_key_value_heads, int64_t{8});
  CHECK_EQ(c.head_dim, int64_t{128});
  CHECK_EQ(c.intermediate_size, int64_t{3072});
  CHECK_EQ(c.vocab_size, int64_t{151936});
  CHECK_EQ(c.max_position_embeddings, int64_t{40960});
  CHECK_EQ(c.hidden_act, std::string("silu"));
  CHECK_NEAR(c.rope_theta, 1e6, 1e-3);
  CHECK_NEAR(c.rms_norm_eps, 1e-6, 1e-18);
  CHECK_TRUE(c.tie_word_embeddings);
  CHECK_FALSE(c.attention_bias);
}

LLMRT_TEST(derived_values_follow_from_explicit_head_dim) {
  const Qwen3Config c = tiny_config();
  CHECK_EQ(c.q_proj_dim(), int64_t{8});    // 2 heads * 4
  CHECK_EQ(c.kv_proj_dim(), int64_t{4});   // 1 kv head * 4
  CHECK_EQ(c.num_key_value_groups(), int64_t{2});
  CHECK_NEAR(c.attention_scale(), 1.0 / std::sqrt(4.0), 1e-15);
  // head_dim 4 == hidden/heads 8/2 == 4, so this config is not "explicit".
  CHECK_FALSE(c.head_dim_is_explicit());
}

LLMRT_TEST(real_config_reports_explicit_head_dim_quirk) {
  const std::string dir = golden::default_model_dir();
  if (!std::ifstream(dir + "/config.json").good()) return;
  const Qwen3Config c = Qwen3Config::from_model_dir(dir);
  // 128 != 1024/16 == 64: the reason q_proj is 2048-wide, not 1024.
  CHECK_TRUE(c.head_dim_is_explicit());
  CHECK_EQ(c.q_proj_dim(), int64_t{2048});
  CHECK_TRUE(c.q_proj_dim() != c.hidden_size);
  CHECK_EQ(c.num_key_value_groups(), int64_t{2});
}

LLMRT_TEST(head_dim_falls_back_when_field_absent) {
  const Qwen3Config c = Qwen3Config::from_json(
      json::parse(tiny_config_json(2, 1, "silu", false, 0, /*include_head_dim=*/false)));
  CHECK_EQ(c.head_dim, int64_t{4});  // hidden_size / num_attention_heads = 8 / 2
}

LLMRT_TEST(kv_heads_default_to_attention_heads_when_absent) {
  const Qwen3Config c =
      Qwen3Config::from_json(json::parse(tiny_config_json(2, 1, "silu", false, 4, true,
                                                          /*include_kv_heads=*/false)));
  CHECK_EQ(c.num_key_value_heads, c.num_attention_heads);
  CHECK_EQ(c.num_key_value_groups(), int64_t{1});
}

LLMRT_TEST(rejects_unsupported_hidden_act) {
  const std::string err = parse_error(tiny_config_json(2, 1, "gelu"));
  CHECK_MSG(!err.empty(), "expected gelu to be rejected");
  CHECK_MSG(err.find("hidden_act") != std::string::npos, err);
}

LLMRT_TEST(rejects_attention_bias) {
  const std::string err = parse_error(tiny_config_json(2, 1, "silu", /*attention_bias=*/true));
  CHECK_MSG(!err.empty(), "expected attention_bias=true to be rejected");
  CHECK_MSG(err.find("attention_bias") != std::string::npos, err);
}

LLMRT_TEST(rejects_heads_not_divisible_for_gqa) {
  const std::string err = parse_error(tiny_config_json(16, 3));
  CHECK_MSG(!err.empty(), "expected 16 heads with 3 kv heads to be rejected");
  CHECK_MSG(err.find("multiple") != std::string::npos, err);
}

LLMRT_TEST(rejects_missing_required_fields) {
  const std::string err = parse_error(R"({"model_type": "qwen3"})");
  CHECK_MSG(!err.empty(), "expected an empty config to be rejected");
}

// ---------------------------------------------------------------------------
// Weight binding
// ---------------------------------------------------------------------------

LLMRT_TEST(binds_synthetic_tiny_model) {
  const std::string path = write_tiny();
  const SafeTensors st = SafeTensors::open(path);
  const Qwen3Weights w = Qwen3Weights::bind(st, tiny_config());

  CHECK_EQ(w.num_layers(), int64_t{1});
  CHECK_TRUE(w.embed_tokens().name == "model.embed_tokens.weight");
  CHECK_TRUE(w.final_norm().name == "model.norm.weight");
  // Tied: lm_head must alias embed_tokens, and the model has no lm_head on disk.
  CHECK_TRUE(w.lm_head_is_tied());
  CHECK_FALSE(w.lm_head_on_disk());
  CHECK_TRUE(&w.lm_head() == &w.embed_tokens());
  // 14 referenced slots but only 13 distinct tensors: lm_head is aliased.
  CHECK_EQ(w.reference_slots(), size_t{14});
  CHECK_EQ(w.all_tensors().size(), size_t{13});
  // embed 512 + norm 16 + layer 1200
  CHECK_EQ(w.resident_bytes(), size_t{1728});

  const LayerWeights& l = w.layer(0);
  CHECK_TRUE(l.q_proj->shape_string() == "[8, 8]");
  CHECK_TRUE(l.k_proj->shape_string() == "[4, 8]");
  CHECK_TRUE(l.o_proj->shape_string() == "[8, 8]");
  CHECK_TRUE(l.q_norm->shape_string() == "[4]");
  CHECK_TRUE(l.down_proj->shape_string() == "[8, 16]");
}

LLMRT_TEST(binding_rejects_missing_tensor) {
  const std::string path = write_tiny([](std::vector<std::pair<std::string, fixture::Entry>>& t) {
    for (auto it = t.begin(); it != t.end(); ++it) {
      if (it->first.find("k_proj") != std::string::npos) {
        t.erase(it);
        return;
      }
    }
  });
  const SafeTensors st = SafeTensors::open(path);
  bool threw = false;
  try {
    Qwen3Weights::bind(st, tiny_config());
  } catch (const Error& e) {
    threw = true;
    CHECK_MSG(std::string(e.what()).find("k_proj") != std::string::npos, e.what());
  }
  CHECK_TRUE(threw);
}

LLMRT_TEST(binding_rejects_wrong_shape) {
  const std::string path = write_tiny([](std::vector<std::pair<std::string, fixture::Entry>>& t) {
    // q_proj should project to heads*head_dim = 8, not 16.
    entry_named(t, "q_proj") = fixture::bf16_zeros({16, 8});
  });
  const SafeTensors st = SafeTensors::open(path);
  bool threw = false;
  try {
    Qwen3Weights::bind(st, tiny_config());
  } catch (const Error& e) {
    threw = true;
    CHECK_MSG(std::string(e.what()).find("q_proj") != std::string::npos, e.what());
  }
  CHECK_TRUE(threw);
}

LLMRT_TEST(binding_rejects_tensor_whose_payload_contradicts_its_shape) {
  const std::string path = write_tiny([](std::vector<std::pair<std::string, fixture::Entry>>& t) {
    entry_named(t, "model.norm.weight").bytes.resize(10);  // claims [8] bf16 (16 bytes)
  });
  bool threw = false;
  try {
    SafeTensors::open(path);
  } catch (const Error& e) {
    threw = true;
    CHECK_MSG(std::string(e.what()).find("occupies") != std::string::npos, e.what());
  }
  CHECK_TRUE(threw);
}

LLMRT_TEST(rejects_header_longer_than_file) {
  const std::string path = "/tmp/llmrt_test_bogus_header.safetensors";
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const uint64_t bogus = 1ull << 40;  // 1 TiB header
    out.write(reinterpret_cast<const char*>(&bogus), sizeof(bogus));
    out.write("{}", 2);
  }
  bool threw = false;
  try {
    SafeTensors::open(path);
  } catch (const Error& e) {
    threw = true;
    CHECK_MSG(std::string(e.what()).find("header length") != std::string::npos, e.what());
  }
  CHECK_TRUE(threw);
}

LLMRT_TEST(rejects_unsupported_dtype_in_header) {
  const std::string path = write_tiny([](std::vector<std::pair<std::string, fixture::Entry>>& t) {
    t[1].second.dtype = "F64";
  });
  bool threw = false;
  try {
    SafeTensors::open(path);
  } catch (const Error& e) {
    threw = true;
    CHECK_MSG(std::string(e.what()).find("F64") != std::string::npos, e.what());
  }
  CHECK_TRUE(threw);
}

// ---------------------------------------------------------------------------
// Real checkpoint binding
// ---------------------------------------------------------------------------

LLMRT_TEST(binds_real_checkpoint_and_accounts_for_tied_lm_head) {
  const std::string ckpt = golden::default_model_dir() + "/model.safetensors";
  if (!std::ifstream(ckpt).good()) {
    std::printf("      (skipped: checkpoint not found)\n");
    return;
  }
  const SafeTensors st = SafeTensors::open(ckpt);
  const Qwen3Config cfg = Qwen3Config::from_model_dir(golden::default_model_dir());
  const Qwen3Weights w = Qwen3Weights::bind(st, cfg);

  CHECK_EQ(w.num_layers(), int64_t{28});
  CHECK_TRUE(w.lm_head_is_tied());
  CHECK_TRUE(w.lm_head_on_disk());
  CHECK_TRUE(&w.lm_head() == &w.embed_tokens());

  // 311 referenced slots (3 global + 11*28) but 310 distinct tensors: the
  // on-disk lm_head copy is aliased to embed_tokens and never loaded.
  CHECK_EQ(w.reference_slots(), size_t{3 + 11 * 28});
  CHECK_EQ(w.all_tensors().size(), size_t{310});
  CHECK_EQ(w.resident_bytes() + w.lm_head().nbytes(), st.data_bytes());

  CHECK_TRUE(w.layer(0).q_norm->shape_string() == "[128]");
  CHECK_TRUE(w.layer(27).down_proj->shape_string() == "[1024, 3072]");
}

int main() { return llmrt_test::run_all("config"); }
