#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "llmrt/common.h"
#include "llmrt/json.h"

#include "test_framework.h"

using namespace llmrt;
using json::Value;

namespace {

// Real Qwen3-0.6B config, asserted against the values verified from the
// checkpoint itself. Skipped (with a notice) when the model is not present.
std::string model_dir() {
  const char* env = std::getenv("LLMRT_MODEL_DIR");
  return env ? env : "/home/dr/models/Qwen3-0.6B";
}

std::string golden_dir() {
  return std::string(LLMRT_SOURCE_DIR) + "/data/golden";
}

bool file_exists(const std::string& p) {
  std::ifstream f(p);
  return f.good();
}

bool parse_throws(const std::string& text) {
  try {
    json::parse(text);
  } catch (const Error&) {
    return true;
  }
  return false;
}

}  // namespace

// ---------------------------------------------------------------------------
// Scalars and containers
// ---------------------------------------------------------------------------

LLMRT_TEST(parses_scalars) {
  CHECK_TRUE(json::parse("null").is_null());
  CHECK_TRUE(json::parse("true").as_bool());
  CHECK_FALSE(json::parse("false").as_bool());
  CHECK_EQ(json::parse("42").as_int(), int64_t{42});
  CHECK_EQ(json::parse("-7").as_int(), int64_t{-7});
  CHECK_NEAR(json::parse("3.5").as_double(), 3.5, 1e-12);
  CHECK_NEAR(json::parse("1e-06").as_double(), 1e-6, 1e-18);
  CHECK_NEAR(json::parse("2.5e3").as_double(), 2500.0, 1e-9);
  CHECK_EQ(json::parse("\"hi\"").as_string(), std::string("hi"));
  CHECK_EQ(json::parse("  \n 42 \t ").as_int(), int64_t{42});
}

LLMRT_TEST(distinguishes_integer_from_float_numbers) {
  CHECK_TRUE(json::parse("42").is_integer());
  CHECK_FALSE(json::parse("42.0").is_integer());
  CHECK_FALSE(json::parse("42e0").is_integer());
  CHECK_TRUE(json::parse("1000000").is_integer());
  // A double is still readable through the generic number accessor.
  CHECK_NEAR(json::parse("42.0").as_number(), 42.0, 1e-12);
}

LLMRT_TEST(parses_nested_containers) {
  const Value v = json::parse(R"({"a": [1, 2, {"b": true}], "c": {"d": null}})");
  CHECK_TRUE(v.is_object());
  CHECK_EQ(v.size(), size_t{2});

  const Value* a = v.find("a");
  CHECK_TRUE(a != nullptr);
  CHECK_TRUE(a->is_array());
  CHECK_EQ(a->size(), size_t{3});
  CHECK_EQ((*a)[0].as_int(), int64_t{1});
  CHECK_EQ((*a)[2].at("b").as_bool(), true);

  CHECK_TRUE(v.at("c").at("d").is_null());
  CHECK_TRUE(v.contains("a"));
  CHECK_FALSE(v.contains("missing"));
  CHECK_TRUE(v.find("missing") == nullptr);
}

LLMRT_TEST(decodes_string_escapes) {
  CHECK_EQ(json::parse(R"("a\tb\nc\\d\"e")").as_string(),
           std::string("a\tb\nc\\d\"e"));
  CHECK_EQ(json::parse(R"("\u0041\u0042")").as_string(), std::string("AB"));
  // U+00E9 -> C3 A9
  CHECK_EQ(json::parse(R"("a\u00e9b")").as_string(), std::string("a\xC3\xA9" "b"));
  // U+1F600 as a surrogate pair -> F0 9F 98 80
  CHECK_EQ(json::parse(R"("\ud83d\ude00")").as_string(),
           std::string("\xF0\x9F\x98\x80"));
}

// ---------------------------------------------------------------------------
// Malformed input
// ---------------------------------------------------------------------------

LLMRT_TEST(rejects_malformed_input) {
  CHECK_TRUE(parse_throws(""));
  CHECK_TRUE(parse_throws("{"));
  CHECK_TRUE(parse_throws("[1,]"));
  CHECK_TRUE(parse_throws("{\"a\":}"));
  CHECK_TRUE(parse_throws("{a: 1}"));
  CHECK_TRUE(parse_throws("tru"));
  CHECK_TRUE(parse_throws("01"));
  CHECK_TRUE(parse_throws("1."));
  CHECK_TRUE(parse_throws("1e"));
  CHECK_TRUE(parse_throws("[1] extra"));
  CHECK_TRUE(parse_throws("\"unterminated"));
  CHECK_TRUE(parse_throws(R"("bad \q escape")"));
}

LLMRT_TEST(error_message_reports_position) {
  try {
    json::parse("{\n  \"a\": ,\n}");
    CHECK_MSG(false, "expected parse to throw");
  } catch (const Error& e) {
    const std::string what = e.what();
    CHECK_MSG(what.find("line 2") != std::string::npos, "no line info: " + what);
  }
}

LLMRT_TEST(typed_accessors_reject_wrong_types) {
  const Value v = json::parse(R"({"s": "x", "n": 1, "b": true})");
  CHECK_TRUE([&] { try { v.at("s").as_int(); } catch (const Error&) { return true; } return false; }());
  CHECK_TRUE([&] { try { v.at("n").as_string(); } catch (const Error&) { return true; } return false; }());
  CHECK_TRUE([&] { try { v.at("b").as_array(); } catch (const Error&) { return true; } return false; }());
  // at() on a missing key throws.
  CHECK_TRUE([&] { try { v.at("nope"); } catch (const Error&) { return true; } return false; }());
}

LLMRT_TEST(get_with_defaults_tolerates_missing_and_null) {
  const Value v = json::parse(R"({"present": 5, "nothing": null})");
  CHECK_EQ(v.get_int("present", -1), int64_t{5});
  CHECK_EQ(v.get_int("absent", -1), int64_t{-1});
  CHECK_EQ(v.get_int("nothing", -1), int64_t{-1});
  CHECK_EQ(v.get_string("absent", "fallback"), std::string("fallback"));
  CHECK_TRUE(v.get_array("absent") == nullptr);
}

// ---------------------------------------------------------------------------
// Real files
// ---------------------------------------------------------------------------

LLMRT_TEST(parses_real_qwen3_config) {
  const std::string path = model_dir() + "/config.json";
  if (!file_exists(path)) {
    std::printf("      (skipped: %s not found)\n", path.c_str());
    return;
  }
  const Value c = json::parse_file(path);

  CHECK_EQ(c.get_string("model_type"), std::string("qwen3"));
  CHECK_EQ(c.get_int("num_hidden_layers", 0), int64_t{28});
  CHECK_EQ(c.get_int("hidden_size", 0), int64_t{1024});
  CHECK_EQ(c.get_int("num_attention_heads", 0), int64_t{16});
  CHECK_EQ(c.get_int("num_key_value_heads", 0), int64_t{8});
  // head_dim is explicit in Qwen3 and must NOT be derived from hidden/heads.
  CHECK_EQ(c.get_int("head_dim", 0), int64_t{128});
  CHECK_EQ(c.get_int("intermediate_size", 0), int64_t{3072});
  CHECK_EQ(c.get_int("vocab_size", 0), int64_t{151936});
  CHECK_EQ(c.get_int("max_position_embeddings", 0), int64_t{40960});
  CHECK_NEAR(c.get_double("rope_theta", 0.0), 1e6, 1e-3);
  CHECK_NEAR(c.get_double("rms_norm_eps", 0.0), 1e-6, 1e-18);
  CHECK_EQ(c.get_bool("tie_word_embeddings", false), true);
  CHECK_EQ(c.get_bool("attention_bias", true), false);
  CHECK_EQ(c.get_string("hidden_act"), std::string("silu"));

  // 16 heads x 128 == 2048 != hidden_size 1024 -- the Qwen3 quirk.
  CHECK_EQ(c.get_int("head_dim", 0) * c.get_int("num_attention_heads", 0), int64_t{2048});
  CHECK_TRUE(c.get_int("head_dim", 0) * c.get_int("num_attention_heads", 0) !=
             c.get_int("hidden_size", 0));
}

LLMRT_TEST(parses_golden_manifest) {
  const std::string path = golden_dir() + "/manifest.json";
  if (!file_exists(path)) {
    std::printf("      (skipped: %s not found -- run tools/gen_oracle.py)\n", path.c_str());
    return;
  }
  const Value m = json::parse_file(path);

  CHECK_EQ(m.at("config").get_int("head_dim", 0), int64_t{128});
  CHECK_EQ(m.at("config").get_int("num_hidden_layers", 0), int64_t{28});
  CHECK_EQ(m.get_string("compute_dtype"), std::string("float32"));
  CHECK_EQ(m.get_string("attn_implementation"), std::string("eager"));

  // `arrays` is an object keyed by array name, not an array.
  const Value* arrays = m.find("arrays");
  CHECK_TRUE(arrays != nullptr);
  CHECK_TRUE(arrays->is_object());
  CHECK_EQ(arrays->size(), size_t{31});
  CHECK_TRUE(arrays->contains("logits"));

  const Value& logits = m.at("arrays").at("logits");
  CHECK_EQ(logits.get_string("dtype"), std::string("f32"));
  CHECK_EQ(logits.at("shape")[0].as_int(), int64_t{60});
  CHECK_EQ(logits.at("shape")[1].as_int(), int64_t{151936});

  // Same token count drives input_ids and the first logits axis.
  CHECK_EQ(m.at("arrays").at("input_ids").at("shape")[0].as_int(),
           logits.at("shape")[0].as_int());
}

LLMRT_TEST(dump_round_trips) {
  const std::string text = R"({"a":[1,2],"b":{"c":"x"},"d":true,"e":null})";
  const Value v = json::parse(text);
  const Value again = json::parse(v.dump());
  CHECK_EQ(again.at("a")[1].as_int(), int64_t{2});
  CHECK_EQ(again.at("b").get_string("c"), std::string("x"));
  CHECK_EQ(again.get_bool("d", false), true);
  CHECK_TRUE(again.at("e").is_null());
  CHECK_EQ(again.size(), size_t{4});
}

int main() { return llmrt_test::run_all("json"); }
