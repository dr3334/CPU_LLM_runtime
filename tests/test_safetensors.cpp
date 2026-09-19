#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "llmrt/common.h"
#include "llmrt/convert.h"
#include "llmrt/safetensors.h"

#include "golden.h"
#include "test_framework.h"

using namespace llmrt;

namespace {

std::string checkpoint() { return golden::default_model_dir() + "/model.safetensors"; }

bool checkpoint_exists() {
  std::ifstream f(checkpoint());
  return f.good();
}

// Bit-exact reference values for IEEE binary16.
struct F16Case {
  uint16_t bits;
  float value;
};

const F16Case kF16Cases[] = {
    {0x0000, 0.0f},
    {0x8000, -0.0f},
    {0x3C00, 1.0f},
    {0x4000, 2.0f},
    {0xC000, -2.0f},
    {0x3555, 0.333251953125f},   // ~1/3
    {0x7BFF, 65504.0f},          // largest finite half
    {0x0001, 5.960464477539063e-08f},  // smallest subnormal: 2^-24
    {0x03FF, 6.097555160522461e-05f},  // largest subnormal: (1023/1024)*2^-14
};

}  // namespace

// ---------------------------------------------------------------------------
// dtype mapping and conversions
// ---------------------------------------------------------------------------

LLMRT_TEST(dtype_from_string_maps_supported_dtypes) {
  CHECK_TRUE(dtype_from_string("F32") == DType::F32);
  CHECK_TRUE(dtype_from_string("F16") == DType::F16);
  CHECK_TRUE(dtype_from_string("BF16") == DType::BF16);
  CHECK_TRUE(dtype_from_string("I8") == DType::I8);
  bool threw = false;
  try {
    dtype_from_string("F64");
  } catch (const Error&) {
    threw = true;
  }
  CHECK_TRUE(threw);
}

LLMRT_TEST(f16_conversion_matches_ieee_reference_values) {
  for (const F16Case& c : kF16Cases) {
    const float got = f16_to_f32(c.bits);
    // Compare bit patterns so that -0.0 and NaN payloads are handled exactly.
    uint32_t gb, wb;
    std::memcpy(&gb, &got, 4);
    float want = c.value;
    std::memcpy(&wb, &want, 4);
    CHECK_MSG(gb == wb, "half 0x" + std::to_string(c.bits) + " -> got " +
                            std::to_string(got) + " want " + std::to_string(want));
  }
}

LLMRT_TEST(f16_conversion_handles_infinities) {
  const float pinf = f16_to_f32(0x7C00);
  const float ninf = f16_to_f32(0xFC00);
  CHECK_TRUE(pinf > 65504.0f);
  CHECK_TRUE(ninf < -65504.0f);
  CHECK_TRUE(pinf == pinf);
  CHECK_TRUE(ninf == ninf);
  // NaN stays NaN.
  CHECK_TRUE(f16_to_f32(0x7E00) != f16_to_f32(0x7E00));
}

LLMRT_TEST(bf16_conversion_is_exact_upper_half_of_f32) {
  // bf16 -> f32 is truncation of the low 16 mantissa bits: the round trip must
  // leave those bits zero and preserve the high 16 bits exactly.
  const float samples[] = {0.0f,  -0.0f, 1.0f,   -1.0f,  0.1f,    -0.1f,
                           3.14159f, 1e-30f, 1e30f, 65504.0f, -1e-30f};
  for (const float f : samples) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint16_t hi = static_cast<uint16_t>(u >> 16);
    const float back = bf16_to_f32(hi);
    uint32_t u2;
    std::memcpy(&u2, &back, 4);
    CHECK_EQ(u2 & 0xFFFFu, 0u);
    CHECK_EQ(u2 >> 16, static_cast<uint32_t>(hi));
  }
}

LLMRT_TEST(convert_to_f32_handles_each_supported_dtype) {
  const float src_f32[] = {1.5f, -2.25f};
  float dst[2] = {0, 0};
  convert_to_f32(src_f32, DType::F32, dst, 2);
  CHECK_NEAR(dst[0], 1.5, 0.0);
  CHECK_NEAR(dst[1], -2.25, 0.0);

  const uint16_t src_bf16[] = {0x3FC0, 0xC010};  // 1.5, -2.25
  convert_to_f32(src_bf16, DType::BF16, dst, 2);
  CHECK_NEAR(dst[0], 1.5, 0.0);
  CHECK_NEAR(dst[1], -2.25, 0.0);

  const uint16_t src_f16[] = {0x3E00, 0xC080};  // 1.5, -2.25
  convert_to_f32(src_f16, DType::F16, dst, 2);
  CHECK_NEAR(dst[0], 1.5, 0.0);
  CHECK_NEAR(dst[1], -2.25, 0.0);

  const int8_t src_i8[] = {-128, 127};
  convert_to_f32(src_i8, DType::I8, dst, 2);
  CHECK_NEAR(dst[0], -128.0, 0.0);
  CHECK_NEAR(dst[1], 127.0, 0.0);
}

LLMRT_TEST(convert_rejects_nibble_packed_i4) {
  bool threw = false;
  float dst = 0;
  const uint8_t src = 0;
  try {
    convert_to_f32(&src, DType::I4, &dst, 1);
  } catch (const Error&) {
    threw = true;
  }
  CHECK_TRUE(threw);
}

// ---------------------------------------------------------------------------
// Real checkpoint
// ---------------------------------------------------------------------------

LLMRT_TEST(opens_real_qwen3_checkpoint) {
  if (!checkpoint_exists()) {
    std::printf("      (skipped: %s not found)\n", checkpoint().c_str());
    return;
  }
  const SafeTensors st = SafeTensors::open(checkpoint());

  // 311 tensors plus the __metadata__ entry, which is not a tensor.
  CHECK_EQ(st.size(), size_t{311});
  CHECK_EQ(st.file_size(), size_t{1503300328});
  CHECK_EQ(st.data_offset(), size_t{8 + 35552});
  CHECK_EQ(st.data_bytes(), size_t{1503264768});
  CHECK_EQ(st.data_offset() + st.data_bytes(), st.file_size());

  // The header itself remains queryable for anything the table does not model.
  CHECK_TRUE(st.header().is_object());
  CHECK_TRUE(st.header().contains("__metadata__"));
}

LLMRT_TEST(checkpoint_is_uniformly_bf16) {
  if (!checkpoint_exists()) return;
  const SafeTensors st = SafeTensors::open(checkpoint());
  const auto hist = st.dtype_histogram();
  CHECK_EQ(hist.size(), size_t{1});
  CHECK_EQ(hist[0].first, std::string("BF16"));
  CHECK_EQ(hist[0].second, size_t{311});
}

LLMRT_TEST(known_tensor_metadata_matches_verified_spec) {
  if (!checkpoint_exists()) return;
  const SafeTensors st = SafeTensors::open(checkpoint());

  const TensorInfo& lm = st.at("lm_head.weight");
  CHECK_EQ(lm.dim(0), int64_t{151936});
  CHECK_EQ(lm.dim(1), int64_t{1024});
  CHECK_EQ(lm.nbytes(), size_t{311164928});
  CHECK_EQ(lm.numel(), size_t{151936} * 1024);
  CHECK_EQ(lm.shape_string(), std::string("[151936, 1024]"));
  CHECK_TRUE(lm.dtype == DType::BF16);

  // The Qwen3 QK-Norm pair, sized by head_dim (128), not hidden_size.
  CHECK_EQ(st.at("model.layers.0.self_attn.q_norm.weight").numel(), size_t{128});
  CHECK_EQ(st.at("model.layers.0.self_attn.k_norm.weight").numel(), size_t{128});
  // q_proj projects to n_heads*head_dim = 2048, twice hidden_size.
  CHECK_EQ(st.at("model.layers.0.self_attn.q_proj.weight").dim(0), int64_t{2048});
  CHECK_EQ(st.at("model.layers.0.self_attn.k_proj.weight").dim(0), int64_t{1024});
  CHECK_EQ(st.at("model.layers.0.mlp.gate_proj.weight").dim(0), int64_t{3072});

  // Layer indices run 0..27 and nothing beyond.
  CHECK_TRUE(st.find("model.layers.27.self_attn.q_proj.weight") != nullptr);
  CHECK_TRUE(st.find("model.layers.28.self_attn.q_proj.weight") == nullptr);
}

LLMRT_TEST(tied_lm_head_and_embeddings_are_identical_bytes) {
  if (!checkpoint_exists()) return;
  const SafeTensors st = SafeTensors::open(checkpoint());
  const TensorInfo& emb = st.at("model.embed_tokens.weight");
  const TensorInfo& lm = st.at("lm_head.weight");
  CHECK_EQ(emb.nbytes(), lm.nbytes());
  // tie_word_embeddings=True: both are stored but must hold the same weights,
  // which is what lets the runtime load only one copy.
  CHECK_EQ(std::memcmp(st.raw(emb), st.raw(lm), emb.nbytes()), 0);
}

LLMRT_TEST(missing_tensor_lookup_is_reported_not_silent) {
  if (!checkpoint_exists()) return;
  const SafeTensors st = SafeTensors::open(checkpoint());
  CHECK_TRUE(st.find("no.such.tensor") == nullptr);
  bool threw = false;
  try {
    st.at("no.such.tensor");
  } catch (const Error& e) {
    threw = true;
    CHECK_MSG(std::string(e.what()).find("no.such.tensor") != std::string::npos,
              "error does not name the tensor");
  }
  CHECK_TRUE(threw);
}

// Cross-checks the reader, the bf16 conversion and the golden fixtures against
// each other: the embedding row for a known token id must equal the embedding
// output the reference model produced for that same token.
LLMRT_TEST(embedding_row_for_known_token_matches_golden) {
  if (!checkpoint_exists()) return;
  golden::Store g;
  if (!g.available()) {
    std::printf("      (skipped: golden fixtures not generated)\n");
    return;
  }
  const golden::Array* emb_out = g.get("embed_out");
  const golden::Array* ids = g.get("input_ids");
  if (emb_out == nullptr || ids == nullptr) return;

  const std::vector<int64_t> offsets = g.prompt_offsets();
  CHECK_TRUE(!offsets.empty());
  const int32_t token = ids->i()[offsets[0]];
  const int64_t hidden = emb_out->shape[1];

  const SafeTensors st = SafeTensors::open(checkpoint());
  const TensorInfo& table = st.at("model.embed_tokens.weight");
  CHECK_EQ(table.dim(1), hidden);

  const uint16_t* row = st.ptr<uint16_t>(table) + static_cast<size_t>(token) * hidden;
  std::vector<float> got(static_cast<size_t>(hidden));
  for (int64_t i = 0; i < hidden; ++i) got[static_cast<size_t>(i)] = bf16_to_f32(row[i]);

  const golden::Diff d = golden::compare(got.data(), emb_out->f(), got.size());
  CHECK_MSG(golden::within(d), "token " + std::to_string(token) + ": " + golden::diff_string(d));
}

int main() { return llmrt_test::run_all("safetensors"); }
