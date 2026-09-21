// Qwen3 forward pass: embedding -> 28 decoder layers -> final norm -> lm_head.
//
// ---------------------------------------------------------------------------
// 层内顺序（逐行从 modeling_qwen3.py 的 Qwen3DecoderLayer.forward 抄下来的）
// ---------------------------------------------------------------------------
//
//   residual = hidden
//   h = input_layernorm(hidden)
//   h = self_attn(h)
//   hidden = residual + h
//   residual = hidden
//   h = post_attention_layernorm(hidden)
//   h = mlp(h)
//   hidden = residual + h
//
// 也就是"norm 在子层之前、残差在子层之后"，注意**不是** pre-norm 之外还有别的
// 花样。这个顺序写错（比如把 norm 放在残差之后）结果会全错但看着像样。
//
// attention 内部（Qwen3Attention.forward）：
//
//   q = q_proj(h) -> view(S, heads, D) -> q_norm -> transpose -> rope
//   k = k_proj(h) -> view(S, kv_heads, D) -> k_norm -> transpose -> rope
//   v = v_proj(h) -> view(S, kv_heads, D) -> transpose          ← v 不过 rope
//   attn = attention(q, k, v)                                   → [S, heads*D]
//   out = o_proj(attn)
//
// ⚠️ QK-Norm 在 RoPE **之前**，而且作用在 head_dim 上（不是 hidden）。这是
//    Qwen3 相对 Llama/Qwen2 新增的东西，放错位置会得到 cosine ~0.87。
//
// ⚠️ v 不过 RoPE。q 和 k 过。
//
// ---------------------------------------------------------------------------
// 三次 head 转置
// ---------------------------------------------------------------------------
//
// q_proj 输出是 [S, heads*D]，而 rope 和 attention 要 [heads, S, D]。中间是
// 真正的数据搬移（不是 reshape），因为 rope 把"位置"当中间轴、attention 要每个
// 头的 K/V 在 seq 方向连续。
//
// 代价（S=512 时）：q 4 MB + k 2 MB + v 2 MB = 8 MB/层，转置读写各一遍所以
// 16 MB 流量，×28 层 = 448 MB —— 约合 f32 权重流量的 19%。
//
// 消掉它们的办法是把激活统一成 [S, heads, D]（见 docs/DECISIONS.md P2），但
// 那要重写 rope 和 attention 两个已验证的算子，留给 Phase 8 用 profiler 实测
// 之后再决定。现在先照参考实现的布局来，好和 modeling_qwen3.py 对照着读。

#include "llmrt/forward.h"

#include <cstring>
#include <string>
#include <utility>

#include "llmrt/common.h"
#include "llmrt/convert.h"
#include "llmrt/ops.h"

namespace llmrt {

namespace {

// The 11 tensors of one layer, in a fixed order.
//
// Kept in one place because both the arena sizing and the view construction
// walk it. If those two lists could drift, the arena would come out short and
// the copy would run off the end of a 2.4 GB allocation. Returned by value --
// eleven pointers is nothing, and handing out a reference to a shared buffer
// would just invite a dangling read.
std::vector<const TensorInfo*> layer_tensors(const LayerWeights& lw) {
  return {lw.input_layernorm, lw.q_proj,           lw.k_proj, lw.v_proj,
          lw.q_norm,           lw.k_norm,          lw.o_proj, lw.post_attention_layernorm,
          lw.gate_proj,        lw.up_proj,         lw.down_proj};
}

// A Tensor over a scratch buffer. Views are built where they are used rather
// than cached, because allocate() resizes those vectors and a resize invalidates
// every pointer taken from them. The cost is a few hundred small allocations per
// forward, against 2.4 GB of weight traffic.
Tensor as_tensor(std::vector<float>& buf, std::vector<int64_t> shape) {
  return Tensor::contiguous(buf.data(), DType::F32, DeviceKind::CPU, std::move(shape));
}

// hidden += proj, elementwise. Not an op -- docs/DECISIONS.md D5 keeps the op
// list to what the plot calls for, and a residual add has no golden of its own.
void add_in_place(std::vector<float>& dst, const std::vector<float>& src, size_t n) {
  for (size_t i = 0; i < n; ++i) dst[i] += src[i];
}

// [seq, heads, dim] -> [heads, seq, dim], i.e. the reference's transpose(1, 2).
//
// The inner loop is over `dim`, so both the source and the destination are
// walked contiguously -- one 512-byte run per (seq, head) pair, which is what
// keeps this from being a cache-line hop per element.
void transpose_heads(const float* src, float* dst, int64_t seq, int64_t heads,
                     int64_t dim) {
  for (int64_t s = 0; s < seq; ++s) {
    for (int64_t h = 0; h < heads; ++h) {
      const float* in = src + (s * heads + h) * dim;
      float* out = dst + (h * seq + s) * dim;
      std::memcpy(out, in, static_cast<size_t>(dim) * sizeof(float));
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// F32Weights
// ---------------------------------------------------------------------------

F32Weights::F32Weights(const SafeTensors& st, const Qwen3Weights& w) {
  // Pass 1: total size. The arena is resized exactly once, before any Tensor
  // exists, so nothing is ever left pointing into a freed buffer.
  size_t total = w.embed_tokens().numel() + w.final_norm().numel();
  for (int64_t l = 0; l < w.num_layers(); ++l) {
    for (const TensorInfo* t : layer_tensors(w.layer(l))) total += t->numel();
  }
  arena.resize(total);

  // Pass 2: copy and convert into the arena, taking a view of each slot.
  float* cursor = arena.data();
  const auto take = [&](const TensorInfo& t) {
    Tensor v = Tensor::contiguous(cursor, DType::F32, DeviceKind::CPU, t.shape);
    convert_to_f32(st.raw(t), t.dtype, cursor, t.numel());
    cursor += t.numel();
    return v;
  };

  // lm_head is tied to embed_tokens, so this is the only place either one is
  // materialised -- one 311 MB copy instead of two.
  embed_tokens = take(w.embed_tokens());
  final_norm = take(w.final_norm());

  layers.resize(static_cast<size_t>(w.num_layers()));
  for (int64_t l = 0; l < w.num_layers(); ++l) {
    const std::vector<const TensorInfo*> ts = layer_tensors(w.layer(l));
    Layer& dst = layers[static_cast<size_t>(l)];
    dst.input_layernorm = take(*ts[0]);
    dst.q_proj = take(*ts[1]);
    dst.k_proj = take(*ts[2]);
    dst.v_proj = take(*ts[3]);
    dst.q_norm = take(*ts[4]);
    dst.k_norm = take(*ts[5]);
    dst.o_proj = take(*ts[6]);
    dst.post_attention_layernorm = take(*ts[7]);
    dst.gate_proj = take(*ts[8]);
    dst.up_proj = take(*ts[9]);
    dst.down_proj = take(*ts[10]);
  }

  // The two passes above must agree on how much there is to copy. They walk the
  // same list, so a mismatch means someone added a `take()` without adding it
  // to layer_tensors() -- which would otherwise be a heap overflow.
  LLMRT_CHECK(static_cast<size_t>(cursor - arena.data()) == total,
              "F32Weights: copied " + std::to_string(cursor - arena.data()) +
                  " elements into an arena sized for " + std::to_string(total));
}

// ---------------------------------------------------------------------------
// Qwen3Forward
// ---------------------------------------------------------------------------

Qwen3Forward::Qwen3Forward(const SafeTensors& st, const Qwen3Weights& w)
    : config_(w.config()), weights_(st, w) {}

void Qwen3Forward::allocate(int64_t seq) {
  const Qwen3Config& c = config_;
  const auto n = [](int64_t rows, int64_t cols) {
    return static_cast<size_t>(rows) * static_cast<size_t>(cols);
  };

  // resize rather than assign: a later run with a shorter prompt reuses the
  // existing allocation instead of freeing and reallocating. Every buffer is
  // fully written before it is read.
  hidden_.resize(n(seq, c.hidden_size));
  normed_.resize(n(seq, c.hidden_size));
  q_.resize(n(seq, c.q_proj_dim()));
  k_.resize(n(seq, c.kv_proj_dim()));
  v_.resize(n(seq, c.kv_proj_dim()));
  q_heads_.resize(n(seq, c.q_proj_dim()));
  k_heads_.resize(n(seq, c.kv_proj_dim()));
  v_heads_.resize(n(seq, c.kv_proj_dim()));
  attn_.resize(n(seq, c.q_proj_dim()));
  proj_.resize(n(seq, c.hidden_size));
  gate_.resize(n(seq, c.intermediate_size));
  up_.resize(n(seq, c.intermediate_size));
  act_.resize(n(seq, c.intermediate_size));
  cos_.resize(n(seq, c.head_dim));
  sin_.resize(n(seq, c.head_dim));
}

void Qwen3Forward::forward_layer(int64_t layer, int64_t seq) {
  const F32Weights::Layer& w = weights_.layers[static_cast<size_t>(layer)];
  const int64_t H = config_.hidden_size;
  const int64_t Nh = config_.num_attention_heads;
  const int64_t Nkv = config_.num_key_value_heads;
  const int64_t D = config_.head_dim;
  const int64_t I = config_.intermediate_size;
  const int64_t A = config_.q_proj_dim();   // heads * head_dim
  const int64_t KVD = config_.kv_proj_dim();
  const float eps = static_cast<float>(config_.rms_norm_eps);
  const size_t hn = static_cast<size_t>(seq) * static_cast<size_t>(H);

  // Every op takes its destination by Tensor&, so each view has to be a named
  // lvalue -- a temporary cannot bind to a non-const reference. Declaring them
  // up front rather than inlining keeps that requirement from being scattered
  // through the body.
  Tensor hidden = as_tensor(hidden_, {seq, H});
  Tensor normed = as_tensor(normed_, {seq, H});
  Tensor q = as_tensor(q_, {seq, A});
  Tensor k = as_tensor(k_, {seq, KVD});
  Tensor v = as_tensor(v_, {seq, KVD});
  Tensor attn = as_tensor(attn_, {seq, A});
  Tensor proj = as_tensor(proj_, {seq, H});
  Tensor gate = as_tensor(gate_, {seq, I});
  Tensor up = as_tensor(up_, {seq, I});
  Tensor act = as_tensor(act_, {seq, I});

  // ---- attention block ----------------------------------------------------
  cpu::rmsnorm(hidden, w.input_layernorm, normed, eps);

  cpu::matmul(normed, w.q_proj, q, true);
  cpu::matmul(normed, w.k_proj, k, true);
  cpu::matmul(normed, w.v_proj, v, true);

  // QK-Norm: per head, over head_dim, BEFORE RoPE. Viewing the projection output
  // as [seq, heads, head_dim] is a free contiguous reshape, so this is the same
  // op with a different last-axis length -- 128 here, 1024 for the block norms.
  //
  // Called in place (x and out are the same tensor). Safe because rmsnorm reads
  // x_row[j] before writing out_row[j] and no iteration needs another row's
  // original value. That saves a [seq, hidden] buffer per call, and there are 57
  // such calls per forward.
  {
    Tensor q_by_head = as_tensor(q_, {seq, Nh, D});
    Tensor k_by_head = as_tensor(k_, {seq, Nkv, D});
    cpu::rmsnorm(q_by_head, w.q_norm, q_by_head, eps);
    cpu::rmsnorm(k_by_head, w.k_norm, k_by_head, eps);
  }

  // [seq, heads, D] -> [heads, seq, D]. v goes through the same transpose but
  // gets no RoPE.
  transpose_heads(q_.data(), q_heads_.data(), seq, Nh, D);
  transpose_heads(k_.data(), k_heads_.data(), seq, Nkv, D);
  transpose_heads(v_.data(), v_heads_.data(), seq, Nkv, D);

  {
    // The table was built once in run() and is shared by every layer: it depends
    // on position and head_dim, not on the layer.
    Tensor qh = as_tensor(q_heads_, {Nh, seq, D});
    Tensor kh = as_tensor(k_heads_, {Nkv, seq, D});
    Tensor cos = as_tensor(cos_, {seq, D});
    Tensor sin = as_tensor(sin_, {seq, D});
    cpu::rope_apply(qh, kh, cos, sin);
  }

  {
    Tensor qa = as_tensor(q_heads_, {1, Nh, seq, D});
    Tensor ka = as_tensor(k_heads_, {1, Nkv, seq, D});
    Tensor va = as_tensor(v_heads_, {1, Nkv, seq, D});
    Tensor ao = as_tensor(attn_, {1, seq, A});
    AttentionParams p;
    p.num_heads = Nh;
    p.num_kv_heads = Nkv;
    p.head_dim = D;
    p.causal = true;  // prefill: position i sees every j <= i
    cpu::attention(qa, ka, va, ao, p);
  }

  cpu::matmul(attn, w.o_proj, proj, true);
  add_in_place(hidden_, proj_, hn);

  // ---- MLP block ----------------------------------------------------------
  cpu::rmsnorm(hidden, w.post_attention_layernorm, normed, eps);

  cpu::matmul(normed, w.gate_proj, gate, true);
  cpu::matmul(normed, w.up_proj, up, true);
  cpu::swiglu(gate, up, act);
  cpu::matmul(act, w.down_proj, proj, true);
  add_in_place(hidden_, proj_, hn);
}

void Qwen3Forward::run(const std::vector<int32_t>& tokens, std::vector<float>& logits,
                       const LayerObserver& observe) {
  const int64_t seq = static_cast<int64_t>(tokens.size());
  LLMRT_CHECK(seq > 0, "Qwen3Forward::run: empty prompt");
  LLMRT_CHECK(seq <= config_.max_position_embeddings,
              "Qwen3Forward::run: " + std::to_string(seq) +
                  " tokens exceeds max_position_embeddings (" +
                  std::to_string(config_.max_position_embeddings) + ")");

  allocate(seq);

  const int64_t H = config_.hidden_size;
  const int64_t D = config_.head_dim;

  // 1. embedding: gather the prompt's rows out of the table. The op also checks
  //    that every id is in range, so a bad token id is caught here.
  //    Tensor stores a non-const void*, hence the cast -- see the note in
  //    tensor.h about that being a deliberate trade-off.
  {
    Tensor ids = Tensor::contiguous(const_cast<int32_t*>(tokens.data()), DType::I32,
                                    DeviceKind::CPU, {seq});
    Tensor emb = as_tensor(hidden_, {seq, H});
    cpu::embedding(weights_.embed_tokens, ids, emb);
  }

  // 2. The rotary table, once for the whole forward pass. Computing it inside
  //    the layer loop would redo the same transcendentals 28 times.
  {
    Tensor cos = as_tensor(cos_, {seq, D});
    Tensor sin = as_tensor(sin_, {seq, D});
    cpu::rope_frequencies(seq, D, static_cast<float>(config_.rope_theta), cos, sin);
  }

  // 3. The stack. The observer sees the hidden state each layer produced, which
  //    is exactly what golden's layer_hidden holds -- that is what makes a
  //    per-layer bisect possible when the logits come out wrong.
  for (int64_t l = 0; l < config_.num_hidden_layers; ++l) {
    forward_layer(l, seq);
    if (observe) observe(l, hidden_);
  }

  // 4. final norm
  Tensor y = as_tensor(normed_, {seq, H});
  {
    Tensor h = as_tensor(hidden_, {seq, H});
    cpu::rmsnorm(h, weights_.final_norm, y, static_cast<float>(config_.rms_norm_eps));
  }

  // 5. lm_head. Tied to embed_tokens, so this reads the same table step 1 did,
  //    from the other side: [seq, hidden] @ [vocab, hidden]^T -> [seq, vocab].
  //    This one matmul reads 311 MB of weights -- a quarter of the model -- even
  //    for a 12-token prompt.
  logits.assign(static_cast<size_t>(seq) * static_cast<size_t>(config_.vocab_size), 0.0f);
  Tensor lg = Tensor::contiguous(logits.data(), DType::F32, DeviceKind::CPU,
                                 {seq, config_.vocab_size});
  cpu::matmul(y, weights_.embed_tokens, lg, true);
}

}  // namespace llmrt
