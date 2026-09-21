# CODEBUDDY.md

This file provides guidance to CodeBuddy Code when working with code in this repository.

## What this project is

A from-scratch C++17 LLM inference runtime for Android, targeting on-device
heterogeneous execution (CPU + OpenCL) with per-layer device placement and
profiling.

Hard constraints, chosen deliberately by the project owner:

- **The inference path must be hand-written.** No MNN / ggml / llama.cpp /
  ONNX Runtime / LiteRT. Every kernel, the KV cache, the memory manager, the
  scheduler and the profiler are ours. "Add a library" is never an acceptable
  answer here.
- **Python is tooling only.** torch/transformers are used to generate reference
  data and nothing else. They must never be linked into `llmrt_core`.
- **CLI first, Android app later.** The runtime is validated as a native binary
  before any APK work.
- **safetensors first.** ONNX loading is intentionally deferred until all other
  features work.

## Target model

`Qwen3-0.6B`, expected at `/home/dr/models/Qwen3-0.6B` (override with
`LLMRT_MODEL_DIR`). Verified spec:

```
28 layers   hidden 1024   16 heads   8 kv heads   head_dim 128
intermediate 3072   vocab 151936   max_pos 40960
rope_theta 1e6   rms_norm_eps 1e-6   silu   no attention bias
single model.safetensors, 311 tensors, all BF16, 1503300328 bytes
```

There is no `Qwen3-0.8B`; the sizes are 0.6B / 1.7B / 4B / 8B / 14B / 32B.

## Build

```bash
./scripts/build_desktop.sh          # -> build/llmrt   (clang, RelWithDebInfo)
BUILD_TYPE=Debug ./scripts/build_desktop.sh
./scripts/build_android.sh          # -> build-android-arm64-v8a/llmrt
ABI=arm64-v8a API=24 ./scripts/build_android.sh
```

Android builds use the NDK's own toolchain file (`$ANDROID_NDK_HOME`, default
`/home/dr/android-sdk/ndk/26.3.11579264`), `arm64-v8a`, `android-24`,
`c++_static`. The output is a PIE binary for `adb push`.

## Test

```bash
ctest --test-dir build --output-on-failure           # all suites
ctest --test-dir build -R test_json --output-on-failure   # one suite
./build/tests/test_json                              # run a suite directly
```

Tests use a small in-tree harness (`tests/test_framework.h`) rather than a
third-party framework: `LLMRT_TEST(name) { ... }` with `CHECK_TRUE`,
`CHECK_EQ`, `CHECK_NEAR`, `CHECK_MSG`. Register a new suite by adding one line
to `tests/CMakeLists.txt`:

```cmake
llmrt_add_test(test_mything test_mything.cpp)
```

Two gotchas that have already cost time:

- **stdout is lost if a test aborts.** When debugging, run the binary under
  `stdbuf -o0 -e0` or the `[ OK ]` lines vanish along with the crash.
- Tests locate the fixtures via the `LLMRT_SOURCE_DIR` compile definition and
  the `tests/golden.h` helper, which reads `data/golden/manifest.json`.
  Suites that need the fixtures skip cleanly when they are absent.

## Golden reference data

`data/golden/` holds every intermediate the forward pass must reproduce,
dumped by the authoritative HuggingFace implementation as raw little-endian
binary plus a JSON manifest. This is the correctness backbone: it turns
"output looks like garbage" into a binary search over named tensors.

```bash
# Regenerate (uses the pinned reference env: torch 2.13, transformers 4.51.0)
PYTHONPATH=/home/dr/mtk_toolkit_enhence/reference \
  /home/dr/mtk_toolkit_enhence/.venv-ref/bin/python tools/gen_oracle.py

# Re-derive the forward pass from raw weights and check every intermediate
PYTHONPATH=/home/dr/mtk_toolkit_enhence/reference \
  /home/dr/mtk_toolkit_enhence/.venv-ref/bin/python tools/check_oracle.py
```

`tools/check_oracle.py` is an executable specification: it reimplements the
forward pass with plain tensor math (no `model.forward`) and asserts it
reproduces all 53 captured intermediates. If a C++ kernel disagrees with the
golden, run it first to confirm which side is wrong.

Observed fp32 round-off of the independent re-derivation: intermediates
`rel ~5e-7`, full-stack logits `rel ~2e-6`. The tiers in
`golden::Tolerance` (cosine > 0.99999, rel < 1e-4) are deliberately looser to
leave room for different accumulation orders.

Python tooling is managed with `uv` (`tools/pyproject.toml`), per the owner's
convention.

## Numerical conventions that must not be broken

These are the details that produce plausible-looking but wrong output, and
they are all confirmed against `modeling_qwen3.py`:

- **`head_dim` is explicit, not `hidden_size / num_attention_heads`.** For
  Qwen3-0.6B it is 128 while `1024/16 = 64`, so `q_proj` projects to
  `16*128 = 2048` — twice `hidden_size`. Never derive it.
- **QK-Norm runs per head over `head_dim`, BEFORE RoPE.** Order is
  `q_proj -> view(seq, heads, head_dim) -> q_norm -> transpose -> rope`.
  Applying it after RoPE is a known way to get cosine ~0.87 instead of ~1.0.
- **RoPE is the half-split (GPT-NeoX) convention**: `rotate_half` pairs the
  first and second halves, `cos`/`sin` are duplicated across halves. Not
  interleaved.
- **Attention scale is `1/sqrt(head_dim)`**, and softmax is computed in fp32.
- **`lm_head` is tied to `embed_tokens`.** The checkpoint stores both, and they
  are byte-identical (asserted by a test); the runtime loads one copy and saves
  311 MB. When tied, `Qwen3Weights::lm_head()` aliases `embed_tokens()`.
- **BF16 -> F32 is exact** (bf16 is the upper 16 bits of f32), so the loader
  introduces no error. All arithmetic is fp32.

### Floating-point rules

The goal is **not** to compute the most accurate value. It is to reproduce the
reference implementation's value. Those are different goals, and chasing
accuracy can move the result *away* from the golden. Every rule below comes
from a case that actually bit us.

| Trap | What goes wrong | Fix |
|---|---|---|
| Operator grouping | `u*g/d` parses as `(u*g)/d`, but the formula is `u*(g/d)`. `*` and `/` share a precedence level and are left-associative. | Copy the reference expression and add parentheses liberally. Extra parens cost nothing. |
| Literal type | `1.0 + std::exp(-g)` promotes the whole expression to `double` | Write `1.0f` / `0.0f`. |
| Approximate intrinsics | `_mm_rsqrt_ps` etc. differ in the last bits | Use exact `<cmath>` functions, i.e. `1/sqrt(x)`, never a hardware rsqrt. |
| Accumulator width | fp32 summing 1024 terms drifts | Match the reference first, widen only if the margin is thin. Measured for rmsnorm's sum of squares: fp32 -> rel 1.13e-06, double -> 4.69e-08. |
| **Compiler contraction** | The compiler may fuse `a*b + c` into a single FMA, turning two roundings into one. | `-ffp-contract` is **on by default** in GCC/Clang, so today's `matmul` inner loop is probably FMA-contracted. Results can then shift with compiler, flags or target. Add `-ffp-contract=off` if bit-stability across machines is ever required. |
| Overflow | `(u*g)/d` reaches `inf` where `u*(g/d)` does not (e.g. `g=-49.4`, `u=1e37`; the true value is representable) | Watch the magnitude of intermediates; prefer the grouping that is also safer. |

Two worked examples from this repo:

- **silu**: `x / (1 + exp(-x))` matches torch bit-for-bit, while the tidier
  `x * sigmoid(x)` differs by ~2.4e-7. The mathematically nicer form is the
  wrong one.
- **swiglu**: rewriting the loop from `(u*g)/d` to `u*(g/d)` — identical in real
  arithmetic — cut the error against golden from 2 ulp to 1 ulp.

**The only reliable check is measurement.** Write the variant, run it against
the golden, compare `rel` with the tolerance in `tests/golden.h`:

```
rel < tol * 0.01   comfortable
rel < tol * 0.1    acceptable
rel > tol * 0.5    fragile -- find another formulation
```

Current margins: rmsnorm 88x, matmul 79x, swiglu 3500x (purely elementwise, so
`exp` is its only error source).

## Decisions and open issues

`docs/DECISIONS.md` records every design decision (what, why, where the code is)
plus the alternatives that were considered and the optimizations that are
deferred with their measured costs. Each entry is bound to the git commit hash
that established it, so the reasoning can be checked against the code that was
actually written at that point.

Read it before proposing a change to the op contracts, the layout conventions or
the weight format — the trade-off has probably already been measured. If you do
change your mind, append a new entry rather than editing the old one, and note
which it supersedes.

## Layout

```
include/llmrt/       public headers
  common.h           DType, DeviceKind, Error, LLMRT_CHECK
  tensor.h           Tensor: non-owning typed view (ptr + offset + shape + strides)
  json.h             hand-written JSON parser
  safetensors.h      mmap'd checkpoint reader
  convert.h          bf16/f16 -> f32
  config.h           Qwen3Config
  model.h            weight binding (Qwen3Weights, LayerWeights)
  ops.h              the op contracts (see below)
  forward.h          F32Weights + Qwen3Forward (the 28-layer stack)
  generate.h         greedy generation, two prompt modes
src/core/            json, tensor, version
src/io/              safetensors, convert
src/model/           config, weights, forward, generate
src/ops/cpu/         rmsnorm, matmul, swiglu, rope, softmax, attention,
                     embedding, argmax
src/cli/             main + one cmd_*.cpp per subcommand (thin, no logic)
tools/               Python: golden generation, oracle checking
tests/               harness + one file per area; golden.h loads fixtures,
                     model_fixture.h loads the checkpoint once per binary
kernels/             (reserved for hand-written OpenCL)
```

`Tensor` is a **view, not an owner**: it holds `data` (host pointer, or a
`cl_mem` once the OpenCL backend lands), an element `offset`, shape and
strides. Ownership stays with whoever allocated the bytes — the memory manager,
or a `std::vector` in a test — so there is exactly one place that decides when
memory is freed. Strides exist from the start because attention reads the same
buffer as `[seq, heads, dim]` and as `[heads, seq, dim]`; the CPU ops currently
require contiguous input and call `require_contiguous()` to reject strided
views loudly rather than reading the wrong elements.

`ops.h` is **the list of primitives each backend implements**, not a list of
"model maths" — `embedding` is a gather and `argmax` is a reduction, and both
are there because a GPU backend wants them natively. What lives in `forward.h` /
`generate.h` instead is sequencing and policy: the layer order, the head
transposes, the chat wrapper, end-of-sequence handling. Ops never allocate;
the model layer owns its scratch.

Implemented today: metadata and loading, all eight CPU ops checked against
golden, the full 28-layer forward, and greedy generation.

```bash
./build/llmrt inspect --summary                            # weights and footprint
./build/llmrt generate --ids 785,6722,315 --mode raw       # continue text
./build/llmrt generate --ids 785,6722,315 --mode chat      # answer a question
```

## Roadmap

Tracking the approved plan. Status as of the layout section above:

1. ✅ Hand-written CPU ops — rmsnorm (which covers qk_norm), matmul, swiglu, rope,
   softmax, attention, embedding, argmax. All eight checked against
   `data/golden/`, each with a negative control proving the check discriminates.
2. 🔶 Full 28-layer forward ✅ (matches golden to rel 9.6e-06 worst case),
   greedy sampling ✅. **Missing: the tokenizer.** Decode is a table lookup and
   is cheap; encode needs the pre-tokenization regex, which uses Unicode
   property escapes (`\p{L}`, `\p{N}`) that `std::regex` cannot express. That is
   an open decision — see `docs/DECISIONS.md`.
   Until then: ids in, ids out, and the chat wrapper is expressed as token ids
   so neither mode needs a tokenizer.
3. ⬜ KV cache and the memory manager. First measured target: generation is
   1.68 s/token today because every step re-forwards the whole prefix. Note the
   KV cache cuts the attention term (O(S²) → O(S)) but **not** the weight
   traffic, which is what actually dominates.
4. ⬜ OpenCL backend: hand-written kernels, `cl_event` profiling, buffer pool.
5. ⬜ Heterogeneous scheduler: `--split 0-13:cpu,14-27:opencl`, boundary
   transfers, double buffering.
6. ⬜ Profiling report: TTFT, prefill/decode tok/s, per-layer and per-transfer
   timings.
7. ⬜ Android CLI, then the APK, then ONNX.

The shared abstraction these hang off is `IBackend` (see the plan file):
ops plus buffer management, so a layer's weights and KV cache live on the
device that executes it, and the scheduler inserts host<->device copies only at
placement boundaries.

Note on OpenCL in WSL: this machine has only the ocl-icd loader, no vendor ICD
(`/etc/OpenCL/vendors` does not exist), so `clGetPlatformIDs` returns
`CL_PLATFORM_NOT_FOUND_KHR` and there is no device to develop against.
`sudo apt install pocl-opencl-icd opencl-headers ocl-icd-opencl-dev clinfo`
fixes that; extracting the .deb without root does not work, because Ubuntu's
POCL 1.8 links against LLVM 11 which is not installed.

An earlier note here claimed `libOpenCL.so` "must be dlopen'ed on Android since
there is no standard ICD loader". That is wrong, or at least not how it is
normally done: llama.cpp's OpenCL backend targets Android by building the
Khronos ICD loader from source, dropping `libOpenCL.so` into the NDK sysroot and
linking against it. It also uses NDK 26.3.11579264, the same one this project
targets, and embeds the kernels into the binary by default
(`GGML_OPENCL_EMBED_KERNELS=ON`). Worth following when Phase 4 starts.
