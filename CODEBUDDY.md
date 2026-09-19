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

## Layout

```
include/llmrt/       public headers
  common.h           DType, DeviceKind, Error, LLMRT_CHECK
  json.h             hand-written JSON parser
  safetensors.h      mmap'd checkpoint reader
  convert.h          bf16/f16 -> f32
  config.h           Qwen3Config
  model.h            weight binding (Qwen3Weights, LayerWeights)
src/core/            json, version
src/io/              safetensors, convert
src/model/           config, weights
src/cli/             main + one cmd_*.cpp per subcommand (thin, no logic)
tools/               Python: golden generation, oracle checking
tests/               harness + one file per area; golden.h loads fixtures
kernels/             (reserved for hand-written OpenCL)
```

Implemented today: the metadata and loading layer. `llmrt inspect` prints the
config, the full tensor table and the memory footprint, and is the fastest way
to check a checkpoint:

```bash
./build/llmrt inspect              # full 311-row tensor table
./build/llmrt inspect --summary    # skip the table
```

## Roadmap (NOT yet implemented)

Tracking the approved plan; nothing below exists in the code yet.

1. Hand-written CPU ops (matmul, rmsnorm, qk_norm, rope, GQA attention, swiglu,
   softmax, embedding), each checked op-by-op against `data/golden/ops/*`.
2. Full 28-layer forward + byte-level BPE tokenizer + greedy sampling.
3. KV cache and the memory manager.
4. OpenCL backend: hand-written kernels, `cl_event` profiling, buffer pool.
5. Heterogeneous scheduler: `--split 0-13:cpu,14-27:opencl`, boundary
   transfers, double buffering.
6. Profiling report: TTFT, prefill/decode tok/s, per-layer and per-transfer
   timings.
7. Android CLI, then the APK, then ONNX.

The shared abstraction these hang off is `IBackend` (see the plan file):
ops plus buffer management, so a layer's weights and KV cache live on the
device that executes it, and the scheduler inserts host<->device copies only at
placement boundaries.

Note on OpenCL: this WSL machine has only the ICD loader, no vendor ICD and no
`/usr/include/CL`, so OpenCL kernels cannot be developed or tested locally
until POCL is installed (`sudo apt install pocl-opencl-icd opencl-headers
ocl-icd-opencl-dev clinfo`). `libOpenCL.so` exists only on an actual device,
where it must be `dlopen`ed (Android ships no standard ICD loader).
