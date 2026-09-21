# CPU_LLM_runtime

从零手写的 LLM 推理运行时（C++17，CPU + 目标 OpenCL），把 Qwen3-0.6B 跑在端侧。
**这是一个学习项目。**

> A from-scratch C++17 inference runtime for Qwen3-0.6B, built to learn on-device
> inference infrastructure layer by layer: every kernel, the KV cache, the memory
> manager, the backend abstraction and the profiler are hand-written. No MNN /
> ggml / llama.cpp / ONNX Runtime / LiteRT — the inference path is ours end to
> end. Correctness is pinned to a golden fixture captured from the HuggingFace
> reference and re-checked against it at every step.

---

## 为什么自己写

不是为了造轮子，是为了**看懂轮子**。端侧推理的性能瓶颈几乎全在"内存怎么搬"上：
权重多大、什么时候读、读几遍、放在几级缓存里。这些东西用现成库跑一遍是看不见的。

所以这个项目给自己定了一条硬规矩：**推理路径不许链接任何第三方推理库**。
Python 侧的 torch / transformers 只用来生成对拍数据，绝不进运行时。

配套的规矩是——**"能跑"不算通过，和参考实现算得一样才算**。
每一个算子的验收标准都是对着一份 golden 数据逐项比对，
而不是"输出看起来像句话"。

---

## 现在能做什么

```bash
# 看模型：config、311 张量表、内存足迹
./build/llmrt inspect --summary

# 生成：id 进，id 出（还没有 tokenizer）
./build/llmrt generate --ids 105538,59975,100132 --mode chat --max-new 12
```

`generate` 有两种 prompt 模式，**模板用 token id 表达，整条路径不经过 tokenizer**：

| 模式 | 做什么 | 实测（输入 `中国的首都是`） |
|---|---|---|
| `--mode raw` | 把 id 原样喂进去，模型续写文本 | `2130` = `'____'` ← 填空格式 |
| `--mode chat` | 套上 `<\|im_start\|>user\n…<\|im_end\|>\n<\|im_start\|>assistant\n` + 空的思考块 | `'中国的首都是北京。<\|im_end\|>'`，**自己命中 eos 停住** |

同一个模型、同一份权重，差的就是那 15 个模板 id。raw 模式选 `____` 是因为训练语料里
满是「中国的首都是____」这种填空题，正确的 `北京` 排在第二、只差 0.26 logits。

### 实测数字

```
28 层逐层 hidden state   vs golden   cos=1.00000000  rel 1.41e-07 ~ 4.84e-06
7 个 prompt 的 logits    vs golden   cos=1.00000000  rel 3.70e-06 ~ 9.62e-06
贪心生成的 token 序列    vs golden   逐 token 一致

生成速度                             1.68 s/token（无 KV cache）
测试                                 8 个套件 / 145 个用例
代码                                 include+src 4501 行 / tests 3630 行
构建                                 桌面 clang + Android arm64（NDK 26.3）双通过
```

---

## 现在还不能做什么

诚实列出，因为"还没做"和"做错了"应该一眼分得清：

| | 状态 |
|---|---|
| **文字进出** | ✗ 没有 tokenizer。id 进、id 出；要看文字得自己查 `vocab.json` |
| **多轮 / 长上下文** | ✗ 没有 KV cache，每生成一个 token 重算整个前缀 |
| **内存管理器** | ✗ 权重是一个 2.384 GB 的静态 arena，激活是固定成员变量 |
| **GPU** | ✗ 只有 CPU。桌面无 OpenCL 设备，Phase 5 才开工 |
| **量化** | ✗ 权重是 f32 |
| **采样策略** | ✗ 只有 greedy。`generation_config.json` 默认的 temperature/top-k/top-p 未实现 |
| **ONNX 加载** | ✗ 刻意放到最后 |

---

## 方法：为什么敢说"算对了"

**手写推理引擎最容易掉进的坑不是"跑不起来"，而是"跑起来了、结果看着像话、其实是错的"。**
QK-Norm 放错位置、RoPE 用成交错、GQA 映射写反 —— 这些都会产出"像模像样的浮点数"。

所以这个项目的做法是：**先有标准答案，再写代码**。

```
HuggingFace 参考实现（钉死版本）
        │  gen_oracle.py 抓每一个中间量
        ▼
data/golden/          32 个二进制数组 + manifest
        │  check_oracle.py 用纯张量运算独立重推整个前向
        │  → 53 项断言全部复现（证明基准本身没错）
        ▼
C++ 算子 / 前向        逐项对拍，判据 cos > 0.99999 且 rel < 1e-4
```

golden 里抓的东西是分层的，所以出错时能**定位**而不是只能看到"结果是垃圾"：

| 抓了什么 | 覆盖 |
|---|---|
| `logits` | 7 个 prompt × 全部位置 |
| `layer_hidden` | 28 层的逐层输出 ← 出错时报"第几层开始偏" |
| `ops/*` | prompt 0 第 0 层每个算子的**输入和输出**（26 个文件） |
| `greedy_ids` | 16 步贪心续写，用来验生成循环 |
| `embed_out` / `final_norm` | 首尾两端 |

逆着写：算子对拍 golden 的中间量，前向对拍 golden 的 logits，生成对拍 golden 的 token 序列。

---

## 一路上撞到的东西

这部分是项目的实际产出。每一条都带实测数字，不是书上抄的。

### 数值层面

**`head_dim` 是显式字段，不是 `hidden/heads`。** Qwen3-0.6B 两者差一倍
（128 vs 1024/16=64），所以 `q_proj` 投影到 2048 —— 两个 `hidden_size`。推导出来的话全错。

**QK-Norm 必须在 RoPE 之前，且作用在 `head_dim` 上。** 放错了 cosine 会掉到 0.87
左右 —— 不是崩，是"看起来还行但全错"。

**softmax 减最大值是恒等变形，不是近似。** `exp(-m)` 在分子分母上抵消
（依据 `exp(a-b)=exp(a)/exp(b)`），收益是让指数落在 `(0,1]` 从而永不上溢。

**运算顺序决定末位。** `u*g/d` 和 `u*(g/d)` 数学等价，但 C 里 `*` `/` 左结合，
前者多一次舍入：实测 2 ulp vs 1 ulp。

**bf16 → f32 就是左移 16 位**，精确无损。查了三个独立实现（MNN、ggml、我们自己的），
写法完全一样。

### 性能层面

**softmax 里 `exp` 占了 71% 的运行时。** 原实现每个元素算两遍 `exp`
（一趟求和、一趟输出），去掉重复后实测 **1.55×**。

**"读 bf16"从来不等于"用 bf16 算"。** 除了 AVX512-BF16 / ARMv8.6 `i8mm` /
RISC-V `zvfbfwma` 这些有原生指令的机器，其余全是加宽成 f32 再算。
MNN 在没有 NEON 时直接 `return false` 并留着 `// TODO: raw cpu version of bf16`；
ggml 的 `ggml_vec_dot_bf16` 有五个架构分支但**没有 NEON**，ARM 上落到标量循环。
省的是字节不是运算 —— 和 matmul 的 **0.25 FLOP/字节** 严丝合缝。

**每 token 一次前向，而权重流量与序列长度无关。** 所以没有 KV cache 时，
生成 N 个 token 就是 N × 2.38 GB 的权重读取 —— 这才是 1.68 s/token 的来源。
KV cache 能省掉注意力的 O(S²)，**省不掉权重**。

### 数据层面

**残差流里有一个 1400 倍的离群通道。** 位置 `[0, 35]` 的值是 ~6900，而整个
`[12, 1024]` 激活的其余部分是 O(5) 量级。后果：

- `maxabs` 在这份数据上是**无效指标** —— 幅值 6900 处 1 ulp 就是 `9.766e-04`，
  所以"整栈 maxabs≈1e-3"其实只是那一个通道差 1 ulp。判据只能用 `rel` 和 `cos`
- 朴素 int8 量化会直接削平这个通道，必须 per-channel 缩放

**28 层累积误差只涨了不到一个数量级。** 单算子 1e-6 量级，端到端最差 9.62e-06 ——
不是指数发散，说明没有系统性偏置。但**余量只剩 10.4×**，是全项目最紧的一处：

```
rmsnorm    1.13e-06   余量   88×
matmul     1.27e-06   余量   79×
swiglu     2.84e-08   余量 3500×
整栈 logits 9.62e-06   余量 10.4×   ← 任何改归约顺序的改动都要重新量这里
```

---

## 构建与运行

```bash
./scripts/build_desktop.sh          # -> build/llmrt
./scripts/build_android.sh          # -> build-android-arm64-v8a/llmrt（arm64 PIE）

ctest --test-dir build                          # 全部（约 99 秒）
ctest --test-dir build -R test_ops_cpu          # 单个套件
LLMRT_GREEDY_STEPS=16 ./build/tests/test_generate   # 跑满 16 步贪心（约 3 分钟）
```

模型放在 `/home/dr/models/Qwen3-0.6B`（可用 `LLMRT_MODEL_DIR` 覆盖）。

### 对拍数据不在仓库里

`data/golden/` 有 39 MB，且可由脚本重新生成，所以没进版本库。**新克隆的仓库跑测试时
相关的用例会自动跳过**（不会误报失败），要跑对拍得先生成：

```bash
PYTHONPATH=<你的 HF transformers 参考实现> \
  <参考实现的 venv>/bin/python tools/gen_oracle.py
```

⚠️ `gen_oracle.py` 里钉死了一个本地参考实现的路径，且版本敏感（transformers 4.51.0 /
torch 2.13，`attn_implementation="eager"`）。换环境要重新生成 —— 这是这个项目目前最大的
复现门槛。

---

## 目录

```
include/llmrt/     公共头
  common.h         DType / DeviceKind / Error / LLMRT_CHECK
  tensor.h         Tensor：非持有型视图（指针 + 偏移 + 形状 + 步长）
  json.h           手写 JSON 解析器
  safetensors.h    mmap 零拷贝读取器
  convert.h        bf16 / f16 -> f32
  config.h         Qwen3Config
  model.h          权重绑定（Qwen3Weights）
  forward.h        28 层前向（F32Weights + Qwen3Forward）
  generate.h       贪心生成 + 两种 prompt 模式
  ops.h            算子契约（8 个）

src/core/          json / tensor / version
src/io/            safetensors / convert
src/model/         config / weights / forward / generate
src/ops/cpu/       rmsnorm / matmul / swiglu / rope / softmax / attention /
                   embedding / argmax
src/cli/           main + 每个子命令一个 cmd_*.cpp

tests/             8 个套件，共 145 个用例；golden.h 惰性加载对拍数据
tools/             gen_oracle.py（抓 golden）/ check_oracle.py（自证 golden）
kernels/           （预留给手写 OpenCL）
docs/DECISIONS.md  决策与未决问题，每条绑定 commit hash
docs/transcripts/  开发会话原始日志
```

`Tensor` 是**视图不是所有者**：`data`（宿主指针，或以后的 `cl_mem`）+ 元素 `offset` +
`shape` + `strides`。所有权只在分配方（现在是模型层，Phase 4 之后是内存管理器），
所以"这块内存归谁释放"只有一个地方要判断。`data` 故意不带类型 —— 同一份描述符能表达
两种设备上的数据，这是异构调度的前提。

---

## 路线图

| Phase | 内容 | 状态 |
|---|---|---|
| 0 | CMake 骨架、零依赖、双构建、golden 流程 | ✅ |
| 1 | JSON / safetensors / bf16→f32 / 配置 / 权重绑定 / `inspect` | ✅ |
| 2 | 8 个手写 CPU 算子，逐个对拍 golden | ✅ |
| 3 | 28 层前向 + 贪心生成 + 两种 prompt 模式 | 🔶 缺 tokenizer 的 encode |
| 4 | **KV cache + 内存管理器** | ⬜ |
| 5 | 手写 OpenCL 后端（kernel + 内核缓存 + 缓冲池） | ⬜ |
| 6 | 异构调度：`--split 0-13:cpu,14-27:opencl` | ⬜ |
| 7 | Profiling：TTFT / tok/s / 逐层与搬移报表 | ⬜ |
| 8 | 性能优化与对比（CPU 分块、GPU 向量化、bf16、量化） | ⬜ |
| 9 | Android CLI：`adb push` + 真机取数 | ⬜ |
| 10 | Android APK（Gradle + JNI + 极简 UI） | ⬜ |
| 11 | ONNX 加载 | ⬜ |

详细的取舍、被推迟的优化（含实测代价）和开放风险记在
[`docs/DECISIONS.md`](docs/DECISIONS.md)，每条绑定一个 commit hash。
给 AI 助手看的工程约定在 [`CODEBUDDY.md`](CODEBUDDY.md)。

---

## 环境

```
C++17，零第三方依赖（标准库之外什么都没有）
桌面   clang（Linux x86-64）
Android NDK 26.3.11579264，arm64-v8a，android-24，c++_static
Python 仅用于生成对拍数据，不参与构建
```

参考实现：HuggingFace `transformers` 的 `modeling_qwen3.py`（4.51.0，`eager` attention）。
模型：[`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B)。
