# 决策与未决问题

这个文件记录两类东西：

1. **已生效的决策** —— 定了什么、为什么、代码在哪。
2. **记录在案的替代方案与被推迟的优化** —— 现在不做，但代价已经量过，不至于将来
   重新发现一遍。

每条都绑定 git commit hash。hash 是"当时实际是怎么写的"的凭据，不是引用文献。

**约定**：改主意时不删旧条目，追加新条目并注明它取代了谁。hash 只能指向已经存在的
提交，所以新条目的 hash 会在落地后的下一次文档更新中回填 —— **允许一次提交的滞后**。

```
代码基线  807365b   28 层前向打通，7 个 prompt 的 logits 全部对上 golden
文档基线  8cd7754   本文件首次提交
```

---

## 一、已生效的决策

### D1 推理路径全手写，不引入第三方推理库

`llmrt_core` 只依赖 C++ 标准库。MNN / ggml / ONNX Runtime 用来**读**，不用来**链**。

理由：项目目的就是走通每一层，拿现成库拼出来没有意义。Python 侧（torch / transformers）
只用于生成 golden 数据，绝不进入运行时。

```
121a59d  骨架 + 零第三方依赖
```

### D2 对拍 golden 是唯一的正确性判据

每个算子都对着 `data/golden/` 逐项比对，"能跑"不算通过。判据在
`tests/golden.h`：`cosine > 0.99999 且 rel < 1e-4`，其中
`rel = max_abs / max|expected|`（**全局归一，不是逐元素比值** —— 逐元素比值在参考值
穿过 0 的地方必然爆掉，attention 上实测会虚报成 4.5e-4）。

golden 本身也是被验证过的：`tools/check_oracle.py` 用纯张量运算独立重推整个前向，
断言能复现全部 53 个中间量。它是一份**可执行的规格**。

```
121a59d  生成 golden
3ebda0d  归档开发会话
```

### D3 Tensor 是视图，不是所有者

`data` + `offset` + `shape` + `strides`，没有 `owning` 标志。`data` 是 `void*`：
CPU 时是宿主指针，OpenCL 时是 `cl_mem`。同一份描述符能表达两种设备的数据，
**这是异构调度的前提**。

代价：`Tensor` 不是 trivially copyable（含两个 `vector`），所以热路径一律传 `const&`。

```
3bcff8f  加入 Tensor 视图类型
```

### D4 算子只收连续、f32、在 CPU 上的张量

不满足就抛，不静默读错。`require_contiguous()` 在每个算子入口调用一次。

代价：跨步视图（转置后的 KV cache）用不了，目前用显式拷贝绕开 —— 记录为 P2。

```
06be20f  rmsnorm
15ab236  matmul
```

### D5 算子不分配内存，scratch 由调用方传入

输出以 `Tensor&` 传而不是返回值，这样 Phase 4 的内存管理器是**唯一**决定字节归属的
地方，OpenCL 后端也能把输出放在设备缓冲上。

```
06be20f  rmsnorm 起就如此
```

### D6 浮点写法以"和参考一致"为准则，不是"算得最准"

写在 `CODEBUDDY.md` 的 "Floating-point rules" 一节。已固化的具体条目：

- `1.0f` 而不是 `1.0`（后者把整段表达式提升到 double）
- 用 `1/sqrt` 不用硬件 rsqrt 近似指令
- swiglu 用 `x/(1+exp(-x))` 而不是 `x*sigmoid(x)`（后者差 ~2.4e-7）
- softmax 减最大值是**恒等变形**：`exp(-m)` 上下抵消，不是近似
- `u*(g/d)` 而不是 `u*g/d`（C 里 `*` `/` 左结合，后者多一次舍入，实测 2 ulp vs 1 ulp）

```
808cd29  记录六类坑
19e7ace  softmax
```

### D7 全序列 logits，不切最后一个位置

golden 的 `logits` 是 `[60, 151936]`，覆盖每个位置。我们的前向也输出 `[S, vocab]`。
真实推理只需要最后一个位置，但那是 P3 的事。

```
121a59d  生成 golden（logits 就是全序列）
```

---

## 二、Phase 2 收尾与端到端 — **已完成**

目标：**手写引擎第一次真的算出概率分布** —— 用 golden 的 `input_ids` 跑完 28 层，
和 `logits.f32.bin` 对拍。

7 个 prompt 的切片位置在 manifest 的 `prompt_offsets [0,12,17,28,46,49,56]`：
1 个随机 token（seed 42，12 个）+ 6 段自然文本（英/中/代码/数列）。

### A1 加 `DType::I32`（索引缓冲的类型） — 已落地

token id 需要一个整数 dtype 才能被 Tensor 描述。定位是**索引缓冲**，不是数值类型：
不做权重类型、不做计算类型，`convert_to_f32` 明确拒绝而不是加宽（静默转 float 在
id 超过 2^24 之前都看不出问题）。

```
1a09368
```

### A2 `embedding` 算子 — 已落地

`out[s][:] = table[ids[s]][:]`，纯 gather，无算术，所以**逐位相同**（实测 memcmp=0）。

三条"没有"写进了注释和测试，因为每一条在别的模型里都有对应做法：没有 padding 行
（config 里没有 `pad_token_id`，实测全零行数为 0）、没有缩放（Qwen3 不是 Gemma）、
没有重归一化。

`ids` 的范围检查**下界和上界同等重要** —— `int32_t` 有符号，负 id 得到负字节偏移，
走向表前面，只查上界会漏。

```
2083c73
```

### A3 `F32Weights` + `Qwen3Forward` — 已落地

前向放 `src/model/`（不是算子层），自持 scratch。见 D8 / D9 的取舍。

三个必须记住的实现细节，都是真的会踩的：

- **arena 必须定长之后再建 Tensor view。** 中途 `vector` realloc 会让所有 view 悬空。
  做法是先算总字节、一次 `resize`、再按游标建 view。类型禁拷贝禁移动 —— 拷贝会
  复制 arena，但 Tensor 里的指针仍指向原对象的内存。
- **`rmsnorm` 可以原地**（`x` 和 `out` 传同一个 Tensor）：实现是先读 `x_row[j]`
  再写 `out_row[j]`，同址安全，每层省两个 `[S,1024]` 缓冲，一次 forward 省 57 次。
- **视图不能缓存为成员。** `allocate()` 会 `resize` 那些 vector，一 resize 指针就
  失效。所以视图在用到的地方现建 —— 代价是每次 forward 几百次小分配，相对 2.4 GB
  的权重流量可以忽略。

层内顺序与 lm_head 同权，都已从 `modeling_qwen3.py` 逐行确认：

```
x = x + attn(input_layernorm(x))
x = x + mlp(post_attention_layernorm(x))
h = final_norm(h);  logits = h @ embed_tokens^T      ← lm_head 与 embed 同权
```

`LayerObserver` 钩子（甲方案）：`run()` 可接受一个回调，每层之后交出 hidden state。
golden 恰好存了 `layer_hidden[28,1,12,1024]`，所以出错时能报"第几层开始偏"，而不是
对着一个错的 logits 数组发呆。

```
807365b
```

### A4 端到端对拍（7 个 prompt） — 已落地

```
28 层逐层 hidden state    cos=1.00000000  rel 1.41e-07 ~ 4.84e-06
7 个 prompt 的 logits     cos=1.00000000  rel 3.70e-06 ~ 9.62e-06
```

```
807365b
```

---

## 三、记录在案的替代方案与被推迟的优化

### P1（替代 B2）权重保持 bf16 + 自写 NEON 加宽内核

**现状 B1**：载入时全部转 f32。

```
safetensors 数据区            1.503 GB (bf16, 311 张量)
去掉同权的 lm_head            1.192 GB (310 张量)
f32 镜像                      2.384 GB
```

**代价不只是内存，是带宽。** 之前量过 matmul 是 **0.25 FLOP/字节** —— 瓶颈是搬权重
不是算。权重减半 → 流量减半 → 内存受限时接近 2× 提速。

**为什么现在不做**：这不是"改个 dtype"，是**要写一套 SIMD 内核**。三个独立实现都
指向同一结论 —— 除了有原生 bf16 指令的机器（AVX512-BF16 / ARMv8.6 `i8mm` /
RISC-V `zvfbfwma`），其余全部是"加宽成 f32 再算 f32"：

- MNN `source/backend/cpu/bf16/BF16Functions.cpp:332`：`#if !defined(MNN_USE_NEON)
  return false;`，同处留着 `// TODO: raw cpu version of bf16`（至今没做）。
  `:340` 的 `gInstance->matmulBytes = 2` 告诉分配器元素是 2 字节；
  `Convolution1x1Strassen.cpp:55` 在**权重准备阶段**调 `MNNPackForMatMul_B`
  把权重重排成 SIMD 打包布局。低精度是一整套独立的函数表，不是类型参数。
- ggml `ggml/src/ggml-cpu/ggml-cpu.c:395`：BF16 的 `vec_dot_type = GGML_TYPE_BF16`，
  也就是说**激活也被下变成 bf16**（比我们想要的更差）。
- ggml `ggml/src/ggml-cpu/vec.cpp:139`：`ggml_vec_dot_bf16` 有 AVX512-BF16 / AVX512F /
  AVX2 / RISC-V / POWER9 五个分支，**没有 NEON 分支** —— ARM 上落到标量循环
  `GGML_BF16_TO_FP32(x[i]) * GGML_BF16_TO_FP32(y[i])`。

**我们比 ggml 有优势**：我们的 matmul 是自己写的，可以做到"权重 bf16 + 激活 f32"，
只加宽权重那一侧，精度损失只有 checkpoint 本身的 bf16，比 ggml 少降一档。

**目标平台是 Android arm64**，所以真要做得自己写 NEON：
`vld1q_u16` → `vshll_n_u16(v, 16)` → `vfmaq_f32`。这也解释了 llama.cpp 在 ARM 上
推荐 Q4_0/Q8_0 而不是 bf16 的原因。

**归入 Phase 8，和量化同批做**（int8/int4 是同一个改动方向的延伸）。

### P1.5 残差流里有一个 1400 倍的离群通道 —— 量化的硬约束

实测 `data/golden/layer_hidden.f32.bin`：**位置 `[0, 35]` 这一个通道的值是 ~6900，
而整个 `[12, 1024]` 激活的其余部分是 O(5) 量级。**

```
层   max|h|      位置     该位置的值
 2  6895.747   [0,35]   6895.7471
 ...
25  6912.265   [0,35]   6912.2646
27  1915.046   [0,35]   1915.0459
```

这个尖峰从第 2 层出现后被残差流一路带着，量级几乎不变（6895 → 6865 → 6912）。

三个后果：

1. **`maxabs` 在这份数据上是无效指标。** 幅值 6900 处 1 ulp = `2^-10` = 9.766e-04，
   所以"整栈 maxabs ≈ 1e-3"其实只是**那一个通道差 1 ulp**，其余元素吻合得好得多。
   这也解释了逐层对拍里 layers 10-25 的 maxabs 完全相同（9.766e-04）——同一个通道、
   同样的舍入方向，不是重复数据。**判据只能用 `rel`（它按 max|expected| 归一，而
   max|expected| 也是这个尖峰）和 `cos`。** 这条反过来印证了 `tests/golden.h` 里
   `rel = max_abs / max|expected|` 的设计。
2. **朴素 int8 量化会直接削平这个通道**（动态范围差 1400 倍）。P1 真要落地时，
   要么 per-channel 缩放，要么把这个通道单独提出来，要么用 Q4_K 之类带 per-block
   scale 的格式。ggml 的 k-quant 就是这个思路。
3. 第 27 层的绝对误差到了 9.277e-03（76 ulp @ 幅值 1915），比第 10 层的 9.8e-04
   涨了 9.5 倍 —— 比 √N 的随机游走预期（4 倍）快一些。rel 仍是 4.8e-06。

### P2（替代 C2）统一成 `[B,S,H,D]`，消掉每层 3 次 head 转置

**现状 C1**：保留 HF 的 `[B,H,S,D]` 中间布局，用显式拷贝转置。

```
q  [S,H,D]  -> [H,S,D]   512*16*128 = 1,048,576 floats   4 MB
k  [S,Kv,D] -> [Kv,S,D]   512* 8*128 =   524,288 floats   2 MB
v  [S,Kv,D] -> [Kv,S,D]   512* 8*128 =   524,288 floats   2 MB
                                  合计 2,097,152 floats = 8 MB
转置是读+写 → 每层 16 MB 流量，×28 层 = 448 MB
对照 f32 权重流量 2.384 GB → 占 19%
```

**替代方案**：把激活统一成 `[B,S,H,D]`（head 在 seq 后面）。于是

- k/v 不用转：`v_proj` 输出 `[S, Kv*D]` → reshape `[S,Kv,D]` 就是它要的
- `q_norm` 原样可用（最后一维还是 D）
- attention **输出也不用转** —— 它天然写成 `[S, H*D]`，那正是 `o_proj` 的输入布局

代价：`rope` 的轴语义要改（position 变成第一轴而非中间轴），attention 的输入布局要改，
两者及其测试都要重写。

**为什么现在不做**：rope 和 attention 是当前**唯一已对拍验证过**的两个算子，端到端
里程碑之前不该重写它们；golden 抓的 `rope_q_in/out` 正好是 `[1,16,12,128]`，改约定要
连测试夹具一起翻新；而 19% 的成本只在 S=512 才显现，S=12（里程碑测试）时为零。

**Phase 8 用实测决定**：届时 profiler 可以直接量出转置占 forward 的比例。

### P3 长上下文不要落地整个 logits

```
logits [S=12,    151936] f32 =     7.3 MB
logits [S=512,   151936] f32 =   311.2 MB
logits [S=40960, 151936] f32 = 24893.2 MB   ← 24.9 GB
```

最大上下文下整个张量 24.9 GB，装不下。推理时永远不需要全序列 logits，
只有最后一个位置有用。真实实现要么 `logits[:, -1, :]` 只投影最后一个位置，
要么把 lm_head 投影和 argmax 融合，连 151936 维的向量都不落地。

（golden 是全序列，因为要验证**每个位置**。测试按 prompt 切片跑，S 最大 18，
所以现在碰不到这个问题。）

### P4 朴素顺序累加

目前所有归约都是顺序累加。实测的代价：

| | maxabs | 说明 |
|---|---|---|
| numpy `@`（BLAS 分块归约） | 1.192e-07 | 1 ulp |
| 我们的顺序累加 | 4.172e-07 | 3.5 ulp |

这和当初 rmsnorm 换 double 累加器后好 24 倍是同一件事：**参考实现的归约比朴素顺序
累加准**。attention 那里不值得改（最内层点积翻倍开销，而 rel 3.3e-07 是容差的 300 倍），
rmsnorm 那里也不值得（内存带宽受限，不是 ALU 受限）。

真要做是 SIMD 分块归约，归入 Phase 8。

### P5 ONNX 加载

刻意推迟到所有其他功能都工作之后。safetensors 优先。

---

## 四、开放风险

### R1 28 层累积误差 — **已量，不需要处理**

实测（`807365b`）：单算子 1e-6 量级，端到端 logits 最差 **rel 9.62e-06**。

**涨了不到一个数量级，不是指数发散** —— 说明没有系统性偏置，只是随机游走式的
累积（误差按 √N 而不是 N 增长，和当初 rmsnorm 那次推导的规律一致）。

但**余量只剩 10.4×**（1e-4 / 9.62e-06），是全项目最紧的一处：

| | rel | 余量 |
|---|---|---|
| rmsnorm | 1.13e-06 | 88× |
| matmul | 1.27e-06 | 79× |
| swiglu | 2.84e-08 | 3500× |
| **整栈 logits** | **9.62e-06** | **10.4×** |

将来任何提高精度的改动（P1 的 bf16 内核、P4 的归约方式）都要重新量这一项。
10× 的余量够用但不再宽裕，**换归约顺序之前必须先跑 `test_forward`**。

### R2 `-ffp-contract` 默认开启

GCC/Clang 默认允许把 `a*b + c` 合并成一条 FMA，把两次舍入变成一次。我们的构建
**没有关掉它**，所以：

- 换编译器、换优化级别、换到 ARM，**结果都可能变**
- 单算子时余量 100~3500× 无所谓；28 层之后还剩多少需要看数（R1）
- 目前已知存在，**不处理**；需要跨平台严格一致时才考虑 `-ffp-contract=off`

### R3 桌面没有可用 OpenCL 设备

只有 ICD loader，`/etc/OpenCL/vendors` 为空，没有 `/usr/include/CL`。Phase 5 之前需要

```
sudo apt install -y pocl-opencl-icd opencl-headers ocl-icd-opencl-dev clinfo
```

否则 OpenCL 内核只能在真机上迭代。

### R4 Android 真机尚未接入

`adb devices` 为空。Phase 9–10 需要。

### R5 公开仓库里的历史包袱

仓库是 public，`docs/transcripts/` 里有原始会话日志。一旦被抓取/克隆则无法收回。
要撤的话：

```
gh repo edit dr3334/CPU_LLM_runtime --visibility private
```

---

## 五、复现基线用到的命令

```bash
# 全量测试（当前 6 套件 / 122 用例）
ctest --test-dir build --output-on-failure

# 单套件
stdbuf -o0 ./build/tests/test_ops_cpu        # stdin 缓冲：测试 abort 时 stdout 会丢

# 双构建
./scripts/build_desktop.sh && ./scripts/build_android.sh

# golden 自证（53 项）
PYTHONPATH=/home/dr/mtk_toolkit_enhence/reference \
  /home/dr/mtk_toolkit_enhence/.venv-ref/bin/python tools/check_oracle.py
```
