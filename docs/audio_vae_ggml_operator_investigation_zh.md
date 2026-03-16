# AudioVAE 的 ggml 算子级优化调查报告

> 状态说明（2026-03-16）：
> 这份文档记录的是一轮已经完成并收尾的 `ggml` CPU 侧优化实验。
> 当前仓库代码里已经撤回：
> - `conv_transpose_1d` micro-kernel 实验路径
> - AudioVAE 专用 instrumentation
> 保留下来的主要是文档结论和 `benchmark/results/` 中的结果文件。

## 1. 目标与边界

本报告的目标是回答一个更具体的问题：

- 为什么 AudioVAE 在 CPU 上“已经能量化，但没有变快”？
- 下一步如果要在 `third_party/ggml` 侧做原型，最值得先改哪几个算子？

本报告基于当前仓库本地源码阅读完成，重点看了：

- `third_party/ggml/src/ggml.c`
- `third_party/ggml/src/ggml-cpu/ggml-cpu.c`
- `third_party/ggml/src/ggml-cpu/ops.cpp`

本阶段结论是：

- 暂时**不建议**先做“大面积图改写”
- 更建议先在 `ggml` CPU 内核上做小而准的原型
- 先做测量，再做 fused / cached / specialized kernel

## 1.1 2026-03-16 进展更新

这份文档里的“先测量，再决定第一批 patch”的路线曾在当前仓库里落地过一轮。

实验期间曾完成的事项：

- 已在 `third_party/ggml/src/ggml-cpu/ggml-cpu.c`
- 已在 `third_party/ggml/src/ggml-cpu/ops.cpp`
- 已在 `benchmark/voxcpm_benchmark.cpp`

当时加入了 opt-in 的 CPU instrumentation，使用：

- `VOXCPM_GGML_PROFILE=1`

即可在 benchmark 结束时输出聚合统计。当时会记录：

- `im2col`
- `mul_mat` 的激活 repack
- `conv_transpose_1d`

并按 benchmark case scope 聚合，例如：

- `audio_vae.encode`
- `audio_vae.decode`

同时也做了一轮实际 benchmark 验证，使用：

- backend: `cpu`
- threads: `8`
- scenario: `short / medium / long`

当前实测结论已经比较明确：

### A. `audio_vae.decode` 的主热点仍然是 `conv_transpose_1d`

在当前 F32 模型、CPU 8 线程下，`audio_vae.decode` 的 tracked hotspot 中：

- `conv_transpose_1d` 约占 `85% ~ 91%`
- `im2col` 约占 `9% ~ 15%`

这说明 decode 侧“先看 `conv_transpose_1d`”的总体方向是对的。

### B. 当前这轮 benchmark 没有触发 `mul_mat` repack 热点

本次验证使用的是当前主模型的 F32 路径，因此 profiler 中没有看到显著的 `mul_mat` activation repack。

这意味着：

- 当时的 instrumentation 已经具备观察 repack 的能力
- 但若要判断 mixed / low-bit 路径到底慢多少，还需要后续用量化模型再跑一轮

所以“encoder 是否主要慢在 `mul_mat pack`”这一点，当前还不能仅凭这次 F32 benchmark 下最终结论。

### C. `conv_transpose_1d` 的 kernel cache 原型已经试过，但未保留

我实际做过一版 `conv_transpose_1d` kernel 预打包缓存原型，思路是：

- 对固定权重 kernel 预先 permute
- forward 时跳过 kernel 重排

但在当前 workload 上，这个 patch **没有带来稳定收益**，因此已经回退，没有保留在当前代码里。

原因从 phase breakdown 也能看出来：

- kernel pack 很小
- 真正重的是 source pack 和 compute

例如 long 场景最后一层样本里，记录到的典型值是：

- `kpack=87us`
- `srcpack=143621us`
- `compute=115569us`

这说明“只缓存 kernel”并没有切中当前 decode 侧最重的成本中心。

### D. 当前文档的推荐顺序需要相应修正

因此，这份文档后续阅读时应按下面顺序理解：

1. instrumentation 路线已经被证明是正确的第一步
2. `conv_transpose_1d` 仍是 decode 主热点
3. 但“kernel cache 作为第一批优化”已经试验过，当前不建议继续投入
4. 下一步更值得研究的是：
   - `conv_transpose_1d` 的 source 重排
   - `conv_transpose_1d` 的 compute 内核
   - 以及量化模型下 encoder 的 `mul_mat pack`

### E. `source pack / dst zero` 并行化原型也已经试过，但同样未保留

在确认 kernel cache 无效之后，我又做过一版更靠近当前主瓶颈的低风险原型：

- 将 `conv_transpose_1d` 的 source 重排改成多线程协作
- 将输出张量清零改成多线程协作

从 profiler 的 phase breakdown 看，这版原型确实能把 `srcpack` 这段单独压低。

但是端到端 `audio_vae.decode` benchmark 仍然没有得到稳定收益，反而在当前机器上更慢，因此这版也已经回退，没有保留在当前代码里。

这进一步说明：

- 当前 decode 侧的主要问题不能只靠“继续改 pack”解决
- 下一步应更直接转向 `conv_transpose_1d` 的 compute 内核本身
- profiler 仍然有价值，因为它已经帮助排除了两条看起来合理、但当前收益不成立的路线

### F. 一轮低风险 compute 专门化原型也已经试过，但同样未保留

在 kernel cache 和 source pack 两条路线都失败后，我又尝试了一版非常保守的 compute 侧原型：

- 针对当前最常见的 `kernel=4` 场景
- 对 `conv_transpose_1d` 内层循环做固定 tap 的专门化
- 目标只是减少循环控制和指针运算开销，不改数值语义

结果仍然是：

- `test_audio_vae` 可以通过
- 但端到端 `audio_vae.decode` benchmark 依然没有得到稳定收益
- 因此这版也已经回退，没有保留在当前代码里

进一步说明：

- 这版 compute 专门化比前两条路线更接近正确方向
- 在 profiler 里，`conv_transpose_1d` tracked time 确实出现了小幅下降
- 但这点下降仍然不足以转化成稳定的端到端 decode 改善
- 因此它仍然不满足“值得留在主线里继续维护”的标准

这意味着当前已经连续排除了三条“低风险、看起来合理”的路线：

1. kernel cache
2. source pack / dst zero 并行化
3. 基于双路 dot 的小规模 compute 专门化

因此下一步如果继续推进 `conv_transpose_1d`，就不应再期待“很小的局部 patch”自动带来收益，而要准备进入更明确的结构性优化，例如：

- 更换内部数据布局
- 设计更适合当前 shape 的专用 micro-kernel
- 或者重新审视是否需要更大粒度的算子表达

对应的下一阶段设计稿已单独整理为：

- `docs/audio_vae_conv_transpose_microkernel_design_zh.md`

### G. `conv_transpose_1d` 专用 micro-kernel 实验已经完成，并最终放弃

在完成设计稿和骨架之后，我曾在 `third_party/ggml/src/ggml-cpu/ops.cpp` 里把实验路径扩成一版可运行的 family 原型：

- 当时默认关闭
- 当时通过环境变量启用：
  - `VOXCPM_GGML_AUDIOVAE_TRANSPOSE_MICROKERNEL=1`
- 当时覆盖：
  - `F32 -> F32`
  - `kernel = 2 * stride`
  - `stride in {2, 3, 5, 6, 7, 8}`

这版原型的核心变化是：

- 不再做 source pack
- 直接从原始 source 布局读取
- 按输出通道 tile 做直算累加

正确性状态：

- 默认路径下 `test_audio_vae` 通过
- 实验路径下 `test_audio_vae` 也通过

profiler 观察：

- 这版原型在命中的 `conv_transpose_1d` 上已经能做到：
  - `kpack=0`
  - `srcpack=0`

说明“绕开 pack、直接进入 compute”这条结构性方向是可行的。

顺序 benchmark 重新跑完之后，结论变成了：

- 新模型 `models/voxcpm1.5.gguf`
  - `short / repeat=3`
    - baseline: `707.854 ms`
    - micro-kernel family: `412.280 ms`
  - `long / repeat=3`
    - baseline: `4978.141 ms`
    - micro-kernel family: `5414.926 ms`
- 老模型 `models/voxcpm-0.5b.gguf`
  - `short / repeat=3`
    - baseline: `291.496 ms`
    - micro-kernel family: `109.785 ms`
  - `long / repeat=3`
    - baseline: `1221.000 ms`
    - micro-kernel family: `1434.086 ms`

也就是说：

- `short` 场景下，这条 family 路线收益非常明显
- 但 `long` 场景下，新旧模型都出现了明确回退

在 `VOXCPM_GGML_PROFILE=1` 下，命中路径仍能确认：

- `kpack=0`
- `srcpack=0`

这说明它不是“方向错误”，而是当前 family 级直算路径在长场景下还没有把缓存/寻址行为组织好。

因此最终结论是：

- 这版原型不值得继续保留在当前代码里
- 原因不是“完全无效”，而是：
  - `short` 场景虽然明显变快
  - 但 `long` 场景下两代模型都回退
  - 同时它引入了额外实现分支和维护负担
- 因此当前已经把这条 micro-kernel 实验路径从代码中撤回
- 当前保留下来的只有：
  - benchmark 数据
  - 这份经验总结

补充校正：

- `models/voxcpm1.5.dump` 对应老版 AudioVAE：
  - `decoder_rates = [7, 7, 6, 3, 2]`
  - `decoder_dim = 2048`
- `models/voxcpm-0.5b.dump` 对应较新的当前主模型 AudioVAE：
  - `decoder_rates = [8, 8, 5, 2]`
  - `decoder_dim = 1536`

因此如果后续要求同一套代码同时兼容新旧模型，那么实验 micro-kernel 路线就不能只盯住：

- `kernel=4, stride=2`

而应该至少扩成一组 `kernel = 2 * stride` family：

- `4/2`
- `6/3`
- `10/5`
- `12/6`
- `14/7`
- `16/8`

这组 family 的实验虽然已经做过，但当前代码中已不再保留对应执行入口。

### H. 这轮实验留下来的真正经验

这次实验真正值得保留的，不是 micro-kernel 代码本身，而是下面这些判断：

1. `conv_transpose_1d` 确实是 decode 侧主热点，继续研究方向没有错。
2. 只做 `kernel cache` 不够，因为 `kernel pack` 不是主要矛盾。
3. 只做 `source pack` 轻量并行化也不够，子阶段下降不等于端到端下降。
4. 直接绕开 `pack` 的 compute 路线在 `short` 场景上能打出收益，说明方向有技术价值。
5. 但当前这版 family 级直算实现对 `long` 场景缓存行为不友好，导致新旧模型都回退。
6. 对当前项目来说，这种“短场景很好、长场景回退、维护成本上升”的 patch 不值得留在主线。

因此，本轮实验的最终价值是：

- 缩小问题空间
- 证明哪些路线不值得继续
- 给后续若真的要重做 `conv_transpose_1d` 优化的人留下 benchmark 和设计边界

而不是把这版代码继续维护下去。

---

## 2. AudioVAE 当前在 ggml 上的实际调用路径

### 2.1 普通卷积

当前规则卷积的计算链路是：

1. `ggml_pad_ext` 做左侧 causal padding
2. `ggml_im2col` 把激活展开
3. `ggml_mul_mat(weight, activations)`
4. `ggml_add` 加 bias

这条路径的好处是：

- 能复用现有 `mul_mat` 量化能力

坏处是：

- 先物化 `im2col`
- 再执行 `mul_mat`
- 两步之间还有激活类型、布局和缓存开销

### 2.2 深度卷积

depthwise conv 当前不是 `ggml` 原生 op，而是项目侧自定义 kernel。

这条路径目前已经支持：

- `F32` / `F16` 权重
- `F32` / `F16` bias

它不是当前 mixed-Q4_K slowdown 的主嫌疑，因为：

- depthwise 权重体积很小
- 也没有走低比特 `mul_mat`

### 2.3 转置卷积

decoder 的上采样路径走：

- `ggml_conv_transpose_1d`

这部分仍然是 AudioVAE decode 侧最重的候选热点之一。

---

## 3. 源码级观察

## 3.1 `ggml_im2col` 本身是严格 shape-sensitive 的

位置：

- `third_party/ggml/src/ggml.c`

1D 情况下有这个断言：

- `GGML_ASSERT(b->ne[1] == a->ne[1]);`

这里的语义是：

- 输入激活 `b` 的 channel 数必须等于 kernel `a` 的 input channel 数

这也是为什么旧版 `voxcpm_tts` 二进制在加载新格式量化模型时会直接在 `ggml_im2col` 处崩掉。

这条观察说明：

- 只要我们继续复用 `im2col`，运行时 shape 解释必须绝对严格
- 一旦后续想做新的 fused conv op，可以考虑绕开这段 shape-only kernel tensor 的构造

## 3.2 CPU 侧 `im2col` 是实打实的“物化激活矩阵”

位置：

- `third_party/ggml/src/ggml-cpu/ops.cpp`
- `ggml_compute_forward_im2col_f32`
- `ggml_compute_forward_im2col_f16`

当前实现特征：

1. 会把输入显式写入目标 buffer，而不是 lazy view。
2. F16 路径会在写出时做 `FP32 -> F16` 转换。
3. 线程划分是 `for (iic = ith; iic < IC; iic += nth)`，也就是按 channel 分摊。
4. 对 1D pointwise (`kernel=1`) 没有专门分支。

这意味着：

- 对很多 AudioVAE 小卷积而言，`im2col` 本身的 copy/gather 成本未必小于真正的算术成本
- pointwise conv 明明本质更像线性层，但还是走了通用 `im2col`

## 3.3 CPU 侧 `mul_mat` 在 low-bit 路径下会 repack 激活

位置：

- `third_party/ggml/src/ggml-cpu/ggml-cpu.c`
- `ggml_compute_forward_mul_mat`

关键行为：

1. `src0` 是权重，`src1` 是激活。
2. 若 `src1->type != vec_dot_type`，会把 `src1` 转换/打包到 `params->wdata`。
3. 这条分支里还有明确断言：`GGML_ASSERT(src1->type == GGML_TYPE_F32);`

对我们当前 AudioVAE mixed 路径的影响是：

- 权重是 low-bit `Q4_K/Q5_K/...`
- 激活是 `F32`
- 因此每个规则卷积 `mul_mat` 都要额外做一次激活 repack

这非常像当前 slowdown 的核心来源之一：

- 低比特权重省下来的带宽和算术，不足以抵消激活 repack + `im2col` 物化

## 3.4 `conv_transpose_1d` 只支持 `F16/F32` kernel，而且每次都会重排数据

位置：

- `third_party/ggml/src/ggml-cpu/ops.cpp`
- `ggml_compute_forward_conv_transpose_1d_f16_f32`
- `ggml_compute_forward_conv_transpose_1d_f32`

当前实现特征：

1. 只支持：
  - `src0=F16, src1=F32, dst=F32`
  - `src0=F32, src1=F32, dst=F32`
2. 每次调用时：
  - 先清空 `params->wdata`
  - 把 kernel 从 `(K x Cout x Cin)` 重排到 `(Cin x K x Cout)`
  - 把 source 从 `(L x Cin)` 重排到 `(Cin x L)`
  - 再执行嵌套 dot 累加
3. 每次调用还会先 `memset(dst->data, 0, ggml_nbytes(dst))`

这说明 decode 侧至少有三个潜在成本：

1. kernel 重排是重复的
2. source 重排是重复的
3. 输出清零和逐点累加写回可能有较大 cache 压力

## 3.5 当前证据说明“问题不只是权重位宽”

真实 benchmark 已经给出一个很重要的反证：

- FP32 最快
- mixed Q4_K 更慢
- AudioVAE=F16 + Q4_K 还要更慢

如果问题主要只是“low-bit 精度不合适”，那么 `AudioVAE=F16` 本该更接近 FP32。

但它依然明显慢于 FP32，说明当前 CPU 开销还有别的主因，例如：

- `conv_transpose_1d` 的实现方式
- `im2col` 的物化
- 小矩阵场景下线程/缓存/重排开销

---

## 4. 为什么当前 mixed 路径会慢

综合源码与实测，当前最可信的解释是：

### 4.1 encoder 侧：`im2col + activation repack + small matmul`

规则卷积低比特化以后，encoder 每层大致变成：

1. 生成 `im2col`
2. 把 `F32` 激活 repack 到 `vec_dot_type`
3. 执行小到中等规模的 low-bit `mul_mat`

对 AudioVAE 这种：

- 层很多
- 规则卷积 kernel 小
- patch 长度不算特别大

的 workload 来说，这三步叠加非常容易吃掉 low-bit 的收益。

### 4.2 decoder 侧：`conv_transpose_1d` 仍然很重

mixed Q4_K 模型里，decoder 最大块的上采样权重仍然是 `F16`。

因此 decode 侧没有享受到“权重显著缩小后直接加速”的效果，反而还要承担：

- 内核重排
- source 重排
- 累加写回

### 4.3 `AudioVAE=F16` 更慢，说明不是“只要降低精度就能更快”

`AudioVAE=F16` 模式主要减少的是：

- 模型加载大小
- 权重内存体积

但它没有减少：

- `im2col`
- `conv_transpose_1d` 的重排逻辑
- `F32` 输出累加

因此在当前 CPU 后端上，它没有变成更快的折中方案。

---

## 5. 候选优化方向

以下方向按建议优先级排序。

## 5.1 P0：先补可观测性，而不是直接改内核

第一优先级不是“马上写新 kernel”，而是先把热点量化清楚。

建议在 `ggml` CPU 路径加最小测量：

1. `GGML_OP_IM2COL` 总耗时
2. `GGML_OP_MUL_MAT` 中 `src1` repack 耗时
3. `GGML_OP_CONV_TRANSPOSE_1D` 总耗时
4. 每个 op 的输入输出 shape、类型、调用次数

目标：

- 判断 encoder 真正慢在 `im2col` 还是 `mul_mat pack`
- 判断 decoder 真正慢在 kernel 重排还是 dot 累加

如果没有这一层测量，后续很容易反复做“看起来合理、实际更慢”的优化。

## 5.2 P1：`conv_transpose_1d` kernel cache 已试验，当前不建议继续作为第一刀

这条路线已经做过一轮原型验证，但在当前 workload 上没有带来稳定收益，因此**不再推荐作为当前第一刀**。

原始理由本身没有错：

1. decoder 的 kernel 是固定权重，不该每次 forward 都重新 permute。
2. 当前实现每次调用都会重排 kernel 到 `params->wdata`。
3. 这类优化对数值语义影响最小。

但这轮 instrumentation 和原型验证说明：

- kernel 重排本身在总成本里占比很小
- 真正更重的是 source 重排和 compute
- 因此“只缓存 kernel”不能显著改变 decode 时延

当前结论：

- 这条路线可以保留为一个已验证过的备选思路
- 但现阶段不建议继续维护这版 patch
- 下一刀不应再停留在 kernel cache

补充更新：

- 其中 source 路径也已经做过一版并行化原型
- profiler 上能降低 `srcpack`，但端到端 decode 没有稳定变快
- compute 路径也做过一版低风险专门化，依然没有稳定收益
- 因此当前应把重点收缩到“更结构化的 compute 方案”，而不是继续堆小 patch

## 5.3 P1：给 1D 规则卷积做 fused kernel，直接绕开 `im2col`

这条路线也很值得做，但复杂度比上一个高。

目标不是一开始就支持所有 low-bit，而是分阶段：

1. 先做 `F16/F32` 权重版本
2. 再做 `Q8_0`
3. 最后再看 `Q4_K/Q5_K`

原因：

- 先证明“绕开 `im2col` 物化”本身能不能带来收益
- 如果连 `F16/Q8_0` 版本都不快，就没必要继续给 `Q4_K` 做更复杂实现

建议支持的第一批场景：

- 1D causal conv
- `stride=1`
- `dilation=1`
- `kernel=1/3/7`

这已经足够覆盖 AudioVAE 大部分普通卷积。

## 5.4 P1：专门做 pointwise conv (`kernel=1`) 的原生 CPU kernel

graph 侧做 pointwise fast path 已经试过，效果不好，原因大概率是：

- 仍然需要显式 `permute/cont/reshape`

但这并不说明 pointwise conv 没有优化空间，而是说明：

- 这种优化应该下沉到内核层，而不是只在图上拼 op

最理想的 pointwise kernel 应该能直接吃：

- `[T, C, B]` 或其现有 stride 布局

避免额外：

- `ggml_cont`
- `ggml_permute`
- `im2col`

## 5.5 P2：让 `mul_mat` 的 activation repack 更便宜

如果继续保留 “规则卷积 = `im2col + mul_mat`” 这条思路，那么另一个方向是降低 `mul_mat` 对激活侧的 repack 成本。

候选思路：

1. 更细粒度的 pack cache
2. 对小矩阵做不同的 chunking 策略
3. 对 `Q4_K/Q5_K x F32` 小矩阵场景加专用内核

但这条路线的风险也更大：

- `ggml_compute_forward_mul_mat` 是全局核心路径
- 一旦改坏，影响面会远大于 `conv_transpose_1d`

因此我不建议把它作为第一刀。

## 5.6 P3：低比特 `conv_transpose_1d`

这是最诱人的方向之一，但不建议现在就做。

原因：

1. 设计和测试成本高
2. 需要新的 low-bit dot kernel 组合
3. 可能牵涉到更多类型分发和 pack 格式

更实际的顺序应该是：

1. 先把现有 F16/F32 `conv_transpose_1d` 做快
2. 确认 decode 侧到底还有多少剩余瓶颈
3. 再决定值不值得继续做 low-bit transpose conv

---

## 6. 推荐原型顺序

建议按下面顺序推进：

### 阶段 1：只做测量

目标：

- 在 `ggml` CPU 路径里记录 `IM2COL`、`MUL_MAT` pack、`CONV_TRANSPOSE_1D` 的时间占比

产出：

- 一份带真实 hotspot 占比的数据表

### 阶段 2：先改 `conv_transpose_1d` 的 kernel 预打包

目标：

- 不改变外部 API
- 只减少每次 forward 的重复 kernel 重排

判断标准：

- 若 decode 明显下降，则说明这条路线有价值

进展更新：

- 该阶段已经做过一次原型验证
- 在当前机器和 workload 上没有得到稳定收益
- 因此这一阶段当前视为“已验证但暂不继续”

替代建议：

- 不再优先投入 kernel cache
- 不再优先投入 source pack 的轻量并行化变体
- 下一轮应直接研究更结构化的 `conv_transpose_1d` compute 方案
- 默认需要考虑更激进的数据布局或专用 micro-kernel
- 不再优先尝试只减少少量循环/指针开销的小 patch

### 阶段 3：做规则 conv 的 fused F16/Q8_0 原型

目标：

- 先验证“去掉 `im2col` 物化”本身的收益

不建议一上来就做：

- `Q4_K` fused conv

因为那样很难判断到底是“low-bit 算法问题”还是“fused 思路本身就不划算”。

### 阶段 4：再看要不要进 low-bit fused conv 或 low-bit transpose conv

只有当前三阶段已经证明：

- 真正的瓶颈位置明确
- F16/Q8_0 原型有正收益

才值得继续做更深的 low-bit kernel。

---

## 7. 对 subtree 维护方式的建议

既然 `ggml` 后续准备作为 subtree 维护，建议尽量按下面原则推进：

1. 第一批 patch 只做：
  - instrumentation
  - `conv_transpose_1d` phase breakdown
2. 尽量不改公共 API，优先改 CPU backend 内部实现
3. 每个 patch 都附带：
  - 独立 benchmark
  - AudioVAE workload 对照数据
4. 若必须新增实验接口，优先放在本 repo 上层 wrapper，而不是直接扩散到 `ggml.h`

这样后续无论是：

- 自己长期维护 patch queue
- 还是尝试向上游整理提交

成本都会低很多。

---

## 8. 当前判断

当前最值得继续的方向不是：

- 再换一种 AudioVAE 权重量化类型

而是：

1. 测清 `IM2COL / MUL_MAT pack / CONV_TRANSPOSE_1D` 三者的真实占比
2. 若继续深挖 `conv_transpose_1d`，应直接进入更结构化的 compute 方案，而不是继续停留在 kernel/source pack 或轻量专门化
3. 再用量化模型验证 encoder 侧是否真的主要慢在 `mul_mat pack`
4. 最后再评估是否需要 fused regular conv

换句话说，真正的下一阶段任务应该是：

- 从“模型策略优化”切换到“带 profiling 证据的算子实现优化”
