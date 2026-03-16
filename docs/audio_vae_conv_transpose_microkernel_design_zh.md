# AudioVAE `conv_transpose_1d` 专用 micro-kernel 设计稿与回顾

> 状态说明（2026-03-16）：
> 这份文档记录了一轮已经完成并最终放弃的 `conv_transpose_1d` micro-kernel 实验。
> 当前仓库代码里已经**撤回**这条实验路径，不再保留相关执行分支。
> 当前仓库代码里也已经撤回了当时配套加入的 instrumentation。
> 保留下来的主要是 benchmark 结果与经验结论。

## 1. 背景

当前 `AudioVAE` decoder 的热点已经比较明确：

- `audio_vae.decode` 的 tracked hotspot 中，`conv_transpose_1d` 约占 `85% ~ 91%`
- 现有 `ggml` CPU 实现里，`kernel pack` 很小
- 更重的是：
  - `source pack`
  - `compute`

此前已经试过三条低风险路线，但都没有得到稳定端到端收益，因此均已回退：

1. kernel cache
2. `source pack / dst zero` 并行化
3. 小规模 compute 专门化

这意味着下一阶段不能继续堆“轻量 patch”，而要直接准备更结构化的 `conv_transpose_1d` compute 方案。

## 1.1 2026-03-16 首个原型进展

当时仓库里曾落下一版**实验性、默认关闭**的 F32 micro-kernel family 原型：

- 位置：
  - `third_party/ggml/src/ggml-cpu/ops.cpp`
- 当时的开关：
  - `VOXCPM_GGML_AUDIOVAE_TRANSPOSE_MICROKERNEL=1`
- 当前覆盖范围：
  - `src0 = F32`
  - `src1 = F32`
  - `dst = F32`
  - `kernel = 2 * stride`
  - `stride in {2, 3, 5, 6, 7, 8}`

这版原型的目的不是直接替换主路径，而是验证两件事：

1. 不做 source pack，是否能在 family 级 shape 上保持正确性
2. 按输出通道 tile 的直算路径，是否能同时服务新旧模型

实验结束后的状态：

- 代码已撤回，不再保留实验路径
- `test_audio_vae` 在实验期间曾验证通过
- profiler 下可观察到：
  - `kpack=0`
  - `srcpack=0`

也就是说，这版原型在实验期间完成了“把 pack 从执行路径里拿掉”的目标。

当前结果需要按“顺序独占 CPU 重跑后的 benchmark”来理解，早先并行跑的对照只应视作探索性数据，现已被下面这组结果取代。

顺序 benchmark 结果：

- 新模型 `models/voxcpm1.5.gguf`
  - `short / repeat=3`
    - baseline: `707.854 ms`
    - micro-kernel family: `412.280 ms`
    - 相对下降约 `41.8%`
  - `long / repeat=3`
    - baseline: `4978.141 ms`
    - micro-kernel family: `5414.926 ms`
    - 相对上升约 `8.8%`
- 老模型 `models/voxcpm-0.5b.gguf`
  - `short / repeat=3`
    - baseline: `291.496 ms`
    - micro-kernel family: `109.785 ms`
    - 相对下降约 `62.3%`
  - `long / repeat=3`
    - baseline: `1221.000 ms`
    - micro-kernel family: `1434.086 ms`
    - 相对上升约 `17.5%`

同时，在实验期间的 `VOXCPM_GGML_PROFILE=1` 命中路径下仍能确认：

- `kpack=0`
- `srcpack=0`

这说明：

- family 路线已经证明“short 场景下很有潜力”
- 但它在 `long` 场景下对新旧模型都出现了明确回退
- 因此当前不仅**不适合默认启用**，而且不值得继续留在主线代码里增加维护负担
- 下一步仍然需要继续做：
  - 更合适的 `oc_tile`
  - 更少的权重寻址开销
  - 可能的 SIMD / unroll
  - 长场景下更合理的 tile / cache 行为
  - 以及 old/new 双模型下继续分场景评估

但基于当前项目目标，这些“下一步”已不再建议继续投入实现，保留为技术备忘即可。

补充校正：

- `models/voxcpm1.5.dump` 对应的是老版 AudioVAE 结构：
  - `decoder_rates = [7, 7, 6, 3, 2]`
  - `decoder_dim = 2048`
  - `sample_rate = 44100`
- `models/voxcpm-0.5b.dump` 对应的是当前这套较新的 AudioVAE 结构：
  - `decoder_rates = [8, 8, 5, 2]`
  - `decoder_dim = 1536`
  - `sample_rate = 16000`

因此这条路线如果未来有人重新验证，不能只盯住单一模型，而要保证至少覆盖下面这组共同 family：

- `kernel=4, stride=2`
- `kernel=6, stride=3`
- `kernel=10, stride=5`
- `kernel=12, stride=6`
- `kernel=14, stride=7`
- `kernel=16, stride=8`

## 2. 当前实现复盘

当前位置：

- `third_party/ggml/src/ggml-cpu/ops.cpp`
- `ggml_compute_forward_conv_transpose_1d_f16_f32`
- `ggml_compute_forward_conv_transpose_1d_f32`

当前 F32 路径本质上做了三件事：

1. 将 kernel 从 `(K x Cout x Cin)` 重排为 `(Cout x K x Cin)` 风格的临时布局
2. 将 source 从 `(L x Cin)` 重排为 `(L x Cin-contiguous)` 的临时布局
3. 对每个输出通道、每个输入时间步、每个 tap 做一轮 `dot(Cin)`

对应的循环结构近似是：

```text
for oc in Cout:
  for t in L:
    for k in K:
      dst[oc][t * stride + k] += dot(src_pack[t][:], kernel_pack[oc][k][:])
```

这条路径的问题不是数值语义，而是数据准备方式：

- 它必须先把 source 整体转成 `Cin` 连续的布局
- 这个全量转置在当前 AudioVAE workload 上非常贵
- 而且后续 compute 仍然只是标量地逐输出通道推进

## 3. AudioVAE 当前最值得优先覆盖的 shape

从当前仓库里的两代模型配置和 decoder 路径可以直接得到：

- `models/voxcpm-0.5b.dump`
  - `decoder_rates = {8, 8, 5, 2}`
- `models/voxcpm1.5.dump`
  - `decoder_rates = {7, 7, 6, 3, 2}`
- `causal_transpose_conv1d()` 调用 `ggml_conv_transpose_1d(weight, x, stride, 0, 1)`
- decoder block 使用的转置卷积 kernel size 为 `2 * stride`

因此当前需要兼容的转置卷积主形状可概括为：

- 新模型 family:
  - stride: `8, 8, 5, 2`
  - kernel size: `16, 16, 10, 4`
- 老模型 family:
  - stride: `7, 7, 6, 3, 2`
  - kernel size: `14, 14, 12, 6, 4`
- kernel relation: `kernel = 2 * stride`

通道规模分别来自两代 decoder channel progression：

- 新模型：
  - `1536 -> 768 -> 384 -> 192 -> 96`
- 老模型：
  - `2048 -> 1024 -> 512 -> 256 -> 128 -> 64`

因此本阶段建议先把“专用 micro-kernel”范围压到：

- dtype:
  - `src0 = F32`
  - `src1 = F32`
  - `dst  = F32`
- op shape:
  - `kernel = 2 * stride`
  - `stride in {2, 3, 5, 6, 7, 8}`
- batch / extra dims:
  - 保持当前 AudioVAE decoder 的常见 3D/4D 调用形态
- 目标：
  - 先覆盖两代模型的共同热点 family
  - 不试图一开始就做通用 `ggml` 级最佳实现

`F16 kernel + F32 activation` 可以作为第二阶段沿用同一骨架扩展，不作为第一版入口。

## 4. 设计目标

本轮 micro-kernel 的目标不是“立即替代整个实现”，而是：

1. 去掉当前最重的 source 全量转置
2. 把 compute 组织成更适合 AudioVAE shape 的固定布局
3. 把后续实验约束在一个独立骨架里，避免反复改动主路径
4. 让未来继续做：
   - F32 首版
   - F16 kernel 扩展
   - SIMD / unroll / tile 调优

都能沿着同一套 plan 结构推进

## 5. 推荐的计算视角

与当前“每个输出点做一次 `dot(Cin)`”不同，下一版更值得尝试的视角是：

```text
for ic in Cin:
  for t in L:
    x = src[ic][t]
    for k in K:
      out_t = t * stride + k
      dst[:, out_t] += x * kernel[k, :, ic]
```

这样做的核心好处是：

- source 按当前原始布局访问即可，不需要整块转置
- 输入访问模式变成“按 channel 顺序扫完整条时间轴”，对当前 source 布局更友好
- 计算核心可以转化成“对一个输出通道 tile 做 AXPY/FMA 累加”

代价是：

- 权重访问模式需要专门设计
- 需要更谨慎地处理输出写回冲突与线程划分

## 6. 计划中的内部布局与分块

### 6.1 plan 级约束

专用 plan 先只描述当前 AudioVAE 热点，不追求通用性：

- `stride`
- `kernel_size`
- `input_channels`
- `output_channels`
- `input_length`
- `output_length`
- `output_channel_tile`
- `time_tile`

### 6.2 初始 tile 建议

第一版骨架建议先固定保守参数：

- `output_channel_tile = 8`
- `time_tile = 1`

原因：

- 先把“source 不转置”这件事验证清楚
- 避免一开始把时间维 blocking 和寄存器 blocking 同时耦合
- 便于 later patch 逐步演进到 `oc_tile=16`、`time_tile=2/4`

### 6.3 预期内核接口

推荐把内部 micro-kernel 约束成下面这种抽象：

```text
for each output-channel block:
  zero local accumulators
  for ic in Cin:
    x = src[ic][t]
    for k in K:
      accum[k][oc_lane] += x * w[k][oc_lane][ic]
  scatter accumulators into dst
```

这样后续可以自然扩展到：

- 标量骨架
- 向量化 `oc_tile`
- 针对 `kernel=4`
- 针对 `kernel=10/16`

## 7. 线程划分建议

此前试过在 `source pack / dst zero` 上做轻量并行化，但端到端无收益。因此下一版线程策略不应建立在“先 pack 再算”的假设上。

建议：

- 仍按输出通道块划分线程
- 每个线程处理一段 `output channel tile`
- 线程内部遍历完整 `Cin x L x K`

这样做的理由：

- 输出通道天然独立，避免对 `dst` 的写冲突
- 不再需要对 source 临时缓存做线程协作
- 更适合后续将 `oc_tile` 做 SIMD 化

不建议第一版就做：

- 按时间维切线程
- 多线程共享同一输出通道块并做归约
- 在骨架阶段引入额外 scratch merge

## 8. 分阶段实施建议

### 阶段 A：只搭骨架，不改变行为

在 `ops.cpp` 中补：

- AudioVAE `conv_transpose_1d` matcher
- plan 结构
- F32 skeleton stub
- 接入点，但默认关闭

目标：

- 编译通过
- 当前 benchmark 行为不变
- 后续实验可以直接在这个骨架上继续填 kernel

### 阶段 B：F32 标量首版

目标：

- 不做 source transpose
- 只覆盖 `kernel = 2 * stride`
- 先验证最后一层 `stride=2, kernel=4`

验收：

- `test_audio_vae` 通过
- profiler 中 `srcpack` 显著下降或消失
- `audio_vae.decode` 至少在一个场景出现稳定改善

实验收尾结论：

- 一版 family 级 F32 首版曾实现过
- 实验路径曾覆盖：
  - `kernel=4, stride=2`
  - `kernel=6, stride=3`
  - `kernel=10, stride=5`
  - `kernel=12, stride=6`
  - `kernel=14, stride=7`
  - `kernel=16, stride=8`
- 它已经把 `srcpack` 从这一路径中去掉
- 也在 `short` 场景上给出过明显正收益
- 但因为 `long` 场景在新旧模型上都回退，且增加维护复杂度，因此该实验现已放弃，代码已回退

### 阶段 C：扩到完整 AudioVAE decoder 热点

按优先级扩：

1. `kernel=4, stride=2`
2. `kernel=10, stride=5`
3. `kernel=16, stride=8`

若 F32 首版证明有效，再考虑：

- `F16 kernel + F32 activation`
- `oc_tile` 向量化
- 更细的 weight 预布局

## 9. 这次骨架刻意不做的事

为了保持 subtree 可维护性，这次设计稿和骨架都不会做下面这些事：

- 不修改 `ggml.h`
- 不增加新的公共 API
- 不改变默认执行路径
- 不引入低比特 `conv_transpose_1d`
- 不引入跨 op / 跨进程缓存
- 不把“实验用元数据”塞进 `ggml_tensor` 公共字段

## 10. 预期落地方式

本次“实现骨架”的完成标准是：

1. 文档明确
2. `ops.cpp` 里出现独立的 AudioVAE micro-kernel plan / matcher / stub
3. 默认路径仍走现有 `ggml_compute_forward_conv_transpose_1d_*`
4. 后续若继续做实验，只需要在 stub 内逐步替换内部计算

换句话说，这一版不是为了立刻提速，而是为了把后面的结构化优化纳入一个可维护、可回退、可持续演化的入口。
