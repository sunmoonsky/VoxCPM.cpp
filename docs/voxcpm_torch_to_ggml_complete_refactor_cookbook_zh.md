# VoxCPM.cpp 完整重构 CookBook

这份文档不是要求你现在就去改动当前项目代码，而是给未来的重构者一份“按重构标准完全重写 VoxCPM.cpp”的落地手册。

它基于两类材料整理而成：

- 通用方法论：[docs/torch_to_ggml_migration_guide_zh.md](./torch_to_ggml_migration_guide_zh.md)
- torch 原版模型语义：`third_party/VoxCPM/src/voxcpm`

本文的目标不是“逐行翻译 PyTorch”，而是给出一套更成熟的目标形态：

1. 先定义 VoxCPM 的模型契约
2. 再定义 GGML runtime 骨架
3. 再按模块迁移核心前向
4. 最后才讨论优化和后端特化

---

## 0. 文档约定

为了让这份 CookBook 更像设计规范而不是随笔，先固定三组读法约定。

### 0.1 术语约定

| 术语 | 本文含义 |
| --- | --- |
| `WeightStore` | 单份 GGUF 权重和 metadata 的统一持有者 |
| `Backend` | 统一管理 buffer type、graph 分配、执行和张量拷贝的后端抽象 |
| `Persistent State` | 跨 step、跨 graph 持久存在的 backend-visible 状态，例如 KV、hidden、prefix patch |
| `Cross-State` | 跨阶段共享、但不属于单个 graph 生命周期的数据对象 |
| `Output Buffer / OutputPool` | 专门承接最终输出、跨阶段可读结果和调试输出的独立层 |
| `Graph Cache` | 以“真实重建条件”为 key 的图缓存，而不是仅按 shape 猜测 |
| `Host Fallback` | 在无法安全保持 backend-resident 时，显式退回 host 持久对象的过渡方案 |

### 0.2 伪代码接口命名约定

下文伪代码统一使用这套动词：

| 前缀 | 语义 |
| --- | --- |
| `initialize_*` | 初始化对象、分配资源、建立所有权 |
| `bind_*` | 绑定共享权重或共享句柄，不转移所有权 |
| `prepare_*` | 准备图输入、临时张量、位置张量或调度参数 |
| `capture_*` | 把单图结果写入持久 state |
| `publish_*` | 把单图结果写入 output pool |
| `export_*` | 把 backend-visible 结果导出成 host-visible API 输出 |

### 0.3 如何阅读这份文档

推荐按下面顺序读：

1. 先读第 2、3 节，把模型契约和 GGUF 契约固定住
2. 再读第 4、5 节，建立 runtime 分层和对象边界
3. 然后读第 6、7 节，看模块模板和主流程
4. 最后读第 9、10、11 节，用它指导真正的迁移、验证和排错

---

## 1. 先说结论

如果按“完全重构”的标准来写 VoxCPM.cpp，推荐的最终结构不是：

- `PyTorch 模块 = 一个 C++ 类`
- `一个模块 = 一次 tensor_set / tensor_get`
- `每个阶段都 materialize 成 std::vector<float>`

而应该是：

- 单份 `WeightStore` 统一持有 GGUF 权重
- 单份 `Backend` 统一管理 `Weights / State / Output / Compute`
- 单份 `DecodeState` 显式持有 `base_kv / residual_kv / lm_hidden / residual_hidden / prefix_patch`
- 图按阶段缓存，但状态不放进 compute arena
- 热路径模块之间默认传 `ggml_tensor *` 或 state handle，而不是 host vector
- 只有最终用户可见输出，例如音频 PCM，才拉回 host

一句话概括：

> VoxCPM 的正确 GGML 化方式，是把它重写成“共享权重池 + 持久状态对象 + 图缓存 + backend-aware 前向骨架”的 runtime，而不是把 torch 前向拆成若干 host-side 小函数。

---

## 2. VoxCPM 的真实推理契约

先把模型说清楚，再写 C++。

### 2.1 顶层模块

VoxCPM 的最小推理闭环包含六块：

1. `AudioVAE`
   - 波形 `<->` 连续 latent
2. `LocalEncoder`
   - 把一个音频 patch `[P, D]` 编成一个 patch-level token
3. `BaseLM`
   - 文本 token 和音频 patch token 的联合自回归语义主干
4. `FSQ`
   - 对 `BaseLM` 输出做轻量标量量化瓶颈
5. `ResidualLM`
   - 声学残差主干
6. `LocDiT + UnifiedCFM`
   - 用 LM hidden 生成下一个连续音频 patch

### 2.2 关键 shape 契约

以下 shape 是 Cookbook 的基础。不要跳过这一步。

| 模块 | torch 语义 shape | 推荐 GGML 契约 |
| --- | --- | --- |
| prompt latent patch | `[B, T, P, D]` | 运行时按 patch 为主，单步常用 `[D, P]` |
| `LocalEncoder` 输入 | `[B, T, P, D]` | 单 patch `[D, P]`，序列 `[D, P, T]` |
| `LocalEncoder` 输出 | `[B, T, H_enc]` | 单 patch `[H_enc]`，序列 `[H_enc, T]` |
| text embedding | `[B, T, H_lm]` | `[H_lm, T]` |
| `BaseLM` / `ResidualLM` 输入输出 | `[B, T, H_lm]` | `[H_lm, T]` 或单步 `[H_lm]` |
| `LocDiT.mu` | `[N, H_dit]` | `[H_dit, N]`，单步常退化为 `[H_dit]` |
| `LocDiT.x / cond` | `[N, D, P]` | `[D, P, N]`，单步常退化为 `[D, P]` |
| `UnifiedCFM` 输出 patch | `[N, D, P]` | `[D, P]` |
| `AudioVAE.decode` 输入 | `[B, D, T_latent]` | `[D, T_latent]` |

注意：

- `ggml` 的关键事实不是“统一都写成 `[C, T, B]`”，而是：`ne[0]` 是最连续维。
- 对 VoxCPM 来说，更稳的做法是模块内部各自固定布局契约，在模块边界明确转换。
- 不要把 `[B, T, P, D] -> [D, P, T, B]` 当成放之四海而皆准的模板。

### 2.3 顶层生成语义

VoxCPM 不是“LM 直接生成离散音频 token”，而是：

1. prompt 文本和 prompt 音频 patch 共同驱动 `BaseLM`
2. `BaseLM + FSQ + ResidualLM` 产生两个 patch-level hidden
3. `LocDiT + UnifiedCFM` 用这两个 hidden 生成“下一个连续 patch”
4. 新 patch 再喂回 `LocalEncoder`，转成新的 LM 输入
5. 如此循环直到 stop

也就是说，VoxCPM 的 decode step 本质上是：

```text
LM hidden -> DiT condition -> continuous patch -> LocalEncoder -> next LM input
```

这点会直接决定 state、graph cache 和 API 边界的设计。

---

## 3. 先定 GGUF 契约，再写 C++

如果没有导出契约，后面的 bug 会全部变成 shape、转置、broadcast 和命名混战。

### 3.1 必须固化的 metadata

GGUF 至少要包含：

- `voxcpm_patch_size`
- `voxcpm_feat_dim`
- `voxcpm_max_length`
- `voxcpm_residual_lm_num_layers`
- `voxcpm_dit_config_cfm_config_sigma_min`
- `voxcpm_dit_config_cfm_config_inference_cfg_rate`
- `voxcpm_audio_vae_config_*`
- MiniCPM / LocEnc / LocDiT / FSQ / AudioVAE 所需全部配置项

### 3.2 推荐的张量命名

建议采用以下命名：

- `token_embd.weight`
- `output_norm.weight`
- `blk.{i}.attn_q.weight`
- `blk.{i}.attn_k.weight`
- `blk.{i}.attn_v.weight`
- `blk.{i}.attn_output.weight`
- `blk.{i}.ffn_gate.weight`
- `blk.{i}.ffn_up.weight`
- `blk.{i}.ffn_down.weight`
- `residual_lm.*`
- `locenc.in_proj.weight`
- `locenc.in_proj.bias`
- `locenc.special_token`
- `locdit.in_proj.weight`
- `locdit.cond_proj.weight`
- `locdit.out_proj.weight`
- `locdit.time_mlp.linear_1.weight`
- `locdit.time_mlp.linear_2.weight`
- `locdit.delta_time_mlp.linear_1.weight`
- `locdit.delta_time_mlp.linear_2.weight`
- `proj.enc_to_lm.weight`
- `proj.lm_to_dit.weight`
- `proj.res_to_dit.weight`
- `stop.stop_proj.weight`
- `stop.stop_head.weight`
- `audio_vae.*`

### 3.3 导出脚本应该一次做掉的事情

导出阶段必须尽量“消化差异”，不要把兼容逻辑堆到 C++ 运行时：

1. 合并 `weight_norm`
2. 线性层必要转置
3. `Snake alpha` 之类广播敏感参数做 shape 修正
4. 拍平 metadata
5. 统一张量命名
6. 导出模块级 trace

实现时建议单独维护一份导出脚本，职责只做：

- 读 torch 权重
- 规整 metadata
- 完成必要转置/weight_norm 合并/broadcast 修正
- 输出 GGUF

### 3.4 模块级 Contract 清单

如果要让别人真的“从 0 开始重写”，只给高层 shape 还不够，必须把模块级 contract 写成表。

#### 3.4.1 BaseLM / ResidualLM

| 项目 | PyTorch | GGUF | GGML 运行时契约 | 备注 |
| --- | --- | --- | --- | --- |
| 输入 | `[B, T, H_lm]` | 无 | `[H_lm, T]` | 单步退化为 `[H_lm]` |
| `embed_tokens.weight` | `[vocab, H_lm]` | `[vocab, H_lm]` 存储后反解为 `token_embd.weight` | 运行时视作 `[H_lm, vocab]` 查表 | embedding 通常需要转成 GGML 友好的 lookup 布局 |
| `q/k/v/o` projection | `[H_out, H_in]` | 建议按线性层统一转成 `[H_in, H_out]` | 输入 `[H_in, T]`，输出 `[H_out, T]` | 这样 matmul/linear 契约统一 |
| `gate/up/down` | `[H_out, H_in]` | 建议转成 `[H_in, H_out]` | 同上 | 与线性层同规约 |
| `norm.weight` | `[H_lm]` | `[H_lm]` | `[H_lm]` | RMSNorm 只读广播 |
| KV cache | torch 内部 cache tuple | 无 | `K/V: [head_dim, max_len, n_kv_heads] * n_layer` | 必须独立于 compute arena |

必须额外写清楚：

- causal 与 non-causal 的 mask 语义
- RoPE positions 的 shape 与 dtype
- GQA/MQA 下 `n_heads != n_kv_heads` 的布局
- 哪些 view 只是 metadata 改写，哪些步骤必须 materialize contiguous

#### 3.4.2 LocalEncoder

| 项目 | PyTorch | GGUF | GGML 运行时契约 | 备注 |
| --- | --- | --- | --- | --- |
| 输入 patch | `[B, T, P, D]` | 无 | 单 patch `[D, P]`，序列 `[D, P, T]` | `P` 是 patch 内 token 数 |
| `in_proj.weight` | `[H_enc, D]` | 建议转成 `[D, H_enc]` | 输入 `[D, P]`，输出 `[H_enc, P]` | 线性层统一规约 |
| `in_proj.bias` | `[H_enc]` | `[H_enc]` | `[H_enc]` | 广播到 token 维 |
| `special_token` | `[1, 1, 1, H_enc]` | 建议压平成 `[H_enc]` | 运行时 repeat 成 `[H_enc, 1]` | 这是典型 alias/broadcast 契约 |
| 内部 transformer 输入 | `[B*T, P+1, H_enc]` | 无 | `[H_enc, P+1]` | 非因果 |
| 输出 | `[B, T, H_enc]` | 无 | 单 patch `[H_enc]`，序列 `[H_enc, T]` | 只取 CLS |

必须额外写清楚：

- CLS token 在 dim=1 拼接
- `forward_sequence()` 里的 patch view 可以 alias 输入序列，不必复制
- 输出 CLS slice 只是 view，不应默认 materialize 新 buffer

#### 3.4.3 FSQ

| 项目 | PyTorch | GGUF | GGML 运行时契约 | 备注 |
| --- | --- | --- | --- | --- |
| 输入/输出 | `[B, T, H_lm]` 或 `[B, H_lm]` | 无 | `[H_lm, T]` 或 `[H_lm]` | 纯 pointwise + linear |
| `in_proj.weight` | `[latent_dim, H_lm]` | 建议转成 `[H_lm, latent_dim]` | 输入 `[H_lm, *]`，输出 `[latent_dim, *]` | |
| `out_proj.weight` | `[H_lm, latent_dim]` | 建议转成 `[latent_dim, H_lm]` | 输入 `[latent_dim, *]`，输出 `[H_lm, *]` | |
| quant step | `round(scale * x) / scale` | 无 | 同表达式 | 推理不要保留 STE |

#### 3.4.4 LocDiT

| 项目 | PyTorch | GGUF | GGML 运行时契约 | 备注 |
| --- | --- | --- | --- | --- |
| `x` / `cond` 输入 | `[N, D, P]` | 无 | 单步 `[D, P]` | `N` 常退化为 1 或 CFG 双分支 |
| `mu` 输入 | `[N, H_dit]` | 无 | 单步 `[H_dit]` | 由 LM hidden 投影得到 |
| `in_proj.weight` | `[H_dit, D]` | 建议转成 `[D, H_dit]` | 输入 `[D, P]`，输出 `[H_dit, P]` | |
| `cond_proj.weight` | `[H_dit, D]` | 建议转成 `[D, H_dit]` | 同上 | |
| `out_proj.weight` | `[D, H_dit]` | 建议转成 `[H_dit, D]` | 输入 `[H_dit, P]`，输出 `[D, P]` | |
| `time_mlp` / `delta_time_mlp` | 两层 MLP | 线性层统一规约 | 输入 `[H_dit]`，输出 `[H_dit]` | time embedding 通常要求 contiguous |
| Transformer token 序列 | `[mu_token, cond_tokens, x_tokens]` | 无 | `[H_dit, 1 + P + P]` | 非因果；顺序不能变 |

必须额外写清楚：

- `t` 与 `dt` 的 dtype 约定
- `sin/cos` embedding 输出布局
- prefix 段和 noisy 段切片时只改 view 还是重新 materialize
- CFG 双分支时输入如何拼成 `[2, ...]`

#### 3.4.5 UnifiedCFM

| 项目 | PyTorch | GGUF | GGML 运行时契约 | 备注 |
| --- | --- | --- | --- | --- |
| `z` | `[B, D, P]` | 无 | 单步 `[D, P]` | 初始噪声 |
| `mu` | `[B, H_dit]` | 无 | `[H_dit]` | 条件向量 |
| `cond` | `[B, D, P]` | 无 | `[D, P]` | prefix patch |
| `t_span` | `[n_timesteps + 1]` | 可选 metadata/常量表 | `[n_timesteps + 1]` 或预表 | graph cache key 必须纳入 timesteps/sampler 条件 |

必须额外写清楚：

- Euler/solver 的步进公式
- `cfg_value`、`use_cfg_zero_star`、`sway_sampling_coef` 是否参与图重建
- 预计算 time table 是 host 常量、state 常量还是 backend 常量输入

#### 3.4.6 AudioVAE

| 项目 | PyTorch | GGUF | GGML 运行时契约 | 备注 |
| --- | --- | --- | --- | --- |
| `encode` 输入 | `[B, 1, T_wave]` | 无 | 可按 `[C, T]` 或 `[T]` 实现 | 先 pad 到 `patch_size * chunk_size` 的倍数 |
| `decode` 输入 | `[B, D, T_latent]` | 无 | `[D, T_latent]` | 通常最终才回 host |
| Conv1d weight | `[OC, IC, K]` | 需按 GGML conv 契约存放 | 运行时视作 conv kernel 布局 | 不要想当然套线性层转置规则 |
| Snake alpha | `[1, C, 1]` | 建议存成广播友好 shape | 运行时应匹配 `[T, C]` 或 `[C, T]` 广播契约 | 这是语音模型最容易踩的 broadcast 坑之一 |

另外要明确一条量化约定：

- 第一版迁移建议先做 F32/F16 正确性闭环。
- 量化属于第二阶段 contract，必须写清“哪些 tensor 可以量化、哪些 tensor 保持 F16/F32、量化块布局是什么”。
- 对 stop head、时间嵌入、小型投影和语音 codec 敏感路径，默认不要在第一版就量化。

---

## 4. 目标 runtime 架构

### 4.1 分层

推荐把“完全重构版 VoxCPM.cpp”拆成下面七层：

1. Contract
2. WeightStore / Loader
3. Backend / Memory
4. Output Buffer
5. Persistent State
6. Graph Cache / Scheduler
7. Runtime API

### 4.2 目录建议

建议从一开始就按职责拆文件，而不是把所有逻辑堆在一个 runtime 文件里：

```text
include/voxcpm/
  contract.h
  weight-store.h
  backend.h
  context.h
  output.h
  state.h
  graph-cache.h
  runtime.h
  minicpm.h
  localenc.h
  locdit.h
  unified_cfm.h
  fsq.h
  components.h
  audio-vae.h

src/
  weight-store.cpp
  backend.cpp
  context.cpp
  output.cpp
  state.cpp
  graph-cache.cpp
  runtime.cpp
  minicpm.cpp
  localenc.cpp
  locdit.cpp
  unified_cfm.cpp
  fsq.cpp
  components.cpp
  audio-vae.cpp
```

### 4.3 权重放置、Buffer Type 与多后端策略

CookBook 不能只写“有个 Backend 抽象”，还必须写清楚数据为什么放在那里。

#### 4.3.1 最小放置原则

不要把“模块在哪个设备上”当成一等设计对象。真正该决定的是：

1. 这个 tensor 会参与什么主算子
2. 这些主算子在哪些 backend 上成熟
3. 这个 tensor 放在哪种 buffer type 时，scheduler 是否更容易减少拷贝

同时要把一条 loader 规则写死：

- `one context per buffer type`
- 不要 `one context per module`

也就是说，权重 metadata、KV/state metadata、output metadata、graph metadata 应该按生命周期和 buffer 类型组织，而不是按 `BaseLM / LocDiT / AudioVAE` 这种模块边界各自建一套上下文。

这是为了避免两个问题：

1. 同一类持久 tensor 被模块边界切碎，导致重复分配和所有权混乱
2. 本来应该共享一个 backend buffer type 的对象，被错误地拆成多份孤立 context

推荐默认规则：

- embedding / linear / attention 主干权重
  - 跟随主 Transformer backend
- KV / persistent state
  - 默认跟随 decode 热路径 backend
- output buffer
  - 如果最终要频繁读 host，单独使用 host-visible buffer 或 staging buffer
- AudioVAE / conv-heavy 路径
  - 取决于目标 backend 对 conv、deconv、depthwise、custom op 的支持情况

#### 4.3.2 至少区分四类 buffer

```cpp
enum class BufferUsage {
    Weights,   // 只读持久
    KVCache,   // 持久可写
    State,     // 跨阶段持久状态
    Output,    // 对外暴露或跨图长期可读结果
    Compute,   // 单图生命周期
};
```

如果实现里没有单独的 `Output` buffer，那么“最后输出”和“跨图共享结果”通常都会被迫借 compute buffer，后面几乎一定要返工。

#### 4.3.3 多后端扩展时必须写清的策略

如果未来支持 CPU / CUDA / Metal / Vulkan，同一份 CookBook 至少要把下面几件事说死：

- 默认基线 backend 是哪个
- 哪些子图允许 offload
- 哪些算子缺失时必须 fallback 到 CPU
- fallback 后 tensor 是回 host，还是通过 scheduler/device copy 转移
- 是否使用 host staging / pinned host buffer
- scheduler 的切图条件和图重建条件是什么

建议写成显式策略对象：

```cpp
struct PlacementPolicy {
    BackendType default_backend = BackendType::CPU;
    bool allow_transformer_offload = true;
    bool allow_audiovae_offload = false;
    bool allow_scheduler = true;
    bool require_host_visible_output = true;
};
```

#### 4.3.4 什么时候进 scheduler

以下情况不要再坚持“先只用 gallocr”：

- decode 链路已经跨 CPU/GPU
- 有阶段间持久 tensor 需要跨 backend 传递
- 存在 AudioVAE / Transformer 混合路径，后端能力不对称
- 需要减少 host 中转和重复 copy

如果还是单后端、shape 稳定、没有跨设备需求，用单后端 allocator 也完全合理。但这应该是显式选择，而不是默认假设。

### 4.4 Output Buffer 作为独立架构层

不要把 output buffer 仅仅理解为“最后读一把 host 内存”。

在完整重构版里，`Output Buffer` 应该和 `State` 一样，是独立架构层，负责：

1. 承接最终用户可见输出
2. 承接跨图、跨阶段、但不适合继续留在 compute arena 的结果
3. 统一管理 host-visible / staging / backend-resident 的可见性策略

推荐最少区分三类输出对象：

- `final_output`
  - 例如 PCM、最终 latent 序列
- `cross_stage_output`
  - 例如阶段间长期保留的 embedding / patch / mask
- `inspectable_output`
  - 例如调试、trace、stop logits 等需要稳定读取的结果

如果这层没有独立出来，runtime 很容易退化成：

`compute tensor = API output = 持久状态`

这种混用在小模型上短期能跑，在多阶段模型上几乎一定会反噬。

### 4.5 语音项目的阶段化 reserve / schedule

VoxCPM 属于典型的语音生成项目。对这类模型，默认不建议一开始就把所有路径拼成一张超级大图。

更稳的默认拆法是：

1. `frontend / audio preprocessing`
2. `prompt encode / latent patch preparation`
3. `prefill transformer path`
4. `decode front half`
   - `lm_to_dit_proj + res_to_dit_proj + UnifiedCFM`
5. `decode back half`
   - `LocalEncoder + enc_to_lm_proj + BaseLM.step + ResidualLM.step`
6. `final AudioVAE.decode`

对每个阶段分别做：

- worst-case reserve
- graph cache key 设计
- backend placement 决策
- output/state 落盘策略

这样做的好处是：

- 每个阶段的 graph lifetime 更清楚
- trace 对拍更容易
- 多后端 fallback 更容易收口
- 不会为了“拼大图”而把跨阶段状态和 compute tensor 混在一起

对于语音项目里的传统 DSP / frontend，默认建议是：

- 先把数值稳定下来
- 再决定是否要并入 graph

不要为了“一张图看起来更完整”而过早把本来稳定的 frontend 算法强塞进主图。

---

## 5. 核心骨架代码

这一节给出“应该写成什么样”的核心代码模板。

说明：

- 以下代码块是目标骨架/伪实现。
- 重点是 ownership、lifetime、state 和模块边界，不是逐字可编译性。
- 真正落地时，请围绕这些边界自行组织 helper 和工程细节。

### 5.1 统一权重池

这一层是整份重构最不该妥协的部分。

后面的接口名都按前面的“文档约定”统一，不再混用 `load_* / from_* / materialize_*`。

```cpp
class VoxCPMWeightStore {
public:
    bool initialize_from_gguf(const std::string & path, VoxCPMBackend & backend) {
        path_ = path;

        gguf_init_params params = {};
        params.no_alloc = true;
        params.ctx = &ggml_ctx_;

        gguf_ctx_ = gguf_init_from_file(path.c_str(), params);
        if (!gguf_ctx_ || !ggml_ctx_) {
            return false;
        }

        buffer_ = backend.alloc_buffer(ggml_ctx_, BufferUsage::Weights);
        if (!buffer_) {
            return false;
        }

        if (!load_tensor_bytes_from_gguf(path, gguf_ctx_, ggml_ctx_, backend, buffer_)) {
            return false;
        }

        return true;
    }

    ggml_tensor * tensor(const char * name) const {
        return ggml_get_tensor(ggml_ctx_, name);
    }

    bool read_u32(const char * key, uint32_t & value) const;
    bool read_f32(const char * key, float & value) const;
    bool read_string(const char * key, std::string & value) const;

    ggml_context * ggml_ctx() const { return ggml_ctx_; }
    gguf_context * gguf() const { return gguf_ctx_; }
    ggml_backend_buffer_t buffer() const { return buffer_; }

private:
    gguf_context * gguf_ctx_ = nullptr;
    ggml_context * ggml_ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::string path_;
};

using SharedWeightStore = std::shared_ptr<VoxCPMWeightStore>;
```

约束只有两条：

- 整个模型只加载一次 GGUF
- 子模块只绑定 tensor 指针，不重新分配整份权重

### 5.2 Backend / Memory

Backend 的职责是把“权重 / 状态 / 计算图”区分开，而不是只包一层 `ggml_backend_t`。

```cpp
enum class BufferUsage {
    Weights,
    KVCache,
    State,
    Output,
    Compute,
};

class VoxCPMBackend {
public:
    explicit VoxCPMBackend(BackendType type, int n_threads);

    ggml_backend_buffer_t alloc_buffer(ggml_context * ctx, BufferUsage usage);
    void reserve_compute_memory(ggml_cgraph * graph, const char * stage);
    void alloc_graph(ggml_cgraph * graph, const char * stage);
    ggml_status compute(ggml_cgraph * graph);

    void tensor_set(ggml_tensor * tensor, const void * data, size_t off, size_t size);
    void tensor_get(const ggml_tensor * tensor, void * data, size_t off, size_t size);
    void tensor_copy(ggml_tensor * src, ggml_tensor * dst);

    bool uses_scheduler() const;
    bool is_gpu() const;
};
```

重点不是 API 多漂亮，而是要守住下面的边界：

- 权重不进 compute allocator
- KV 不进 compute arena
- decode state 不借用 graph 临时输出
- graph reserve 和 graph alloc 分开

### 5.3 持久 state，而不是散落的 host vector

完整重构版应该从第一天开始把 `base_kv / residual_kv / lm_hidden / residual_hidden / prefix_patch` 定义成正式 state 对象。

```cpp
struct VoxCPMPersistentState {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    ggml_tensor * lm_hidden = nullptr;        // [H_lm]
    ggml_tensor * residual_hidden = nullptr;  // [H_lm]
    ggml_tensor * prefix_patch = nullptr;     // [D, P]
};

struct VoxCPMDecodeState {
    MiniCPMKVCache base_kv;
    MiniCPMKVCache residual_kv;
    VoxCPMPersistentState persistent;

    int current_position = 0;
    int streaming_prefix_len = 3;

    VoxCPMCachedGraph base_step_graph;
    VoxCPMCachedGraph residual_step_graph;
};
```

初始化时要单独给 state 分配 buffer：

```cpp
static VoxCPMPersistentState create_persistent_state(
    VoxCPMBackend & backend,
    int hidden_size,
    int feat_dim,
    int patch_size) {

    VoxCPMPersistentState s;

    ggml_init_params params = {};
    params.mem_size = 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;

    s.ctx = ggml_init(params);
    s.lm_hidden       = ggml_new_tensor_1d(s.ctx, GGML_TYPE_F32, hidden_size);
    s.residual_hidden = ggml_new_tensor_1d(s.ctx, GGML_TYPE_F32, hidden_size);
    s.prefix_patch    = ggml_new_tensor_2d(s.ctx, GGML_TYPE_F32, feat_dim, patch_size);
    s.buffer = backend.alloc_buffer(s.ctx, BufferUsage::State);

    return s;
}
```

这个设计的意义是：

- `lm_hidden` 和 `prefix_patch` 不再跟着某张 graph 一起销毁
- decode step 可以直接读写 state buffer
- 以后要接 scheduler、CUDA、Metal 时，不需要改 API 边界

### 5.4 Graph Cache

图缓存要按“真实重建条件”来建 key，而不是只看 shape。

```cpp
struct VoxCPMGraphKey {
    int seq_len = 0;
    int n_timesteps = 0;
    float cfg_value = 0.0f;
    bool is_prefill = false;
    bool with_prefix = false;

    bool operator==(const VoxCPMGraphKey &) const = default;
};

struct VoxCPMCachedGraph {
    std::unique_ptr<VoxCPMContext> context;
    ggml_cgraph * graph = nullptr;

    ggml_tensor * input0 = nullptr;
    ggml_tensor * input1 = nullptr;
    ggml_tensor * input2 = nullptr;
    ggml_tensor * input3 = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * aux_output = nullptr;
};
```

推荐缓存这些阶段：

- `embedding(seq_len)`
- `locenc_sequence(seq_len)`
- `enc_to_lm_proj(seq_len)`
- `fsq(seq_len)`
- `prefill(seq_len, mask pattern 可选)`
- `decode_front_half(n_timesteps, cfg_value)`
- `base_step(position 或 batch-shape)`
- `residual_step(position 或 batch-shape)`
- `stop_predictor()`

图缓存的 key 不要只看 shape。至少还要纳入：

- 输入输出模式
- 是否依赖外部持久 state
- sampler 条件，例如 `cfg_value / n_timesteps / sway_sampling_coef`
- output mask 或 stop 策略
- backend/scheduler 模式

否则 cache 命中看似正确，实际可能复用到不兼容的图。

### 5.5 Output Buffer、Cross-State 与阶段性 Host Fallback

这是当前文档最该讲透的一块。

#### 5.5.1 Output buffer 不是“顺手读一下 compute tensor”

以下三类东西都不应该默认直接借 compute arena：

1. 最终用户可见输出
   - 例如 PCM、最终 latent 序列
2. 跨图长期可读结果
   - 例如下游阶段要消费的 encoder 输出、prefix patch 历史
3. 需要在 graph 结束后继续保留的中间结果
   - 例如 stop logits、跨阶段条件张量

推荐单独定义：

```cpp
struct DecodeOutputView {
    ggml_tensor * patch = nullptr;       // backend-visible
    ggml_tensor * stop_logits = nullptr; // backend-visible
};

struct HostDecodeOutput {
    std::vector<float> patch;
    std::array<float, 2> stop_logits {};
};

struct VoxCPMOutputPool {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    ggml_tensor * patch_output = nullptr;     // [D, P]
    ggml_tensor * stop_logits = nullptr;      // [2]
    ggml_tensor * latent_seq = nullptr;       // [D, T_total * P]

    void publish_decode_outputs(
        VoxCPMBackend & backend,
        ggml_tensor * patch_src,
        ggml_tensor * stop_src) {
        backend.tensor_copy(patch_src, patch_output);
        backend.tensor_copy(stop_src, stop_logits);
    }

    DecodeOutputView view_decode_outputs() const {
        return DecodeOutputView {
            .patch = patch_output,
            .stop_logits = stop_logits,
        };
    }
};

static VoxCPMOutputPool create_output_pool(
    VoxCPMBackend & backend,
    int feat_dim,
    int patch_size,
    int max_latent_patches) {
    VoxCPMOutputPool pool;

    ggml_init_params params = {};
    params.mem_size = 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;

    pool.ctx = ggml_init(params);
    pool.patch_output = ggml_new_tensor_2d(pool.ctx, GGML_TYPE_F32, feat_dim, patch_size);
    pool.stop_logits  = ggml_new_tensor_1d(pool.ctx, GGML_TYPE_F32, 2);
    pool.latent_seq   = ggml_new_tensor_2d(pool.ctx, GGML_TYPE_F32, feat_dim, max_latent_patches * patch_size);
    pool.buffer = backend.alloc_buffer(pool.ctx, BufferUsage::Output);

    return pool;
}

HostDecodeOutput export_decode_output_to_host(
    const VoxCPMOutputPool & output_pool,
    VoxCPMBackend & backend);
```

#### 5.5.2 Cross-state 应该是什么

如果模型是多阶段的，阶段间共享的数据应该被显式建模为 `cross-state`，而不是散落成若干 host vector。

对于 VoxCPM，这类状态至少可以包括：

- `prefix_patch`
- prompt latent history
- 未来如果有 encoder-decoder 扩展，可包含 cross embedding
- 与 cross-stage 结果配套的长度、mask、序列映射信息

推荐形式：

```cpp
struct VoxCPMCrossState {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    ggml_tensor * prefix_patch = nullptr;   // [D, P]
    ggml_tensor * latent_hist = nullptr;    // [D, T_total * P]
    ggml_tensor * seq_meta = nullptr;       // 例如长度、offset、mask 索引
};
```

#### 5.5.3 什么时候允许 host fallback

理想方案当然是热路径始终 backend-resident，但完整重构版也应该承认：有时阶段性 host fallback 是合理妥协。

只有同时满足下面条件，host fallback 才算可接受：

1. 该结果跨图持久存在
2. 当前 runtime 还没有安全的 backend-resident 跨图绑定协议
3. fallback 被封装成专门的 state/output 对象
4. 文档明确它是过渡方案，而不是默认接口

不推荐的做法：

```text
tensor_get -> std::vector<float> -> 下个模块 tensor_set
```

可以接受的过渡做法：

```text
tensor_get -> CrossState.host_staging_buffer -> 下一阶段显式 input
```

区别在于：

- 前者是散落的模块边界
- 后者是有生命周期、所有权和升级目标的正式状态对象

#### 5.5.4 CookBook 里应该给读者的默认建议

- 能用 backend-resident tensor/state 直传，就不要回 host
- 如果必须回 host，必须落到专门的 output/cross-state 对象
- 不要把一次合理 fallback 演化成默认 API 形态
- 一旦 fallback 进入热路径，就应该把“去掉这次 host 中转”列入下一阶段重构目标

---

## 6. 模块迁移模板

这一节只放“最值得写进 Cookbook 的核心代码”。

### 6.1 MiniCPM 只保留统一骨架

MiniCPM 是共享 backbone。Cookbook 不必把所有 attention 细节重新抄一遍，但一定要把接口定准。

```cpp
class MiniCPMModel {
public:
    bool bind_weights(
        const SharedWeightStore & weights,
        const std::string & prefix,
        VoxCPMBackend & backend);

    ggml_tensor * forward(
        VoxCPMContext & ctx,
        ggml_tensor * input,       // [H, T]
        ggml_tensor * positions,   // [T]
        MiniCPMKVCache & kv_cache,
        bool is_causal,
        bool write_kv_cache = true,
        ggml_tensor * attention_mask = nullptr);

    ggml_tensor * forward_step(
        VoxCPMContext & ctx,
        ggml_tensor * input,       // [H]
        int position,
        ggml_tensor * pos_tensor,  // [1]
        MiniCPMKVCache & kv_cache,
        bool is_causal,
        bool write_kv_cache = true);
};
```

约束：

1. `BaseLM`、`ResidualLM`、`LocEnc`、`LocDiT` 共享同一套 backbone 代码
2. 只有 prefix 不同、配置不同、权重不同
3. KV cache 必须是显式对象，不能偷放进 graph tensor

### 6.2 LocalEncoder

`LocalEncoder` 的职责很单纯：

- patch 内非因果编码
- 取 CLS token
- 变成 patch-level token

推荐直接写双接口：`forward_patch()` 和 `forward_sequence()`。

```cpp
class LocEncModel {
public:
    bool bind_weights(const SharedWeightStore & weights, VoxCPMBackend & backend);

    // input: [D, P]
    // output: [H_enc]
    ggml_tensor * forward_patch(VoxCPMContext & ctx, ggml_tensor * input) {
        ggml_tensor * x = linear(ctx, input, w_.in_proj_weight, w_.in_proj_bias); // [H_enc, P]

        ggml_tensor * cls = ggml_repeat(
            ctx.raw_context(),
            w_.special_token,  // [H_enc]
            ggml_new_tensor_2d(ctx.raw_context(), GGML_TYPE_F32, x->ne[0], 1));

        ggml_tensor * tokens = ggml_concat(ctx.raw_context(), cls, x, 1); // [H_enc, P+1]

        ggml_tensor * pos = build_positions(ctx, tokens->ne[1]);
        ggml_tensor * hidden = encoder_.forward(ctx, tokens, pos, *scratch_kv_cache_, false, false);

        return ggml_view_1d(ctx.raw_context(), hidden, hidden->ne[0], 0);
    }

    // input: [D, P, T]
    // output: [H_enc, T]
    ggml_tensor * forward_sequence(VoxCPMContext & ctx, ggml_tensor * input) {
        std::vector<ggml_tensor *> cols;
        cols.reserve(input->ne[2]);

        for (int t = 0; t < input->ne[2]; ++t) {
            ggml_tensor * patch = ggml_view_2d(
                ctx.raw_context(), input, input->ne[0], input->ne[1],
                input->nb[1], t * input->nb[2]);
            cols.push_back(forward_patch(ctx, patch));
        }

        return concat_patch_outputs(ctx, cols); // 项目 helper，不是 ggml 原生 API；可用 ggml_concat/view/permute 组合实现
    }
};
```

说明：

- 第一版可以先做“逐 patch loop”，保证正确性。
- 第二版再优化成真正的 batched patch encoder graph。
- `scratch_kv_cache_` 只服务模块内部，不暴露到 runtime API。

### 6.3 FSQ

FSQ 是最适合先迁移的模块之一，因为它简单、闭环清楚、trace 好对拍。

```cpp
class FSQ {
public:
    ggml_tensor * forward(VoxCPMContext & ctx, ggml_tensor * input) {
        ggml_tensor * h = linear(ctx, input, w_.in_proj_weight, w_.in_proj_bias);
        h = ggml_tanh(ctx.raw_context(), h);
        h = scalar_round(ctx, h, config_.scale);   // round(scale * h) / scale
        h = linear(ctx, h, w_.out_proj_weight, w_.out_proj_bias);
        return h;
    }
};
```

推理迁移不要保留 torch 训练里的 STE 技巧。

### 6.4 LocDiT

`LocDiT` 的关键不是算子多少，而是 prefix 语义必须对。

它的逻辑是：

1. `x` 投影成 noisy token 序列
2. `cond` 投影成 prefix token 序列
3. `mu + time_embed(t) + delta_time_embed(dt)` 组成首 token
4. 拼成 `[global_token, cond_tokens, noisy_tokens]`
5. 跑一个非因果 MiniCPM
6. 丢弃 prefix，只取 noisy token 段
7. `out_proj` 回到 `[D, P]`

```cpp
class LocDiTModel {
public:
    ggml_tensor * forward(
        VoxCPMContext & ctx,
        ggml_tensor * x,    // [D, P]
        ggml_tensor * mu,   // [H_dit]
        ggml_tensor * t,    // [1]
        ggml_tensor * cond, // [D, P]
        ggml_tensor * dt) { // [1]

        ggml_tensor * x_proj = linear(ctx, x, w_.in_proj_weight, w_.in_proj_bias);       // [H_dit, P]
        ggml_tensor * c_proj = linear(ctx, cond, w_.cond_proj_weight, w_.cond_proj_bias); // [H_dit, P]

        ggml_tensor * t_emb  = timestep_mlp(ctx, sinusoidal_embedding(ctx, t, config_.hidden_size, 1000.0f));
        ggml_tensor * dt_emb = timestep_mlp(ctx, sinusoidal_embedding(ctx, dt, config_.hidden_size, 1000.0f),
                                            true);
        ggml_tensor * global = ggml_add(ctx.raw_context(), mu, t_emb);
        global = ggml_add(ctx.raw_context(), global, dt_emb); // [H_dit]
        global = ggml_reshape_2d(ctx.raw_context(), global, global->ne[0], 1);

        ggml_tensor * tokens = concat_dim1(ctx, {global, c_proj, x_proj}); // [H_dit, 1 + P + P]
        ggml_tensor * pos = build_positions(ctx, tokens->ne[1]);

        ggml_tensor * hidden = decoder_.forward(ctx, tokens, pos, *scratch_kv_cache_, false, false);
        ggml_tensor * noisy_hidden = slice_dim1(ctx, hidden, 1 + c_proj->ne[1], x_proj->ne[1]); // [H_dit, P]

        return linear(ctx, noisy_hidden, w_.out_proj_weight, w_.out_proj_bias); // [D, P]
    }
};
```

Cookbook 必须强调：

- `LocDiT` 是“带前缀条件的非因果 Transformer”
- 它不是普通 UNet，也不是普通 decoder-only LM
- `global_token / cond_tokens / noisy_tokens` 的顺序不能写错

### 6.5 UnifiedCFM

`UnifiedCFM` 自己不拥有权重，它只是 solver，引用 `LocDiTModel`。

```cpp
class UnifiedCFM {
public:
    explicit UnifiedCFM(LocDiTModel & estimator, const CFMConfig & config)
        : estimator_(estimator), config_(config) {}

    ggml_tensor * forward(
        VoxCPMContext & ctx,
        ggml_tensor * z,           // [D, P]
        ggml_tensor * mu,          // [H_dit]
        int patch_size,
        ggml_tensor * cond,        // [D, P]
        int n_timesteps,
        float cfg_value,
        float temperature,
        float sway_sampling_coef,
        bool use_cfg_zero_star,
        ggml_tensor * cached_time_table = nullptr) {

        ggml_tensor * x = z;
        std::vector<float> t_span = cached_time_table
            ? export_time_table_to_host(cached_time_table)
            : compute_t_span(n_timesteps, sway_sampling_coef);

        for (int i = 0; i < n_timesteps; ++i) {
            const float t0 = t_span[i + 0];
            const float t1 = t_span[i + 1];
            const float dt = t0 - t1;

            ggml_tensor * v = compute_velocity_with_cfg(
                ctx, x, mu, cond, t0, dt, cfg_value, use_cfg_zero_star);

            ggml_tensor * step = ggml_scale(ctx.raw_context(), v, dt);
            x = ggml_sub(ctx.raw_context(), x, step);
        }

        return x;
    }
};
```

这里最关键的设计点有两个：

1. `UnifiedCFM` 是算法层，不应该偷偷持有权重
2. 常见 timesteps 的 time table 可以预计算成缓存输入

### 6.6 顶层 Runtime

完整重构版的顶层 runtime，建议写成这样：

```cpp
class VoxCPMRuntime {
public:
    bool initialize(const SharedWeightStore & weights, VoxCPMBackend & backend) {
        backend_ = &backend;
        weights_ = weights;

        if (!base_lm_.bind_weights(weights, "", backend)) return false;
        if (!residual_lm_.bind_weights(weights, "residual_lm", backend)) return false;
        if (!locenc_.bind_weights(weights, backend)) return false;
        if (!locdit_.bind_weights(weights, backend)) return false;
        if (!fsq_.bind_weights(weights)) return false;

        components_ = VoxCPMComponents::create(
            weights, base_lm_.config().hidden_size, base_lm_.config().vocab_size, scale_emb());
        if (!components_) return false;

        feat_decoder_ = std::make_unique<UnifiedCFM>(locdit_, read_cfm_config(*weights));
        return true;
    }

    VoxCPMDecodeState create_decode_state() const;
    VoxCPMOutputPool create_output_pool() const;
    void prefill(VoxCPMDecodeState & state, const PrefillInput & input);
    DecodeOutputView decode_step(
        VoxCPMDecodeState & state,
        VoxCPMOutputPool & output_pool,
        ggml_tensor * noise_patch);
};
```

---

## 7. Prefill 和 Decode 的标准写法

这一节是整份 Cookbook 的核心。

### 7.1 Prefill

prefill 的职责不是“先随便跑一遍”，而是把 decode 所需的所有持久状态一次建好。

```cpp
void VoxCPMRuntime::prefill(VoxCPMDecodeState & state, const PrefillInput & in) {
    // 1. prompt latent -> patch 序列 [D, P, T]
    ggml_tensor * feat_seq = prepare_prompt_patch_sequence(in);

    // 2. LocalEncoder + enc_to_lm_proj
    ggml_tensor * feat_hidden = locenc_.forward_sequence(prefill_ctx, feat_seq);             // [H_enc, T]
    ggml_tensor * feat_embed  = components_->enc_to_lm_proj()->forward(prefill_ctx, feat_hidden); // [H_lm, T]

    // 3. 文本 embedding
    ggml_tensor * text_embed = components_->embed_tokens()->forward(prefill_ctx, in.text_ids); // [H_lm, T]

    // 4. 按 mask 混合成 combined_embed
    ggml_tensor * combined = blend_text_and_audio(prefill_ctx, text_embed, feat_embed, in.text_mask, in.audio_mask);

    // 5. BaseLM full forward
    ggml_tensor * positions = build_positions(prefill_ctx, in.seq_len);
    ggml_tensor * base_hidden = base_lm_.forward(prefill_ctx, combined, positions, state.base_kv, true, true);

    // 6. 只对音频位做 FSQ
    ggml_tensor * fsq_hidden = fsq_.forward(prefill_ctx, base_hidden);
    base_hidden = blend_audio_only(prefill_ctx, base_hidden, fsq_hidden, in.text_mask, in.audio_mask);

    // 7. ResidualLM full forward
    ggml_tensor * residual_in = ggml_add(prefill_ctx.raw_context(), base_hidden, masked_audio_embed(prefill_ctx, feat_embed, in.audio_mask));
    ggml_tensor * residual_hidden = residual_lm_.forward(prefill_ctx, residual_in, positions, state.residual_kv, true, true);

    // 8. 把最后一步 hidden 和最后一个 prefix patch 写入持久 state
    capture_last_column_to_state(*backend_, base_hidden, state.persistent.lm_hidden);
    capture_last_column_to_state(*backend_, residual_hidden, state.persistent.residual_hidden);
    capture_last_patch_to_state(*backend_, feat_seq, state.persistent.prefix_patch);

    state.current_position = in.seq_len;
}
```

prefill 阶段必须完成三件事：

1. 填满 `base_kv`
2. 填满 `residual_kv`
3. 落盘 `lm_hidden / residual_hidden / prefix_patch`

### 7.2 Decode Step

单步 decode 的标准语义如下：

```cpp
DecodeOutputView VoxCPMRuntime::decode_step(
    VoxCPMDecodeState & state,
    VoxCPMOutputPool & output_pool,
    ggml_tensor * noise_patch) {
    // 1. 读 state 里的 lm_hidden / residual_hidden
    ggml_tensor * lm_hidden  = state.persistent.lm_hidden;
    ggml_tensor * res_hidden = state.persistent.residual_hidden;
    ggml_tensor * prefix     = state.persistent.prefix_patch;

    // 2. 投到 DiT hidden
    ggml_tensor * dit_1 = components_->lm_to_dit_proj()->forward(step_ctx, lm_hidden);
    ggml_tensor * dit_2 = components_->res_to_dit_proj()->forward(step_ctx, res_hidden);
    ggml_tensor * mu = ggml_add(step_ctx.raw_context(), dit_1, dit_2);

    // 3. 用 UnifiedCFM 生成下一个 patch
    ggml_tensor * next_patch = feat_decoder_->forward(
        step_ctx, noise_patch, mu, config_.patch_size, prefix,
        config_.cfm_steps, config_.cfg_value, 1.0f,
        config_.sway_sampling_coef, config_.use_cfg_zero_star,
        get_cached_time_table(config_.cfm_steps));

    // 4. stop predictor 先判停
    ggml_tensor * stop_logits = components_->stop()->forward(step_ctx, lm_hidden);

    // 5. 把新 patch 再编码回 LM embedding
    ggml_tensor * patch_hidden = locenc_.forward_patch(step_ctx, next_patch);
    ggml_tensor * curr_embed = components_->enc_to_lm_proj()->forward(step_ctx, patch_hidden);

    // 6. BaseLM forward_step + FSQ
    ggml_tensor * pos = prepare_position_scalar(step_ctx, state.current_position);
    ggml_tensor * next_lm = base_lm_.forward_step(step_ctx, curr_embed, state.current_position, pos, state.base_kv, true, true);
    next_lm = fsq_1d(step_ctx, next_lm);

    // 7. ResidualLM forward_step
    ggml_tensor * residual_in = ggml_add(step_ctx.raw_context(), next_lm, curr_embed);
    ggml_tensor * next_res = residual_lm_.forward_step(step_ctx, residual_in, state.current_position, pos, state.residual_kv, true, true);

    // 8. 更新持久 state
    capture_tensor_to_state(*backend_, next_lm,  state.persistent.lm_hidden);
    capture_tensor_to_state(*backend_, next_res, state.persistent.residual_hidden);
    capture_tensor_to_state(*backend_, next_patch, state.persistent.prefix_patch);

    // 9. 发布到独立 output pool，而不是直接借 compute tensor 充当 API 输出
    output_pool.publish_decode_outputs(*backend_, next_patch, stop_logits);

    state.current_position += 1;

    return output_pool.view_decode_outputs();
}
```

这段逻辑里，真正重要的是设计选择，而不是每一行具体 API：

- `prefix_patch` 是显式持久 state
- `lm_hidden / residual_hidden` 是显式持久 state
- `BaseLM` 与 `ResidualLM` 各有独立 KV
- patch 生成和 LM 状态推进是同一条逻辑链
- `decode_step()` 返回的是 output pool 视图；如果 API 需要 host 数据，再显式调用 `export_decode_output_to_host()`

---

## 8. 不该带进推理端的训练逻辑

torch 原版里下面这些逻辑不要照搬进推理迁移：

1. `UnifiedCFM.compute_loss()` 的训练分支
   - `training_cfg_rate`
   - `noise_cond_prob_range`
   - `noise_cond_scale`
   - `ratio_r_neq_t_range`
   - `adaptive_loss_weighting`
   - `jvp`
   - `mean_mode`
2. FSQ 的 STE 写法
3. LoRA 注入与训练态权重装配
4. `torch.compile` / triton
5. 训练专用 badcase retry 策略
6. 训练态冻结参数逻辑

推理端只保留纯数值路径。

---

## 9. 推荐迁移顺序

如果真的按这份 Cookbook 来重写，建议顺序如下。

### 阶段一：先搭骨架

1. `Contract`
2. `WeightStore`
3. `Backend`
4. `OutputBuffer`
5. `Context`
6. `MiniCPMKVCache`
7. `DecodeState`

### 阶段二：先迁移简单模块

1. `Embedding`
2. `LinearProjection`
3. `StopPredictor`
4. `FSQ`

### 阶段三：迁移 backbone

1. `MiniCPMModel`
2. `LocEnc`
3. `LocDiT`

### 阶段四：迁移生成链

1. `UnifiedCFM`
2. `prefill`
3. `decode_step`
4. `AudioVAE`

### 阶段五：最后才做优化

1. batched `LocEnc`
2. 更彻底的 graph reuse
3. 减少 host 往返
4. 调 scheduler / offload
5. fused projection / backend-specific repack

---

## 10. 模块级验证方法

不要一开始就跑整模型。

推荐验证顺序：

1. `Embedding`
2. `LinearProjection`
3. `FSQ`
4. `StopPredictor`
5. `LocalEncoder.forward_patch`
6. `LocalEncoder.forward_sequence`
7. `MiniCPM.forward`
8. `MiniCPM.forward_step`
9. `LocDiT.forward`
10. `UnifiedCFM.forward`
11. `prefill`
12. `decode_step`
13. `AudioVAE.decode`
14. 全链路 TTS

每个模块都至少导出三类 trace：

1. 输入张量
2. 输出张量
3. 中间关键节点

例如：

- `LocalEncoder`
  - `in_proj` 输出
  - `cls_output`
- `LocDiT`
  - `x_proj`
  - `cond_proj`
  - `combined token`
  - `out_proj`
- `Decode step`
  - `mu`
  - `next_patch`
  - `curr_embed`
  - `next_lm_hidden`
  - `next_residual_hidden`
  - `stop_logits`

### 10.1 推荐排查顺序

当某个模块第一次对不上时，推荐按下面顺序查：

1. 先查导出后的真实 tensor shape
2. 再查 GGML 侧 layout/stride/contiguous 前提
3. 再查 broadcast 是否对齐
4. 再查 `reshape/view/permute/concat` 是否漏了一步
5. 再查是不是把训练逻辑误带进推理
6. 最后才判断是否真的缺算子

这个顺序很重要。很多所谓“GGML 没这个算子”，最后其实都是 layout 或 broadcast 写错。

### 10.2 语音模型的额外验证点

对 VoxCPM 这类语音生成模型，除了常规 trace，还要单独验证：

1. AudioVAE `encode/decode` 的长度对齐
2. `patch_size * chunk_size` 的补齐规则
3. Snake/Conv 的 broadcast 行为
4. `prefix_patch` 在多步 decode 中是否被正确更新
5. `base_kv` 与 `residual_kv` 是否独立推进
6. stop 判停是否稳定

---

## 11. 常见反模式与重构时的排查清单

### 11.1 不要把三种 context 混成一个东西

一定要分清：

- weight context
- graph context
- KV/state/output context

如果把它们混成一个对象来思考，常见后果是：

- graph metadata 被误算进权重常驻内存
- 持久状态被误当成一次性 graph tensor
- compute arena 覆盖掉跨步结果

### 11.2 不要只看模块名猜 shape

真正决定实现的不是“这个模块听起来像什么”，而是：

- 导出后的真实权重 shape
- GGUF 中的实际存储 shape
- 运行时算子要求的输入布局

重构时要强制自己先看导出结果，再看训练侧模块名。

### 11.3 不要先怀疑缺算子

当模块结果第一次不对时，默认先怀疑：

1. 布局
2. stride
3. broadcast
4. 少了 view/reshape/permute

只有这些都排除后，才去决定要不要补自定义 op。

### 11.4 不要把导出期修正留到运行时

以下修正尽量放到导出阶段一次做完：

- weight_norm 合并
- 线性层转置
- broadcast 友好 shape 修正
- metadata 拍平
- 命名映射

运行时越接近“绑定 tensor 指针 + build graph”，后续越稳。

### 11.5 不要为了像训练 batch 而硬拼不自然的图

如果执行模式天然是：

- 双分支
- CFG 双前向
- 多阶段 pipeline
- patch-by-patch decode

那就按这个执行模式设计图。

“更像训练代码”的 batch 图不一定更快，也不一定更省内存。

### 11.6 语音项目往往不止一个 state

VoxCPM 这种模型至少天然有：

- `base_kv`
- `residual_kv`
- `prefix_patch`
- latent history
- output buffer

如果后面再加 encoder-decoder 或 cross attention，还会继续增加 cross-state。

不要试图把它们全塞进一个“大缓存对象”里不分生命周期。

---

## 12. 这份 Cookbook 最后要守住的五条红线

1. 不允许每个模块各自重新加载 GGUF
2. 不允许把 KV 或 decode state 偷放进 compute allocator
3. 不允许把模块边界默认实现成 host/vector 边界
4. 不允许把训练逻辑残留到推理主路径
5. 不允许在布局契约不清楚时直接翻译算子

---

## 13. 最简版总流程

如果要把整份文档压缩成一段伪代码，应该是下面这样：

```text
GGUF -> WeightStore.initialize_from_gguf()
Runtime.initialize(weights, backend)
Runtime.create_decode_state(base_kv, residual_kv, persistent_state)
Runtime.create_output_pool(patch_output, stop_logits, latent_seq)

prompt wav -> AudioVAE.encode -> latent patches [D, P, T]
text ids + latent patches -> prefill:
    LocalEncoder -> enc_to_lm_proj
    token_embd
    BaseLM full forward
    FSQ(audio positions only)
    ResidualLM full forward
    capture {lm_hidden, residual_hidden, prefix_patch, base_kv, residual_kv}

for each decode step:
    mu = lm_to_dit_proj(lm_hidden) + res_to_dit_proj(residual_hidden)
    next_patch = UnifiedCFM(LocDiT, mu, prefix_patch, noise)
    stop = StopPredictor(lm_hidden)
    curr_embed = enc_to_lm_proj(LocalEncoder(next_patch))
    lm_hidden = FSQ(BaseLM.forward_step(curr_embed, base_kv))
    residual_hidden = ResidualLM.forward_step(lm_hidden + curr_embed, residual_kv)
    prefix_patch = next_patch
    publish {next_patch, stop} -> OutputPool
    if user API requires host-visible data:
        export OutputPool -> host output

all patches -> concat [D, T_total * P] -> AudioVAE.decode -> PCM
```

这就是完整重构版 VoxCPM.cpp 应该收敛到的主架构。
