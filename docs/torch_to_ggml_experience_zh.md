# Torch -> GGML 迁移指南（以 llama.cpp 架构为主线）

## 0. 文档范围

这份文档基于四个视角整理，但主线会以 `llama.cpp` 为准：

- `third_party/llama.cpp` 中已经被大规模验证过的 GGML 架构模式
- `third_party/whisper.cpp` 中更贴近语音模型落地的前处理、状态与调度设计
- 当前仓库 `VoxCPM.cpp` 的真实 GGML 实现
- 仓库内已有的 GGML 复盘文档与优化报告

其中：

- `llama.cpp` 负责回答“一个成熟 GGML runtime 应该怎样组织权重、buffer type、memory、scheduler 与输出边界”
- `whisper.cpp` 负责补充“语音项目的 frontend、state 拆分、阶段化 reserve/schedule”
- `VoxCPM.cpp` 主要作为本地案例：既提供可复用经验，也提供需要避免继承的反例，尤其是运行时内存占用和 Host/Device 往返方面

---

## 1. 先讲结论：Torch -> GGML 最稳的迁移路线

如果下一个项目要从 PyTorch 落到 GGML，最稳的顺序不是“先把所有算子都翻译出来”，而是：

1. 先定义 GGUF 命名、元数据和张量布局规则
2. 再按 `model loader / memory / output buffer / scheduler` 的边界搭 runtime 骨架
3. 再决定每类权重和状态该落到哪种 buffer type / device
4. 再逐模块做 PyTorch -> GGML 前向翻译
5. 每翻一个模块就用 trace / 对拍固定数值
6. 最后再做图复用、fuse、pipeline parallel、多后端这类性能工作

`llama.cpp` 和本仓库的经验共同说明：骨架错了，后面所有优化都会变成返工；骨架对了，算子翻译反而是最容易收敛的一部分。

### 1.1 如果目标是“最优迁移指南”，应优先继承 llama.cpp 的哪些原则

相比“从某个业务项目迁移出来的经验总结”，`llama.cpp` 更适合作为默认答案的地方在于：

- 它把 `model loader`、`memory`、`output materialization`、`scheduler` 做成了清晰分层，而不是把这些职责散落在各模块之间
- 它按 `buffer type` 和算子支持能力决定张量放置，而不是先按模块名拍脑袋分 CPU/GPU
- 它把输出缓冲单独管理，只在真正需要对外暴露结果时才 materialize 到 host
- 它默认把 reserve、重建、reuse 视为 runtime 设计问题，而不是“上线前再补”的小优化

因此，这份文档后面的内容应该这样理解：

- 凡是 `llama.cpp` 已经给出稳定答案的地方，应优先把它当默认方法论
- `whisper.cpp` 主要补充语音项目特有问题
- `VoxCPM.cpp` 更适合作为“本地已踩坑样本”和“验证过的局部实现”，而不是所有项目都应照搬的骨架

---

## 2. 当前 VoxCPM 的 GGML 架构和对象边界

### 2.1 两阶段模型是总前提

VoxCPM 明确采用 GGML 的“两阶段模型”：

- `Context` 只存元数据，不存实际张量数据
- `Backend Buffer` 负责真正分配设备内存
- `Graph Allocator / Scheduler` 负责中间计算内存

对应材料：

- `docs/GGML_CORE_CONCEPTS.md:7-35`
- `docs/GGML_CORE_CONCEPTS.md:195-203`
- `docs/GGML_BEST_PRACTICES.md:374-419`

VoxCPM 自己的封装也完全围绕这个原则：

- `VoxCPMContext` 默认 `no_alloc=true`，只保存 tensor metadata，图上下文额外带一个可复用的 `graph_buffer_`，见 `src/context.cpp:10-31`
- `calc_context_size()` 统一按“tensor overhead + graph overhead + margin”估算 metadata 内存，见 `include/voxcpm/common.h:74-88`
- `VoxCPMBackend` 负责 backend、buffer、gallocr/scheduler、tensor set/get/copy，见 `include/voxcpm/backend.h:45-144`

### 2.2 运行时对象边界

当前工程已经把职责分得比较清楚：

- `VoxCPMWeightStore`
  - 单次加载 GGUF metadata 与整份权重 buffer
  - 向各模块发放 tensor 指针和 metadata 读取接口
  - 见 `src/weight-store.cpp:52-107`
- `VoxCPMBackend`
  - 初始化 CPU/CUDA/Vulkan backend
  - 管理 `ggml_gallocr`，以及 GPU 场景下的可选 `ggml_backend_sched`
  - 见 `src/backend.cpp:252-307`, `src/backend.cpp:428-569`
- `VoxCPMContext`
  - 创建 graph context、tensor、graph
  - 见 `src/context.cpp:73-132`
- 模块层
  - `MiniCPMModel` 负责 Transformer 主干，见 `src/minicpm.cpp:525-846`
  - `LocEncModel` 负责局部编码，见 `src/localenc.cpp:56-153`
  - `LocDiTModel` 负责局部生成器估计器，见 `src/locdit.cpp:146-318`
  - `FSQ` 负责量化瓶颈，见 `src/fsq.cpp:152-193`
  - `AudioVAE` 负责语音编解码，见 `src/audio-vae.cpp`
  - `LinearProjection / StopTokenPredictor / Embedding` 负责辅助组件，见 `src/components.cpp:78-228`
- `VoxCPMRuntime`
  - 把上述模块组织成 `prefill/decode` 流程
  - 持有各种 shape-stable graph cache
  - 见 `src/voxcpm.cpp:140-205`, `src/voxcpm.cpp:214-565`, `src/voxcpm.cpp:959-1120`

### 2.3 当前最重要的架构收敛：共享权重池

这次仓库里最关键的架构经验，不是某个算子，而是”统一权重加载器 + 集中权重所有权 + 多模块共享 tensor 指针”。

原因和修正过程在文档里写得非常清楚：

- 重构前每个模块都可能对整份 GGUF 再做一次 `ggml_backend_alloc_ctx_tensors(...)`
- 结果不是”只分配自己的权重”，而是”再给整份模型分配一遍权重 buffer”
- 峰值 RSS 因此飙到接近 `14 GiB`

对应材料：

- `docs/voxcpm_shared_weight_store_refactor.md:26-70`
- `docs/voxcpm_shared_weight_store_refactor.md:100-143`
- `docs/voxcpm_shared_weight_store_refactor.md:147-159`

这个模式和 `llama.cpp` 是一致的。`llama.cpp` 的核心是”统一 model loader 负责权重所有权”，而不是”单份 metadata / 单份 buffer”。实际上，split GGUF 场景下会有多份 GGUF context，mmap 路径下也可能按文件片段创建多个 buffer。但所有权重都由统一 loader 管理，模块只持有 tensor 指针，见：

- `third_party/llama.cpp/src/llama-model-loader.cpp:568-634`（split GGUF 多 context 加载）
- `third_party/llama.cpp/src/llama-model.cpp:7527-7532`（buffer 按 `ctx_map × files` 组织）
- `third_party/llama.cpp/src/llama-model.cpp:7559-7580`（mmap 多 buffer 场景）

### 2.4 llama.cpp 比“共享权重池”更进一步的地方：按 buffer type 组织权重

如果目标不是“先能跑”，而是“尽量接近成熟 GGML runtime 的最优设计”，那只做共享权重池还不够。`llama.cpp` 还有两条更关键的原则：

- one context per buffer type，而不是 one context per module
- 权重放置由“张量角色 + 主算子 + 目标设备支持能力”共同决定

这两点在 `llama_model_loader::create_tensor()` 里非常明确：

- 为每个 `buffer type` 创建独立 `ggml_context`，见 `third_party/llama.cpp/src/llama-model-loader.cpp:1014-1041`
- 根据 tensor 属于 input/output/repeating layer，分别选择 `buft_list_input / buft_list_output / buft_list_layer`，见 `third_party/llama.cpp/src/llama-model-loader.cpp:1102-1117`
- 再结合主算子兼容性选最终 buffer type，见 `third_party/llama.cpp/src/llama-model-loader.cpp:1143-1147`

这个方法论非常重要，因为它直接影响：

- 权重总内存是否会重复分配
- scheduler 是否能更自然地把算子调度到“权重所在设备”
- Host/Device 之间是否会因为错误放置而反复搬运中间结果

---

## 3. 内存管理：哪些做法值得直接复用

### 3.1 分开三类内存：权重、KV、计算图

这是最核心的工程纪律。

VoxCPM 与 GGML 文档都强调三类内存必须逻辑分离：

- 权重 buffer：只读、持久
- KV cache buffer：可写、跨 step 持久
- compute buffer：临时中间结果，由 allocator/scheduler 统一复用

参考：

- `docs/GGML_CORE_CONCEPTS.md:237-257`
- `docs/logical-hatching-zebra.md:2620-2626`
- `src/minicpm.cpp:162-198`
- `src/backend.cpp:394-410`

下一个项目里，凡是“跨图/跨步仍然要活着”的张量，都不要交给同一个 gallocr 临时管理。

### 3.2 不要重复 alloc 整份 GGUF

这是 VoxCPM 已经踩穿过的坑。

直接经验：

- `gguf_init_from_file(..., no_alloc=true, ctx=&ggml_ctx_ptr)` 得到的是整份 GGUF tensor metadata context
- 只要对这个 context 调一次 `ggml_backend_alloc_ctx_tensors(...)`
- 就是对整份模型做一次完整权重分配

参考：

- `src/weight-store.cpp:52-89`
- `docs/voxcpm_shared_weight_store_refactor.md:47-60`

### 3.3 Graph Context 要复用，不要在热路径频繁 malloc/free

VoxCPM 的 `VoxCPMContext` 对 graph 类型使用成员级 `graph_buffer_`，这就是一个简单但有效的 graph metadata 池，见 `src/context.cpp:16-31`。

对应的复盘文档也明确指出：

- graph context 频繁创建销毁会带来碎片和性能问题
- 应优先做实例内复用，必要时再做 `thread_local`

参考：

- `docs/logical-hatching-zebra.md:2677-2771`

### 3.4 单 gallocr 连续执行多张图时，别假设前一张图的中间结果还安全

这是 GGML 新手最容易掉进去的坑之一。

`ggml_gallocr_alloc_graph_impl` 会先清空自己的生命周期状态，因此：

- 图 A 算完后留下的中间结果
- 如果只是 compute arena 里的普通 tensor
- 图 B 再 `alloc_graph` 时就可能被覆盖

参考：

- `docs/GGML_BEST_PRACTICES.md:619-643`
- `docs/logical-hatching-zebra.md:2634-2675`

正确做法只有两种：

- 要么把跨阶段依赖合并成一张大图
- 要么把需要跨图持久存在的结果拷到独立 buffer

### 3.5 KV cache 应按固定容量预分配

VoxCPM 的 `MiniCPMKVCache` 直接为每层创建 K/V tensor，并分配专属 buffer，见 `src/minicpm.cpp:162-191`。

`llama.cpp` 的做法也一致，只是复杂得多：

- 按 buffer type 分 context
- 按 layer/filter/offload 策略把 KV 放到不同设备
- 统一清零初始化
- 支持 `kv_unified / n_ctx_seq` 等不同模式
- 还有 recurrent / hybrid memory 等多种 memory 类型

参考：

- `third_party/llama.cpp/src/llama-kv-cache.cpp:41-70`
- `third_party/llama.cpp/src/llama-kv-cache.cpp:102-200`
- `third_party/llama.cpp/src/llama-context.cpp:1252`（`kv_unified` 判断）
- `third_party/llama.cpp/src/llama-memory-hybrid.cpp`（hybrid memory 实现）

可以直接抽象出经验：

- KV 是”状态内存”，不是”中间计算内存”
- 它应该按固定容量预分配（服从模型最大上下文），而不是按当前 batch 临时长大
- 不同模型架构可能有不同的 memory 语义，要根据实际情况选择

---

## 4. 性能优化：VoxCPM 与 llama.cpp 给出的共同答案

### 4.1 先做 graph reserve，再做反复执行

VoxCPM 现在几乎所有热路径都遵循：

1. 构图
2. `reserve_compute_memory()`
3. 每次执行前 `alloc_graph()`
4. `tensor_set()`
5. `compute()`
6. `tensor_get()`

典型位置：

- `src/voxcpm.cpp:268-565`
- `src/voxcpm.cpp:771-956`

`llama.cpp` 也是先 reserve worst-case graph，再反复执行，甚至会做多轮 reserve 以稳定 split/buffer 大小，见：

- `third_party/llama.cpp/src/llama-context.cpp:395-643`
- `third_party/llama.cpp/src/llama-context.cpp:1194-1226`

经验是：

- GGML 优化先做“形状稳定化”和“内存稳定化”
- 再去谈 kernel/fusion，收益更扎实

### 4.2 只缓存形状稳定的图

VoxCPM 的 runtime cache 做得很克制，采用”shape-bucket 多图缓存”策略，按形状分桶缓存多张图：

- patch graph
- sequence graph
- embedding graph
- 若干按 `seq_len` 或 `(timesteps, cfg)` 分桶的图
- decode state 自己持有的 step graph

参考：

- `src/voxcpm.cpp:214-243`
- `src/voxcpm.cpp:268-565`
- `docs/voxcpm_runtime_optimization_report_zh.md:61-99`
- `docs/voxcpm_runtime_optimization_report_zh.md:129-169`

**注意**：`llama.cpp` 的 graph reuse 策略不同，它采用的是”严格条件下复用上一张图”而非”shape-bucket 多图缓存”。复用条件不仅看 shape，还要检查 `sampler`、`output` 位图、`gtype`、`arch`、`loras`、`cross` 等拓扑相关条件，见：

- `third_party/llama.cpp/src/llama-graph.h:562-623`（`allow_reuse()` 检查十几个条件）
- `third_party/llama.cpp/src/llama-context.cpp:1187-1213`（只复用前一张图，失败则重建）

这背后的方法论很重要：

- “固定形状小图 cache”收益高、风险低
- “引用外部状态的图”不要做全局 cache，要下沉到 state owner

### 4.3 先消灭 host 侧碎片化，再谈更深的 kernel 优化

VoxCPM 当前一轮优化并没有先改 GGML 内核，而是先处理：

- 小图反复构建
- host `std::vector` 往返
- 逐 patch 计算改成 sequence 大图
- decode 热路径拆成更清晰、更少的阶段

参考：

- `docs/voxcpm_runtime_optimization_report_zh.md:7-16`
- `docs/voxcpm_runtime_optimization_report_zh.md:100-128`
- `docs/voxcpm_decode_refactor_summary_zh.md:50-84`

这个顺序非常值得复用，因为它几乎不改变数值路径，验证成本最低。

### 4.4 scheduler 的定位：VoxCPM 与 llama.cpp 的差异

VoxCPM 当前 backend 的策略是：

- 默认仍有单 backend + gallocr 路径
- GPU 场景在特定 stage 可选启用 `ggml_backend_sched`
- 只对少数 decode step graph 走 scheduler

参考：

- `src/backend.cpp:57-66`
- `src/backend.cpp:263-307`
- `src/backend.cpp:438-569`
- `docs/voxcpm_cuda_optimization_notes_zh.md:62`

而 `llama.cpp` 则更彻底地以 scheduler 为 compute graph 分配和执行的默认主路径：

- context 构造时默认创建 scheduler，见 `third_party/llama.cpp/src/llama-context.cpp:418`
- `sched_reserve()` 在初始化末尾无条件调用，见 `third_party/llama.cpp/src/llama-context.cpp:355`

但 scheduler 不是整个 runtime 的唯一骨架，以下模块仍在 scheduler 外：

- output buffer（`buf_output`），见 `third_party/llama.cpp/src/llama-context.cpp:261-270`
- memory 模块（KV cache 等），见 `third_party/llama.cpp/src/llama-context.cpp:281`
- 权重加载（由 model loader 管理）

方法论上可以总结为：

- 如果当前是单后端、图形状稳定、也不需要跨设备切图，先用 gallocr 往往最直接（VoxCPM 当前做法）
- 一旦开始需要异构放置、extra buffer type、跨设备 op 支持探测或 scheduler 切图，scheduler 就应该更早进入主路径（llama.cpp / whisper.cpp 做法）
- 无论选哪条路，都要明确 scheduler 只负责 compute graph 的分配与调度，不接管权重所有权和业务状态生命周期

### 4.5 把 Host/Device 边界当成一等设计对象，而不是模块接口细节

这是当前文档最需要向 `llama.cpp` 对齐的地方之一。

`llama.cpp` 的默认思路不是“每个模块都 `tensor_set()` 输入、`compute()`、`tensor_get()` 输出”，而是：

- 先为 graph 输出准备独立的 output buffer，见 `third_party/llama.cpp/src/llama-context.cpp:261-269`
- output buffer 需要扩容时单独重分配，而不是把 compute arena 当结果容器，见 `third_party/llama.cpp/src/llama-context.cpp:1914-1949`
- 如果设备支持 host buffer，会优先给输出和中间搬运选更合适的 host buffer type，见 `third_party/llama.cpp/src/llama-context.cpp:296-302`
- 权重加载阶段也会探测 async upload / host buffer / events 支持，以减少权重上传成本，见 `third_party/llama.cpp/src/llama-model-loader.cpp:1398-1468`

反过来看，如果 runtime API 天然以 `std::vector<float>` 为模块边界，就很容易把“模块边界”错误地实现成“Host/Device 边界”。

VoxCPM 当前 runtime 里就能看到这种模式：

- `encode_feature_sequence()`：`tensor_set -> compute -> tensor_get`，见 `src/voxcpm.cpp:634-645`
- `run_embedding()`：`tensor_set -> compute -> tensor_get`，见 `src/voxcpm.cpp:649-658`
- `run_projection_2d()`：再次把上一步结果从 host 塞回 device，见 `src/voxcpm.cpp:689-708`
- `run_decode_front_half()`：同一阶段多个输入都从 host 回灌，输出再拉回 host，见 `src/voxcpm.cpp:872-908`

因此，对“最优 Torch -> GGML 迁移指南”来说，更好的默认原则应该是：

- 模块内部尽量直接传 `ggml_tensor*` / device-resident state，而不是 `std::vector<float>`
- 只有最终用户可见输出、日志采样结果、或必须跨 runtime 生命周期保存的状态，才 materialize 到 host
- 如果两个阶段总是串联执行，应优先考虑一张大图、共享 graph owner，或者至少共享 device-resident output/input tensor
- Host/Device 传输统计应尽早接入，因为它和峰值内存一样，属于架构指标，不只是 profiling 附属品

---

## 5. PyTorch -> GGML 翻译方法论

### 5.1 先定布局，再写前向

最容易犯的错误不是算子写错，而是“默认沿用 PyTorch 维度心智”。

VoxCPM 现有文档已经把最关键的布局差异讲清楚了：

- 在当前 VoxCPM 的 Transformer/Embedding 主路径里，3D 激活经常从 PyTorch `[B, T, C]` 映射成 GGML `[C, T, B]`
- 对线性层这类 2D 视图，常可把激活看成 `[C, B]`
- 线性层权重在 GGML 里常存成 `[in_dim, out_dim]`
- `ggml_mul_mat(weight, input)` 在这类布局下通常得到 `[out_dim, batch]`

参考：

- `docs/logical-hatching-zebra.md:2949-2959`
- `docs/logical-hatching-zebra.md:3119-3149`
- `include/voxcpm/components.h:47-64`
- `include/voxcpm/components.h:228-249`

但这里一定要注意：这不是 GGML 的通用硬规则，而是“当前模块选定的布局契约”。

更准确地说，`ggml` 核心定义的是通用的 n 维张量表示，也就是 `shape + stride`，并要求算子按各自的 shape/stride 契约工作；它并不提供一个对所有模型、所有模块都成立的统一“业务语义维度顺序”。

像 AudioVAE 这类卷积/语音路径，就会为了卷积核布局、broadcast 和 `im2col` 选择不同的维度组织方式。因此更稳的方法论不是“把所有 3D 张量都变成同一种顺序”，而是：

- 每个子模块先固定自己的输入/输出布局契约
- 导出脚本、GGUF 存储、运行时 `view/permute/repeat` 全部服从这份契约
- 只在模块交界处做明确的布局转换

压缩版可以记成一句话：

- `ggml` 规定的是通用 shape/stride 机制，不是统一语义维度顺序；布局应按子模块和算子契约决定，而不是套固定的 `[B, T, C] -> [C, T, B]` 模板

建议每个模块翻译前先写一页“小抄”：

- PyTorch 输入形状
- GGML 输入形状
- PyTorch 权重形状
- GGUF 存储形状
- `mul_mat / get_rows / repeat / view / permute` 后的形状

没有这一步，后面 debug 基本都会浪费在 shape 上。

### 5.2 先做命名映射和导出脚本，前向实现反而会变简单

VoxCPM 的迁移经验说明：

- `scripts/convert_voxcpm_to_gguf.py` 不是“附属工具”，而是迁移骨架的一部分
- 它承担了权重命名、元数据拍平、dtype 约定、特殊张量形状修正

关键参考：

- AudioVAE Conv1d 不转置，见 `scripts/convert_voxcpm_to_gguf.py:47-53`
- Snake alpha 为了 GGML broadcast 需要改形状，见 `scripts/convert_voxcpm_to_gguf.py:56-87`
- weight_norm 先在导出阶段还原，见 `scripts/convert_voxcpm_to_gguf.py:90-147`
- GGUF metadata 全量拍平写入，见 `scripts/convert_voxcpm_to_gguf.py:610-660`
- PyTorch 名称到 GGUF 名称映射，见 `scripts/convert_voxcpm_to_gguf.py:294-419`

直接经验：

- 所有“导出时一次性修正”的问题，都不要留到运行时
- 运行时越像“纯 tensor pointer 绑定 + graph build”，代码越稳

### 5.3 算子翻译优先级

最推荐的迁移顺序：

1. `Embedding`
  - `ggml_get_rows`
2. `Linear`
  - `ggml_mul_mat + bias add`
3. `RMSNorm / LayerNorm`
4. 激活函数
  - `silu / gelu / tanh / sigmoid`
5. reshape/view/permute/concat
6. attention
7. 卷积和自定义算子

VoxCPM 的组件层就是这样落的：

- `Embedding::forward()` 用 `ggml_get_rows`，见 `src/components.cpp:217-227`
- 线性投影统一是 `ggml_mul_mat + add_bias`，见 `src/components.cpp:100-107`
- FSQ 先 `mul_mat` 再 `tanh` 再量化，见 `src/fsq.cpp:174-188`

### 5.4 对卷积/语音模块，不要先假设要“补 GGML 算子”

AudioVAE 的经验很值钱：

- 一部分 Conv1d 可以用 `im2col + mul_mat` 重写，见 `src/audio-vae.cpp:145-160`
- 深度卷积如果现成算子不合适，可以先落自定义 op，见 `src/audio-vae.cpp:85-142`
- 很多问题的关键不是“GGML 不支持”，而是“你还没把布局和广播关系说清楚”

这对语音模型尤其重要，因为它们比标准 Transformer 更依赖：

- broadcast 规则
- 卷积核布局
- 通道维和时间维的明确约定

### 5.5 先做 trace 对拍，再做性能优化

仓库里最成熟的经验不是理论，而是测试策略：

- AudioVAE 用 PyTorch trace 对拍，误差控制到 `max_diff < 0.0002`，见 `docs/logical-hatching-zebra.md:3058-3086`
- Components、MiniCPM、LocEnc、LocDiT、UnifiedCFM 都有独立测试

这个方法论可以直接照搬：

1. 先冻结 reference 条件：`eval()`、固定输入、对齐 dtype，优先准备一条 CPU/F32 baseline
2. 从 PyTorch 导出模块级输入输出 trace
3. 先让 GGML 模块单测过
4. 再往上拼大图
5. 最后再做 cache / fuse / offload

---

## 6. 已经踩过的坑和修正

### 6.1 误把 weight context 当 graph context

这是大坑。

参考：

- `docs/logical-hatching-zebra.md:2868-2884`

结论：

- Weight context 不该混入 graph overhead 心智
- Graph context / Weight context / KV context 必须分开估算

### 6.2 多图共享 gallocr 导致中间结果被覆盖

参考：

- `docs/logical-hatching-zebra.md:2634-2675`

结论：

- gallocr 负责“单图生命周期优化”
- 不是“跨图状态存储”

### 6.3 维度认知错误，导致文档和 GGUF 实际权重不符

LocEnc / LocDiT 的 `in_proj` 曾被误写成 `[feat_dim * patch_size, hidden_size]`，后来通过真实 GGUF 校正为 `[feat_dim, hidden_size]`。

参考：

- `docs/logical-hatching-zebra.md:2773-2835`

这类坑说明：

- 不要只看 PyTorch module 名字猜 shape
- 一定要回到实际导出后的 GGUF tensor shape 验证

### 6.4 把训练逻辑残留到 GGML 推理实现

FSQ 的经验就是：

- GGML 当前实现目标是纯推理
- 训练分支、STE、反向友好逻辑都应该剥掉

参考：

- `docs/logical-hatching-zebra.md:2886-2905`

### 6.5 为了“伪 batch”而引入不必要的中间图结构

UnifiedCFM 之前走 cond/uncond 拼 batch=2 的路线，后来被改成显式双前向。

参考：

- `docs/voxcpm_runtime_optimization_report_zh.md:39-60`

结论：

- 如果 GGML 图和你目标执行模式天然不一致，不要为了“看起来更像 PyTorch batch”硬拼
- 明确拆成两条路径，有时更快也更稳

---

## 7. llama.cpp 给下一个项目的可复用经验

### 7.1 它最值得借鉴的不是某个 kernel，而是分层

`llama.cpp` 的成熟点主要在这几层：

- model loader 层
- backend / buffer type 选择层
- memory / KV cache 层
- context / scheduler reserve 层
- graph build 层

参考：

- `third_party/llama.cpp/src/llama-model-loader.cpp`
- `third_party/llama.cpp/src/llama-model.cpp:7523-7653`
- `third_party/llama.cpp/src/llama-kv-cache.cpp:41-200`
- `third_party/llama.cpp/src/llama-context.cpp:218-260`
- `third_party/llama.cpp/src/llama-context.cpp:395-643`

这和 VoxCPM 后来的收敛方向是一致的：先把层次拆清，再谈性能。

### 7.2 它对 KV cache 的处理比“简单 3D tensor”更工程化

`llama.cpp` 不只是建几个 K/V tensor，它还会处理：

- per-buffer-type context
- offload
- unified / multi-stream
- layer filter / layer reuse
- state write/read

参考：

- `third_party/llama.cpp/src/llama-kv-cache.cpp:20-214`
- `third_party/llama.cpp/src/llama-memory-hybrid.cpp:11-120`

下一个项目如果只是单流推理，可以先学 VoxCPM 的简单版；一旦进入多序列、滑窗、混合记忆，就该参考 llama.cpp 的 memory 抽象方式。

### 7.3 reserve 是一等公民，不是”临近上线再补”

`llama.cpp` 在 context 构造期就做 sampler 初始化、图 reserve 和 buffer 尺寸稳定化，尽量前置稳定图结构。

但需要注意：运行时如果 `can_reuse()` 失败，仍会触发 `reset()` → `sched_reset()` → `build_graph()` → `alloc_graph()`，见 `third_party/llama.cpp/src/llama-context.cpp:1192-1209`。所以更准确的表述是”尽量前置稳定图结构，减少额外 re-reserve”，而非”构造期 reserve 后就完全避免重建”。

参考：

- `third_party/llama.cpp/src/llama-context.cpp:66-83`（sampler 前置初始化）
- `third_party/llama.cpp/src/llama-context.cpp:395-643`（sched_reserve 实现）
- `third_party/llama.cpp/src/llama-context.cpp:1187-1213`（can_reuse 失败时的重建逻辑）

这是一个很强的工程信号：

- 如果图结构会变，就把”稳定图结构”当作架构任务，而不是小优化
- 同时要对运行时重建有预案，确保重建路径正确且高效

---

## 8. whisper.cpp 给语音项目补上的经验

现在仓库里已经有 `third_party/whisper.cpp`，它提供了比 `llama.cpp` 更贴近语音模型的本地参照，尤其是在前处理、状态对象和分阶段调度上。

### 8.1 传统 DSP frontend 应优先稳定在 C++ 侧，而不是急着并进 GGML 图

`whisper.cpp` 的做法非常明确：

- Hann window、FFT、mel filter、clamp/normalize 都在 C++ 侧实现
- mel 结果作为后续 encoder 的稳定输入
- 前处理多线程化，但不强行并入 GGML 图

参考：

- `third_party/whisper.cpp/src/whisper.cpp:3023-3259`
- `third_party/whisper.cpp/include/whisper.h:273-307`

这对“前端主要由确定性 DSP 预处理组成”的语音 Torch -> GGML 项目非常重要。真正应该先稳定的是：

- PCM -> feature 的数值一致性
- 帧长、hop、padding、窗函数、mel filter 的约定
- feature 张量的最终布局

而不是一开始就追求“整条音频链都在 GGML 图里”。

但这条经验不要过度泛化：

- 如果前端本身是可学习模块，或者它明显受益于图内调度、缓存或后端 offload，就不必强行留在 C++ 侧
- VoxCPM 的 AudioVAE 就说明，学习型语音模块完全可以作为 GGML 图的一部分落地

### 8.2 语音状态对象通常不止一个 KV cache

`whisper_state` 里并不是只有“模型权重 + 一个 decode cache”，而是显式拆成：

- self-attention KV
- cross-attention KV
- flash attention padding buffer
- mel
- batch
- decoder 状态
- 分阶段 scheduler

参考：

- `third_party/whisper.cpp/src/whisper.cpp:834-935`
- `third_party/whisper.cpp/src/whisper.cpp:3374-3544`

这个经验可以直接迁移到下一类语音/多模态项目：

- 不要把所有 runtime state 都混成一个“大上下文”
- 先把“语音特征状态”“交叉注意力状态”“自回归状态”“辅助对齐状态”拆开
- 这样后面做复用、清理和异构放置时才不会互相污染

### 8.3 语音模型很适合按阶段分别 reserve/schedule

`whisper.cpp` 把计算拆成了四类图，并分别初始化 scheduler：

- conv
- encode
- cross
- decode

参考：

- `third_party/whisper.cpp/src/whisper.cpp:3472-3542`
- `third_party/whisper.cpp/src/whisper.cpp:1976-2036`
- `third_party/whisper.cpp/src/whisper.cpp:2038-2456`
- `third_party/whisper.cpp/src/whisper.cpp:2458-2992`

这和 VoxCPM 当前“小图缓存 + 分阶段执行”的方向高度一致，但 whisper.cpp 给出了一个更明确的语音版范式：

- 把前端卷积、编码器、交叉注意力、解码器拆成不同 reserve 单元
- 每段单独测算 compute buffer
- 保证运行时尽量零额外分配

如果下一个模型是 encoder-decoder 结构，这个模式比“从一开始就拼成一张超级大图”更稳。

### 8.4 权重放置要按算子支持能力，而不是按模块名拍脑袋

`whisper.cpp` 在选 weight buffer type 时，会根据目标设备是否支持 `MUL_MAT` / `GET_ROWS` 等操作做判断，而不是简单地说“能上 GPU 的都上 GPU”。

参考：

- `third_party/whisper.cpp/src/whisper.cpp:1361-1472`

这条经验很实用：

- 迁移时先识别模型真正的主算子
- 再看这些主算子在候选后端上是否成熟
- 不成熟的模块宁可先留在 CPU/default buffer，也不要为了“看起来更先进”把整块逻辑推到一个算子支持不完整的后端

### 8.5 转换脚本不只导权重，还要导语音资产

`whisper.cpp` 的 `convert-pt-to-ggml.py` 明确把这些东西一并导出：

- hparams
- mel filters
- tokenizer vocab
- model variables

参考：

- `third_party/whisper.cpp/models/convert-pt-to-ggml.py:17-32`
- `third_party/whisper.cpp/models/convert-pt-to-ggml.py:219-247`

这一点很值得和 VoxCPM 的 GGUF 脚本一起看：

- `whisper.cpp` 证明“非权重资产也必须成为模型契约的一部分”
- `VoxCPM` 则进一步证明“最好把这份契约做成自描述的 GGUF metadata + tensor 命名规则”

也就是说，下一个项目最推荐的不是照搬 whisper 的旧 `ggml-model.bin` 形式，而是吸收它“把 frontend 资产也一起固化导出”的思想，再用 GGUF 做更现代的承载。

### 8.6 结合 whisper.cpp 与 VoxCPM，语音迁移的更稳路线

对下一个语音 Torch -> GGML 项目，最值得直接复用的组合是：

- 用 `whisper.cpp` 学 frontend、state 拆分、分阶段 scheduler
- 用 `llama.cpp` 学 buffer type、memory reserve、backend 分层
- 用 `VoxCPM.cpp` 学 GGUF 契约、共享权重池、卷积/广播/trace 对拍

换句话说，真正可复用的方法论不是“复制某一个仓库的目录结构”，而是把三者的优势拼起来：

- 先把卷积/频谱/广播维度规则写清楚
- 再把导出脚本做成一等公民
- 再把 runtime state 拆成独立生命周期对象
- 再做模块级 trace 对拍

---

## 9. 给下一个 Torch -> GGML 项目的落地清单

### 9.1 设计阶段

- 定义每个模块的 PyTorch shape、GGML shape、GGUF weight shape
- 规定 GGUF 命名和 metadata key
- 明确哪些是持久权重、哪些是状态、哪些是临时计算

### 9.2 导出阶段

- 在 Python 侧完成 weight_norm 合并、必要转置、广播友好形状修正
- 把所有配置拍平成 GGUF metadata
- 导出模块级 trace

### 9.3 C++/GGML 实现阶段

- 先做统一 `ModelLoader / WeightStore`
- 让权重按 `buffer type` 建 context，而不是按模块建 context
- 再做 `Memory` 模块，把 KV / recurrent state / cross state 从 compute arena 中独立出来
- 再做 `Backend + Scheduler`
- 再定义 output materialization 策略：哪些结果留在 device，哪些结果进入独立 output buffer
- 再做模块级 `load_from_store()`
- 再做模块级 `forward()`
- 模块间默认传 tensor/view/state，避免把 `std::vector<float>` 作为热路径接口
- 每个模块先单测再集成

### 9.4 优化阶段

- 先做 graph reserve
- 再做 graph reuse 或 shape-stable graph cache
- 尽早做 host/device 传输统计
- 再消灭不必要的 `tensor_get -> std::vector -> tensor_set`
- 最后再考虑 fuse、自定义 op、scheduler、多后端

---

## 10. 最终方法论总结

一句话总结这次 `llama.cpp` 主线 + `whisper.cpp` 补充 + `VoxCPM` 本地经验给出的结论：

> Torch -> GGML 不是“把 PyTorch 算子逐个翻译成 C API”，而是“先重建内存与布局模型，再把前向计算映射到这个模型里”。

真正决定成败的五件事是：

- 布局先行
- 生命周期分离
- 导出脚本前置
- trace 对拍驱动
- 把 Host/Device 边界收紧到最少

而真正值得延后做的事是：

- 大规模 fusion
- 多后端调度
- 激进 offload
- 复杂 kernel 优化

先把这四件事做对，下一个 Torch -> GGML 项目会顺很多。
