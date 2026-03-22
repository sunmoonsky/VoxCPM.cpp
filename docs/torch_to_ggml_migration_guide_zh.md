# Torch -> GGML 通用迁移指南

这份文档的目标不是复盘某个具体项目，而是给出一套更通用、可复用、尽量接近成熟 GGML runtime 的迁移方法论。

默认参考主线：

- `third_party/llama.cpp`：通用 runtime 骨架、权重放置、memory、scheduler、graph reuse
- `third_party/whisper.cpp`：语音 frontend、state 拆分、阶段化 reserve/schedule

同时，这份文档会直接纳入本仓库里已经被验证过的坑点和反例，但会改写成更通用的表述，而不是按项目复盘的写法堆引用。

---

## 1. 一句话结论

Torch -> GGML 不是“把 PyTorch 算子逐个翻译成 C API”，而是：

1. 先定义模型契约
2. 再搭 runtime 骨架
3. 再翻译前向
4. 最后才做性能工作

真正决定成败的核心原则是：

- 布局先行
- 生命周期分离
- 权重放置前置设计
- trace 对拍驱动
- Host/Backend 边界尽量收紧

---

## 2. 推荐的目标架构

一个更稳的 GGML runtime，建议至少拆成下面五层。

### 2.1 Contract 层

负责定义模型文件和运行时的共同语言：

- GGUF tensor 命名
- metadata key
- 每个模块的输入/输出 shape
- 权重在 GGUF 中的存储 shape
- dtype 约定
- layout / stride / contiguous 前提
- view / alias 关系
- 共享权重关系
- 量化 tensor 的 type 与块布局约束

这一步要在写 C++ 前完成。没有契约，后面所有 bug 都会变成 shape、转置、broadcast 混战。

### 2.2 Model Loader 层

这是 `llama.cpp` 最值得继承的部分之一。

默认原则：

- 统一 loader 持有全部权重所有权
- 模块只拿 tensor 指针，不拥有整份权重 buffer
- one context per buffer type，而不是 one context per module
- 权重放置由“张量角色 + 主算子 + 设备支持能力”共同决定

不要默认把“某个模块放 GPU”当成正确抽象。真正该决定的是：

- 这个 tensor 会参与什么 op
- 这些 op 在候选 backend 上是否成熟
- 这个 tensor 放到哪个 buffer type 后，scheduler 是否更容易把相关 op 放在同一设备

### 2.3 Memory 层

至少要把下面几类内存彻底分开：

- 权重：只读、持久
- 状态：KV、recurrent state、cross state、缓存特征等
- compute：graph 中间结果
- output：对外暴露结果的缓冲

原则很简单：

- 权重不进入 compute allocator
- 状态不交给 compute arena 托管
- output 不要默认从 compute arena 临时借出来

### 2.4 Graph / Scheduler 层

推荐目标不是“每次现场 malloc 一张图”，而是：

- 先 reserve worst-case graph
- 对 shape-stable 场景复用 graph
- 图重建条件要明确，不要只靠 shape 猜
- 需要异构放置、切图、offload 时，尽早进入 scheduler 主路径

如果当前只是单后端、图稳定、没有明显跨设备需求，直接走单后端 `gallocr` 路径也可以。但要注意：`scheduler` 内部本身也是基于 `gallocr` 做分配；这里真正的区别，不是“要不要 allocator”，而是“是否需要 backend-aware 的切图、拷贝和多 buffer type 分配”。一旦项目进入多后端或复杂 offload，scheduler 不应等到最后才补。

### 2.5 Runtime API 层

热路径 API 应尽量以 backend-resident tensor/state 为边界，而不是 `std::vector<float>`。

这里更准确地说，`ggml_tensor` 的元数据对象始终在 host 侧；所谓 backend-resident，指的是它绑定的数据 buffer 落在目标 backend 或 host-side backend buffer 中，而不是每一层接口都 materialize 成普通 host 向量。

更好的默认设计：

- 在 owner、context、graph lifetime 都明确时，模块间可以传 `ggml_tensor *`、view、或 state handle
- 只有最终用户可见输出才 materialize 到 host
- 跨图、跨阶段默认优先传 state handle、持久 buffer，或显式 materialized output
- 如果两个阶段天然串联，应优先共享 graph owner 或 backend-resident tensor

另外，compute allocator 里的输出并不是“绝对不能读”；在同一轮图执行完成、且相关 buffer 还没被下一次 `alloc_graph/reserve` 覆盖前，通常可以读取。这里真正要避免的是把这类短生命周期结果误当成长期稳定的 API/output buffer 使用。并且“可读取”还受同步是否完成、buffer 是否 host-visible、以及是否需要 staging / `tensor_get` 影响。

一个非常常见的反模式是：

`tensor_get -> std::vector -> tensor_set`

如果这条链出现在热路径里，通常意味着：

- Host/Backend 边界往返偏多
- output buffer 设计不清晰
- 模块边界被错误实现成了数据回传边界

### 2.6 为什么这套架构能支持 CPU / CUDA / Metal

如果参考 `llama.cpp` 的做法，这套方法就不是“只为某一个设备写一套 runtime”，而是把 runtime 先设计成 backend-aware，再让不同设备作为后端接入。

但这不意味着“前向一旦抽象对了，所有后端都会自动可用”。更准确的前提是：目标 backend 已经实现了你依赖的关键算子、buffer type、拷贝路径和必要同步语义；否则仍然需要回退、改写图，或者补后端能力。

`llama.cpp` 为了支持 CPU、CUDA、Metal 这类不同后端，核心上做了下面几件事：

- 不把设备当成模块属性，而是把“tensor 应该落到哪种 buffer type、由哪个 backend 执行”当成一等设计对象
- 不按模块分配上下文，而是按 buffer type 建立上下文和后端缓冲，这样 CPU、CUDA、Metal 都可以走同一套权重/状态/图管理骨架
- 权重放置不是写死规则，而是结合 tensor 角色、主算子类型、目标 backend 的算子支持能力来选择；如果首选后端不支持，会回退到可执行的放置方案
- 用 scheduler 负责图切分、backend assignment 和必要拷贝，让图尽量按现有 tensor/backend 约束落到可执行的设备上，并尽量减少无意义的数据来回搬运
- 把 host buffer 和 device/backend buffer 明确区分开；当具体后端实现提供 `host buffer type` 时，还可以使用更适合传输的 host-side staging buffer 或 pinned buffer 机制来降低装载成本
- 把 CPU 保留为稳定的基线和兜底路径，同时把 GPU offload、层分配、tensor split、KV/offload 策略做成显式配置，而不是写死在模型实现里
- 单独管理 output 路径和可见结果，避免为了取一点最终输出而把整条中间链路都频繁拉回 host
- 提供按 buffer type / device 统计内存占用的能力，这样放置策略、offload 策略和峰值内存问题才有办法被持续校正

因此，真正支持多设备推理的关键，不是“额外写一套 CUDA 版、Metal 版前向”，而是先把 runtime 的抽象边界放对：

- 前向逻辑只描述张量关系和算子契约
- buffer 放置决定数据在哪
- scheduler 决定图怎么切、op 去哪算
- output 策略决定什么东西必须回到 host

如果这四层边界是清楚的，那么在大多数 Transformer / LLM / GGUF / 多后端 runtime 场景下，CPU-only、CUDA offload、Metal offload，甚至多设备混合放置，通常都可以沿着同一套主架构扩展。

需要再补一句现实约束：能否“沿着同一套主架构扩展”不等于“性能和功能自然等价”。后端算子覆盖率、view/contiguous 要求、异步拷贝能力和 scheduler 的切图效果，都会影响最终可行性和收益。

---

## 3. 推荐迁移顺序

### 3.1 先定契约，再写代码

每个模块先写清楚五件事：

- PyTorch 输入 shape
- GGML 输入 shape
- PyTorch 权重 shape
- GGUF 存储 shape
- 模块间是否需要布局转换

注意：更准确地说，`ggml` 核心定义的是以固定上限维度实现的 `shape + stride` 张量表示，并要求算子按各自的 shape/stride 契约工作；它并不提供一个对所有模型、所有模块都成立的统一“业务语义维度顺序”。

还要记住一个非常具体的实现细节：在 `ggml` 里，`ne[0]` 是最内层、也是默认最连续的那一维，`nb[]` 描述各维 stride。很多 `view/permute/reshape` 可以只改元数据，但并不是所有算子都接受任意 stride；不少算子会要求 contiguous 或特定布局。

因此，不要把 `[B, T, C] -> [C, T, B]` 当成通用硬规则。正确做法是每个子模块自己固定布局契约，交界处明确转换。

压缩版可以直接记成：

- `ggml` 管的是 shape/stride，`ne[0]` 是最内层连续维；它不管统一语义维度顺序，而且很多算子仍有 contiguous/布局前提，所以布局必须按子模块和算子契约定，不要套固定模板。

### 3.2 先写导出脚本，再写前向

Python 导出阶段应该尽量一次性做完这些事：

- weight_norm 合并
- 必要转置
- broadcast 友好形状修正
- metadata 拍平
- 权重命名映射
- 模块级 trace 导出

运行时越接近“绑定 tensor 指针 + build graph”，后续通常越稳。但不要把这句话理解成“导出脚本能吸收一切差异”：backend-specific repack、量化块布局适配，以及动态 mask / position / state 元信息，合理地留在加载期或执行期也完全正常。

### 3.3 先搭 runtime 骨架，再翻模块

顺序建议：

1. `ModelLoader / WeightStore`
2. `Memory`
3. `Backend + Scheduler`
4. `Output Buffer`
5. 模块级 `load_from_store()`
6. 模块级 `forward()`

不要反过来先把模块都写完，再回头补 loader 和 memory。那样通常会返工。

### 3.4 算子翻译优先级

通用原则不是按某个固定算子列表排优先级，而是优先翻译：

1. 最能跑通模型主干闭环的算子族
2. 最能暴露布局契约和 broadcast 问题的算子族
3. 最能验证 runtime 骨架是否成立的算子族

如果目标是 Transformer 类模型，常见默认顺序可以是：

1. `Embedding`
2. `Linear`
3. `Norm`
4. 常用激活函数
5. `reshape / view / permute / concat`
6. attention
7. 卷积和自定义 op

如果目标是 conv-first、audio frontend-heavy、VAE/CNN 主干模型，卷积、`reshape/view/permute`、broadcast 和局部状态处理往往应该更早进入主路径。

---

## 4. 验证方法

迁移时最可靠的推进方式不是“先跑整模型”，而是模块级 trace 对拍。

推荐流程：

1. 固定 reference 条件：`eval()`、固定输入、对齐 dtype
2. 优先准备一条 CPU/F32 baseline
3. 从 PyTorch 导出模块级输入输出 trace
4. 先让单模块 GGML 对拍通过
5. 再向上拼更大图
6. 先验证无 cache 的全量路径，再单独验证 cache / incremental 路径，最后再做 offload、fusion

这一步必须回答两个问题：

- 数值路径是否一致
- 偏差来自布局/实现错误，还是来自 dtype/后端差异

如果 reference 本身不稳定，后面的 debug 会非常低效。

---

## 5. 性能优化顺序

推荐顺序如下。

### 5.1 先做内存稳定化

- reserve worst-case graph
- 固定 KV/state 容量
- 独立 output buffer
- 尽量减少 graph 重建

### 5.2 再收紧 Host/Backend 边界

- 统计 `tensor_set` / `tensor_get`
- 找出 `tensor_get -> std::vector -> tensor_set`
- 合并总是串联执行的阶段
- 优先让中间结果留在 backend-resident tensor/state 中

### 5.3 再做 graph reuse

graph reuse 不应该只看 shape。至少还要看：

- 输入输出模式
- 是否依赖外部状态
- sampler / output mask / 拓扑条件
- backend/scheduler 切图条件

### 5.4 最后才做更深的优化

- 自定义 op
- fused kernel
- aggressive offload
- pipeline parallel
- 更复杂的多后端调度

先把内存和边界理顺，收益通常比先改 kernel 更稳。

---

## 6. 常见反模式

下面这些做法非常常见，也非常容易拖慢项目。

### 6.1 按模块分配整份权重

错误思路：

- 每个模块自己 `gguf_init_from_file`
- 每个模块自己 `alloc_ctx_tensors`

正确思路：

- 统一 loader 持有权重
- 模块只拿 tensor 指针

这是一个非常真实的坑。只要你把“同一份包含整模型 tensor 定义的 `ggml_context`”交给多个模块分别做 `alloc_ctx_tensors`，结果就不会是“每个模块只拿到自己的小片权重”，而是“整份模型被重复分配多次”。这里要明确区分三件事：`gguf_context` 保存文件元数据，`ggml_context` 持有 tensor 定义，backend buffer 才承载实际数据。这种错误在模型刚接通时不容易第一时间暴露，因为功能上可能仍然正确，但 RSS 会异常偏高，后面再去做量化、offload、scheduler 都会被这个错误基线污染。

### 6.2 把状态交给 compute allocator

错误思路：

- 让 KV 或跨步结果留在 compute arena

正确思路：

- KV/state 独立分配
- compute allocator 只管单图生命周期

这个坑最容易出现在“先把功能跑通”的阶段。很多人会先把跨步结果也当普通 graph tensor 用，短期看代码更省事，但一旦切到多图执行或单个 gallocr 连续服务多张图，中间结果就会被下一张图覆盖。最容易被误判的是：功能在某些小 case 下能跑，但一进入真实 decode 或多阶段流水线就开始出现不稳定、脏数据或数值漂移。

### 6.3 把模块边界做成 Host/Backend 边界

错误思路：

- 每个模块输入输出都是 `std::vector<float>`

正确思路：

- 热路径传 tensor/state
- 只在必要时拉回 host

这是很多业务项目从“先通”走向“可用”时的主要性能障碍。表面上看，模块接口全是 `std::vector<float>` 很干净；实际上这经常意味着每个阶段都在重复做：

- 把输入从 host 送进当前 backend
- 在 backend 上算一小张图
- 再把结果拉回 host
- 下个阶段再送回 backend

这种设计会带来两类问题：

- Host/Backend 往返频繁，硬件执行效率很难稳定
- 中间结果在 host 和 backend 侧各保留一份，峰值内存也会更高

如果一个模型天然是多阶段串行的，这类反模式尤其明显，因为每个阶段都看起来“局部合理”，但整条链路的往返成本会非常大。

### 6.3.1 一个真实的例外：为什么成熟 runtime 也会暂时退回 host

这里要特别补一个现实世界里很常见、而且容易让人困惑的例外。

以 `llama.cpp` 当前的 encoder-decoder / cross-attention 路径为例，它整体 runtime 设计是非常接近本指南推荐方向的；但在某些 enc-dec 模型上，encoder 输出给 decoder 的那条 cross embedding 路径，仍然会先同步、拷到 host 持久缓冲，再作为 decoder 的输入重新喂回图里。

这件事表面上看，正好违反了前面说的：

- 热路径尽量保持 backend-resident
- 不要把模块边界实现成 Host/Backend 边界
- 尽量避免 `tensor_get -> std::vector -> tensor_set`

但它之所以会出现在成熟项目里，通常不是因为作者不知道最佳实践，而是因为当时的 runtime 抽象还没有把“跨图、跨阶段、跨后端持久状态”这件事彻底做完。

更具体地说，这类项目往往同时满足下面几个条件：

- encoder 和 decoder 不是一张大图，而是两张独立图、分阶段执行
- graph/scheduler 会在不同阶段做 reset、重建、重新 alloc 或 graph reuse 判定
- encoder 输出张量原本只是上一张图的结果 tensor，它的生命周期默认服从 compute allocator / graph allocator，而不是自动升级为“下一阶段可长期借用的状态对象”
- decoder 侧输入接口仍然是“host 准备数据，再通过 input tensor 写入”的心智，而不是“直接绑定上一阶段的 backend tensor 作为外部输入”
- 除了 cross embedding 本身，decoder 往往还需要额外的 host-side 元信息，例如 encoder 端保留下来的序列映射、cross-attention mask 构造信息、输出条目和输入序列的对应关系

当这些条件叠在一起时，如果强行让 decoder 直接引用 encoder 图里的输出 tensor，就会立刻遇到几类很难一次做对的问题：

- 这个 tensor 在下一次 graph alloc / reserve 后是否还有效
- 它的 buffer 是否还归当前 scheduler / graph owner 管
- encoder 和 decoder 如果落在不同 backend / buffer type，上下游怎么安全衔接
- graph reuse 时，新的 decoder 图是否仍然能复用旧的 cross tensor 绑定关系
- cross-attention 需要的 mask 和序列映射如果还是 host 侧构造，那数据边界到底应该在哪里收口

在这种情况下，一个“看起来不优雅但很稳”的折中就是：

1. encoder 图执行完成后，显式同步
2. 把需要跨阶段保留的 embedding 复制到 host 持久缓冲
3. 同时保存 decoder 构造 cross mask 所需的序列映射等元信息
4. decoder 图启动时，把这份 host 缓冲当成普通输入重新喂进去

这样做的代价当然很明显：

- 多了一次 Host/Backend 往返
- embedding 会在 host 和 backend 侧各保留一份
- 破坏了理想情况下“阶段间直接共享 backend-resident tensor”的链路

但它能立刻换来三件很实际的收益：

- 生命周期清楚：跨阶段的数据变成显式的持久对象，而不是偷偷借用上一张图里的临时结果
- 正确性更容易保证：不用同时解决 graph allocator 生命周期、backend 所有权和跨图绑定问题
- 改动面更小：可以复用现有“input tensor 由 host 填充”的主框架，而不用立刻发明一整套外部 backend tensor 输入协议

所以，这不是“最佳实践错了”，而是：

- 最佳实践描述的是长期应该收敛到的架构边界
- 现实项目在架构尚未补齐时，可能会有一个阶段性的 host fallback

真正需要警惕的，不是“代码里出现了一次 host copy”，而是下面两种情况：

- 本来只是阶段性折中，却被误当成长期正确接口，最后整个 runtime 都围绕 host 中转设计
- 明明跨阶段数据已经是核心热路径，却迟迟没有把它从 host fallback 升级成正式状态对象

更理想、也更符合本指南的方法，通常是把这条链路继续收敛成一个显式 `cross-state`：

- 它独立于 compute arena 生命周期，不会因为下一次 graph alloc 被覆盖
- 它拥有自己的 backend buffer 或持久 output buffer，而不是借上一张图的临时结果
- decoder 输入接口能够直接绑定这份持久 tensor/view，而不必强制先 materialize 到 host
- 与 cross-attention 相关的序列映射、mask 元信息也和这份状态对象一起管理
- graph reuse 条件里显式纳入这份 cross-state 的 shape、backend、mask 语义，而不是只按 token 数和 batch 形状猜

如果短期内做不到完整重构，更稳的中间状态也至少应该是：

- 明确把它标成“阶段性 fallback”
- 把 host 持久缓冲封装成专门的 cross-state，而不是散落成若干 `std::vector<float>`
- 明确这条路径只服务于哪些模型或哪些阶段
- 让后续重构目标清晰指向“去掉这次 host 中转”，而不是继续在它上面叠更多特殊逻辑

压缩成一句话就是：

- 当 runtime 还没有能力安全地跨图持久化 backend-resident 中间结果时，一次显式的 host fallback 可以是合理妥协；但它应该被当成过渡方案，而不是新的默认架构。

### 6.4 先写运行时修正逻辑，再补导出脚本

错误思路：

- 广播修正、转置修正、weight_norm 还原都放 C++ 运行时

正确思路：

- 尽量在 Python 导出阶段一次性消化

这是另一个极易低估的坑。很多 shape 修正、broadcast 修正、weight_norm 还原、特殊权重转置，如果留到 C++ 运行时处理，短期看只是多写几行逻辑，长期看会造成三件事：

- 运行时代码越来越像“模型兼容层”，而不是稳定前向
- 同一个修正逻辑会散落在多个模块里，难以验证
- 很难判断某个数值问题到底是导出错了，还是运行时又做了一次额外修正

更稳的做法是：导出阶段尽量把权重和 metadata 变成“可直接执行”的状态，运行时只做绑定和构图。

### 6.5 为了“像 PyTorch batch”硬拼不自然的图

错误思路：

- 明明执行模式天然分两条路径，还是强行伪 batch

正确思路：

- 按实际执行模式设计图
- 明确接受双前向或多阶段图

这是推理迁移里常见的思维误区。训练代码里很多模式天然以 batch 为中心，但到了推理端，尤其是带 CFG、双分支、条件/无条件路径时，硬把它们拼成一个“更像训练代码”的 batch 图，未必更快，也未必更省内存。很多时候，显式双前向反而更容易验证、cache、更少中间结构，也更容易配合现有 scheduler。

---

## 7. 已验证坑点

除了上面的通用反模式，还可以把已经反复出现、且迁移时特别容易忽视的坑单独记下来。

### 7.1 把 weight context、graph context、KV context 混成一种心智

这是很多内存估算错误的源头。

这三者虽然都叫“context”，但职责完全不同：

- weight context：保存权重 tensor metadata，服务于持久权重
- graph context：保存图里 tensor 和 graph metadata
- KV/state context：服务于跨步持久状态

如果把它们混成同一种对象去思考，就很容易出现两类问题：

- graph overhead 被错误地算进权重上下文
- 状态对象被错误地当成一次性 graph tensor

结果通常是内存预估不准、对象边界不清、后续重构困难。

### 7.2 只看 PyTorch 模块名猜 shape，不回到导出后的真实 tensor

这是布局迁移时非常危险的习惯。

模块名往往会让人产生“它应该长这样”的错觉，但真正决定运行时实现的，是导出后的真实权重 shape、GGUF 中的最终存储方式，以及相关算子的输入契约。只看训练侧模块名，很容易把 patch 维、feat 维、hidden 维混淆，最后文档、导出脚本、运行时三边都自洽但互相不一致。

更稳的原则是：

- 先看导出后的真实 tensor shape
- 再写运行时实现
- 最后再回头写文档总结

### 7.3 把训练期逻辑残留到推理实现

迁移到 GGML 时，目标通常是推理，不是训练兼容。

很多训练侧逻辑，例如：

- STE
- 反向友好的 surrogate 实现
- 只为训练稳定性存在的分支

如果原样带进推理端，会让实现和验证都变复杂。更糟的是，这些逻辑经常会掩盖真正需要保留的那条纯推理数值路径。

因此，迁移时最好主动问一句：

- 这段逻辑是推理必需，还是训练遗留？

### 7.4 先急着补“缺失算子”，而不是先审查布局和 broadcast

当一个模块第一次对不上时，最容易下意识得出“GGML 没这个算子”。

但实际项目里，很多所谓“算子缺失”最后都不是因为算子真的不存在，而是因为：

- 权重布局没理清
- 输入输出语义维没理清
- broadcast 形状不对
- 某一步少了 `reshape/view/permute`

卷积、语音模块尤其容易掉进这个坑，因为它们对通道维、时间维、kernel 维和 broadcast 的要求更敏感。

更稳的排查顺序应该是：

1. 先查布局
2. 再查 shape/broadcast
3. 再查是否能用现有算子重写
4. 最后才决定要不要补自定义 op

---

## 8. 语音和多模态项目的额外建议

如果是语音或多模态项目，除了通用原则，还建议注意下面三点。

### 8.1 传统 DSP frontend 先稳定数值，不必急着进图

对 Hann、FFT、mel、normalize 这类确定性前处理，先保证数值一致性，往往比一开始强行并图更重要。

### 8.2 runtime state 往往不止一个 KV

语音项目里常见的状态包括：

- self-attention KV
- cross-attention KV
- frontend feature state
- decoder state
- 对齐/辅助缓存

这些状态不要混成一个“大逻辑对象”。物理上是否拆成多个独立 context / buffer，可以由实现根据生命周期、所有权和后端边界决定。

### 8.3 分阶段 reserve/schedule 往往比超级大图更稳

对 encoder-decoder、语音生成、带 frontend 的项目，按阶段拆：

- conv/frontend
- encoder
- cross
- decoder

通常比一开始就拼一张超级大图更稳、更容易验证。

---

## 9. 一个可直接执行的落地清单

### 9.1 设计阶段

- 定义每个模块的 PyTorch shape、GGML shape、GGUF shape
- 定义命名规则和 metadata key
- 明确权重、状态、compute、output 的边界
- 明确哪些阶段必须保持在 backend-resident tensor/state 内

### 9.2 导出阶段

- 做权重重排和形状修正
- 导出完整 metadata
- 导出模块级 trace
- 确认导出后的真实 tensor shape

### 9.3 实现阶段

- 先做统一 loader
- 再做 memory
- 再做 backend/scheduler
- 再做 output buffer
- 再做模块级 forward
- 模块间尽量传 tensor/state

### 9.4 验证阶段

- 先对拍单模块
- 再对拍组合模块
- 最后对拍整链路
- 区分数值误差和布局错误

### 9.5 优化阶段

- 先 reserve
- 再压缩 Host/Backend 往返
- 再做 graph reuse
- 最后做深层优化

---

## 10. 最后总结

最稳的 Torch -> GGML 迁移，不是“先补齐所有算子”，而是先把下面这条主线做对：

`Contract -> Loader -> Memory -> Output -> Scheduler -> Module Forward -> Validation -> Optimization`

如果只能记住几句话，那就是：

- 先定布局，不要边写边猜
- 先定权重和状态放哪，再谈算子翻译
- 在大多数推理热路径里，少过 host、尽量保持 backend-resident，通常比“模块很多但接口全是 vector”更重要
- trace 对拍是主流程，不是补救手段
- 对 Transformer / LLM / GGUF / 多后端 runtime 场景，`llama.cpp` 通常更适合当默认骨架；具体业务项目更适合当案例和反例
