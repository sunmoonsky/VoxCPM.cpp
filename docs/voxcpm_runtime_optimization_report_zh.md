# VoxCPM Runtime 优化阶段报告

## 1. 背景

本轮工作的目标是处理 `src/` 中由内存访问、临时张量和过碎图边界带来的 CPU 利用率问题。

起点观察：

- 8 线程运行时，CPU 利用率常见只有约 `60% ~ 70%`
- `prefill()` 和 `decode()` 路径存在大量：
  - 小图反复构建
  - host 端 `std::vector` 往返
  - 同形状图重复 `alloc_graph + compute`
  - 一些每次前向都会重新构建的临时张量

本轮优先策略不是直接改 GGML 内核，而是先清理 VoxCPM 自己 runtime 层的结构性开销。

---

## 2. 已完成优化

### 2.1 MiniCPM：单次前向复用 causal mask

涉及文件：

- [include/voxcpm/minicpm.h](/home/orangepi/Codes/ggbond/VoxCPM.cpp/include/voxcpm/minicpm.h)
- [src/minicpm.cpp](/home/orangepi/Codes/ggbond/VoxCPM.cpp/src/minicpm.cpp)

改动内容：

- `forward()` 内只生成一次 causal mask，所有 layer 复用
- 消除了每层重复构建 mask 的临时张量

效果判断：

- 这是低风险、正确性稳定的清理项
- 对 decode 单步不是决定性大头，但减少了无意义临时张量

### 2.2 UnifiedCFM：去掉伪 batch=2 路径

涉及文件：

- [include/voxcpm/locdit.h](/home/orangepi/Codes/ggbond/VoxCPM.cpp/include/voxcpm/locdit.h)
- [src/locdit.cpp](/home/orangepi/Codes/ggbond/VoxCPM.cpp/src/locdit.cpp)
- [src/unified_cfm.cpp](/home/orangepi/Codes/ggbond/VoxCPM.cpp/src/unified_cfm.cpp)

改动内容：

- 之前 `UnifiedCFM` 会把 cond/uncond 拼成 batch=2
- 然后 `LocDiT::forward()` 又在 C++ 侧按 batch 串行拆开
- 现在改为显式 cond/uncond 双前向，去掉：
  - `repeat`
  - 伪 batch 张量
  - `cpy + sync` 之类中间张量

效果判断：

- 直接打到了 decode 最热的 `unified_cfm` 路径
- 是本轮 decode 优化的重要组成部分

### 2.3 Runtime：固定形状小图 cache

涉及文件：

- [include/voxcpm/voxcpm.h](/home/orangepi/Codes/ggbond/VoxCPM.cpp/include/voxcpm/voxcpm.h)
- [src/voxcpm.cpp](/home/orangepi/Codes/ggbond/VoxCPM.cpp/src/voxcpm.cpp)

改动内容：

- 为以下高频、稳定形状路径加入 lazy cache：
  - `run_locenc_patch()`
  - `run_stop_predictor()`
  - `run_locenc_patch_to_lm_embed()`

目的：

- 避免每次都重新创建相同 graph/context
- 保留每次 `alloc_graph + tensor_set + compute`
- 去掉最明显的“小图反复建图”开销

### 2.4 Runtime：按 `seq_len` 缓存 prefill 子图

涉及文件：

- [include/voxcpm/voxcpm.h](/home/orangepi/Codes/ggbond/VoxCPM.cpp/include/voxcpm/voxcpm.h)
- [src/voxcpm.cpp](/home/orangepi/Codes/ggbond/VoxCPM.cpp/src/voxcpm.cpp)

改动内容：

- 为以下 prefill 子图加入按长度 lazy cache：
  - `run_embedding(token_count)`
  - `enc_to_lm projection(seq_len)`
  - `run_fsq_2d(seq_len)`

效果判断：

- 这几段本来就不是 prefill 主瓶颈
- 但它们足够稳定，属于“该做且风险很低”的清理项

### 2.5 LocEnc：从逐 patch 小图改成 sequence 大图

涉及文件：

- [include/voxcpm/localenc.h](/home/orangepi/Codes/ggbond/VoxCPM.cpp/include/voxcpm/localenc.h)
- [src/localenc.cpp](/home/orangepi/Codes/ggbond/VoxCPM.cpp/src/localenc.cpp)
- [src/voxcpm.cpp](/home/orangepi/Codes/ggbond/VoxCPM.cpp/src/voxcpm.cpp)
- [include/voxcpm/minicpm.h](/home/orangepi/Codes/ggbond/VoxCPM.cpp/include/voxcpm/minicpm.h)
- [src/minicpm.cpp](/home/orangepi/Codes/ggbond/VoxCPM.cpp/src/minicpm.cpp)

改动内容：

- 新增 `LocEncModel::forward_sequence()`
- `encode_feature_sequence()` 不再：
  - `for each patch -> run_locenc_patch()`
- 改为：
  - 一次把 `[feat_dim, patch_size, seq_len]` 输入送进一张 sequence 图

配套调整：

- 给 `MiniCPM` 前向增加 `write_kv_cache` 开关
- `LocEnc/LocDiT` 这类一次性编码路径关闭 scratch KV 写入
- 避免在单个大图里展开多个分支时互相踩 KV

效果判断：

- 这是 prefill 方向最关键的一刀
- 从 runtime 结构上消除了 “逐 patch 一次 compute” 的主要碎片化问题

### 2.6 Decode：缓存两张稳定大图

涉及文件：

- [include/voxcpm/voxcpm.h](/home/orangepi/Codes/ggbond/VoxCPM.cpp/include/voxcpm/voxcpm.h)
- [src/voxcpm.cpp](/home/orangepi/Codes/ggbond/VoxCPM.cpp/src/voxcpm.cpp)

改动内容：

- 为以下 decode 热路径加入 cache：
  - `run_unified_cfm()`
  - `run_decode_front_half()`

原因：

- 两者都不依赖具体 decode state 的 KV cache
- 图形状稳定
- 是 decode 最大头

效果判断：

- 这是 decode 侧收益最大的一刀

### 2.7 Decode：state-owned LM step cache

涉及文件：

- [include/voxcpm/voxcpm.h](/home/orangepi/Codes/ggbond/VoxCPM.cpp/include/voxcpm/voxcpm.h)
- [src/voxcpm.cpp](/home/orangepi/Codes/ggbond/VoxCPM.cpp/src/voxcpm.cpp)

改动内容：

- `base_lm_step` / `residual_lm_step` 图不能做 runtime 全局 cache
- 因为它们直接引用每个 `VoxCPMDecodeState` 自己的 KV cache
- 现在改为：
  - `VoxCPMDecodeState` 内维护按 position 缓存的 step graph

效果判断：

- 对真实 `decode()` 主路径有小幅收益
- 是正确但边际收益已经较小的一刀

---

## 3. 回归验证

已通过的关键测试：

- `ctest --test-dir build --output-on-failure -R "test_(minicpm|localenc|locdit|unified_cfm|voxcpm)"`

验证结论：

- `MiniCPM`
- `LocEnc`
- `LocDiT`
- `UnifiedCFM`
- `VoxCPM prefill/decode`

均未出现回归。

---

## 4. Benchmark 结果

测试环境：

- 模型：`models/voxcpm1.5.gguf`
- prompt audio：`third_party/whisper.cpp/samples/jfk.wav`
- backend：CPU
- threads：8
- scenario：`medium`
- repeat：3
- warmup：1

### 4.1 Prefill 总体

来自：

- `/tmp/voxcpm_bench_prefill.json`

结果：

- `voxcpm.prefill`：`1473.780 ms`
- `prefill.locenc_all`：`917.490 ms`

结论：

- 当前 `prefill` 的最大头仍然是 `LocEnc`
- 在这个场景下约占总 prefill 的 `62%`

### 4.2 Prefill 分解

来自：

- `/tmp/voxcpm_bench_prefill_breakdown.json`

结果：

- `prefill.locenc_all`：`927.940 ms`
- `prefill.base_lm`：`411.763 ms`
- `prefill.residual_lm`：`147.282 ms`
- `prefill.enc_to_lm_proj`：`1.076 ms`
- `prefill.fsq`：`0.616 ms`
- `prefill.text_embedding`：`0.023 ms`

结论：

- prefill 侧真正值得继续打的只有两个大头：
  - `LocEnc`
  - `base_lm`
- `projection / embedding / fsq` 已经小到可以忽略

### 4.3 Decode：本轮前后的关键对比

本轮前的 early decode 关键数字：

- `voxcpm.decode_step.early`：`661.454 ms`
- `decode.unified_cfm.early`：`535.041 ms`

本轮后的 early decode 关键数字：

来自：

- `/tmp/voxcpm_bench_decode_early.json`
- `/tmp/voxcpm_bench_decode_state_cached_early.json`

结果：

- `voxcpm.decode_step.early`：`528.694 ms`
- `decode.front_half_total.early`：`417.211 ms`
- `decode.unified_cfm.early`：`417.277 ms`

收益：

- `decode_step`：从 `661.454 ms` 降到 `528.694 ms`
  - 下降约 `20.1%`
- `unified_cfm`：从 `535.041 ms` 降到 `417.277 ms`
  - 下降约 `22.0%`

结论：

- decode 侧本轮最成功的优化就是：
  - 去掉伪 batch
  - 缓存 `unified_cfm`
  - 缓存 `decode_front_half`

### 4.4 Decode：当前 early 分解

来自：

- `/tmp/voxcpm_bench_decode_rest_early.json`
- `/tmp/voxcpm_bench_decode_state_cached_early.json`

结果：

- `decode.stop_predictor.early`：`0.038 ms`
- `decode.locenc_patch_to_lm.early`：`15.446 ms`
- `decode.base_lm_step_fsq.early`：`84.163 ms`
- `decode.residual_lm_step.early`：`94.955 ms`
- `voxcpm.decode_step.early`：`528.694 ms`

结论：

- 当前 decode 最大头仍然是 front-half / unified CFM
- `base_lm_step` 和 `residual_lm_step` 仍然有成本，但已经不是第一大头
- `stop_predictor` 和 `locenc_patch_to_lm` 都不是关键瓶颈

---

## 5. 重要解释

### 5.1 为什么 `decode.base_lm_step_fsq` benchmark 没明显下降？

因为 benchmark helper 和真实 `decode()` 路径不完全相同。

本轮对 `base_lm_step` / `residual_lm_step` 做的是：

- state-owned cache
- 绑定真实 `VoxCPMDecodeState`

它直接优化的是真实 `decode()` 主路径。

但 `benchmark_run_base_lm_decode_step()` 和 `benchmark_run_residual_lm_decode_step()` 仍是单独 helper 调用，不会自动经过 `decode()` 新接入的 state-owned cache 逻辑。

因此判断这刀是否有效，应主要看：

- `voxcpm.decode_step.early`

而不是只看独立 helper benchmark。

### 5.2 为什么没有直接迁移到 `ggml_backend_sched`？

因为本轮主要问题不在 scheduler 缺失，而在 runtime 自己的：

- 碎图边界
- 中间张量
- host 端往返
- 重复构图

在这些结构性问题没清理前，先上 scheduler 不是收益最高的一刀。

---

## 6. 当前结论

本轮优化后，可以比较明确地下结论：

1. `prefill` 的主瓶颈仍然是 `LocEnc`
2. `decode` 的主瓶颈已经明确集中在 front-half / `UnifiedCFM`
3. decode 侧通过 runtime 结构优化已经拿到了约 `20%` 的真实单步收益
4. 再往下继续抠 `LM step` 不是不行，但已经进入边际收益区

---

## 7. 建议的下一步

### 7.1 如果继续做 CPU 推理优化

优先级建议：

1. 继续深挖 `LocEnc`
2. 再考虑 `base_lm` prefill 路径
3. decode 侧如继续做，则应下探 `MiniCPM/GGML` 内层算子，而不是继续加 runtime cache

### 7.2 如果目标是端到端 TTS 时延

还应单独评估：

- AudioVAE decode

因为当前这份报告主要覆盖的是：

- `VoxCPMRuntime::prefill()`
- `VoxCPMRuntime::decode()`

并不等同于完整音频重建链路。

---

## 8. 结语

这轮优化已经把“低风险且高确定性”的 runtime 结构问题清掉了大半：

- 小图反复构建
- 部分长度稳定子图不缓存
- `LocEnc` 逐 patch 小图
- `UnifiedCFM` 伪 batch
- decode front-half 大图不缓存

其中 decode 侧收益已经被 benchmark 明确证实。  
后续若再继续追求大幅提升，就需要更深入地进入：

- `LocEnc` / `MiniCPM` 模型内部
- 或 GGML 更底层的算子路径

这将是下一阶段的工作边界。
