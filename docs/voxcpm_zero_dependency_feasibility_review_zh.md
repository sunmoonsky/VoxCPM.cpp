# VoxCPM 零依赖封装与 `voxcpm-server` 可行性评审

## 结论摘要

这次核查的结论可以先概括为三句话：

1. 当前仓库默认构建出来的 `voxcpm` 相关可执行文件，不是“除了 ggml 以外零依赖”，也不是“单文件直接运行”的形态。
2. `VoxCPM.cpp` 的代码结构本身支持做成 CPU-only 的单文件可执行程序，例如未来的 `voxcpm-server`，Linux 上也可以做到真正静态链接。
3. 但这件事目前还没有被工程化收口；要做到稳定、可发布、跨平台，还需要补构建选项、裁剪 `ggml` 的动态加载路径、修正跨平台构建细节，并补上 server 目标本身。

这意味着：`voxcpm.cpp` 不是“天然已经无依赖封装好”，但“改造后做到 Linux 单文件运行、Windows 尽量无额外 DLL”是可行的。

## 评审对象

本评审同时核查了以下两部分：

- 当前仓库实际构建行为
- 文档 `<LOCAL_PLAN_PATH>/adaptive-gathering-map.md` 的正确性

## 当前工程的真实状态

### 1. `voxcpm` 本体目前是静态库

顶层 CMake 明确把 `voxcpm` 定义成静态库：

- [CMakeLists.txt](${REPO_ROOT}/CMakeLists.txt#L52)

也就是说，项目核心库本身不是 `.so/.dylib/.dll`，而是 `libvoxcpm.a`。

### 2. 但 `ggml` 默认不是静态构建

`ggml` 子项目在非 MinGW 平台默认 `BUILD_SHARED_LIBS=ON`：

- [third_party/ggml/CMakeLists.txt](${REPO_ROOT}/third_party/ggml/CMakeLists.txt#L68)
- [third_party/ggml/CMakeLists.txt](${REPO_ROOT}/third_party/ggml/CMakeLists.txt#L82)

所以默认构建时，`voxcpm_tts` 会链接到：

- `libggml.so`
- `libggml-cpu.so`
- `libggml-base.so`
- 以及系统动态库 `libstdc++.so.6`、`libm.so.6`、`libgcc_s.so.1`、`libc.so.6`
- 当前默认还会带上 `libgomp.so.1`，因为 `ggml` 默认启用了 OpenMP

因此默认产物显然不是“零动态依赖”。

### 3. `json` 目前是 vendored，但保留在线抓取回退

顶层 CMake 对 `nlohmann_json` 的逻辑是：

- 若 `third_party/json` 存在，则直接 `add_subdirectory`
- 若不存在，则使用 `FetchContent` 在线下载

对应代码在：

- [CMakeLists.txt](${REPO_ROOT}/CMakeLists.txt#L37)

所以文档里把 `json` 说成“已 vendored”在当前工作区下基本成立，但严格说法应该是：

“优先使用 vendored `third_party/json`，缺失时会回退到网络下载。”

### 4. 当前还没有 `voxcpm-server`

现有 examples 只有：

- `voxcpm_tts`
- `voxcpm_quantize`
- `voxcpm_imatrix`

没有 HTTP server 目标，这一点原文档判断是对的。

## 我实际验证过的构建结果

### 默认构建

使用接近默认配置构建 `voxcpm_tts` 后，产物存在以下动态依赖：

- `libggml.so`
- `libggml-cpu.so`
- `libggml-base.so`
- `libstdc++.so.6`
- `libm.so.6`
- `libgcc_s.so.1`
- `libc.so.6`
- `libgomp.so.1`

所以默认构建结果不能被描述成“无依赖”或“只有 ggml 依赖”。

### 静态 `ggml` 构建

我额外验证了：

```bash
cmake -S . -B build-static-review \
  -DVOXCPM_BUILD_TESTS=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_STATIC=ON \
  -DGGML_OPENMP=OFF \
  -DVOXCPM_VULKAN=OFF \
  -DVOXCPM_NATIVE=OFF
```

这时 `ggml` 系列库会变成静态库，但最终 `voxcpm_tts` 仍然是动态链接 ELF，仍依赖：

- `libstdc++.so.6`
- `libm.so.6`
- `libgcc_s.so.1`
- `libc.so.6`

这说明“只打开 `GGML_STATIC=ON` 就能得到完全无依赖二进制”是不成立的。

### 全静态链接验证

当我进一步加入：

```bash
-DCMAKE_EXE_LINKER_FLAGS='-static -static-libstdc++ -static-libgcc'
```

之后，`voxcpm_tts` 在当前 Linux/aarch64 环境中可以被构建成：

- `statically linked`
- `ldd` 输出为“不是动态可执行文件”

这说明：

- Linux 上“单文件直接运行”的 CPU-only 可执行程序是可做到的
- 但不是当前默认行为
- 也不是只靠一个 `GGML_STATIC` 选项就能稳定得到

## 对原计划文档的逐项评审

### 正确的部分

以下判断基本正确：

1. `voxcpm-server` 需要新增实现，当前仓库没有现成 server 目标。
2. 如果目标是 HTTP-only，可以选择 header-only 的 HTTP 库来避免引入 OpenSSL 一类额外依赖。
3. CPU-only 路线是最适合做“单文件分发”的实现路线。
4. Linux / Windows / macOS 都可以做发行构建，但三者的“无依赖”定义并不相同。

### 不准确或不完整的部分

#### 1. “当前几乎可以做到零依赖”表述过于乐观

原文：

- [adaptive-gathering-map.md](<LOCAL_PLAN_PATH>/adaptive-gathering-map.md#L55)

问题在于它忽略了：

- `ggml` 默认共享库构建
- OpenMP 默认引入 `libgomp`
- `libstdc++/libgcc/libc` 并不会因为 `GGML_STATIC` 自动消失

更准确的说法应当是：

“当前代码结构具备改造成单文件可执行程序的条件，但默认构建结果并非零依赖，需要额外的静态链接与裁剪改造。”

#### 2. `libdl` 的描述不够准确

原文把 `libdl` 解释成“仅 backend plugins 需要”：

- [adaptive-gathering-map.md](<LOCAL_PLAN_PATH>/adaptive-gathering-map.md#L60)

但当前 `ggml` 在 Linux 上会直接给 `ggml` 目标链接 `dl`：

- [third_party/ggml/src/CMakeLists.txt](${REPO_ROOT}/third_party/ggml/src/CMakeLists.txt#L243)

而且 `ggml` 本体源文件里包含了 `ggml-backend-dl.cpp`：

- [third_party/ggml/src/CMakeLists.txt](${REPO_ROOT}/third_party/ggml/src/CMakeLists.txt#L224)

所以“CPU-only 就天然不需要 `dl`”在当前代码下并不成立。

#### 3. 文档漏掉了 OpenMP 依赖

原文依赖表中没有提到 `libgomp`：

- [adaptive-gathering-map.md](<LOCAL_PLAN_PATH>/adaptive-gathering-map.md#L22)

但 `ggml` 默认 `GGML_OPENMP=ON`，并会在 CPU backend 中链接 OpenMP：

- [third_party/ggml/CMakeLists.txt](${REPO_ROOT}/third_party/ggml/CMakeLists.txt#L240)

这会让默认构建额外依赖 `libgomp.so.1`。

#### 4. “设置 `VOXCPM_STATIC` 后直接 `set(CMAKE_EXE_LINKER_FLAGS \"-static\" FORCE)`”不够稳妥

原文建议：

- [adaptive-gathering-map.md](<LOCAL_PLAN_PATH>/adaptive-gathering-map.md#L156)

这有几个问题：

- 对 macOS 不适用
- 对 Windows/MSVC 不适用
- 对 Linux 也不总是足够，还可能需要 `-static-libstdc++ -static-libgcc`
- 直接覆盖 `CMAKE_EXE_LINKER_FLAGS` 可维护性较差

更稳妥的方式应当是平台分支处理，并只对目标或发行 preset 生效。

#### 5. “CPU-only 模式下，无需 `libdl`”这条目前做不到

原文：

- [adaptive-gathering-map.md](<LOCAL_PLAN_PATH>/adaptive-gathering-map.md#L120)

这在“目标状态”里可以作为改造目标，但不能写成“当前现状”。当前 `ggml` 的 registry / backend-dl 路径仍然在参与构建和链接。

#### 6. “Windows/macOS 无依赖”需要更谨慎表述

原文：

- [adaptive-gathering-map.md](<LOCAL_PLAN_PATH>/adaptive-gathering-map.md#L112)

更准确的版本应该是：

- Linux：有机会做到真正静态单文件 ELF
- Windows：有机会做到无需额外安装 DLL 的单个 `.exe`，但要分 MinGW 和 MSVC 两套逻辑
- macOS：通常不追求完全无系统动态库，更现实的目标是“无额外第三方 dylib 依赖，可直接运行的 app 或可执行文件”

## `voxcpm.cpp` 能不能做到无依赖封装库

答案分两层。

### 1. 如果你说的是“单个可执行文件直接运行”

答案是：可以，但需要改造。

现有代码并没有必须依赖外部服务、外部 Python runtime 或额外 daemon 的设计障碍。`voxcpm_tts` 已经证明：

- 模型加载、推理、音频输出都可以在单进程内完成
- 后续增加 `voxcpm-server` 只是再包一层 HTTP 接口

因此，做一个 CPU-only 的 `voxcpm-server` 单可执行文件，是技术上可行的。

### 2. 如果你说的是“单个可安装库，对外不暴露其他依赖”

答案是：当前还不行，但可以改。

原因有两个：

1. `voxcpm` 公开链接了 `ggml`、`ggml-base`、`nlohmann_json`
2. 当前 install 只安装 `voxcpm` 和头文件，没有把完整依赖关系导出成一个可消费的 package

相关位置：

- [CMakeLists.txt](${REPO_ROOT}/CMakeLists.txt#L78)
- [CMakeLists.txt](${REPO_ROOT}/CMakeLists.txt#L111)

所以当前它更像“项目内部静态库”，而不是“对外完整封装好的发布库”。

## 如果要做到你要的目标，需要改什么

### 必做项

1. 增加 `voxcpm-server` 可执行目标。
2. 增加正式的发行构建选项，例如：
   - `VOXCPM_SERVER`
   - `VOXCPM_CPU_ONLY`
   - `VOXCPM_STATIC`
   - `VOXCPM_PORTABLE`
3. 在发行构建中统一设置：
   - `BUILD_SHARED_LIBS=OFF`
   - `GGML_OPENMP=OFF`
   - `VOXCPM_VULKAN=OFF`
   - `VOXCPM_NATIVE=OFF`
4. 为 Linux 发行版补真正的静态链接配置，而不是只依赖 `GGML_STATIC=ON`。
5. 重新梳理 `ggml` 的 `dlopen/libdl` 路径，在 CPU-only 静态模式下尽量裁掉。

### 强烈建议项

1. 不要直接用全局字符串覆盖 `CMAKE_EXE_LINKER_FLAGS`，改用平台条件和目标级配置。
2. 为 Windows 分别明确：
   - MinGW 路径
   - MSVC 路径
3. 为 macOS 明确目标不是“绝对静态”，而是“不依赖额外第三方动态库”。
4. 补 `install(EXPORT ...)` 和 package config，让 `voxcpm` 真正成为可被外部项目消费的封装库。

### 如果要交付“单库”

有两种路线：

1. 保持 `voxcpm + ggml` 为多个静态库，但把 install/export/package 做完整。
2. 进一步把依赖合并，产出一个更接近“单一静态库”的交付物。

对当前仓库来说，第一条更现实，第二条改造更深。

## 推荐的目标定义

如果要降低风险，建议把目标明确成下面这种表述：

### 近期目标

产出一个 CPU-only 的 `voxcpm-server`：

- Linux 上可做成单个静态可执行文件
- Windows 上尽量做成无需额外 DLL 的单个 `.exe`
- macOS 上做成无需额外第三方动态库的可执行程序或 app

### 中期目标

把 `voxcpm` 整理成可安装、可导出的封装库：

- 安装头文件
- 安装静态库及依赖
- 导出 CMake package
- 明确 ABI / API 边界

## 最终结论

对“我们的 `voxcpm.cpp` 能不能做到无依赖封装库？”这个问题，最准确的回答是：

- 现在默认构建结果：不能这么说。
- 作为单文件可执行程序：可以做到，Linux 上已经验证存在可行路径。
- 作为完整对外封装库：当前还不够，需要补打包和依赖收口。

对 `<LOCAL_PLAN_PATH>/adaptive-gathering-map.md` 这份文档的评价则是：

- 方向正确
- 适合作为“初步方案草稿”
- 但不能直接当成已验证的实施文档

在真正立项改造前，建议先把下面三条写进新的实施方案里：

1. 默认构建不是零依赖。
2. `GGML_STATIC=ON` 不等于最终产物完全静态。
3. CPU-only 静态发行模式下，必须额外处理 OpenMP、`dlopen/libdl`、跨平台链接选项和 `VOXCPM_NATIVE` 的可移植性问题。
