# AscendNPU-IR 项目 regbase(a5) 架构 Pass 完整分析报告

> 生成时间：2026-08-31
> 项目路径：`/home/zhaojs/workspace/codes/AscendNPU-IR`
> 分析范围：与 triton regbase(a5) 架构相关的所有 pass

---

## 目录

1. [项目整体结构概览](#1-项目整体结构概览)
2. [regbase/a5 相关文件完整路径列表](#2-regbasea5-相关文件完整路径列表)
3. [Triton 相关 Pass 目录结构](#3-triton-相关-pass-目录结构)
4. [regbase(a5) 架构下完整 Pass 列表（按执行顺序）](#4-regbasea5-架构下完整-pass-列表按执行顺序)
5. [Pass 注册/组装关键代码位置](#5-pass-注册组装关键代码位置)
6. [Pass 定义文件（TableGen .td）](#6-pass-定义文件tablegen-td)
7. [RegBase(A5) 特有 Pass 与差异点](#7-regbasea5-特有-pass-与差异点)
8. [文档资源](#8-文档资源)
9. [总结](#9-总结)

---

## 1. 项目整体结构概览

```
AscendNPU-IR/                          # MLIR-based IR for Ascend NPU
├── bishengir/                         # 源代码主目录（即 AscendNPU IR）
│   ├── include/                       # 头文件 + TableGen .td 定义
│   ├── lib/                           # 实现代码
│   │   ├── Conversion/                # 方言间转换
│   │   ├── Dialect/                   # 各方言（HFusion/HIVM/HIVMAVE/Triton/HACC/...）
│   │   │   └── */Pipelines/           # 各方言的 pipeline 组装
│   │   ├── Tools/bishengir-compile/   # 编译器驱动
│   │   │   └── regbase/               # ★ RegBase(A5) 专用 pipeline
│   │   └── Template/                  # RegBase Cube 模板（含 a5 hpp）
│   ├── triton/                        # Triton 上游代码（lib/Dialect, lib/Conversion）
│   ├── tools/                         # 可执行工具入口
│   └── test/                          # 测试用例（含 regbase/ 目录）
├── docs/                              # 文档（en + zh_cn）
└── third-party/                       # LLVM 等第三方
```

**关键事实**：`regbase` = `Register-based` 编程模型，对应 **A5 芯片（Ascend950 系列）** 和 **Ascend310B 系列**。判定函数 `hacc::utils::isRegBasedArch()` = `isAscend310B() || isAscend950()`（`bishengir/lib/Dialect/HACC/Utils/Utils.cpp:339-341`）。

### 1.1 顶层关键文件

| 文件 | 说明 |
|------|------|
| `README.md` / `README_zh.md` | 项目介绍 |
| `CMakeLists.txt` | 顶层构建配置 |
| `bishengir/CMakeLists.txt` | bishengir 子项目构建 |
| `bishengir/triton/CMakeLists.txt` | Triton 子项目构建 |
| `AGENTS.md` | 智能体指南 |
| `LICENSE` / `NOTICE` | 许可证与声明 |

### 1.2 A5 芯片特性（摘自 `docs/source/en/introduction/architecture.md:35-41`）

A5 芯片继承 RegBase（`Register-based`）编程模型（来自 310B 芯片）。相比 A2/A3 的 `Memory-based` 模型：
- 硬件层增加寄存器层
- Cube 和 Vector 核之间增加数据通路，为 CV 融合提供更多优化机会
- 增加 `Warp Scheduler` 引入 SIMT 能力
- 引入 `ND-DMA` 等新硬件指令

AscendNPU IR 对 A5 的支持：
- **SIMD 编译**：VF 融合、向量化、mask 优化、Combine 优化
- **SIMT 编译**：将社区 `TritonGPU` dialect lowering 到 HIVM
- **混合 SIMD/SIMT 编译**
- Layout 优化、共享内存分配、核心指令映射等 Ascend 友好优化算法

---

## 2. regbase/a5 相关文件完整路径列表

### 2.1 核心 pipeline 组装文件（最关键）

| 文件绝对路径 | 作用 |
|------|------|
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp` | ★ RegBase 主 pipeline 组装（buildBiShengHIRPipeline 等） |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Tools/bishengir-compile/regbase/BiShengIRCompileMain.cpp` | RegBase 编排+重试/fallback 逻辑（runRegBasePipeline） |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Tools/bishengir-compile/regbase/Driver.cpp` | RegBase 编译驱动入口（runRegBaseCompile） |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Tools/bishengir-compile/regbase/Utility.cpp` | inferMixedCV/inferLayoutOptimization 等 |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HFusion/Pipelines/regbase/HFusionRegbasePipelines.cpp` | ★ HFusion RegBase pipeline |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Pipelines/regbase/HIVMRegbasePipelines.cpp` | ★ HIVM RegBase pipeline |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HIVMAVE/Pipelines/HIVMAVEPipelines.cpp` | HIVMAVE lowering pipeline（regbase 条件分支） |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/Triton/Pipelines/TritonPipelines.cpp` | Triton lowering pipeline（SIMT 路径） |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HACC/Pipelines/HACCPipelines.cpp` | HACC→LLVM pipeline（host 编译） |

### 2.2 RegBase 专用 Transforms

**HFusion Normalize（11 个文件）**：
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HFusion/Transforms/regbase/Normalize/Normalize.cpp`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HFusion/Transforms/regbase/Normalize/NormalizeArithmetic.cpp`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HFusion/Transforms/regbase/Normalize/NormalizeAtomic.cpp`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HFusion/Transforms/regbase/Normalize/NormalizeCasting.cpp`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HFusion/Transforms/regbase/Normalize/NormalizeComparison.cpp`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HFusion/Transforms/regbase/Normalize/NormalizeMath.cpp`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HFusion/Transforms/regbase/Normalize/NormalizeReduction.cpp`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HFusion/Transforms/regbase/Normalize/NormalizeScalar.cpp`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HFusion/Transforms/regbase/Normalize/NormalizeTraitsBase.cpp`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HFusion/Transforms/regbase/Normalize/NormalizeTrig.cpp`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HFusion/Transforms/regbase/Normalize/NormalizeTypeConversion.cpp`

**HIVM Normalize（13 个文件）**：
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/regbase/Normalize/Normalize.cpp`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/regbase/Normalize/NormalizeArithmetic.cpp`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/regbase/Normalize/NormalizeAtomic.cpp`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/regbase/Normalize/NormalizeCasting.cpp`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/regbase/Normalize/NormalizeComparison.cpp`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/regbase/Normalize/NormalizeMath.cpp`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/regbase/Normalize/NormalizeReduction.cpp`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/regbase/Normalize/NormalizeRelu.cpp`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/regbase/Normalize/NormalizeScalar.cpp`

**其他 RegBase Transforms**：
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/regbase/CVPipelining.cpp`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/regbase/PlanMemory.cpp`

### 2.3 RegBase 专用头文件

- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Tools/bishengir-compile/regbase/PassPipeline.h`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Tools/bishengir-compile/regbase/Driver.h`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Tools/bishengir-compile/regbase/Utility.h`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/Pipelines/regbase/Passes.h`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/Transforms/regbase/NormalizePatterns.h`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/Transforms/regbase/NormalizeTraitsBase.h`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/Transforms/regbase/NormalizeUtils.h`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/Transforms/regbase/RegBaseArchUtils.h`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/Transforms/regbase/PlanMemory.h`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Transforms/regbase/Normalize/NormalizeArithmeticTemplate.h`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Transforms/regbase/Normalize/NormalizeAtomicTemplate.h`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Transforms/regbase/Normalize/NormalizeCastingTemplate.h`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Transforms/regbase/Normalize/NormalizeComparisonTemplate.h`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Transforms/regbase/Normalize/NormalizeMathTemplate.h`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Transforms/regbase/Normalize/NormalizeReductionTemplate.h`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Transforms/regbase/Normalize/NormalizeScalarTemplate.h`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Transforms/regbase/Normalize/NormalizeTrigTemplate.h`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Transforms/regbase/Normalize/NormalizeTypeConversionTemplate.h`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Transforms/regbase/Normalize/Utils/CastingTemplateHelpers.h`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Transforms/regbase/Normalize/Utils/Kinds.h`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Transforms/regbase/Normalize/Utils/MathTemplateHelpers.h`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Transforms/regbase/Normalize/Utils/ReductionTemplateHelpers.h`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Transforms/regbase/Normalize/Utils/ScalarTemplateHelpers.h`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Transforms/regbase/Normalize/Utils/TrigTemplateHelpers.h`

### 2.4 RegBase 专用方言/转换

- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.cpp`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.td`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Conversion/AscendDPXToHIVMRegbaseIntrins/AscendDPXToHIVMRegbaseIntrins.cpp`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Conversion/AscendDPXToHIVMRegbaseIntrins/AscendDPXToHIVMRegbaseIntrins.h`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Conversion/HIVMToStandard/regbase/HIVMToStandard.cpp`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Utils/RegbaseUtils.cpp`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/Utils/RegbaseUtils.h`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HIVMRegbaseIntrins/Utils/RegbaseUtils.cpp`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVMRegbaseIntrins/Utils/RegbaseUtils.h`

### 2.5 A5 专用模板

- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Template/lib/RegBase/Cube/include/catlass/gemm/tile/copy_gm_to_l1_a5.hpp`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Template/lib/RegBase/Cube/include/catlass/gemm/tile/copy_l0c_to_gm_a5.hpp`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Template/lib/RegBase/Cube/include/catlass/gemm/tile/copy_l1_to_bt_a5.hpp`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Template/lib/RegBase/Cube/include/catlass/gemm/tile/copy_l1_to_l0a_a5.hpp`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Template/lib/RegBase/Cube/include/catlass/gemm/tile/copy_l1_to_l0b_a5.hpp`

### 2.6 测试文件

- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/test/regbase/`（含 Conversion/Dialect 等子目录）
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/test/Dialect/HFusion/regbase/Normalize/`（16 个 mlir 测试）
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/test/Dialect/HIVM/regbase/Normalize/`（17 个 mlir 测试）
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/test/Dialect/HIVM/regbase/CVPipelining/`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/test/Conversion/HFusionToHIVM/gather-regbase.mlir`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/test/Conversion/HFusionToHIVM/matmul-regbase.mlir`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/test/Conversion/HIVMToLLVM/regbase-convert-llvm.mlir`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/test/Conversion/HIVMToLLVM/regbase-convert-llvm_signless.mlir`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/test/Conversion/HIVMToLLVM/regbase-type-conversion-nested_block.mlir`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/test/Dialect/HIVM/arith-constant-op-to-hivm-regbase-broadcast.mlir`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/test/Dialect/HIVM/legalize-vector-storage-regbase.mlir`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/test/Dialect/HIVM/plan-memory-regbase.mlir`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/test/Dialect/Tensor/propagate-reshape-a5-regbase.mlir`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/test/Dialect/Tensor/propagate-reshape-regbased-unit-dim.mlir`
- `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/test/Dialect/HFusion/FlattenOps/hfusion-flatten-tidy-regbase.mlir`

---

## 3. Triton 相关 Pass 目录结构

```
bishengir/triton/                     # Triton 上游代码
├── lib/
│   ├── Dialect/
│   │   ├── Triton/IR, Transforms
│   │   ├── TritonGPU/{IR, Transforms, Pipeliner, WarpSpecialization}
│   │   ├── TritonNvidiaGPU/
│   │   └── Gluon/
│   ├── Conversion/
│   │   ├── TritonToTritonGPU/
│   │   ├── TritonGPUToLLVM/
│   │   └── TritonInstrumentToLLVM/
│   └── Target/, Analysis/, Instrumentation/

bishengir/lib/Dialect/Triton/         # BiShengIR 对 Triton 的扩展
├── IR/
├── Transforms/
└── Pipelines/
    └── TritonPipelines.cpp           # buildLowerTritonPipeline
```

### 3.1 TritonGPU Transforms（上游 pass，在 regbase pipeline 中被调用）

位于 `bishengir/triton/lib/Dialect/TritonGPU/Transforms/`：
- `AccelerateMatmul.cpp` — 矩阵乘加速
- `Coalesce.cpp` — 合并访问
- `CoalesceAsyncCopy.cpp` — 异步拷贝合并
- `CombineTensorSelectAndIf.cpp` — tensor.select + if 合并
- `DecomposeScaledBlocked.cpp` — 分解缩放阻塞
- `F32DotTC.cpp` — F32 dot TC
- `FuseNestedLoops.cpp` — 嵌套循环融合
- `HoistTMEMAlloc.cpp` — TMEM alloc 提升
- `OptimizeAccumulatorInit.cpp` — 累加器初始化优化
- `OptimizeDotOperands.cpp` — dot 操作数优化
- `OptimizeThreadLocality.cpp` — 线程局部性优化
- `Prefetch.cpp` — 预取
- `ReduceDataDuplication.cpp` — 减少数据重复
- `RemoveLayoutConversions.cpp` — 移除布局转换
- `ReorderInstructions.cpp` / `ReorderInstructionsImpl.cpp` — 指令重排
- `Utility.cpp` — 工具函数
- `Pipeliner/` — 流水线器
- `WarpSpecialization/` — Warp 特化

---

## 4. regbase(a5) 架构下完整 Pass 列表（按执行顺序）

### 4.0 整体编排（`runRegBasePipeline` @ `BiShengIRCompileMain.cpp:325`）

根据 `CompileFlow` 分三种路径（`BiShengIRCompileMain.cpp:42-54`）：

| 编译流 | 条件 | Pipeline 顺序 |
|--------|------|---------------|
| **Mixed** | `enableSimdSimtMixCompile` | `buildBiShengHIRPipeline` → SIMT 模块 `buildBiShengTTIRPipeline` → `buildBiShengHIRFinishPipeline` → `buildFinalHIVMPipelines` → `buildBiShengHIRAVEToLLVMPipeline` |
| **PureSimt** | `pureSimt` | `buildBiShengTTIRPipeline` |
| **Simd**（默认） | 其他 | `buildBiShengHIRPipeline` → `buildFinalHIVMPipelines` → `buildBiShengHIRAVEToLLVMPipeline` |

最后统一调用外部 `hivmc-a5` 工具完成 LLVM IR → 二进制。

**重试/Fallback 逻辑**（`BiShengIRCompileMain.cpp:364-479`）：
- 最多重试 6 次（tuning 模式 1 次）
- 检测 `ub overflow` / `cc overflow` / `cbuf overflow` 三种内存溢出
- Fallback 策略：逐步禁用 multi-buffer（UB→L0C→L1），然后禁用 auto-multi-buffer，切换 VF fusion 模式，禁用 VF reachable check

---

### 4.1 阶段 A: `buildBiShengHIRPipeline`（HIR 前端）

**代码位置**：`PassPipeline.cpp:464-535`

| # | Pass 名称 | 说明 | 条件 |
|---|-----------|------|------|
| 1 | `hacc-append-device-spec` | 追加设备规格信息 | 非 host 编译 |
| 2 | `canonicalize-module` | 模块级规范化 | 总是 |
| 3 | **HFusion RegBase Pipeline** | 见 §4.2 | enableHfusionCompile |
| 4 | `convert-hfusion-to-hivm` | HFusion→HIVM 转换 | enableHIVMCompile |
| 5 | `triton-global-kernel-args-to-hivm` | Triton kernel 参数转换 | enableTritonKernelCompile |
| 6 | `convert-tensor-to-hivm` | Tensor→HIVM | 总是 |
| 7 | `convert-to-hivm-op` | 其他 op→HIVM | 总是 |
| 8 | `canonicalizer` | 规范化 | enableRegBaseHIVMPipe |
| 9 | **HIVMTensorOptimizations** | 见 §4.3 | enableHIVMCompile |
| 10 | `hivm-aggregated-decompose-op` | 混合 CV 分解 | MixedCV |
| 11 | **buildDelayedHFusionRegBaseVectorizePipeline** | 延迟向量化 | MixedCV 且非混合编译 |
| 12 | `auto-scope` | 自动作用域标记 | SIMD/SIMT 混合 |
| 13 | `legalize-bool-for-simtvf` | SIMT VF bool 合法化 | SIMD/SIMT 混合 |
| 14 | `insert-memory-semantic-for-simtvf` | 插入内存语义 | SIMD/SIMT 混合 |
| 15 | `transform-op-for-simt` | SIMT op 变换 | SIMD/SIMT 混合 |
| 16 | `outline-scope` (outlineMarkedScopesOnly) | 外联标记的 scope | SIMD/SIMT 混合 |
| 17 | `insert-alloc-base-placeholder` | 插入分配基占位符 | SIMD/SIMT 混合 |
| 18 | `infer-simt-vf-memory-effect` | 推断 SIMT VF 内存效果 | SIMD/SIMT 混合 |
| 19 | `infer-simt-vf-mem-scope-hint` | 推断 SIMT VF 内存作用域提示 | SIMD/SIMT 混合 |
| 20 | `split-simt-module` | 拆分 SIMT 模块 | SIMD/SIMT 混合 |

---

### 4.2 阶段 B: `buildHFusionPipelines`（HFusion RegBase）

**代码位置**：`HFusionRegbasePipelines.cpp:497-560`

执行流程：`preProcess` → `canonicalizationPipeline` → `preFlattenPass` → `flattenAndFold` → `inferAndOutlineOp` → `postProcessOutlinedKernel` → `hfusionAutoSchedulePipeline` → `postProcess` → `hfusionAutoVectorizePipeline`

#### 4.2.1 preProcess（`:146-205`）

| # | Pass 名称 | 说明 | 条件 |
|---|-----------|------|------|
| 1 | `hfusion-uplift-while-to-for` | while 循环提升为 for | 总是 |
| 2 | `hfusion-legalize-bool` (clamp) | bool 合法化（clamp 模式） | 总是 |
| 3 | `erase-symbol` | 删除符号 | 非 enableSymbolAnalysis |
| 4 | `fuse-reduction-into-loop` | reduction 融合到循环 | Triton+RegBase+enableFuseReductionIntoLoop |
| 5 | `hfusion-legalize-scalar` | 标量合法化 | Triton+RegBase |
| 6 | `convert-arith-to-hfusion` | Arith→HFusion | Triton+RegBase+flatten |
| 7 | `hfusion-generalize` | HFusion 泛化 | Triton+RegBase+flatten |
| 8 | `hfusion-fold-unit-dims` | 折叠单位维度 | Triton+RegBase+flatten |
| 9 | `convert-arith-to-hfusion` | Arith→HFusion | 总是 |
| 10 | `convert-math-to-hfusion` | Math→HFusion | 总是 |
| 11 | `convert-linalg-to-hfusion` | Linalg→HFusion | 总是 |
| 12 | `symbol-dce` | 符号死代码消除 | Triton |
| 13 | `convert-gpu-to-hfusion` | GPU→HFusion | Triton |
| 14 | `adapt-triton-kernel` | 适配 Triton kernel | Triton |
| 15 | `convert-tensor-to-hfusion` | Tensor→HFusion | 总是 |
| 16 | `canonicalize-tensor-reshape` | tensor reshape 规范化 | 总是 |
| 17 | `cse` + `canonicalizer` + `normalize-tensor-ops` | 规范化 pipeline | 总是 |
| 18 | `convert-arith-to-hfusion` | 再次 Arith→HFusion | 总是 |
| 19 | `convert-generic-to-named-op` | 泛型→命名 op | 总是 |
| 20 | `hfusion-legalize-bf16` | BF16 合法化 | 总是 |
| 21 | `hfusion-legalize-fp8` | FP8 合法化 | 总是 |
| 22 | `hfusion-decompose` | HFusion 分解 | 总是 |
| 23 | `hfusion-normalize-slice-ops` | slice op 规范化 | 总是 |
| 24 | `hfusion-generic-unroller` | 泛型展开 | 总是 |
| 25 | `hfusion-normalize-ops` | ★ RegBase 专用 Normalize | 总是 |
| 26 | `hfusion-legalize-bool` | bool 合法化 | 总是 |
| 27 | `hfusion-simplify-ops` | 简化 ops | 总是 |
| 28 | `hfusion-inline-brc` | 内联 brc | 总是 |
| 29 | `hfusion-normalize-ops` | 再次 Normalize | 总是 |

#### 4.2.2 preFlattenPass（`:207-229`）

| # | Pass 名称 | 说明 |
|---|-----------|------|
| 1 | `bubble-up-extract-slice` | 提取 slice 上浮 |
| 2 | `cse` + `canonicalizer` + `normalize-tensor-ops` | 规范化 |
| 3 | `linalg-fold-unit-extent-dims` | 折叠单位维度 |
| 4 | 规范化 pipeline | |
| 5 | `convert-arith-to-hfusion` | Arith→HFusion |
| 6 | `hfusion-compose-multi-reduce` | 多 reduce 组合 |
| 7 | `propagate-symbol` | 符号传播 | enableSymbolAnalysis |
| 8 | `unfold-symbolic-int` | 符号 int 展开 | enableSymbolAnalysis |
| 9 | `propagate-reshape` | reshape 传播 |
| 10 | `hfusion-simplify-ops` | 简化 |
| 11 | 规范化 pipeline | |

#### 4.2.3 flattenAndFold（`:241-263`）

| # | Pass 名称 | 说明 |
|---|-----------|------|
| 1 | `fold-tensor-empty` | 折叠 tensor.empty |
| 2 | `hfusion-flatten-ops` (Tidy, **registerBased=true**) | ★ RegBase 专用 flatten |
| 3 | `canonicalize-tensor-reshape` | reshape 规范化 |
| 4 | 规范化 pipeline (AfterFlattenBeforeAutoSchedule) | |
| 5 | `hfusion-cache-io-for-return-arg` | 缓存 IO |
| 6 | `fold-tensor-empty` | 折叠 tensor.empty |
| 7 | 规范化 pipeline | |

#### 4.2.4 inferAndOutlineOp（`:265-286`）

| # | Pass 名称 | 说明 |
|---|-----------|------|
| 1 | `hfusion-fold-symbolic-dim` | 折叠符号维度 |
| 2 | `hfusion-infer-fusion-kind` | 推断融合类型 |
| 3 | `hfusion-fuse-ops` | ★ op 融合 |
| 4 | 规范化 pipeline | |
| 5 | `hfusion-outline-single-op` | 外联单个 op |
| 6 | 规范化 pipeline | |
| 7 | `hfusion-unfold-symbolic-dim` | 展开符号维度 |
| 8 | `hfusion-drop-symbols` | 删除符号 |
| 9 | `hfusion-eliminate-duplicate-funcs` | 消除重复函数 |

#### 4.2.5 postProcessOutlinedKernel（`:231-239`）

| # | Pass 名称 | 说明 |
|---|-----------|------|
| 1 | `hfusion-downgrade-fp64` | FP64 降级 |
| 2 | `trickle-concat-down` | concat 下传 |
| 3 | `bubble-pad-up` | pad 上浮 |
| 4 | `hfusion-legalize-bool` | bool 合法化 |
| 5 | `fold-tensor-empty` | 折叠 tensor.empty |
| 6 | `normalize-last-dim-unaligned-tensor-op` | 末维非对齐规范化 |

#### 4.2.6 hfusionAutoSchedulePipeline（`:305-344`）

| # | Pass 名称 | 说明 | 条件 |
|---|-----------|------|------|
| 1 | `hfusion-reorder-ops` | BFS 重排 | enableOpsReorder |
| 2 | 规范化 pipeline | | |
| 3 | `hfusion-decompose` (AFTER_HFUSION_FLATTEN) | 分解 | |
| 4 | **`hfusion-auto-schedule`** | ★★★ 核心自动调度 | 总是 |
| 5 | `hfusion-decompose-multi` | 多分解 | |
| 6 | `convert-generic-to-named-op` | 泛型→命名 | 非 RegBase |
| 7 | `hfusion-reorder-ops` | 再次重排 | enableOpsReorder |
| 8 | **hfusionTilingOptimizationPipeline** | 见下 | |
| 9 | `hfusion-wrap-host-func` | 包装 host 函数 | 非 multiKernel |

**hfusionTilingOptimizationPipeline**（`:289-303`）：
- `constantize-tiling-data` → 规范化 → `hfusion-pack-tiling-data` → `convert-arith-to-affine` → 规范化 → `scf-for-loop-canonicalization` → 规范化

#### 4.2.7 postProcess（`:346-371`）

| # | Pass 名称 | 说明 |
|---|-----------|------|
| 1 | `hfusion-inline-brc` | 内联 brc |
| 2 | `hfusion-normalize-ops` | Normalize |
| 3 | `hfusion-add-ffts-addr` | 添加 FFTS 地址 |
| 4 | `hfusion-hoist-tensor-empty` | 提升 tensor.empty |
| 5 | `hfusion-decompose` (AFTER_HFUSION_FLATTEN) | 分解 |

#### 4.2.8 hfusionAutoVectorizePipeline（`:389-495`）— RegBase 核心向量化

| # | Pass 名称 | 说明 | 条件 |
|---|-----------|------|------|
| 1 | 规范化 pipeline | | |
| 2 | `hfusion-fold-extract-insert-pair` | 折叠 extract-insert 对 | |
| 3 | `hivm-sink-op-to-consumer-in-loop` | op 下沉到消费者 | |
| 4 | **hfusionVectorizeManualScopePipeline** | 手动 scope 向量化 | |
| 5 | `vf-fusion` | ★ VF 融合 | enableSIMDVFFusion |
| 6 | `hfusion-flatten-ops` (registerBased=true) | ★ RegBase flatten | RegBase+flatten |
| 7 | 规范化 pipeline | | |
| 8 | `hivm-fuse-transpose-into-load` | transpose 融入 load | |
| 9 | `hfusion-pre-vectorization-fusion` | 预向量化融合 | |
| 10 | `prepare-i1nx1-for-vectorization` | i1nx1 准备 | |
| 11 | 规范化 pipeline | | |
| 12 | **`hfusion-auto-vectorize-v2`** | ★★★ 自动向量化 V2 | enableAutoVectorizeV2 |
| 13 | `outline-vector-function` | 外联向量函数 | enableAutoVectorizeV2 |
| 14 | **`hfusion-auto-vectorize`** | ★★★ 自动向量化 V1 | 非 enableAutoVectorizeV2 |
| 15 | `hfusion-auto-vectorize-verifier` | 向量化验证 | |
| 16 | `tree-reduce-v2` | 树形 reduce V2 | enableTreeReduce |
| 17 | `convert-hfusion-to-vector` | ★ HFusion→Vector | |
| 18 | `remove-mask-from-unaligned-reduction-loop` | 移除非对齐 mask | |
| 19 | `lower-vector-mask` | 降低 vector mask | |
| 20 | `inliner` | 内联 | enableSIMDVFFusion |
| 21 | `pull-slice-into-vector-function` | slice 拉入 VF | |
| 22 | `hfusion-simplify-vf-arg` | 简化 VF 参数 | |
| 23 | `loop-invariant-subset-hoisting` | 循环不变子集提升 | |
| 24 | 规范化 pipeline | | |
| 25 | `loop-invariant-promotion` | 循环不变提升 | enableLoopInvariantPromotion |
| 26 | `remove-redundant-write-and-read-pair` | 移除冗余读写对 | |
| 27 | `scf-for-loop-canonicalization` | for 循环规范化 | |
| 28 | 规范化 pipeline | | |

**hfusionVectorizeManualScopePipeline**（`:373-387`）：
- `outline-scope` (outlineMarkedScopesOnly) → `hfusion-vectorize-ops` (forManualScope) → `lower-vector-mask` → `hfusion-simplify-vf-arg` → 规范化 → `materialize-vector-write-to-destination`

---

### 4.3 阶段 C: HIVM RegBase Pipeline

#### 4.3.1 buildHIVMTensorOptimizations（`HIVMRegbasePipelines.cpp:685-690`）

| # | Pass 名称 | 说明 |
|---|-----------|------|
| 1 | `hivm-init-entry-kernel` | 初始化入口 kernel |
| 2 | `hivm-normalize-ops` | HIVM Normalize |
| 3 | **hivmPreBufferizationOptimizationPipeline** | 见 §4.3.2 |

#### 4.3.2 hivmPreBufferizationOptimizationPipeline（`:320-481`）

| # | Pass 名称 | 说明 | 条件 |
|---|-----------|------|------|
| 1 | `propagate-reshape` (forHIVM) | reshape 传播 | 非 RegBase |
| 2 | `remove-redundant-loop-init` | 移除冗余循环初始化 | |
| 3 | `hivm-normalize-matmul` | matmul 规范化 | |
| 4 | `hivm-normalize-convops` | conv 规范化 | |
| 5 | `hivm-insert-fixpipe` | 插入 fixpipe | |
| 6 | `hivm-inline-fixpipe` | 内联 fixpipe | |
| 7 | **hivmCVCommunicationPipeline** | CV 通信 pipeline | |
| 8 | `insert-cv-tight-coupled-buffer` (onlyInsert) | 插入紧耦合缓冲 | |
| 9 | `hivm-insert-convert-layout` | 插入布局转换 | layoutOpt |
| 10 | `hivm-propagate-convert-layout` | 布局转换传播 | layoutOpt |
| 11 | `hivm-combine-optimized-convert-layout` | 合并优化布局转换 | layoutOpt |
| 12 | `hivm-convert-layout-to-transpose` | 布局转 transpose | layoutOpt |
| 13 | `hivm-tile-batchmm-into-loop` | batchmm 分块到循环 | |
| 14 | `hivm-normalize-matmul` | 再次 matmul 规范化 | |
| 15 | `hivm-insert-l12ub-for-debug` | ★ A5 专用 L1/L2/UB 调试 | Ascend950 |
| 16 | `hivm-insert-nz2nd-for-debug` | 非 A5 NZ→ND 调试 | 非 Ascend950 |
| 17 | `hivm-insert-fixpipe` | 再次插入 fixpipe | |
| 18 | `hivm-inline-fixpipe` | 再次内联 fixpipe | |
| 19 | **hivmCVCommunicationPipeline** | 再次 CV 通信 | |
| 20 | `insert-cv-tight-coupled-buffer` (onlyInsert) | 再次紧耦合缓冲 | |
| 21 | `hivm-clone-tensor-empty` | 克隆 tensor.empty | |
| 22 | `insert-workspace-for-mix-cv` | 插入 mix-cv workspace | |
| 23 | `hivm-bind-workspace-arg` | 绑定 workspace 参数 | |
| 24 | `hivm-infer-func-core-type` | 推断函数核类型 | |
| 25 | `auto-blockify-parallel-loop` | 自动分块并行循环 | Triton+autoBlockify |
| 26 | `hivm-mark-tightly-coupled-buffer` | 标记紧耦合缓冲 | Skew 模式 |
| 27 | `hivm-mark-multi-buffer` | ★ 标记多缓冲 | |
| 28 | **canonicalizationHIVMPipeline** | HIVM 规范化 | |
| 29 | `hivm-inline-otf-broadcast` | 内联 OTF 广播 | |
| 30 | **`cv-pipelining`** | ★★★ CV 软件流水线 | MixedCV+非 Off |
| 31 | `hivm-mark-multi-buffer` | 再次标记多缓冲 | MixedCV+非 Off |
| 32 | `hivm-bind-sub-block` (partition-and-bind) | 分区绑定子块 | |
| 33 | `hivm-infer-vf-mode` | 推断 VF 模式 | |
| 34 | **`hivm-plan-memory-regbase`** | ★★★ RegBase 内存规划 | |
| 35 | `hivm-mark-tightly-coupled-buffer` | 标记紧耦合缓冲 | |
| 36 | `hivm-hoist-tightly-coupled-alloc` | 提升紧耦合分配 | |
| 37 | **Cross-Core Auto-Sync STEP 1** | 跨核自动同步（见下） | |
| 38 | `hivm-insert-infer-workspace-size-func` | 插入 workspace 大小推断 | Triton |
| 39 | `hivm-split-mix-kernel` | ★ 拆分 mix kernel | |
| 40 | `mark-simt-scope-no-inline` | 标记 SIMT scope 不内联 | |
| 41 | `inline-scope` | 内联 scope | |
| 42 | `hivm-bind-sub-block` (tile) | 分块绑定子块 | 非 skipHIVMBindSubBlock |
| 43 | **hivmWorkspacePipeline** | workspace pipeline | |
| 44 | `fold-tensor-empty` | 折叠 tensor.empty | |
| 45 | **canonicalizationHIVMPipeline** | 规范化 | |
| 46 | `loop-invariant-code-motion` | 循环不变代码外提 | enableCodeMotion |
| 47 | `loop-invariant-subset-hoisting` | 循环不变子集提升 | enableCodeMotion |
| 48 | **canonicalizationHIVMPipeline** | 规范化 | enableCodeMotion |
| 49 | `hfusion-simplify-vf-arg` | 简化 VF 参数 | |
| 50 | `hivm-clone-tensor-empty` | 克隆 tensor.empty | |
| 51 | `hivm-inline-otf-load-store` | 内联 OTF load/store | |

**hivmCVCommunicationPipeline**（`:78-96`）：
- Triton: `insert-load-store-for-mix-cv`
- A5 (Ascend950): `insert-cv-tight-coupled-buffer` + `insert-load-store-for-scalar`
- 其他: `insert-load-store-for-mix-cv` + `insert-load-store-for-scalar`

**Cross-Core Auto-Sync STEP 1**（`:127-219`）三选一：
- **INJ 模式**（非 crossCoreGSS）: `canonicalizationHIVMPipeline` → `hivm-mark-real-core-type` → `hivm-inject-block-sync` → `hivm-mark-real-core-type` (remove)
- **GSS 模式**（crossCoreGSS, 非 delayed）: `canonicalizationHIVMPipeline` → `hivm-mark-real-core-type` → `hivm-cross-core-gss` → `hivm-mark-real-core-type` (remove)
- **Delayed GSS 模式**（crossCoreGSS + delayed）: `canonicalizationHIVMPipeline` → `hivm-mark-real-core-type` → `hivm-cross-core-gss` (enableCVPatterns=false) → `insert-anchors-and-backup` → `hivm-mark-real-core-type` (remove)

**hivmWorkspacePipeline**（`:294-311`）：
- `hivm-bind-workspace-arg` (enableSubWorkspace) → `hivm-plan-memory-regbase` (GLOBAL_WORKSPACE_PLAN) → `hivm-insert-infer-workspace-size-func` (Triton) → `lower-memref-ext`

**canonicalizationHIVMPipeline**（`:50-61`）：
- `convert-arith-to-affine` → `scf-canonicalize-iter-arg` → `canonicalizer` → `scf-for-loop-canonicalization` → `cse` → `canonicalizer` → `hivm-opt-single-point` → `canonicalizer` → `memref-dead-store-elimination` → `canonicalizer`

#### 4.3.3 buildLowerHIVMPipelines（`HIVMRegbasePipelines.cpp:692-705`）

| # | Pass 名称 | 说明 |
|---|-----------|------|
| 1 | **bufferizationPipeline** | 见下 |
| 2 | `sub-block-guard-cleanup` | 子块 guard 清理 |
| 3 | **hivmPostBufferizationOptimizationPipeline** | 见 §4.3.5 |
| 4 | `inline-scope` (forceInline) | 强制内联 scope |
| 5 | `inject-ir` | 注入 IR |

**bufferizationPipeline**（`:222-276`）：

| # | Pass 名称 | 说明 | 条件 |
|---|-----------|------|------|
| 1 | `optimize-dps-op-with-yielded-insert-slice` | DPS op 优化 | Triton |
| 2 | `hivm-clone-tensor-empty` | 克隆 tensor.empty | Triton |
| 3 | `hivm-sink-op-to-consumer-in-loop` | op 下沉 | Triton |
| 4 | `hfusion-merge-vf` (level=1) | VF 合并 level 1 | vfMergeLevel==1 |
| 5 | `hfusion-simplify-vf-arg` | 简化 VF 参数 | |
| 6 | `hfusion-fold-extract-insert-pair` | 折叠 extract-insert | |
| 7 | `hivm-expose-memref-write-to-tensor` | 暴露 memref 写到 tensor | |
| 8 | `hivm-tensor-copy-insertion` | tensor copy 插入 | |
| 9 | `one-shot-bufferize` | ★ 一次性 bufferize | |
| 10 | `hfusion-merge-vf` (level=2) | VF 合并 level 2 | vfMergeLevel==2 |
| 11 | **canonicalizationHIVMPipeline** | 规范化 | |
| 12 | `convert-to-hivm-op` | →HIVM op | Triton |
| 13 | `drop-equivalent-buffer-results` | 丢弃等价缓冲结果 | |
| 14 | **canonicalizationHIVMPipeline** | 规范化 | |
| 15 | `drop-equivalent-buffer-results` | 再次丢弃 | |
| 16 | `convert-to-hivm-op` | →HIVM op | 非 Triton |

#### 4.3.4 buildConvertToHIVMPipeline（`HIVMRegbasePipelines.cpp:659-683`）

| # | Pass 名称 | 说明 | 条件 |
|---|-----------|------|------|
| 1 | `canonicalizer` | 规范化 | enableRegBaseHIVMPipe |
| 2 | `convert-hfusion-to-hivm` | HFusion→HIVM | |
| 3 | `triton-global-kernel-args-to-hivm` | Triton 参数转换 | Triton |
| 4 | `convert-tensor-to-hivm` | Tensor→HIVM | |
| 5 | `convert-to-hivm-op` | →HIVM op | |
| 6 | `propagate-reshape` (forHIVM) | reshape 传播 | 非 enableRegBaseHIVMPipe |

#### 4.3.5 hivmPostBufferizationOptimizationPipeline（`:513-657`）

| # | Pass 名称 | 说明 | 条件 |
|---|-----------|------|------|
| 1 | `hivm-lift-zero-rank` | 零秩提升 | |
| 2 | `map-for-to-forall` | for→forall 映射 | |
| 3 | `hivm-map-forall-to-blocks` | forall→blocks 映射 | |
| 4 | `hivm-decompose-op` | HIVM 分解 | |
| 5 | `hivm-sync-block-hoisting` | sync block 提升 | |
| 6 | `hivm-bind-sync-block-lock-arg` | 绑定 sync block lock | |
| 7 | `hivm-outline-alloc-in-VF` | ★ RegBase VF 分配外联 | RegBase |
| 8 | `hivm-outline-copy-in-VF` | ★ RegBase VF 拷贝外联 | RegBase |
| 9 | `drop-equivalent-buffer-results` | 丢弃等价缓冲 | RegBase |
| 10 | `inline-scope` | 内联 scope | RegBase |
| 11 | `bind-buffer` | 绑定缓冲 | |
| 12 | `hivm-infer-mem-scope` | 推断内存作用域 | |
| 13 | `hivm-decompose-op` | 再次分解 | |
| 14 | `hivm-sync-block-hoisting` | sync block 提升 | |
| 15 | `hivm-bind-sync-block-lock-arg` | 绑定 sync block lock | |
| 16 | `hivm-aggregated-decompose-op` (多阶段) | 聚合分解 | |
| 17 | `hivm-recognize-deinterleave-op` | 识别 deinterleave | 非 RegBase |
| 18 | `hivm-aggregated-decompose-op` | 聚合分解 | |
| 19 | **alignStoragePipeline** | 存储对齐 | |
| 20 | `hivm-aggregated-decompose-op` (AFTER_HIVM_STRIDE_ALIGNMENT) | 聚合分解 | |
| 21 | `canonicalizer` | 规范化 | |
| 22 | `hivm-infer-data-layout` | 推断数据布局 | |
| 23 | `hivm-aggregated-decompose-op` (AFTER_INFER_HIVM_DATA_LAYOUT) | 聚合分解 | |
| 24 | `canonicalizer` | 规范化 | |
| 25 | `hivm-auto-infer-buffer-size` | 自动推断缓冲大小 | |
| 26 | `hivm-set-buffer-size` | 设置缓冲大小 | |
| 27 | `hivm-flatten-ops` | HIVM flatten | |
| 28 | `hivm-aggregated-decompose-op` (AFTER_HIVM_FLATTEN_OPS) | 聚合分解 | |
| 29 | `hivm-reduce-rank-subview` | 降秩 subview | |
| 30 | `hivm-lift-lowest-stride` | 提升最低 stride | |
| 31 | `hivm-alloc-extra-buffer` | 分配额外缓冲 | |
| 32 | `hivm-outline-alloc-in-VF` | ★ RegBase VF 分配外联 | RegBase |
| 33 | `hivm-outline-copy-in-VF` | ★ RegBase VF 拷贝外联 | RegBase |
| 34 | `drop-equivalent-buffer-results` | 丢弃等价缓冲 | RegBase |
| 35 | `hivm-infer-mem-scope` | 推断内存作用域 | |
| 36 | **canonicalizationHIVMPipeline** | 规范化 | |
| 37 | `hivm-mark-multi-buffer` | 标记多缓冲 | |
| 38 | `hivm-vf-operand-substitution` | VF 操作数替换 | enableVFOperandSubstitution |
| 39 | **`hivm-plan-memory-regbase`** | ★★★ RegBase 内存规划 | |
| 40 | **Cross-Core Auto-Sync STEP 2** | 跨核自动同步第二步 | |
| 41 | `hivm-lower-to-loops` | HIVM 降低到循环 | |
| 42 | `hivm-decompose-op` | 分解 | |
| 43 | `hivm-sync-block-hoisting` | sync block 提升 | |
| 44 | `hivm-bind-sync-block-lock-arg` | 绑定 sync block lock | |
| 45 | `hivm-infer-mem-scope` | 推断内存作用域 | |
| 46 | `create-preload` | 创建 preload | enablePreload |
| 47 | **hivmIntraCoreSyncPipeline** | 核内同步 | |
| 48 | `lower-memref-ext` | 降低 memref ext | |
| 49 | `hivm-enable-multi-buffer` | 启用多缓冲 | |
| 50 | `hivm-lower-multi-buffer-counter` | 降低多缓冲计数器 | |
| 51 | `hivm-lift-lowest-stride` | 提升最低 stride | |
| 52 | **canonicalizationHIVMPipeline** | 规范化 | |
| 53 | `normalize-arith` | Arith 规范化 | RegBase+非 DirectHIVMLowering |
| 54 | `lift-arith-indexcast` | 提升 arith indexcast | RegBase+非 DirectHIVMLowering |
| 55 | `peel-loops-containing-transpose` | 剥离含 transpose 循环 | RegBase+非 DirectHIVMLowering |
| 56 | `canonicalizer` | 规范化 | RegBase+非 DirectHIVMLowering |
| 57 | `normalize-vector` | Vector 规范化 | RegBase+非 DirectHIVMLowering |
| 58 | `cse` | CSE | RegBase+非 DirectHIVMLowering |
| 59 | `arith-vector-mask-analyze` | Arith vector mask 分析 | RegBase+非 DirectHIVMLowering |

**alignStoragePipeline**（`:484-493`）：
- `hivm-align-alloc-size` → `hivm-pre-mark-stride-align` → `hivm-mark-stride-align` → `fold-alloc-reshape` → `hivm-enable-stride-align`

**hivmIntraCoreSyncPipeline**（`:99-115`）：
- GraphSyncSolver 模式: `hivm-graph-sync-solver`
- InjectSync 模式: `hivm-inject-sync`

**Cross-Core Auto-Sync STEP 2**（`:162-203`，Delayed GSS 模式）：
- `canonicalizationHIVMPipeline` → `hivm-mark-real-core-type` → `hivm-delayed-cross-core-gss` → `hivm-mark-real-core-type` (remove) → `insert-anchors-and-backup` (cleanup)

**syncBlockLockPipeline**（`:496-507`）：
- Prepare: `hivm-sync-block-hoisting` → `hivm-bind-sync-block-lock-arg`
- Finalize: `hivm-insert-infer-sync-block-lock-num-and-init-func` → `hivm-lower-create-sync-block-lock` → `hivm-mark-sync-block-lock-with-subblock` → `hivm-insert-free-lock-var-before-return`

---

### 4.4 阶段 D: `buildFinalHIVMPipelines`（`PassPipeline.cpp:341-352`）

| # | Pass 名称 | 说明 | 条件 |
|---|-----------|------|------|
| 1 | **buildDelayedHFusionRegBaseVectorizePipeline** | 延迟向量化 | SIMD/SIMT 混合 |
| 2 | **`hivm::regbase::buildLowerHIVMPipelines`** | 同 §4.3.3 | |

**buildDelayedHFusionRegBaseVectorizePipeline**（`PassPipeline.cpp:301-339`）：
1. `hivm-aggregated-decompose-op` (NO_CONSTRAINT)
2. `convert-hivm-to-hfusion` — HIVM→HFusion 回转
3. **`hfusion::regbase::buildHFusionRegBasePipeline`** — 见下
4. `hivm-infer-func-core-type`
5. `convert-hfusion-to-hivm`
6. `convert-tensor-to-hivm`

**buildHFusionRegBasePipeline**（`HFusionRegbasePipelines.cpp:562-595`）：
1. (flatten) `propagate-reshape` (forRegbased) → `fold-tensor-empty` → `canonicalize-tensor-reshape` → 规范化 → `hfusion-flatten-ops` (registerBased=true) → 规范化
2. `fold-tensor-empty`
3. `hfusion-normalize-ops`
4. `hfusion-inline-brc`
5. **hfusionAutoVectorizePipeline**（同 §4.2.8）

---

### 4.5 阶段 E: `buildBiShengHIRAVEToLLVMPipeline`（`PassPipeline.cpp:220-236`）

| # | Pass 名称 | 说明 | 条件 |
|---|-----------|------|------|
| 1 | **buildLowerHACCToLLVMPipeline** | HACC→LLVM | compileHost |
| 2 | **`hivmave::buildLowerAVEPipelines`** | AVE 降低 | enableHIVMCompile |
| 3 | **buildLowerToLLVMPipeline** | →LLVM | lowerToLLVM |

**buildLowerHACCToLLVMPipeline**（`HACCPipelines.cpp:29-35`）：
- `parameter-packing` → `convert-hacc-to-llvm`

---

### 4.6 阶段 F: `buildLowerAVEPipelines`（HIVMAVE，`HIVMAVEPipelines.cpp:85-88`）

**条件**：`!enableDirectHIVMLowering && isRegBasedArch(target)` 时执行

| # | Pass 名称 | 说明 |
|---|-----------|------|
| 1 | `convert-vector-to-hivmave` | Vector→HIVMAVE |
| 2 | `convert-arith-to-hivmave` | Arith→HIVMAVE |
| 3 | `ave-i1op-soft-impl` | i1 op 软实现 |
| 4 | `hivmave-complex-reduction-intermediate-lowering` | 复数 reduction 中间降低 |
| 5 | `canonicalizer` | 规范化 |
| 6 | `ave-process-vsstb` | vsstb 处理 |
| 7 | `optimize-reduction-loop` | reduction 循环优化 |
| 8 | `ave-loop-optimize` | AVE 循环优化 | enableAveLoopOptimize |
| 9 | `legalize-opt-hivmave` | HIVMAVE 合法化 |
| 10 | `hivmave-replace-with-vector-scalar` | vector-scalar 替换 |
| 11 | `ave-process-membar` | membar 处理 |
| 12 | `annotation-lowering` | annotation 降低 |
| 13 | `combine-ave-ops` | AVE op 合并 |
| 14 | `hivmave-scalar-broadcast-to-vload` | 标量广播→vload |
| 15 | `ave-plt-to-pge` | PLT→PGE |
| 16 | `ave-plt-to-pltm` | PLT→PLTM |
| 17 | `hivm-legalize-loop-iter-arg` | 循环 iter arg 合法化 |
| 18 | `duplicate-unit-mask-broadcast` | 复制 unit mask 广播 |
| 19 | `analyze-vector-layout` | 分析 vector 布局 |
| 20 | `canonicalizer` | 规范化 |
| 21 | `ave-normalize-ops` | AVE Normalize |
| 22 | `remove-vector-layout-attr` | 移除 vector 布局属性 |

---

### 4.7 阶段 G: `buildLowerToLLVMPipeline`（`PassPipeline.cpp:239-299`）

| # | Pass 名称 | 说明 | 条件 |
|---|-----------|------|------|
| 1 | `annotation-lowering` | annotation 降低 | |
| 2 | `hivm-memref-alloc-to-alloca` | memref alloc→alloca | |
| 3 | `hivm-insert-init-and-finish-for-debug` | 插入调试 init/finish | nest<FuncOp> |
| 4 | `hivm-mark-disable-load` | 标记禁用 load | |
| 5 | **addSyncBlockLockFinalizePasses** | sync block lock 终结 | |
| 6 | `convert-hivm-to-std` | ★ HIVM→Standard | |
| 7 | `convert-hivmave-to-std` | HIVMAVE→Standard | |
| 8 | `fix-call-unknown-loc` | 修复未知调用位置 | nest<FuncOp> |
| 9 | `expand-strided-metadata` | 展开 strided metadata | |
| 10 | `convert-hivmave-to-ave-intrin` | HIVMAVE→AVE 内联函数 | |
| 11 | `hoist-vstas` | 提升 vstas | |
| 12 | `adapt-gpu-kernel` | 适配 GPU kernel | PureSimt+DPX |
| 13 | `hoist-call-scalar-to-caller` | 提升标量调用 | PureSimt+DPX |
| 14 | `dpx-div-optimization` | DPX 除法优化 | PureSimt+DPX+SIMTFastDiv |
| 15 | `convert-ascend-dpx-to-hivmregbaseintrins` | ★ RegBase 专用 DPX 转换 | |
| 16 | `decompose-frem` | 分解 frem | |
| 17 | `convert-scf-to-cf` | SCF→CF | |
| 18 | `lower-affine` | 降低 affine | |
| 19 | `arith-expand-ops` | Arith 展开 | |

**addSyncBlockLockFinalizePasses**（`HIVMRegbasePipelines.cpp:509-511`）：
- `hivm-insert-infer-sync-block-lock-num-and-init-func` → `hivm-lower-create-sync-block-lock` → `hivm-mark-sync-block-lock-with-subblock` → `hivm-insert-free-lock-var-before-return`

---

### 4.8 阶段 H: `buildBiShengTTIRPipeline`（SIMT/Triton 路径，`PassPipeline.cpp:396-456`）

| # | Pass 名称 | 说明 | 条件 |
|---|-----------|------|------|
| 1 | `simt-vf-sub-tiling` | SIMT VF 子分块 | 混合+enableSimtVFSubTiling |
| 2 | `materialize-simt-vf-mem-scope` | 物化 SIMT VF 内存作用域 | 混合 |
| 3 | `convert-hivm-to-tritongpu` | HIVM→TritonGPU | 混合 |
| 4 | `hacc-append-device-spec` | 追加设备规格 | 非 host |
| 5 | `canonicalize-module` | 模块规范化 | |
| 6 | **buildLowerTritonPipeline** | ★ 见 §4.9 | |
| 7 | `cse` | CSE | |
| 8 | `scp` | SCCP | |
| 9 | `triton-remap` | Triton 重映射 | 非 DPX |
| 10 | `canonicalizer` | 规范化 | |
| 11 | `cse` | CSE | |
| 12 | `canonicalizer` | 规范化 | |
| 13 | `convert-scf-to-cf` | SCF→CF | |
| 14 | `convert-control-flow-to-llvm` | CF→LLVM | |
| 15 | `convert-arith-to-llvm` | Arith→LLVM | |
| 16 | **buildLowerToLLVMPipeline** | 同 §4.7（simtConfig） | |

---

### 4.9 阶段 I: `buildLowerTritonPipeline`（`TritonPipelines.cpp:99-225`）

| # | Pass 名称 | 说明 | 条件 |
|---|-----------|------|------|
| 1 | `convert-non-power-two-tensors` | 非二次幂 tensor 转换 | |
| 2 | `set-bishengir-simt-opt-attr` | 设置 SIMT 优化属性 | |
| 3 | `set-allow-global-scratch-attr` | 设置全局 scratch 属性 | |
| 4 | `adapt-triton-ir-kernel` | 适配 Triton IR kernel | |
| 5 | `simt-auto-blockify` | SIMT 自动分块 | enableSIMTAutoBlockify |
| 6 | `optimize-loads` | 优化 loads | |
| 7 | `loop-restructure-arange-optimization` | arange 循环重构优化 | |
| 8 | `tile-dot-loads` | ★ dot loads 分块 | |
| 9 | `enable-ascend-dpx-mma` | 启用 DPX MMA | |
| 10 | `hoist-and-fuse-dot-chains` | 提升+融合 dot 链 | |
| 11 | `optimize-math` | 数学优化 | enableOptimizeMath |
| 12 | `enable-ascend-dpx-mma` | 再次启用 DPX MMA | |
| 13 | `legalize-f16-for-triton` | F16 合法化 | |
| 14 | `cse` | CSE | |
| 15 | `scp` | SCCP | |
| 16 | `canonicalizer` | 规范化 | |
| 17 | `fix-fused-cat` | 修复融合 cat | |
| 18 | `triton-rewrite-tensor-pointer` | 重写 tensor pointer | |
| 19 | `triton-rewrite-tensor-descriptor-to-pointer` | 重写 tensor descriptor | |
| 20 | `rewrite-slice-op-to-triton` | slice→Triton 重写 | |
| 21 | `expand-gather-op-sources` | 扩展 gather 源 | numWarps>1 |
| 22 | `remove-annotation-mark` | 移除 annotation mark | |
| 23 | **`convert-triton-to-triton-gpu`** | ★★★ TTIR→TTGIR | |
| 24 | **buildTritonGPUOptimizationPipeline** | ★ 见 §4.9.1 | |
| 25 | `lower-dot-buffers-and-shared-mem` | ★ 降低 dot buffers+共享内存 | |
| 26 | `simt-fast-div` | SIMT 快速除法 | SIMTFastDiv+DPX |
| 27 | `convert-scf-to-cf` | SCF→CF | |
| 28 | `allocate-ascend-shared-memory` | 分配 Ascend 共享内存 | |
| 29 | `triton-gpu-global-scratch-allocation` | 全局 scratch 分配 | enableGlobalScratchAllocation |
| 30 | `populate-shared-memory-offset-to-dpx` | 共享内存偏移→DPX | SIMTFastDiv+DPX |
| 31 | `convert-index-to-llvm` | Index→LLVM | |
| 32 | `convert-proton-to-proton-gpu` | Proton→ProtonGPU | |
| 33 | `cse` | CSE | |
| 34 | `allocate-proton-shared-memory` | 分配 Proton 共享内存 | |
| 35 | **`convert-triton-ascend-gpu-to-llvm`** | ★ TritonAscendGPU→LLVM | |
| 36 | `cse` | CSE | enableSinkDPXLoad |
| 37 | `sink-dpx-load` | DPX load 下沉 | enableSinkDPXLoad |
| 38 | `flatten-memdesc-args` | 展平 memdesc 参数 | |
| 39 | `cse` | CSE | |
| 40 | `allocate-proton-ascend-global-scratch-buffer` | 分配 Proton 全局 scratch | |
| 41 | `convert-proton-ascend-gpu-to-llvm` | ProtonAscendGPU→LLVM | |
| 42 | **`convert-triton-gpu-to-llvm`** | ★★★ TritonGPU→LLVM | |
| 43 | `canonicalizer` | 规范化 | |
| 44 | `get-triton-metadata` | 获取 Triton 元数据 | |
| 45 | `convert-arith-to-llvm` | Arith→LLVM | |

#### 4.9.1 buildTritonGPUOptimizationPipeline（`TritonPipelines.cpp:53-89`）

| # | Pass 名称 | 说明 |
|---|-----------|------|
| 1 | `triton-gpu-coalesce` | TritonGPU 合并 |
| 2 | `convert-dot-input-to-linear-layout` | dot 输入→线性布局 |
| 3 | `triton-gpu-remove-layout-conversions` | 移除布局转换 |
| 4 | `triton-gpu-optimize-thread-locality` | 线程局部性优化 |
| 5 | `triton-gpu-remove-layout-conversions` | 再次移除布局转换 |
| 6 | `triton-loop-aware-cse` | 循环感知 CSE |
| 7 | `triton-gpu-fuse-nested-loops` | 嵌套循环融合 |
| 8 | `canonicalizer` | 规范化 |
| 9 | `triton-loop-invariant-code-motion` | 循环不变代码外提 |
| 10 | `canonicalizer` | 规范化 |
| 11 | `triton-gpu-combine-tensor-select-and-if` | tensor.select+if 合并 |
| 12 | `triton-gpu-schedule-loops` | 循环调度 |
| 13 | `canonicalizer` | 规范化 |
| 14 | `triton-loop-aware-cse` | 循环感知 CSE |
| 15 | `triton-gpu-remove-layout-conversions` | 移除布局转换 |
| 16 | `triton-gpu-reduce-data-duplication` | 减少数据重复 |
| 17 | `triton-gpu-reorder-instructions` | 指令重排 | 非 disableReorderInstruction |
| 18 | `decompose-reduction` | reduction 分解 | 非 disableDecomposeReduction |
| 19 | `optimize-layouts` | 布局优化 |
| 20 | `triton-loop-aware-cse` | 循环感知 CSE |
| 21 | `symbol-dce` | 符号死代码消除 |
| 22 | `scp` | SCCP |
| 23 | `cse` | CSE |
| 24 | `canonicalizer` | 规范化 |

---

### 4.10 阶段 J: `buildBiShengHIRFinishPipeline`（`PassPipeline.cpp:459-462`）

| # | Pass 名称 | 说明 |
|---|-----------|------|
| 1 | `write-back-shared` | 写回共享 |

---

### 4.11 阶段 K: 外部工具 `hivmc-a5`

**代码位置**：`BiShengIRCompileMain.cpp:159-242`（`runExternalHIVMC`）

调用 `hivmc-a5` 二进制（`getHIVMCName()` 返回 `"hivmc-a5"`，`:155`），完成 LLVM IR → 二进制。

A5 特有 bitcode 加载（`:95-111`）：
- `meta_op.aic.c310.bc` / `meta_op.aiv.c310.bc`（Ascend950）
- `meta_op.mix.aic.c310.bc` / `meta_op.mix.aiv.c310.bc`（Ascend950）
- `host.bc`

---

## 5. Pass 注册/组装关键代码位置

### 5.1 PassRegistration（pass 注册）

| 位置 | 说明 |
|------|------|
| `bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp:599-601` | `registerBiShengIRCompilePass()` 注册 `bishengir-compile-regbase` pass |
| `bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp:548` | pass argument: `"bishengir-compile-regbase"` |
| `bishengir/lib/Tools/bishengir-compile/PassPipeline.cpp:248` | 非 regbase 版本 `bishengir-compile` pass 注册 |

### 5.2 PassPipelineRegistration（pipeline 注册）

| 位置 | pipeline 名称 | 说明 |
|------|---------------|------|
| `bishengir/lib/Dialect/HFusion/Pipelines/regbase/HFusionRegbasePipelines.cpp:605-611` | `lower-hfusion-regbase-pipeline` | HFusion RegBase pipeline |
| `bishengir/lib/Dialect/HIVM/Pipelines/regbase/HIVMRegbasePipelines.cpp:711-718` | `lower-hivm-pipeline` | HIVM pipeline |
| `bishengir/lib/Dialect/HIVM/Pipelines/regbase/HIVMRegbasePipelines.cpp:720-726` | `convert-to-hivm-pipeline` | ConvertToHIVM pipeline |
| `bishengir/lib/Dialect/HIVMAVE/Pipelines/HIVMAVEPipelines.cpp:94-100` | `lower-ave-pipeline` | AVE pipeline |
| `bishengir/lib/Dialect/Triton/Pipelines/TritonPipelines.cpp:232-240` | `lower-triton-pipeline` | Triton pipeline |
| `bishengir/lib/Dialect/HFusion/Pipelines/HFusionPipelines.cpp:326` | `lower-hfusion-pipeline` | 非 regbase HFusion pipeline |
| `bishengir/lib/Dialect/HIVM/Pipelines/HIVMPipelines.cpp:552` | `lower-hivm-pipeline` | 非 regbase HIVM pipeline |
| `bishengir/lib/Dialect/HIVM/Pipelines/ConvertToHIVMPipeline.cpp:51` | `convert-to-hivm-pipeline` | 非 regbase ConvertToHIVM |
| `bishengir/lib/ExecutionEngine/ExecutionEnginePipelines.cpp:97` | CPURunner pipeline | CPU runner |

### 5.3 Pipeline 组装函数

| 位置 | 函数 | 说明 |
|------|------|------|
| `bishengir/lib/Tools/bishengir-compile/regbase/BiShengIRCompileMain.cpp:325-500` | `runRegBasePipeline` | ★ 主编排（重试 6 次，fallback 逻辑） |
| `bishengir/lib/Tools/bishengir-compile/regbase/BiShengIRCompileMain.cpp:297-321` | `runMixedPipelines` | Mixed 编译流 |
| `bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp:464-535` | `buildBiShengHIRPipeline` | HIR 前端 pipeline |
| `bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp:396-456` | `buildBiShengTTIRPipeline` | Triton/SIMT pipeline |
| `bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp:220-236` | `buildBiShengHIRAVEToLLVMPipeline` | AVE→LLVM pipeline |
| `bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp:239-299` | `buildLowerToLLVMPipeline` | →LLVM pipeline |
| `bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp:341-352` | `buildFinalHIVMPipelines` | 最终 HIVM pipeline |
| `bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp:301-339` | `buildDelayedHFusionRegBaseVectorizePipeline` | 延迟向量化 |
| `bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp:459-462` | `buildBiShengHIRFinishPipeline` | 完成 pipeline |
| `bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp:86-97` | `setupHFusionPipelineOptions` | HFusion 选项设置 |
| `bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp:99-149` | `setupHIVMPipelineOptions` | HIVM 选项设置 |
| `bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp:151-218` | `setupHIVMAVEPipelineOptions` | HIVMAVE 选项设置 |
| `bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp:356-394` | `setupLowerTritonPipelineOptions` | Triton 选项设置 |
| `bishengir/lib/Dialect/HFusion/Pipelines/regbase/HFusionRegbasePipelines.cpp:497-560` | `hfusion::regbase::buildHFusionPipelines` | HFusion RegBase pipeline |
| `bishengir/lib/Dialect/HFusion/Pipelines/regbase/HFusionRegbasePipelines.cpp:562-595` | `hfusion::regbase::buildHFusionRegBasePipeline` | HFusion RegBase 向量化 pipeline |
| `bishengir/lib/Dialect/HIVM/Pipelines/regbase/HIVMRegbasePipelines.cpp:659-683` | `hivm::regbase::buildConvertToHIVMPipeline` | ConvertToHIVM pipeline |
| `bishengir/lib/Dialect/HIVM/Pipelines/regbase/HIVMRegbasePipelines.cpp:685-690` | `hivm::regbase::buildHIVMTensorOptimizations` | HIVM tensor 优化 |
| `bishengir/lib/Dialect/HIVM/Pipelines/regbase/HIVMRegbasePipelines.cpp:692-705` | `hivm::regbase::buildLowerHIVMPipelines` | HIVM lowering pipeline |
| `bishengir/lib/Dialect/HIVMAVE/Pipelines/HIVMAVEPipelines.cpp:85-88` | `hivmave::buildLowerAVEPipelines` | AVE lowering pipeline |
| `bishengir/lib/Dialect/Triton/Pipelines/TritonPipelines.cpp:99-225` | `bishengir::triton::buildLowerTritonPipeline` | Triton lowering pipeline |
| `bishengir/lib/Dialect/Triton/Pipelines/TritonPipelines.cpp:53-89` | `buildTritonGPUOptimizationPipeline` | TritonGPU 优化 pipeline |
| `bishengir/lib/Dialect/HACC/Pipelines/HACCPipelines.cpp:29-35` | `hacc::buildLowerHACCToLLVMPipeline` | HACC→LLVM pipeline |

### 5.4 架构判定与配置

| 位置 | 说明 |
|------|------|
| `bishengir/lib/Dialect/HACC/Utils/Utils.cpp:339-341` | `isRegBasedArch` = `isAscend310B \|\| isAscend950` |
| `bishengir/lib/Dialect/HACC/Utils/Utils.cpp:305-333` | `isAscend950` — 列出所有 Ascend950 系列设备 |
| `bishengir/lib/Tools/bishengir-compile/BiShengIRCompileConfig.cpp:372-386` | `applyArchDependentCompileDefaults` — A5 默认参数 |
| `bishengir/lib/Tools/bishengir-compile/BiShengIRCompileConfig.cpp:395-414` | `createFromCLOptions(bool regbase)` — regbase 配置创建 |
| `bishengir/tools/bishengir-compile/bishengir-compile.cpp:103-191` | 主入口 — regbase 目标调用 `runRegBaseCompile` |
| `bishengir/lib/Tools/bishengir-compile/regbase/Driver.cpp:61-85` | `runRegBaseCompile` — 驱动入口 |

### 5.5 A5 默认参数（`applyArchDependentCompileDefaults`）

| 参数 | A5 默认值 | 说明 |
|------|-----------|------|
| `set-workspace-multibuffer` | 2 | 双缓冲（A3 为 4） |
| `limit-auto-multi-buffer-buffer` | NO_LIMIT | 不限制（A3 为 only-cube） |
| `enable-hivm-unit-flag-sync` | true | 启用 unit flag sync |
| `enable-preload` | true | 启用 preload |
| `enable-lib-call-no-inline` | true | 启用 lib call no inline |

---

## 6. Pass 定义文件（TableGen .td）

| 文件绝对路径 | Pass 数 | 说明 |
|------|---------|------|
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/Transforms/Passes.td` | ~50 | HFusion 所有 pass |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td` | ~90 | HIVM 所有 pass（含 `hivm-plan-memory-regbase`, `cv-pipelining` 等 RegBase 专用） |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVMAVE/Transforms/Passes.td` | ~17 | HIVMAVE pass |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/Triton/Transforms/Passes.td` | ~30 | Triton 扩展 pass |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Conversion/Passes.td` | ~25 | 跨方言转换 pass（含 `convert-ascend-dpx-to-hivmregbaseintrins`） |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/HACC/Transforms/Passes.td` | 2 | HACC pass |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/Scope/Transforms/Passes.td` | 3 | Scope pass |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/Tensor/Transforms/Passes.td` | ~13 | Tensor pass |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/Vector/Transforms/Passes.td` | 4 | Vector pass |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/Arith/Transforms/Passes.td` | 3 | Arith pass |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/Symbol/Transforms/Passes.td` | 5 | Symbol pass |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/AscendDPX/Transforms/Passes.td` | 2 | DPX pass |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/Annotation/Transforms/Passes.td` | 1 | Annotation pass |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/Analysis/VFFusion/Passes.td` | 1 | VFFusion pass |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Transforms/Passes.td` | ~8 | 通用 Transforms pass |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/ExecutionEngine/Passes.td` | 4 | EE pass |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/MemRef/Transforms/Passes.td` | 3 | MemRef pass |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/SCF/Transforms/Passes.td` | 4 | SCF pass |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/LLVMIR/Transforms/Passes.td` | 1 | LLVMIR pass |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Tools/bishengir-compile/Options.td` | ~80+ | 编译选项定义（非 pass，但控制 pipeline 行为） |

### 6.1 各 Dialect Pass 详细列表

#### HFusion Passes（`Passes.td`）
```
hfusion-fuse-ops, hfusion-auto-schedule, hfusion-auto-vectorize, hfusion-generic-unroller,
hfusion-auto-vectorize-verifier, hfusion-pre-vectorization-fusion, hfusion-auto-vectorize-v2,
tree-reduce-v2, hfusion-add-ffts-addr, outline-vector-function, convert-generic-to-named-op,
hfusion-flatten-ops, hfusion-inline-brc, hfusion-outline-single-op, hfusion-simplify-ops,
hfusion-normalize-ops, hfusion-normalize-slice-ops, hfusion-pack-tiling-data, constantize-tiling-data,
infer-fusion-kind, adapt-triton-kernel, hfusion-legalize-scalar, hfusion-legalize-bf16,
hfusion-legalize-fp8, hfusion-legalize-bool, hfusion-reorder-ops, hfusion-downgrade-fp64,
hfusion-infer-out-shapes, hfusion-compose-multi-reduce, hfusion-decompose-multi, hfusion-cache-io,
hfusion-cache-io-for-return-arg, hfusion-hoist-tensor-empty, hfusion-wrap-host-func,
hfusion-fold-symbolic-dim, hfusion-unfold-symbolic-dim, hfusion-drop-symbols,
hfusion-eliminate-duplicate-funcs, hfusion-decompose, hfusion-fold-unit-dims,
hfusion-uplift-while-to-for, loop-invariant-promotion, hfusion-generalize,
hfusion-fold-extract-insert-pair, prepare-i1nx1-for-vectorization, hfusion-simplify-vf-arg,
hfusion-merge-vf, pull-slice-into-vector-function, hfusion-vectorize-ops,
remove-mask-from-unaligned-reduction-loop, remove-redundant-write-and-read-pair
```

#### HIVM Passes（`Passes.td`）— 含 RegBase 专用
```
hivm-infer-func-core-type, convert-to-hivm-op, enable-hivmc-compatible-print,
hivm-normalize-matmul, hivm-normalize-convops, hivm-normalize-bitwise-select,
hivm-normalize-ops, triton-global-kernel-args-to-hivm, hivm-infer-mem-scope,
hivm-mark-multi-buffer, hivm-enable-multi-buffer, hivm-add-ffts-to-syncblocksetop,
hivm-lower-multi-buffer-counter, hivm-memref-alloc-to-alloca, hivm-clone-tensor-empty,
hivm-infer-data-layout, hivm-expose-memref-write-to-tensor, hivm-infer-vf-mode,
hivm-vf-operand-substitution, hivm-plan-memory, hivm-plan-memory-regbase (★RegBase),
hivm-inject-sync, hivm-inject-block-sync, hivm-graph-sync-solver, hivm-cross-core-gss,
hivm-delayed-cross-core-gss, insert-anchors-and-backup, hivm-decompose-op,
hivm-aggregated-decompose-op, hivm-lower-to-loops, hivm-recognize-deinterleave-op,
hivm-recognize-discontinuous-store, hivm-opt-single-point, hivm-alloc-extra-buffer,
hivm-outline-alloc-in-VF (★RegBase), hivm-outline-copy-in-VF (★RegBase),
hivm-auto-infer-buffer-size, hivm-opt-func-output, hivm-insert-infer-task-type-func,
hivm-insert-vf-mode-func, hivm-mark-tightly-coupled-buffer, hivm-hoist-tightly-coupled-alloc,
hivm-split-mix-kernel, hivm-mark-real-core-type, hivm-set-buffer-size,
hivm-map-forall-to-blocks, hivm-flatten-ops, hivm-sink-op-to-consumer-in-loop,
hivm-align-alloc-size, hivm-pre-mark-stride-align, hivm-mark-stride-align,
hivm-enable-stride-align, hivm-lift-lowest-stride, hivm-reduce-rank-subview,
hivm-inline-otf-broadcast, hivm-inline-load-copy, hivm-init-entry-kernel,
hivm-inline-fixpipe, hivm-insert-fixpipe, hivm-tile-batchmm-into-loop, hivm-lift-zero-rank,
insert-load-store-for-mix-cv, insert-cv-tight-coupled-buffer, insert-load-store-for-scalar,
hivm-insert-infer-workspace-size-func, hivm-bind-workspace-arg, hivm-bind-sync-block-lock-arg,
hivm-sync-block-hoisting, hivm-insert-infer-sync-block-lock-num-and-init-func,
hivm-lower-create-sync-block-lock, hivm-insert-free-lock-var-before-return,
hivm-auto-infer-buffer-size, insert-workspace-for-mix-cv, hivm-inline-otf-load-store,
arith-vector-mask-analyze, hivm-annotate-vf-alias, hivm-bind-sub-block,
partition-and-bind-sub-block, cv-pipelining (★RegBase), auto-blockify-parallel-loop,
compose-collapse-expand, infer-simt-vf-memory-effect, infer-simt-vf-mem-scope-hint,
materialize-simt-vf-mem-scope, simt-vf-sub-tiling, split-simt-module,
mark-simt-scope-no-inline, auto-scope, insert-alloc-base-placeholder, write-back-shared,
tile-cube-vector-loop, convert-non-contiguous-reshape-to-copy, hivm-insert-convert-layout,
hivm-propagate-convert-layout, legalize-bool-for-simtvf, insert-memory-semantic-for-simtvf,
hivm-convert-layout-to-transpose, hivm-combine-optimized-convert-layout, create-preload,
hivm-remove-layout-annotation, hivm-remove-copy-ops, hivm-fuse-transpose-into-load,
hivm-tensor-copy-insertion, hivm-insert-init-and-finish-for-debug, hivm-mark-disable-load,
hivm-mark-sync-block-lock-with-subblock, hivm-insert-nz2nd-for-debug,
hivm-insert-l12ub-for-debug (★A5)
```

#### HIVMAVE Passes（`Passes.td`）
```
ave-normalize-ops, hivmave-replace-with-vector-scalar, ave-plt-to-pge, ave-plt-to-pltm,
ave-process-vsstb, ave-loop-optimize, ave-i1op-soft-impl, ave-process-membar,
combine-ave-ops, legalize-opt-hivmave, optimize-reduction-loop,
hivmave-scalar-broadcast-to-vload, hoist-vstas, duplicate-unit-mask-broadcast,
analyze-vector-layout, remove-vector-layout-attr,
hivmave-complex-reduction-intermediate-lowering
```

#### Triton Passes（`Passes.td`）
```
decompose-frem, convert-non-power-two-tensors, set-bishengir-simt-opt-attr,
set-allow-global-scratch-attr, adapt-triton-ir-kernel, simt-auto-blockify,
enable-ascend-dpx-mma, optimize-loads, loop-restructure-arange-optimization,
decompose-reduction, legalize-f16-for-triton, optimize-layouts,
convert-dot-input-to-linear-layout, tile-dot-loads, optimize-math,
hoist-and-fuse-dot-chains, dump-fractal-layout, fix-fused-cat,
rewrite-slice-op-to-triton, expand-gather-op-sources,
populate-shared-memory-offset-to-dpx, simt-fast-div,
lower-dot-buffers-and-shared-mem, flatten-memdesc-args,
get-triton-metadata, adapt-gpu-kernel, triton-remap, remove-annotation-mark
```

#### Conversion Passes（`Passes.td`）
```
convert-arith-to-affine, convert-arith-to-hfusion, convert-math-to-hfusion,
convert-linalg-to-hfusion, convert-gpu-to-hfusion, convert-arith-to-hivm-llvm,
convert-hfusion-to-hivm, convert-hivm-to-std, convert-hivm-to-tritongpu,
convert-ascend-dpx-to-hivmregbaseintrins (★RegBase), convert-tensor-to-hfusion,
convert-tensor-to-hivm, lower-memref-ext, convert-hfusion-to-vector,
convert-triton-ascend-gpu-to-llvm, convert-proton-ascend-gpu-to-llvm,
allocate-proton-ascend-global-scratch-buffer, convert-hacc-to-llvm,
convert-vector-to-hivmave,
convert-hivmave-to-std, fix-call-unknown-loc, convert-arith-to-hivmave,
convert-hivmave-to-ave-intrin
```

---

## 7. RegBase(A5) 特有 Pass 与差异点

相比 membase（A2/A3）pipeline，RegBase(A5) 的关键差异：

### 7.1 RegBase 专用 Pass

| Pass | 说明 | 代码位置 |
|------|------|----------|
| `hivm-plan-memory-regbase` | 寄存器层感知的内存规划，替代 `hivm-plan-memory` | `HIVMRegbasePipelines.cpp:432,306,623` |
| `cv-pipelining` | Cube/Vector 软件流水线（A5 默认 Skew/preload 模式） | `HIVMRegbasePipelines.cpp:417` |
| `convert-ascend-dpx-to-hivmregbaseintrins` | DPX→RegBase 内联函数转换 | `PassPipeline.cpp:294` |
| `hivm-insert-l12ub-for-debug` | A5 专用 L1/L2/UB 调试插入（非 A5 用 `hivm-insert-nz2nd-for-debug`） | `HIVMRegbasePipelines.cpp:356` |
| `insert-cv-tight-coupled-buffer` + `insert-load-store-for-scalar` | A5 CV 通信路径 | `HIVMRegbasePipelines.cpp:88-90` |
| `hivm-outline-alloc-in-VF` | RegBase 专用 VF 分配外联 | `HIVMRegbasePipelines.cpp:524,592` |
| `hivm-outline-copy-in-VF` | RegBase 专用 VF 拷贝外联 | `HIVMRegbasePipelines.cpp:526,593` |
| `hfusion-flatten-ops` (registerBased=true) | RegBase 专用 flatten | `HFusionRegbasePipelines.cpp:420,540,581` |
| `propagate-reshape` (forRegbased=true) | RegBase 专用 reshape 传播 | `HFusionRegbasePipelines.cpp:528,570` |

### 7.2 RegBase 专用 Normalize 框架

HFusion 和 HIVM 各有一套完整的 Normalize 框架，位于 `regbase/Normalize/` 目录：

**HFusion Normalize**（11 个子模块）：
- NormalizeArithmetic — 算术运算规范化
- NormalizeAtomic — 原子操作规范化
- NormalizeCasting — 类型转换规范化
- NormalizeComparison — 比较运算规范化
- NormalizeMath — 数学运算规范化
- NormalizeReduction — 规约运算规范化
- NormalizeScalar — 标量运算规范化
- NormalizeTrig — 三角函数规范化
- NormalizeTypeConversion — 类型转换规范化
- NormalizeTraitsBase — 特征基类
- Normalize — 主入口

**HIVM Normalize**（9 个子模块）：
- 与 HFusion 类似，额外包含 NormalizeRelu

**模板化框架**（`include/bishengir/Transforms/regbase/Normalize/`）：
- `NormalizeArithmeticTemplate.h` / `NormalizeAtomicTemplate.h` / `NormalizeCastingTemplate.h`
- `NormalizeComparisonTemplate.h` / `NormalizeMathTemplate.h` / `NormalizeReductionTemplate.h`
- `NormalizeScalarTemplate.h` / `NormalizeTrigTemplate.h` / `NormalizeTypeConversionTemplate.h`
- `Utils/` 下有 `CastingTemplateHelpers.h`, `Kinds.h`, `MathTemplateHelpers.h`, `ReductionTemplateHelpers.h`, `ScalarTemplateHelpers.h`, `TrigTemplateHelpers.h`

### 7.3 A5 默认参数差异

| 参数 | A5 默认值 | A3 默认值 | 说明 |
|------|-----------|-----------|------|
| `set-workspace-multibuffer` | 2 | 4 | 双缓冲 vs 四缓冲 |
| `limit-auto-multi-buffer-buffer` | NO_LIMIT | only-cube | 不限制 vs 仅 cube |
| `enable-hivm-unit-flag-sync` | true | false | 启用 unit flag sync |
| `enable-preload` | true | false | 启用 preload |
| `enable-lib-call-no-inline` | true | false | 启用 lib call no inline |

### 7.4 SIMT/Triton 支持（A5 新增）

A5 新增 SIMT 编译支持，通过以下 pass 实现：
- `buildBiShengTTIRPipeline` — Triton/SIMT 编译路径
- `buildLowerTritonPipeline` — Triton IR → LLVM
- `split-simt-module` — 拆分 SIMT 模块
- `auto-scope` — 自动作用域标记
- `outline-scope` — 外联 scope
- `adapt-gpu-kernel` — 适配 GPU kernel
- `triton-remap` — Triton 重映射
- `convert-triton-ascend-gpu-to-llvm` — TritonAscendGPU→LLVM
- `convert-hivm-to-tritongpu` — HIVM→TritonGPU

### 7.5 SIMD/SIMT 混合编译（A5 新增）

- `materialize-simt-vf-mem-scope` — 物化 SIMT VF 内存作用域
- `infer-simt-vf-memory-effect` — 推断 SIMT VF 内存效果
- `infer-simt-vf-mem-scope-hint` — 推断 SIMT VF 内存作用域提示
- `legalize-bool-for-simtvf` — SIMT VF bool 合法化
- `insert-memory-semantic-for-simtvf` — 插入内存语义
- `transform-op-for-simt` — SIMT op 变换
- `insert-alloc-base-placeholder` — 插入分配基占位符
- `mark-simt-scope-no-inline` — 标记 SIMT scope 不内联
- `simt-vf-sub-tiling` — SIMT VF 子分块

### 7.6 RegBase 专用方言

- **HIVMRegbaseIntrins** — RegBase 内联函数方言
  - IR 定义：`bishengir/include/bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.td`
  - 实现：`bishengir/lib/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.cpp`
  - 工具：`RegbaseUtils.cpp/.h`

---

## 8. 文档资源

| 文档路径 | 说明 |
|----------|------|
| `/home/zhaojs/workspace/codes/AscendNPU-IR/docs/source/en/introduction/architecture.md` | A5 芯片特性与 RegBase 编程模型说明（第 35-41 行） |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/docs/source/zh_cn/introduction/architecture.md` | 中文版架构说明 |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/docs/source/en/developer_guide/passes/HIVMPasses.md` | HIVM pass 自动生成文档 |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/docs/source/en/developer_guide/passes/HFusionPasses.md` | HFusion pass 文档 |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/docs/source/en/developer_guide/passes/HACCPasses.md` | HACC pass 文档 |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/docs/source/en/developer_guide/passes/ScopePasses.md` | Scope pass 文档 |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/docs/source/en/developer_guide/passes/SymbolPasses.md` | Symbol pass 文档 |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/docs/source/en/developer_guide/passes/AnnotationPasses.md` | Annotation pass 文档 |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/docs/source/en/developer_guide/features/CV/CVOptimization.md` | CV 优化文档 |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/docs/source/en/developer_guide/features/CVPipeline/CVPipelining.md` | CV 流水线文档 |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/docs/source/en/developer_guide/features/AutoFlatten/AutoFlatten.md` | 自动 Flatten 文档 |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/docs/source/en/developer_guide/features/AutoSchedule/HFusion_AutoSchedule.md` | 自动调度文档 |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/docs/source/en/developer_guide/features/AutoBlockify/AutoBlockify.md` | 自动分块文档 |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/docs/source/en/developer_guide/conversion/triton_interface.md` | Triton 接口文档 |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/docs/source/images/introduction/architecture_A5.png` | A5 架构图 |
| `/home/zhaojs/workspace/codes/AscendNPU-IR/docs/source/images/introduction/architecture_A5_zh.png` | A5 架构图（中文） |

---

## 9. 总结

RegBase(A5) 架构的 pass pipeline 是一个**多阶段、多分支**的复杂系统：

### 9.1 整体流程

```
bishengir-compile (检测 regbase 目标)
  → runRegBaseCompile (Driver.cpp)
    → inferLayoutOptimization + inferMixedCV (Utility.cpp)
    → runRegBasePipeline (BiShengIRCompileMain.cpp)
      ├─ Simd 流: buildBiShengHIRPipeline → buildFinalHIVMPipelines → buildBiShengHIRAVEToLLVMPipeline
      ├─ PureSimt 流: buildBiShengTTIRPipeline
      └─ Mixed 流: buildBiShengHIRPipeline → buildBiShengTTIRPipeline(SIMT) → buildBiShengHIRFinishPipeline → buildFinalHIVMPipelines → buildBiShengHIRAVEToLLVMPipeline
    → runExternalHIVMC (hivmc-a5)
```

### 9.2 核心阶段

1. **HIR 前端**（`buildBiShengHIRPipeline`）：HFusion 优化 → HIVM 转换 → HIVM tensor 优化
2. **HFusion RegBase**（`buildHFusionPipelines`）：preProcess → flatten → autoSchedule → autoVectorize
3. **HIVM 优化**（`buildHIVMTensorOptimizations`）：CV 通信 → 内存规划 → 跨核同步 → mix kernel 拆分
4. **HIVM lowering**（`buildLowerHIVMPipelines`）：bufferization → 后端优化 → sync → multi-buffer
5. **AVE lowering**（`buildLowerAVEPipelines`）：Vector→HIVMAVE → AVE 优化
6. **LLVM lowering**（`buildLowerToLLVMPipeline`）：HIVM→Standard → AVE→Standard → LLVM
7. **Triton/SIMT**（`buildLowerTritonPipeline`）：Triton IR → TritonGPU → LLVM
8. **外部工具**（`hivmc-a5`）：LLVM IR → 二进制

### 9.3 RegBase 专用核心 Pass

- **`hivm-plan-memory-regbase`** — 寄存器层感知的内存规划（出现 3 次）
- **`cv-pipelining`** — Cube/Vector 软件流水线
- **`convert-ascend-dpx-to-hivmregbaseintrins`** — DPX→RegBase 内联函数
- **`hfusion-flatten-ops` (registerBased=true)** — RegBase 专用 flatten
- **RegBase Normalize 框架** — HFusion/HIVM 各一套完整 Normalize
- **`hivm-outline-alloc-in-VF` / `hivm-outline-copy-in-VF`** — RegBase 专用 VF 处理

### 9.4 统计

- **Pass 总数**：约 300+ 个（含上游 MLIR/Triton pass）
- **RegBase 专用 pass**：约 15 个核心 pass + 20 个 Normalize 子模块
- **Pipeline 注册**：6 个 PassPipelineRegistration + 2 个 PassRegistration
- **Pipeline 组装函数**：约 20 个 build* 函数
- **TableGen .td 文件**：21 个
- **编译流**：3 种（Simd / PureSimt / Mixed）
- **重试次数**：最多 6 次（含 fallback 策略）

---

> 本文档由 华为云码道（CodeArts）代码智能体 自动生成，基于对 `/home/zhaojs/workspace/codes/AscendNPU-IR` 项目的彻底分析。
