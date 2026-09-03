# `hfusion-compose-multi-reduce` Pass 深度分析报告

> 生成时间：2026-08-31
> 项目路径：`/home/zhaojs/workspace/codes/AscendNPU-IR`
> 分析对象：`hfusion-compose-multi-reduce` pass

---

## 目录

1. [Pass 的注册位置和定义位置](#1-pass-的注册位置和定义位置)
2. [Pass 的详细作用说明](#2-pass-的详细作用说明)
3. [处理的具体 IR 模式（Before → After）](#3-处理的具体-ir-模式before--after)
4. [具体测试案例分析](#4-具体测试案例分析)
5. [Pass 在 Pipeline 中的位置和前后文](#5-pass-在-pipeline-中的位置和前后文)
6. [核心实现关键代码片段](#6-核心实现关键代码片段)
7. [辅助依赖](#7-辅助依赖)
8. [总结](#8-总结)

---

## 1. Pass 的注册位置和定义位置

| 项目 | 位置 |
|------|------|
| **TableGen 定义** | `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/Transforms/Passes.td:545` |
| **头文件声明（工厂函数）** | `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/Transforms/Passes.h:221` |
| **核心实现文件** | `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HFusion/Transforms/ComposeMultiReduce.cpp` |
| **工厂函数实现** | `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HFusion/Transforms/ComposeMultiReduce.cpp:647` |
| **测试文件** | `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/test/Dialect/HFusion/hfusion-compose-multi-reduce.mlir` |
| **中文文档** | `/home/zhaojs/workspace/codes/AscendNPU-IR/docs/source/zh_cn/developer_guide/passes/hfusion_passes.md:43` |
| **英文文档** | `/home/zhaojs/workspace/codes/AscendNPU-IR/docs/source/en/developer_guide/passes/HFusionPasses.md:50` |
| **Pipeline 调用 1（主 pipeline）** | `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HFusion/Pipelines/HFusionPipelines.cpp:137` |
| **Pipeline 调用 2（regbase pipeline）** | `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/lib/Dialect/HFusion/Pipelines/regbase/HFusionRegbasePipelines.cpp:221` |
| **ReduceCompose 属性定义** | `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionAttrs.td:73` |

### 1.1 TableGen 定义（Passes.td:545-558）

```tablegen
def ComposeMultiReduce : Pass<"hfusion-compose-multi-reduce", "func::FuncOp"> {
  let summary = "Compose multi reduce optimization";
  let constructor = "mlir::hfusion::createComposeMultiReduce()";
  let dependentDialects = [];
  let options =
      [Option<"maxCompose", "max-compose", "int", "-1",
              "Maximum reduce composed into single operation, -1 is limitless">,
       Option<"maxDistDiff", "max-dist-diff", "int", "-1",
              "Maximum distance difference from common ancestor">,
       Option<
           "aggressive", "aggressive", "bool", "false",
           "Aggressive mode will try to reshape if shape are loosely matched">,
  ];
}
```

### 1.2 三个选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `max-compose` | -1（无限制） | 单个合并后的 reduce 最多可包含多少个原始 reduce |
| `max-dist-diff` | -1（无限制） | 允许的两个 reduce 之间距共同祖先的最大距离差 |
| `aggressive` | false | 激进模式：当形状"松散匹配"时自动插入 ExpandShape/CollapseShape 来对齐形状 |

---

## 2. Pass 的详细作用说明

**核心功能：** 将多个独立的 `linalg.reduce` 操作合并（compose）为一个多输入多输出的 `linalg.reduce` 操作。

**动机：** 当 IR 中存在多个 reduce 操作，它们：

1. 具有兼容的输入形状和相同的归约维度（dimensions）
2. 彼此之间没有数据依赖（即不存在 producer-consumer 关系）
3. 距离相近（在 DAG 中距共同祖先的距离差在阈值内）

那么这些 reduce 可以合并为一个 reduce，其 region 内包含所有原始 reduce 的计算逻辑。这样可以：

- 减少循环迭代次数（多个归约共享一次遍历）
- 提高数据局部性
- 为后续的算子融合和调度优化提供更好的基础

### 2.1 处理流程（`runOnOperation`，ComposeMultiReduce.cpp:48-59）

```
1. 构建 ReachabilityAnalyzer（可达性分析器，用于依赖检查和距离计算）
2. initOpt()：初始化选项（-1 转为 INT_MAX）
3. collectReduceOperations()：收集函数内所有 linalg.reduce 操作
4. groupCompatibleReduceOps()：将兼容的 reduce 分组
   - isValidForGrouping()：跳过动态形状的 reduce
   - tryAddToExistingGroup()：尝试加入已有分组
     - canGroupWithExisting()：
       - checkShapeCompatibility()：检查形状兼容性
         - 严格模式：形状和归约维度必须完全一致
         - 激进模式：允许通过 reshape 对齐的松散匹配
       - checkGroupDependencies()：检查数据依赖和距离
         - hasDataDependency()：有数据依赖则不能合并
         - getShortestPathFromAncestor()：距离差超过阈值则不能合并
5. composeGroupedOperations()：对每个分组（size > 1）执行合并
   - composeReduceOps()：
     a. aggressive 模式下先 adjustOperandsForAggressive()：插入 ExpandShape 对齐形状
     b. collectComposedReduceInfo()：收集所有输入和输出
     c. createComposedReduceOp()：创建新的多输入多输出 reduce，标记 reduce_composed 属性
     d. mergeReduceRegions()：合并所有原始 reduce 的 region 体
     e. replaceAndEraseOldOps()：用新 reduce 的结果替换旧 reduce 的结果，删除旧 op
     f. ensureTopologicalOrder()：确保拓扑序正确
```

---

## 3. 处理的具体 IR 模式（Before → After）

### 3.1 模式 1：严格模式（默认或 aggressive=false）

**Before（两个独立的 reduce）：**

```mlir
%reduced_0 = linalg.reduce ins(%input0 : tensor<MxNxf32>) outs(%init0 : tensor<Mxf32>) dimensions = [1]
  (%in: f32, %init: f32) {
    %0 = arith.addf %in, %init : f32
    linalg.yield %0 : f32
  }
%reduced_1 = linalg.reduce ins(%input1 : tensor<MxNxf32>) outs(%init1 : tensor<Mxf32>) dimensions = [1]
  (%in: f32, %init: f32) {
    %0 = arith.mulf %in, %init : f32
    linalg.yield %0 : f32
  }
```

**After（合并为一个多输入多输出 reduce）：**

```mlir
%reduced:2 = linalg.reduce ins(%input0, %input1 : tensor<MxNxf32>, tensor<MxNxf32>)
                        outs(%init0, %init1 : tensor<Mxf32>, tensor<Mxf32>)
                        dimensions = [1] {hfusion.reduce_composed = ""}
  (%in0: f32, %in1: f32, %init0: f32, %init1: f32) {
    %0 = arith.addf %in0, %init0 : f32
    %1 = arith.mulf %in1, %init1 : f32
    linalg.yield %0, %1 : f32, f32
  }
```

**关键点：**

- 两个 reduce 的输入形状和归约维度必须完全相同
- 新 reduce 的 region 包含所有原始 reduce 的计算
- 新 reduce 被标记 `hfusion.reduce_composed` 属性
- block 参数顺序：先所有 input，再所有 output（init）

### 3.2 模式 2：激进模式（aggressive=true）

当两个 reduce 的形状不同但可以通过 reshape 对齐时，pass 会自动插入 `tensor.expand_shape` 和 `tensor.collapse_shape` 来对齐形状后再合并。

**Before：**

```mlir
// reduce A: 输入 tensor<24x1024xf32>, 归约维度 [0], 输出 tensor<1024xf32>
// reduce B: 输入 tensor<24x32x32xf32>, 归约维度 [0], 输出 tensor<32x32xf32>
// 两者形状不同，但 24x1024 和 24x32x32 可以通过 reshape 互相转换
```

**After：**

```mlir
// 1. 对 reduce A 的输入插入 expand_shape: tensor<24x1024xf32> -> tensor<24x32x32xf32>
// 2. 合并后的 reduce: 输入为两个 tensor<24x32x32xf32>
// 3. 对 reduce A 的输出插入 collapse_shape: tensor<32x32xf32> -> tensor<1024xf32>
```

---

## 4. 具体测试案例分析

测试文件：`/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/test/Dialect/HFusion/hfusion-compose-multi-reduce.mlir`

### 4.1 案例 1：动态形状不合并（测试文件第 1-20 行）

**输入：**

```mlir
func.func @reduction_tile(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
                          %arg2: tensor<?xf32>, %arg3: tensor<?xf32>)
  -> (tensor<?xf32>, tensor<?xf32>) {
  %reduced = linalg.reduce ins(%arg0 : tensor<?x?xf32>) outs(%arg2 : tensor<?xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %0 = arith.addf %in, %init : f32
      linalg.yield %0 : f32
    }
  %reduced_0 = linalg.reduce ins(%arg1 : tensor<?x?xf32>) outs(%arg3 : tensor<?xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %0 = arith.addf %in, %init : f32
      linalg.yield %0 : f32
    }
  return %reduced, %reduced_0 : tensor<?xf32>, tensor<?xf32>
}
```

**期望输出（CHECK）：** 仍然是两个独立的 `linalg.reduce`（因为动态形状被 `isValidForGrouping` 跳过）。

### 4.2 案例 2：Group Norm 反向传播合并（测试文件第 82-156 行）

这是一个真实的 GroupNorm 反向传播 kernel。输入中有两个 reduce（第 132 行和第 140 行）：

**输入（关键部分）：**

```mlir
%reduced = linalg.reduce ins(%25 : tensor<768x4x1x49152xf32>)
                       outs(%collapsed_12 : tensor<768x4xf32>) dimensions = [2, 3]
  (%in: f32, %init: f32) {
    %31 = arith.addf %in, %init : f32
    linalg.yield %31 : f32
  }
// ... 中间有一些 expand/collapse/reshape 操作 ...
%reduced_20 = linalg.reduce ins(%26 : tensor<768x4x1x49152xf32>)
                          outs(%collapsed_3 : tensor<768x4xf32>) dimensions = [2, 3]
  (%in: f32, %init: f32) {
    %31 = arith.addf %in, %init : f32
    linalg.yield %31 : f32
  }
```

**期望输出（CHECK）：**

```
// CHECK: %[[reduced:.*]]:2 = linalg.reduce ins(
// CHECK-NOT: linalg.reduce
```

两个 reduce 合并为一个 `%reduced:2 = linalg.reduce`（2 个输出），不再有其他独立的 reduce。

### 4.3 案例 3：激进模式合并不同形状（测试文件第 163-190 行，AGGR 前缀）

**输入（关键部分）：**

```mlir
// reduce 1: 输入 tensor<24x32x32xf32>, 归约 [0], 输出 tensor<32x32xf32>
%reduced = linalg.reduce ins(%3 : tensor<24x32x32xf32>) outs(%5 : tensor<32x32xf32>) dimensions = [0]
  (%in: f32, %init: f32) { ... }

// reduce 2: 输入 tensor<24x1024xf32>, 归约 [0], 输出 tensor<1024xf32>
%reduced_2 = linalg.reduce ins(%arg1 : tensor<24x1024xf32>) outs(%7 : tensor<1024xf32>) dimensions = [0]
  (%in: f32, %init: f32) { ... }
```

**期望输出（AGGR-CHECK）：**

```
// AGGR: linalg.reduce
// AGGR-NOT: linalg.reduce
```

在 aggressive 模式下，虽然形状不同（`24x32x32` vs `24x1024`），但通过 reshape 可以对齐，所以合并为一个 reduce。

---

## 5. Pass 在 Pipeline 中的位置和前后文

该 pass 在 `preFlattenPass` 函数中被调用（两个 pipeline 文件中的位置和上下文完全一致）：

### 5.1 `preFlattenPass` 的完整 pass 序列（HFusionPipelines.cpp:123-147）

| 顺序 | Pass | 说明 |
|------|------|------|
| 1 | `createBubbleUpExtractSlicePass` | 将 extract_slice 上浮 |
| 2 | `canonicalizationPipeline` | 规范化 |
| 3 | `createLinalgFoldUnitExtentDimsPass` | 折叠单位维度 |
| 4 | `canonicalizationPipeline` | 规范化 |
| 5 | `createArithToHFusionConversionPass` | 将 arith 操作转为 hfusion |
| **6** | **`createComposeMultiReduce` (aggressive=true)** | **本 pass：合并多个 reduce** |
| 7 | `createPropagateSymbolPass` / `createUnfoldSymbolicIntPass`（可选） | 符号分析 |
| 8 | `createPropagateReshapePass` | 传播 reshape |
| 9 | `createSimplifyOpsPass` | 简化操作 |
| 10 | `canonicalizationPipeline` | 规范化 |

### 5.2 在 pipeline 中的位置（`buildHFusionPipelines`，HFusionPipelines.cpp:288-319）

```
preProcess → canonicalization →
  preFlattenPass（含 ComposeMultiReduce）→ flattenAndFold →
  inferAndOutlineOp → postProcessOutlinedKernel →
  [MultiKernel 时再次] preFlattenPass → flattenAndFold →
  hfusionAutoSchedulePipeline →
canonicalization → postProcess
```

### 5.3 前后文分析

- **之前**：`createArithToHFusionConversionPass` 将 arith 操作转换为 hfusion 操作，这可能会产生多个可合并的 reduce
- **之后**：`createPropagateReshapePass` 传播 reshape 操作（aggressive 模式下 ComposeMultiReduce 可能插入新的 reshape），然后 `flattenAndFold` 进行展平和折叠

**注意：** 在 pipeline 中，`aggressive` 选项被设置为 `true`（`composeOptions.aggressive = true`，第 136 行），这意味着实际使用时启用了激进模式，允许对不同形状但可通过 reshape 对齐的 reduce 进行合并。

---

## 6. 核心实现关键代码片段

### 6.1 合并的核心逻辑（composeReduceOps，第 476-491 行）

```cpp
void composeReduceOps(SmallVector<linalg::ReduceOp> &group) const {
    OpBuilder builder(group[0]->getParentRegion());

    SmallVector<linalg::ReduceOp> adjustedGroup;
    if (aggressive) {
      adjustOperandsForAggressive(builder, group, adjustedGroup);
      std::swap(group, adjustedGroup);
    }

    ComposedReduceInfo info = collectComposedReduceInfo(group);
    linalg::ReduceOp newReduceOp =
        createComposedReduceOp(builder, group[0], info);
    mergeReduceRegions(builder, group, newReduceOp, info);
    replaceAndEraseOldOps(group, newReduceOp);
    ensureTopologicalOrder(newReduceOp);
}
```

### 6.2 创建合并后的 reduce（createComposedReduceOp，第 513-523 行）

```cpp
linalg::ReduceOp
createComposedReduceOp(OpBuilder &builder, linalg::ReduceOp firstOp,
                       const ComposedReduceInfo &info) const {
    builder.setInsertionPointAfter(firstOp);
    auto newReduceOp = builder.create<linalg::ReduceOp>(
        firstOp->getLoc(), TypeRange(info.allOutputs), info.allInputs,
        info.allOutputs, firstOp.getDimensions());

    newReduceOp->setAttr(ReduceComposeAttr::name, builder.getUnitAttr());
    return newReduceOp;
}
```

### 6.3 Block 参数映射（createBlockArgumentMapping，第 602-619 行）

这个函数建立了旧 reduce 的 block 参数到新 reduce 的 block 参数的映射。关键逻辑是：新 block 的参数顺序是先所有 input，再所有 output，而每个原始 reduce 的参数是 [input0, input1, ..., init0, init1, ...]，需要正确映射。

```cpp
for (int64_t j = 0; j < resultNum; ++j, ++globalIdx) {
    mapping.map(oldBlock.getArgument(j), newBlock->getArgument(globalIdx));
    mapping.map(oldBlock.getArgument(resultNum + j),
                newBlock->getArgument(allInputSize + globalIdx));
}
```

---

## 7. 辅助依赖

| 依赖 | 位置 | 说明 |
|------|------|------|
| `ReachabilityAnalyzer` | `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/Utils/ReachabilityAnalyzer.h` | 构建操作间的可达性矩阵，用于检查数据依赖和计算距离 |
| `areLooseReassociationsCompatible` | MLIR 上游 `mlir/Dialect/Utils/ReshapeOpsUtils.h` | 检查两个形状是否可以通过松散的 reassociation（reshape）对齐 |
| `createNewReshapingOp` | `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/Tensor/Transforms/PropagateReshape/Utils.h:89` | 创建 reshape 操作的工具函数 |
| `ReduceComposeAttr` | `/home/zhaojs/workspace/codes/AscendNPU-IR/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionAttrs.td:73` | 标记合并后的 reduce 的属性 |

---

## 8. 总结

`hfusion-compose-multi-reduce` 是一个**归约操作合并优化 pass**，其核心思想是：

1. **输入**：函数中多个独立的 `linalg.reduce` 操作
2. **输出**：合并后的单个多输入多输出 `linalg.reduce` 操作（标记 `hfusion.reduce_composed` 属性）
3. **合并条件**：
   - 非动态形状
   - 形状兼容（严格模式要求完全一致；激进模式允许通过 reshape 对齐）
   - 归约维度相同
   - 无数据依赖
   - 距离差在阈值内
4. **合并方式**：将所有原始 reduce 的输入/输出拼接为新 reduce 的输入/输出，将所有原始 reduce 的 region 体合并到新 reduce 的 region 中
5. **激进模式**：当形状不同但可 reshape 对齐时，自动插入 `tensor.expand_shape`/`tensor.collapse_shape` 来对齐形状

该 pass 在 pipeline 中位于 `preFlattenPass` 阶段，在 arith 到 hfusion 转换之后、reshape 传播之前，实际使用时 `aggressive=true`。

---

> 本文档由 华为云码道（CodeArts）代码智能体 自动生成，基于对 `/home/zhaojs/workspace/codes/AscendNPU-IR` 项目的彻底分析。
