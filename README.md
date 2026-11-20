# A5 vLLM + Mooncake KV Pool baseline

归档 NDS / MAC-3 已验证的 Qwen2.5-72B-Instruct dummy 权重双机 PD-colocated 实验：vLLM-Ascend 的 Mooncake 原生 SDK 导入修正、回归测试、AISBench 合成负载、缓存副本审计、性能报告与复现脚本。

2026-10-12，两台物理 A5、每机 TP4、实际输入/输出 1024/10 tokens、并发 25、三轮 1200/1200 正式请求成功。目标四卡显示 Warning，但实际计算、通信、数据校验和推理通过。

| 路径 | 冷 TTFT → DRAM TTFT | 比值 |
|---|---:|---:|
| 本机源 DRAM | 1619.1 → 198.0 ms | 8.18× |
| 跨节点目标 DRAM | 1620.3 → 189.2 ms | 8.57× |

跨节点三轮均值为 226.1 / 184.5 / 156.9 ms，轮间 CV 18.42%。这是充分预热与完整外部缓存命中的 dummy 模型速度基线，不评价回答质量或长期 SLO。完整条件和失败记录见 [baseline 报告](reports/BASELINE.md)。原始逐请求证据已交付于 Multica MAC-3 评论附件，此仓库保留可复现代码与汇总。

## 代码

- `patches/kv-import-before-device.patch`：将 `ReplicateConfig` 从模块顶层延迟到 Store 初始化后的 `put()` 中导入，同时新增独立进程回归测试。
- `src_overrides/`：与真实 A5 验证相同的修正后后端源码；保留上游许可证。
- `tests/e2e/`：独立子进程导入回归，覆盖原生 SDK 提前初始化的 heap abort。
- `benchmarks/`：双机服务、AISBench 数据生成、Warning 可用性检查、实际 DRAM 副本审计和结果校验。
- `validation/`：2026-10-12 已收集的 Store 组件单测及新增回归结果。
- `docs/AI-ENABLEMENT.md`：固定版本、配置、运行、回收与停止方法。

## 应用修正

在干净的 vLLM-Ascend v0.23.0 源码根目录应用补丁，或使用与原实验一致的父镜像构建覆盖层：

```bash
git apply --check /path/to/this-repo/patches/kv-import-before-device.patch
git apply /path/to/this-repo/patches/kv-import-before-device.patch
```

```bash
docker build -f Dockerfile.kv-import-fix -t nds-vllm-source:kv-import-fix-20261012 .
```

父镜像需已按使能手册准备；此仓库不包含容器镜像、模型资产或训练权重。重新构建的 image ID 可能不同，应记录新 ID 并确保两机一致。

原修正已完成真实 A5 Store 单测 **313 passed / 10 skipped**、新增回归 **1 passed** 及双机负载验证。完整上游仓库测试未运行，补丁未上游合入。为版本归档，实验脚本的机器地址、节点名称和远端目录已参数化；本次只对这些配置调整执行本地语法及离线分析验证，没有重新宣称 NPU 实测。

保留上游文件的原始版权声明；vLLM-Ascend 的许可证位于 `third_party/LICENSE.vllm-ascend`。仓库上传不代表向上游提交 PR 或完成上游验收。
## MAC-16 阶段实验

1024/C1、32输出的T0/T1/T2校准与原始脱敏数据见 [报告](reports/mac16/BASELINE.md)，复现见 [使能手册](docs/MAC16-AI-ENABLEMENT.md)。T3/T4及长上下文扫描待稳定双机资源；不能视为五档全部完成。
