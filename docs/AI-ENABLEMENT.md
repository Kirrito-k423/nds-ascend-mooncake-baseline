# A5 双机使能与复现

本手册面向已有授权的 A5 环境。两台物理节点各使用四张 NPU，每台运行完整模型的 prefill 与 decode；使用 dummy 权重，只准备配置和 tokenizer。Warning 可参与实验：记录状态与任务占用，先验证选卡空闲、矩阵计算、TP4 HCCL 和非均匀数据远端读取，再运行完整负载。

## 固定环境

原实测使用驱动 26.2.0.b020、CANN/NNAL 9.1.0、Python 3.12.13、torch 2.10.0、torch_npu 2.10.0.post4、vLLM/vLLM-Ascend v0.23.0 源码构建、Mooncake NPU 0.3.11.post1、AISBench 3.1.0。numpy 1.26.4、transformers 5.5.4、tokenizers 0.22.2；源码构建设置 `VLLM_VERSION=0.23.0`。

Qwen2.5-72B-Instruct 配置与 tokenizer revision：`495f39366efef23836d0cfae4fbe635880d2be31`。AISBench commit：`8affdc1bccfa0437a912c757db4fe7a25b0329d2`。不下载训练权重，不把 dummy 输出用于业务精度评价。

原父镜像：`nds-vllm-source:mac3-20261012`，ID `sha256:afac49518da833bb9dc52d2faaa72215068da0cc2b52c8ae7f618dd5be6291df`。原修正镜像：`nds-vllm-source:kv-import-fix-20261012`，ID `sha256:64155cd1bb77edf3c927a23c83cdad49e9d5394bd4b90113bc02ce017897c270`。源节点客户端镜像：`nds-vllm-aisbench:typical72b-20261012`，ID `sha256:35c865f8e79e81926ca9d6aa3ab642f49a77b41b30db4a99ebc3b915994434e7`。镜像构建与依赖安装的原始记录在 MAC-3 已交付使能手册和证据附件中。

## 部署参数

从管理入口读取当前机器与可用资源，分别新建机器预约与独立任务 ID。把 `benchmarks/` 内文件放到两机的同一个项目目录；源机还需客户端镜像及模型配置/tokenizer，目标机也需要模型配置/tokenizer。文件布局为 `<项目目录>/run_cross_node.py`、`aisbench_config.template.py`、`replica_audit.py` 等，以及 `<项目目录>/Qwen2.5-72B-Instruct/`。

通过管理入口提交的两个任务都必须显式提供以下非敏感部署参数。这些值来自本轮实际选机，不保存在仓库中。

| 参数 | 用途 |
|---|---|
| `NDS_SOURCE_HOST` | 源物理节点可直连的 IPv4 地址 |
| `NDS_DESTINATION_HOST` | 不同目标物理节点可直连的 IPv4 地址 |
| `NDS_PROJECT_DIR` | 两机已经准备好的项目绝对目录 |
| `SHW_RESULTS_DIR` | 管理入口为每个任务提供的独立结果目录 |
| `NDS_INFERENCE_IMAGE` | 可选；默认已修正镜像 tag |
| `NDS_SOURCE_LABEL` / `NDS_DESTINATION_LABEL` | 可选记录标签，默认 source / destination |

任务的远端 shell 内设置并导出参数，再分别运行 `submit_cross_source.sh` 与 `submit_cross_destination.sh`。先提交源任务，再提交目标任务；保持同一顶层运行前台跟踪两个任务，直到结束并归档。不要用本地任务进程的环境代替远端任务的配置。

固定端口：两端 API 18172、源 Master 50272/指标 50273、目标临时控制 18174。固定容器名用于一次实验；提交前确认没有同名运行，避免冲突。两个任务的设备范围必须一致且已经授权，包装脚本使用卡 0–3。

脚本设置 BF16、TP4、dummy、max_model_len 25600、max_num_batched_tokens 4096、max_num_seqs 32、gpu_memory_utilization 0.9。两端每 worker 100 GiB DRAM，`kv_both`、异步加载、注册 buffer、非 layerwise、关闭 SSD。默认 graph 模式；Ascend/HIXL 使用 `ASCEND_LOCAL_COMM_RES={"version":"1.3"}`。

## 验证与测量

最小远端传输入口为 `remote_store_source.sh` / `remote_store_client.sh`，使用相同部署参数；目标不贡献 DRAM。SDK 必须先 `torch.npu.set_device(0)`，再导入 Mooncake，否则原环境可能在操作完成后发生 heap abort。

正式测量前，两端先运行独立预热集与批大小 1–25。每轮目标 C 冷计算、源 A 冷计算、源 A 非测量 prime、源 A 本机 DRAM、目标 A 跨节点 DRAM；每阶段前清 NPU 前缀并保留外部 DRAM。prime 用于补齐末块部分命中，不能计入正式耗时。

三轮正式测量共 1200 请求；另有预热与 prime。每个请求校验输入 1024、输出 10，命中阶段为 NPU 本机缓存 0 / 外部缓存 1023；审计真实 replica 的端点均在源节点。目标也贡献 DRAM，所以必须检查副本位置，不能只看 Pool 命中。

下载并展开源节点归档后运行：

```bash
python3 benchmarks/analyze_cross_node.py /path/to/source-archive/results
```

成功需 `formal_requests=1200`、`formal_successful_requests=1200`、`all_request_and_cache_checks_passed=true`。原始已测汇总在 `reports/validated-cross-node-summary.json`；严格分析还需要 MAC-3 附件中的原始请求与服务日志。

## 清理与维护

正常结束或异常退出均回收本项目的服务。分别执行 `verify_cleanup.py source|destination <本次清理结果文件>`，确认自建容器不存在、端口关闭、卡 0–3 空闲，再释放自己的预约。不要停止其他任务、修改共享驱动或覆盖历史基线。原实验已释放资源并停用半小时检查；复测需新的任务与预约。

参数化后的脚本配置改动只有本地语法和离线分析验证，原生后端修正及回归测试与实际 A5 已测文件保持一致。部署到新机器或更改依赖后，重新执行受影响的功能与性能验证。
