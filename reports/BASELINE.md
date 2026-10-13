# A5 双机 Qwen2.5-72B + Mooncake KV Pool 功能与性能基线

2026-10-12，已在两台物理 A5 上完成 Qwen2.5-72B-Instruct（dummy 权重）+ AISBench 合成数据 + PD-colocated 测量。目标机器四张卡全程显示 Warning，实际 TP4、跨节点数据校验和完整推理均通过。三轮 1200/1200 正式请求成功；源节点冷 TTFT 1619.1 ms、本机 DRAM 198.0 ms；目标节点冷 TTFT 1620.3 ms、跨节点 DRAM 189.2 ms。分别减少约 8.18×、8.57×。

这是固定环境、完整缓存命中和充分预热下的功能与速度基线。跨节点轮均值为 226.1 / 184.5 / 156.9 ms，轮间 CV 18.42%，不能据此断言跨节点优于本机或给出长期稳定性承诺。模型使用真实 72B 结构、BF16 计算和 TP4 通信，但使用随机 dummy 权重；不评价回答质量或真实权重的业务精度。

## 环境与配置

| 项目 | 本次实测条件 |
|---|---|
| 源节点 / 目标节点 | source A5 / destination A5，两台物理机器 |
| NPU / 选卡 | Ascend950DT_9582，96 GiB HBM/卡；每机卡 0–3 |
| 设备状态 | 源节点 OK；目标节点四卡均 Warning，开始与结束均记录 |
| 主机 / 驱动 / CANN | openEuler 24.03 LTS-SP3 / 26.2.0.b020 / 容器 CANN 9.1.0 |
| Python / torch / torch_npu | 3.12.13 / 2.10.0 / 2.10.0.post4 |
| vLLM / vLLM-Ascend | v0.23.0 源码构建，加下述 isolated SDK 导入修正 |
| Mooncake NPU / AISBench | 0.3.11.post1 / 3.1.0，AISBench commit `8affdc1bccfa0437a912c757db4fe7a25b0329d2` |
| 模型 | Qwen2.5-72B-Instruct，revision `495f39366efef23836d0cfae4fbe635880d2be31`；只取配置与 tokenizer |
| 权重 / 结构 | `--load-format dummy`，BF16；按配置计算 72,706,203,648 参数，80 层 |
| 部署 | 每机完整 prefill + decode；TP4；默认 graph 模式，未启用 enforce-eager |
| 服务配置 | max_model_len 25600，max_num_batched_tokens 4096，max_num_seqs 32，gpu_memory_utilization 0.9，seed 42 |
| 请求 | 实际输入 1024 tokens、输出 10 tokens；并发 25；每阶段 100 请求，三轮 |
| KV Pool | 每个 worker 100 GiB DRAM，即每机 400 GiB；`kv_both`，`load_async=true`，`register_buffer=true`，非 layerwise；关闭 SSD |
| 传输 / Master | `protocol=ascend`；`ASCEND_LOCAL_COMM_RES={"version":"1.3"}`；源节点单独 Master，P2PHANDSHAKE |
| CPU / 内存 | 每个推理容器 32 CPU、800 GiB host memory 上限；客户端 16 CPU、32 GiB 上限；未验证严格 NUMA 绑定 |
| 两机推理 image ID | `sha256:64155cd1bb77edf3c927a23c83cdad49e9d5394bd4b90113bc02ce017897c270` |
| 客户端 image ID | `sha256:35c865f8e79e81926ca9d6aa3ab642f49a77b41b30db4a99ebc3b915994434e7` |

输入沿用 AISBench SyntheticDataset 的 tokenid 合成路径。冻结 A0–A2、C0–C2 和两套独立预热集；Qwen chat template 加入后严格校验 1024 tokens。预热集与测量集首个缓存 block 不重合。无真实训练权重下载。镜像固定了 numpy 1.26.4、transformers 5.5.4、tokenizers 0.22.2；全部版本和公共文件来源/哈希在证据包的 manifest 与项目配置中。

官方参考场景为 A2 双机、Qwen2.5-72B-Instruct、1024/10 tokens、100 请求、并发 25，公开 TTFT 为冷 2322 ms / 本机 739 ms / 跨节点 948 ms，即约 3.14× / 2.45×。本报告使用 A5、dummy 权重、独立预热与 DRAM prime，不能把参考倍数当成本次验收阈值或直接推断硬件优劣。[官方 vLLM-Ascend PD-colocated Mooncake 双机教程](https://docs.vllm.ai/projects/ascend/en/v0.23.0/tutorials/features/pd_colocated_mooncake_multi_instance.html)

## 功能验证

| 验证 | 实际结果 |
|---|---|
| Warning 四卡可用性 | 四 rank FP16 整数矩阵乘与 CPU double 参考逐项相等；HCCL AllReduce 4096 元素的求和全部正确；退出 0 |
| 原生 Mooncake 远端读取 | 目标节点不贡献 DRAM，1 MiB / 8 MiB 非均匀 payload 各 8 次；16/16 逐字节与 SHA-256 一致；两端退出 0 |
| SDK 导入回归 | 原后端模块的独立进程导入可复现 heap abort / 133；修正后实际 NPU 回归通过；新增 pytest 1 passed |
| Store 组件单测 | 313 passed，10 skipped；跳过项保留在日志；未运行整个仓库测试集 |
| 服务启动与核心路径 | 两端加载完整 dummy 模型，graph 初始化、独立预热及业务请求通过 |
| AISBench 请求 | 19 个阶段共 1900/1900 成功，其中正式测量 1200/1200；各阶段实际输入/输出均为 1024/10 |
| 批大小预热 | 两端分别覆盖 1–25，325 请求/节点，预热不计入正式性能 |
| 冷请求缓存 | 六个冷阶段全部无外部 KV Pool 加载 |
| DRAM 命中 | 三个本机与三个跨节点阶段，共 600/600 请求，均为 `vllm_cached=0`、`kvpool_cached=1023` |
| 实际跨节点归属 | 每个命中阶段从真实加载 key 中查询 1200 个 memory replica key；所有副本端点均在源节点 source A5；目标请求在另一物理节点 destination A5 执行 |
| 清理 | 自建 source/destination 容器消失，API/控制/Master 端口关闭，卡 0–3 空闲，两机自己的预约均已释放 |

目标机本身也贡献 400 GiB DRAM，因此仅看到 Pool 命中不足以证明远端路径。本次同时校验服务所在物理节点、加载日志与 SDK `batch_get_replica_desc` 返回的真实 memory descriptor / transport endpoint。副本抽查覆盖每阶段 1200 个实际读取 key；它不是所有 KV key 的逐一全量清单。单独的非均匀 payload 用例补充验证了源 DRAM → 目标 NPU 的数据内容。

1024 输入命中 1023 tokens 为完整可复用前缀；最后一个输入 token 重新计算以生成 logits。每阶段前用 `reset_prefix_cache?reset_external=false` 清理本机 NPU 前缀，保留外部 DRAM；两端副本归属校验通过，目标没有把测量转成本机 DRAM 命中。

## 性能结果

数值为 AISBench 三轮的阶段均值取平均，每轮 100 请求；统计来自原始 CSV 与逐请求 details JSONL，不把服务日志中的调度时间当作 TTFT。

| 路径 | TTFT 均值 ms | 轮间 CV | TPOT 均值 ms/token | 请求吞吐 req/s | 输出吞吐 token/s |
|---|---:|---:|---:|---:|---:|
| 源节点冷计算 | 1619.1 | 0.06% | 239.4 | 6.57 | 65.68 |
| 源节点本机 DRAM | 198.0 | 9.14% | 27.1 | 51.35 | 513.50 |
| 目标节点冷计算 | 1620.3 | 0.11% | 239.7 | 6.56 | 65.61 |
| 目标节点跨节点 DRAM | 189.2 | 18.42% | 25.3 | 53.99 | 539.89 |

本机倍数用同一源节点冷基线计算：1619.1 / 198.0 = **8.18×**。跨节点倍数优先用同一目标节点冷基线计算：1620.3 / 189.2 = **8.57×**；若采用官方表格的源冷基线口径，则为 **8.56×**。以上是平均 TTFT 的比值，不是吞吐倍数，也不是通信带宽倍数。

| 轮次 | 源冷 TTFT ms | 本机 DRAM TTFT ms | 目标冷 TTFT ms | 跨节点 DRAM TTFT ms | 跨节点 P50 / P99 ms |
|---|---:|---:|---:|---:|---:|
| 0 | 1618.1 | 218.8 | 1622.1 | 226.1 | 181.1 / 458.8 |
| 1 | 1619.6 | 189.0 | 1618.7 | 184.5 | 191.5 / 238.5 |
| 2 | 1619.7 | 186.1 | 1620.1 | 156.9 | 163.3 / 194.7 |

目标冷请求 P99 分别为 3537.5 / 3525.1 / 3528.7 ms；本机 DRAM P99 分别为 292.2 / 252.1 / 244.5 ms。P50/P99 保留逐轮数据，没有把逐轮分位数平均后冒充合并分位数。正式阶段未出现已检查的 token_bin_counts Triton 编译告警；跨节点仍呈现轮间波动，当前没有足够证据确定其原因。

最小远端传输用例：1 MiB 五次计时平均 0.331 ms，payload 吞吐约 3.17 GB/s；8 MiB 平均 0.393 ms，约 21.32 GB/s。两种 payload 各有三次预热，数据校验在计时区间之外。这是本配置的 payload 微基准，不能代表链路峰值或完整 72B 服务吞吐。

## 测量方法与必要修正

两端先分别执行短请求、独立的冷预热与本机 DRAM 预热，再覆盖批大小 1–25。正式每轮顺序为目标 C 冷基线 → 源 A 冷基线 → 源 A 的非测量 DRAM prime → 源 A 本机 DRAM → 目标 A 跨节点 DRAM。每轮使用独立数据集，阶段之间清理 NPU 前缀。prime 补齐先前已观察到的末块部分命中，其 300 次请求不计入性能统计。此基线适用于已充分预置的全命中缓存，不代表首次填充或部分命中负载。

实测发现，原生 Mooncake SDK 若在 NPU context 初始化前被加载，独立程序可能在完成操作后以 `corrupted size vs. prev_size` / 133 退出。仅改析构方式不能解决。最小 SDK 用例先 `torch.npu.set_device(0)` 再导入 Mooncake 后通过。后端模块顶层 `ReplicateConfig` 导入同样可复现异常，已将其延迟到 Store 初始化完成后的 `put()` 内。补丁未修改算子、模型参数或共享驱动/CANN，已附源码、回归测试及原镜像对照。[Mooncake 上游同类导入问题](https://github.com/kvcache-ai/Mooncake/issues/1338)

修正前，完整目标推理在默认 graph、双端 eager、切换目标卡 4–7 时均出现 HCCL watchdog / ERR02005；这些轮次已保留且不计入基线。修正后，相同默认 graph、两端卡 0–3 的完整三轮全部通过。SDK 导入异常有独立复现证据；它与先前 HCCL 错误之间的具体内部因果链仍需上游确认。目标 Warning 始终存在，不能据先前失败归因 Warning 或认定机器不可用。

任务停止时，两端日志出现 AsyncLLM output_handler 的 EngineDeadError 和 Python resource_tracker 清理提示，均在完成全部测量、开始主动关闭引擎之后；正式阶段未出现这类错误。已独立验证容器、端口和选定卡清理完成，不把停止期提示描述为无告警。

## 交付、复现与边界

测量区间：2026-10-12 20:49–21:07（北京时间）。正式任务 `nds-mac3-72b-cross-src-20261012-05` / `nds-mac3-72b-cross-dst-20261012-06` 均 succeeded、exitCode=0、archiveReady=true。隔离项目目录与镜像保留用于复测；安装方式、命令、预约与停止方法见 `docs/AI-ENABLEMENT.md`。

Multica 问题评论中的证据 ZIP 分片共同包含两个最终任务的全部归档文件、逐请求记录、各阶段缓存日志、副本审计、Master 指标、前后设备状态、修正补丁、回归日志和复现脚本；另附失败历史精选包与完整历史归档哈希/任务 ID。原始证据文件有 SHA-256 清单。解压最终分片到同一目录后可运行 `python3 benchmarks/analyze_cross_node.py final/source/results` 重做严格校验。

本次交付完成后停用每半小时检查规则；不把规则停用前的配置快照误解为仍需继续实验。智能体描述与运行指令均已持久加入 Warning 可参与实验、按实际计算/通信/业务结果判断可用性的规则。系统驱动、CANN 和原镜像均保留。

尚未验证：真实训练权重的语义精度、业务数据分布、长上下文、多种并发、混合命中率、长期稳定性、完整上游仓库测试与补丁合入。项目性能回退阈值 10% 只是已有建议，未得到项目验收确认；本次不宣称满足未定义的 SLO。
