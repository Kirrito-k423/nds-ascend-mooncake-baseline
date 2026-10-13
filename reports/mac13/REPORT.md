# MAC-13：DeepSeek V4.1-Flash W8A8 单机 A5 部署边界

**当前实测软件栈无法启动目标模型，失败发生在模型配置解析，尚未进入权重加载、KV 分配或长上下文请求。** 这不是一次 OOM 结论，也不能据此证明所有 A5 版本都不支持目标。8 张 Warning 卡的实际计算和 HCCL 校验全部通过。后续 MAC-15 暂无可接收的 V4.1 W8A8 运行配置。

检查时间：2026-10-13；最终任务退出于 12:14:27，北京时间。机器 `a5-b07-6`，设备 0–7，Ascend950DT。完整原始 JSON、逐卡 CSV、启动和失败历史日志、metadata 与 header 在本 issue 附件证据包中。本文和 `summary.json` 保留同一结论。

## 实际验证结果

| 层级 | 结果 | 依据 |
|---|---|---|
| 官方模型身份 | 已确认 | `deepseek-ai/DeepSeek-V4.1-Flash`，revision `5254448103b6254fc646dacbca1cbbf20e9216c1` |
| W8A8 资产 | metadata 已捕获，存在一致性问题 | Eco-Tech revision `155aced390c4692698400b744546a8bc396699c6`；见下文 |
| 八卡计算和通信 | 8/8 通过 | 固定整数输入的 FP16 矩阵乘与 CPU FP64 参考逐元素相等；4096 元素 HCCL all-reduce 全卡等于 28 |
| 目标引擎加载 | 失败，退出码 1 | Transformers 不识别 `deepseek_v41`；当前 vLLM registry 没有 `DeepseekV41ForCausalLM` |
| 8K / 32K / 100K / 200K 请求 | 均未执行 | 引擎初始化失败；每档尝试数为 0，实际输入/输出 token 和成功率为 null |
| KV bytes/token、激活峰值、graph/workspace、并发 | 未测得 | 没有 target worker、Available KV cache memory 或 KV token capacity 记录 |
| 精度 | 不作模型精度结论 | 没有真实权重推理；八卡小计算校验仅证明该测试的数据正确 |

镜像固定为 `sha256:64155cd1bb77edf3c927a23c83cdad49e9d5394bd4b90113bc02ce017897c270`，即已有 `nds-vllm-source:kv-import-fix-20261012`。实测 Python 3.12.13、torch 2.10.0、torch_npu 2.10.0.post4、vLLM `0.23.0+nds1.empty`、vLLM-Ascend `0.23.0+nds1`、Transformers 5.5.4、tokenizers 0.22.2。CANN latest 链接指向 `cann-9.1.0`，宿主驱动为 26.2.0.b020。镜像没有可读的源码 Git revision，以 image ID 和包版本固定本次栈。

最终探针使用 TP8 / DP1 / EP enabled、BF16 非量化张量、`quantization=ascend`、`load_format=dummy`、上下文上限 8192、并发 1、内存利用率 0.9、KV dtype auto、eager、CPU weight offload 0。保留了原 W8A8 `config.json`，未改层数、宽度、专家数或架构。这里只运行了初始化探针：远端只有 config；其他量化 metadata 和所有 tensor header 已在本地捕获。**没有构造或验证完整 fake W8A8 checkpoint，没有任何目标权重布局分配实测。**

基线要求设置 `VLLM_VERSION=0.23.0`。第一次 IPv4 重跑未设置它，遇到 plugin 对不存在的 `get_physical_gpu_ids_for_local_dp_rank` 的引用；补齐后最终失败明确落到 `deepseek_v41` 配置识别。该中间失败保留在证据包中，不作为最终容量原因。

## 逐卡 HBM 记录

单位严格区分：npu-smi 的 98304 MB 按 MiB 口径记录为 96 GiB；torch 的 total 是运行时可管理容量，并非另一份物理容量。两种读数不重复相减，也不再次除 TP。

| rank/设备 | 物理 HBM GiB | 启动前已用 MiB | 启动前空闲 GiB | smoke 中 torch total GiB | smoke 中 torch free GiB | 启动后已用 MiB |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 96 | 4773 | 91.339 | 91.75 | 90.548 | 4773 |
| 1 | 96 | 4765 | 91.347 | 91.75 | 90.557 | 4765 |
| 2 | 96 | 4767 | 91.345 | 91.75 | 90.555 | 4767 |
| 3 | 96 | 4766 | 91.346 | 91.75 | 90.557 | 4766 |
| 4 | 96 | 4774 | 91.338 | 91.75 | 90.547 | 4774 |
| 5 | 96 | 4764 | 91.348 | 91.75 | 90.556 | 4764 |
| 6 | 96 | 4763 | 91.349 | 91.75 | 90.557 | 4763 |
| 7 | 96 | 4766 | 91.346 | 91.75 | 90.556 | 4766 |

全机物理容量 768 GiB，最终启动前实际空闲合计 **730.756 GiB**。smoke 的每 rank PyTorch allocator 峰值为 20,996,608 bytes，reserved 峰值 25,165,824 bytes；这不是模型激活峰值或模型运行时预算。HCCL 单次结果包含首次初始化耗时，未作为通信性能基线。

每 rank 的目标权重、激活、运行时/graph/workspace、KV 已分配字节与 token capacity 均标为未测得；没有把未建立的 worker 写成成功分配了零字节。主机 RAM 总计约 2.43 TB，库存检查 available 约 2.40 TB；系统盘可用约 3.17 TB。这些资源没有解决模型识别失败。

## 完整源布局核查与容量风险

通过 HTTP Range 只读获取了 **274 个被索引分片的 safetensors header**，共 143,496 个 tensor，所有 tensor 的 shape × dtype 字节与 data_offsets 一致，所有索引 tensor 都能定位到对应 header。没有读取训练 tensor payload，也没有核验完整训练分片的 SHA256。

| 源 checkpoint 组成 | header 数据区字节 |
|---|---:|
| 主干及其他张量 | 557,966,183,872 |
| MTP/DSpark | 17,286,159,752 |
| Engram 表及 scale | 221,190,580,800 |
| Vision / Aligner | 970,506,240 |
| **总计** | **797,413,430,664 = 742.649 GiB** |

总量与 checkpoint index 的 `total_size` 完全一致。若这些存储张量原样全部驻留，理想八等分也是每 rank 92.831 GiB，总 payload 比当前全机空闲多 **11.893 GiB**，尚未加入激活、通信和 KV。这是**源布局容量对照**，不是已发生的 OOM：运行时 dtype 转换、padding、专家分片、共享/复制和可选草稿的加载策略均未实测。不能用它生成真实每 rank 权重预算或并发数。

已固定的 W8A8 资产有两项具体不一致：

- README 描述 BF16 Engram 和无 embedding scale；实际量化描述为 `int8_sym / group_size=32`。四个真实 header 分别是 I8 的 `[384006168,256]` / `[384016682,256]` 表，以及 F32 的 `[384006168,8]` / `[384016682,8]` scale。源格式必须以这一 revision 的 tensor header 为准。
- `aurora_export_manifest.json` 内的 index SHA256 为 `b996ddcd393e1512cca8cf210d54f63e3caaf2f472b48192befdcc69546a7e44`，实际 index 为 `0dc009ac93014cb0848acec2140223455b14733a82cfdbefc97c1b8c19559dfb`；README 哈希也不同。config 和 quant description 哈希匹配。**不能把此资产签核为导出清单完整性通过。**

官方模型的 552B 指 backbone，还另有 Engram、视觉和草稿；原生权重使用混合 FP8/FP4，不可与 W8A8 字节量混用。[官方 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) 的 890 B/token 是其 FP4 主 KV 和共享结构下的全局 KV 描述。200,000 × 890 = 178,000,000 bytes（169.754 MiB）仅是该公开描述的算术量，**不是当前 Ascend 引擎的实测物理 KV，更不能再因“1/8”重复折算**。

## 后续实验交接与复现

MAC-15 的目标配置交接为 **blocked：本轮没有成功运行的 V4.1 W8A8 服务、受支持请求长度或 KV capacity**，不能启动 200K 并发扫描。

固定的 vLLM-Ascend 源码 `33a8b92bdc00dc83c6088d51a397f0cab2f98761` 的 `.github/vllm-main-verified.commit` 指向 vLLM `8cc9aa5ad350f6f80cf3bebb298d410803ff75a9`。这是待适配版本候选，未在本次 A5 上构建或验证。其 [V4.1 文档](https://github.com/vllm-project/vllm-ascend/blob/33a8b92bdc00dc83c6088d51a397f0cab2f98761/docs/source/tutorials/models/DeepSeek-V4.1-Flash.md) 的 A3 W8A8 单机配置使用 Engram host offload；末尾 A5 回归示例是原生 FP8、16K 和 Engram host offload。两者均不证明本任务要求的 W8A8、全权重留 HBM、200K 单机配置。文档主段的 V1 限制和末尾 V2 示例也不可混配为已验证方案。

继续目标实验需先解决 checkpoint metadata/导出清单一致性，并在隔离镜像建立匹配的 V4.1 A5/W8A8 实现；dummy 必须校验实际 tensor dtype、shape、scale 和 Engram 分片，加载后再采集真实权重/激活/KV 分配并逐级请求。双机 A5 在该 W8A8/no-offload 条件下也没有本次核实的官方配置，未执行双机对照。

替代模型另列：原 `1b05f19` 中的 Qwen2.5-72B-Instruct dummy / TP4 / max_model_len 25600 / Mooncake KV Pool 已在 MAC-3 验证 1024 输入、10 输出和并发 25，可用于方法复核；本轮没有重跑该服务，不把它算作 V4.1、200K 或输出 32 的验收。官方支持矩阵中的 A5 DeepSeek V4-Flash 可作为另外一个待验证候选，也不能替代本目标结论。

只下载 metadata 和 header 的复现命令：

```bash
python experiments/mac13/capture_sources.py --revision 5254448103b6254fc646dacbca1cbbf20e9216c1 --out evidence/sources
python experiments/mac13/capture_w8a8.py --revision 155aced390c4692698400b744546a8bc396699c6 --out evidence/sources/w8a8
python experiments/mac13/capture_headers.py --revision 155aced390c4692698400b744546a8bc396699c6 --index evidence/sources/w8a8/quant_model_weights.safetensors.index.json --out evidence/sources/w8a8/all-headers
python experiments/mac13/analyze_evidence.py --evidence evidence --out reports/mac13
```

下载本 issue 的证据 ZIP 后，也可直接离线复核，无需解压大批 metadata 文件：

```bash
python experiments/mac13/analyze_evidence.py --archive MAC13-V41-A5-evidence-20261013.zip --out verified
```

远端先通过已核实中台选择/预约并检查占用，把 config 和探针脚本准备在自己的 `NDS_PROJECT_DIR`。`run_preflight.sh` 的完整命令保留在源码；它使用固定 IPv4 rendezvous 端口 18213，启动前检查端口空闲。若已有八卡 smoke 结果，仅需 `run_startup_only.sh`。32 GiB host memory cap 是此次初始化探针的隔离限额，不是完整模型部署预算。

任务 `mac13-preflight-20261013` 的首次 rendezvous 超时退出 124；中台中断期间用既有严格 SSH 入口前台回收并完成 `mac13-ssh-retry-20261013`，八卡通过、未设置版本变量的启动退出 1。中台以相同 serviceId 恢复后，最终任务 `mac13-startup-final-20261013` 退出 1，原任务和最终任务均确认 `archiveReady=true`。没有重启中台或改其持久记录。

两个自有探针容器已由 trap 删除，结束时 8 卡均无 NPU 进程，原有两个容器仍在；预约 `mac13-v41-b076-20261013` 已在 12:15:10 释放，中台显示该机 ready、无 task/reservation 占用。没有启动待留存服务。复现中断时只停止 `nds-mac13-v41-preflight` 或 `nds-mac13-v41-startup`，确认退出并收齐结果后通过预约 API 释放自己的 reservation；不要删除中台记录或停止其他项目。
