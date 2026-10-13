# MAC-17：V4-Flash 单机 A5 复现手册

本手册复现完整结构 fake 权重容量实验。真实业务精度、并发基线与生产配置需另行验证。机器由 simpleHtmlWatch 实时清单选择并预约，以下命令在已获授权的八卡空闲 A5 上执行；工作目录、镜像与预约由执行者提供，不包含机器地址或凭据。

## 固定输入与环境

- 模型 `deepseek-ai/DeepSeek-V4-Flash`，revision `60d8d70770c6776ff598c94bb586a859a38244f1`。
- 43 主层、1 MTP、4096 hidden、256 experts；不改 config。原生 MXFP4 routed experts、MXFP8 linear 与 UE8M0 scale，运行时 `--quantization fp8` 选择 AscendFp8Config。
- 八卡 Ascend950DT，96 GiB/card；驱动 `26.2.0.b020`，CANN/NNAL `9.1.0`。
- 已验证镜像 `nds-vllm-source:kv-import-fix-20261012`，image ID `sha256:64155cd1bb77edf3c927a23c83cdad49e9d5394bd4b90113bc02ce017897c270`。来源见本仓 Dockerfile 和 provenance；tag 本身不足以固定环境。不同 ID 必须重新验证。
- Python 3.12.13，torch 2.10.0，torch_npu 2.10.0.post4，vLLM 0.23.0+nds1.empty，vLLM-Ascend 0.23.0+nds1，transformers 5.5.4，tokenizers 0.22.2，Mooncake NPU 0.3.11.post1；运行时设置 `VLLM_VERSION=0.23.0`。

容器的 `/opt/nds-venv` 隔离依赖是已有基线。此次没有变更主机 Python、驱动、CANN 或其他业务容器；新宿主 Python 如有需要按项目规范用 pyenv+venv 配置，勿覆盖系统环境。

## 准备 metadata 和脚本

在有网络的运行端，仅下载白名单文件；代理可改为空或已有授权代理。

```bash
python3 experiments/mac17/prepare_model.py --out model-metadata --proxy http://127.0.0.1:7890
```

将 `model-metadata/` 放入目标独立目录的 `model/`，将 `experiments/mac17/*.py` 放入 `scripts/`。原始 config/tokenizer 的 SHA256 在生成的 `metadata-manifest.json`；目标目录不应有真实 safetensors 权重。源 checkpoint 的 46 shard headers 用 `capture_headers.py` 另外采集、核算；它不是推理必需的权重输入。

完整 fake 由 `mac17_worker.AuditWorker` 在真实 dummy loader 构造的原始 tensor 上填合法数值；后续转置、NZ packing、scale 重排、KV 分组均由固定运行时执行。保持 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`。执行前核对八卡空闲；Warning 允许参加，先用 `run_preflight.sh` 验证计算/HCCL。

## 运行案例

通过中台提交时，将以下 export 前缀和 `run_case.sh` 正文作为同一个 shell；中台注入 `SHW_RESULTS_DIR`。下列 `NDS_PROJECT_DIR` 使用实际已授权的独立目录。

```bash
export NDS_PROJECT_DIR=/your/authorized/mac17-project
export NDS_IMAGE=nds-vllm-source:kv-import-fix-20261012
export NDS_CONTAINER=nds-mac17-v4flash-base
export NDS_SERVER_ARGS='--max-model-len 200032 --lengths 8192,32768,100000,200000'
# 后接 experiments/mac17/run_case.sh 的完整正文
```

200K 通过后单独运行条件性 1M；两案例须顺序执行。

```bash
export NDS_CONTAINER=nds-mac17-v4flash-million
export NDS_SERVER_ARGS='--max-model-len 1000032 --lengths 1000000'
```

固定核心参数是 TP1、DP8、EP enabled、完整 MTP1、block=128、max_num_seqs=1、batch_tokens=2048、utilization=0.9、eager、CPU offload=0、prefix cache disabled、hybrid KV manager enabled。单请求长度包含 input+output。`run_server.py` 先经 `/tokenize` 获取合法 token ID，再以精确长度 ID 列表请求 `/v1/completions`，校验 usage，而非靠文本长度估算。成功条件是 result exit_code=0，每项实际 prompt_tokens 等于指定长度且 completion_tokens≥32。

## KV Pool 最小案例

```bash
export NDS_CONTAINER=nds-mac17-v4flash-pool
export NDS_SERVER_ARGS='--pool --lengths 32768 --repeats 16'
```

connector 为 `AscendStoreConnector`，role=`kv_both`，Mooncake backend，非 layerwise、load_async、register_buffer。创建自有 master 50217、metrics 50218、API 18218；执行前检查这些端口空闲。master 配置由目标本机自动获取 IP，1 GiB DRAM/rank，SSD offload disabled。六组 hybrid KV 的原始 token 传输粒度是 16,384；8K 推理成功不足以证明缓存转移。

`pool_audit.py` 读取实际注册 NPU storage，对 SDK put/get 的所有 buffer 字节算 SHA256，get 前将目标 buffer 首64字节填 0xA5，验证恢复结果是否等于对应成功 put。它增加同步和 CPU 副本，只用于功能校验。`pool-bytes.jsonl` 必须有成功 put、get、匹配的 poisoned get；仅连接成功或重复输出相同不算通过。缓存字节、SDK 传输字节与 HBM allocation 分别核算。

## 回收、停止与离线验收

案例前台等待完成，中台回收 stdout/stderr/results 后再下载。`run_case.sh` 会结束并等待自有 vLLM/master、删除同名自有容器、等待 HBM sampler，记录 npu-before/after、container-exit-code 与 host-hbm.jsonl。异常时只停止已确认属于本案例的容器，例如 `docker stop --time 15 nds-mac17-v4flash-pool`，再回收日志；不要按 executable 名称停止他人进程或中台 daemon。释放对应预约并核对原有容器、NPU占用、端口。

解压交付证据后，在仓库执行：

```bash
python3 experiments/mac17/verify_evidence.py --root evidence/mac17
python3 experiments/mac17/analyze.py --evidence evidence/mac17 --out analysis-recomputed.json
```

报告里的 rank budgets 来自原始记录，`analysis.json` 保存精确 bytes。历史 DP8/1M 记录曾把 local ordinal 0 当文件名；按记录内 dp_rank 可恢复八卡阶段信息，但其 tensor inventory 是覆盖后的单 rank 样本。后续修正为 DP×TP+TP rank，并补齐八个独立 inventory。所有失败保留，不能把旧诊断任务与成功案例混合为同一次运行。
