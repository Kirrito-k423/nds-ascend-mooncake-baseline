# MAC-16 内网 AI 实验交接：继续 T3/T4

交接日期：2026-10-14。适用任务：Qwen2.5-72B-Instruct dummy BF16、TP4、PD-colocated 的五档 KV 延迟基线。**当前模型 T0/T1/T2 有校准证据；T3/T4 模型有效请求均为 0。下一步先解决初始化与模型形状读回，再测 SSD 模型性能。** 单卡 SSD 成功不能替代四卡模型验收。

本文按开发快照 `e35bcd3faa4bd24080ef49c3e36b40a033865bf2` 核对代码；后续迁移、调度工作可能并行推进。它是可复现的交接基点，不代表原执行者已停止或此后没有新结果。本文新增说明和校验清单，没有实施 HIXL 修复、单机编排或四 worker 批量探针。

## 1. 给接手 AI 的任务

> 先读本文、`docs/MAC16-AI-ENABLEMENT.md`、`reports/mac16/w1/NATIVE-SSD.md`，再读 `benchmarks/kv_tier/{run,node,store_agent,instrument,analyze}.py`。固定代码、镜像、模型元数据指纹，使用用户提供的内网机器与实际数据地址。在独立分支、私有 Master、独立结果目录中执行。先复核四卡计算/HCCL与单卡 HIXL/SSD，再补 `10 MiB × 8 key / rank × 4 worker` 的本机及跨机读回。通过后跑 1024 输入、32 输出、C1、三轮 T3/T4，与同节点同模式 T0 配对。失败时保留全部日志，区分环境缺失、初始化、读回错误与性能问题；不能降低模型规模、跳过位置/盘读/数据校验来获得“成功”。只有一台机器时先补单机入口推进 T3。交付代码 commit、环境清单、原始数据、失败记录及通过/不通过结论。

原任务 3 天预算截止为 **2026-10-16 17:34:44（北京时间）**。接手时重新核对剩余预算与用户的新排期；不可把旧预约、旧资源快照或旧 deadline 当作新的机器授权。外网 B02 迁移与内网实验应使用不同 run ID、端口/容器命名和结果目录，不混写结果。

## 2. 已验证情况与结论边界

以下数字均为**历史实测**，不是本次内网设备验证。

| 项目 | 已有证据 | 尚不能得出的结论 |
|---|---|---|
| MAC-3 基线 | 完整 72B dummy TP4，1024/10、C25，1200/1200 正式请求成功；导入顺序补丁通过设备回归 | 不能与 MAC-16 的 1024/32、C1 直接比较绝对时延 |
| MAC-16 T0/T1/T2 | 17/17 请求成功，其中 12 个正式请求、5 个预热/prime；每种正式条件 n=3；外部加载 320 MiB/请求 | 全五档已完成、尾延迟稳定、业务精度正确 |
| T0 source → T1 本机 DRAM | TTFT 均值 170.227 → 53.038 ms；T1 TPOT 约 18.193 ms/token | 该收益可直接外推到其他长度、并发或 SSD |
| T0 destination / T2 | T0 为 515.05/156.55/436.09 ms；T2 为 995.52/52.30/53.72 ms | 远端稳定加速，或两机硬件固有相差两倍 |
| W1 单卡 SSD-only | 1/3/10 MiB 各三轮全量字节、SHA、返回长度一致；盘读每次为 payload + 4096 B | 模型四 worker、多 key、远端 SSD 或 direct 已通过 |
| W1 SDK 读回均值 | DRAM：0.242/0.195/0.354 ms；SSD：1.589/2.171/4.196 ms | 这些是 `get_into + NPU synchronize`，不是模型 TTFT 或纯盘读时延 |
| 64 MiB SSD 单对象 | `FileStorage::AllocateBatch` 的 `sizes[i] <= kMaxSliceSize` 断言失败 | 不能从参考源码推定部署库准确上限；10 MiB 单对象实际已通过 |
| B02 迁移准备 | B02-1 有精确镜像；六个模型元数据文件已打包；B02-2/3 在检查时缺镜像；未见三台已挂载共享文件存储 | 准备包等于模型部署或设备验证完成 |
| 模型 T3/T4、direct | T3/T4 请求 0；无经验证的 SSD→HBM direct 通路 | 不填写性能收益，不称五档验收通过 |

T0 destination 的慢请求主要落在服务端 prefill 阶段，排队不足 0.5 ms；客户端与服务端差值约 2–3 ms。但 prefill 计时仍包含未拆清的阶段，不能直接归因于算子、HCCL或外部训练。T0 当前还会保存新 KV，后续需把回写、计算和通信分开计时，或增加双方一致的关闭回写诊断对照。

## 3. 代码与证据入口

仓库：`https://github.com/Kirrito-k423/nds-ascend-mooncake-baseline`（私有，需要已有读取权限）。原实验 PR：[MAC-16 #4](https://github.com/Kirrito-k423/nds-ascend-mooncake-baseline/pull/4)，分支 `mac-16-five-tier-kv`；交接分支 `mac-16-intranet-handoff` 基于上述固定 commit。

| 文件/目录 | 已开发的能力 | 验证边界或接手注意事项 |
|---|---|---|
| `patches/kv-import-before-device.patch`、`src_overrides/` | 延后导入原生 `ReplicateConfig`，避免过早初始化；Docker 覆盖层 | MAC-3 已设备验证。保留 `set_device` 后再导入 SDK 的顺序；不要当作 r4 的 HIXL 故障已经修复 |
| `tests/e2e/.../test_mooncake_backend_import.py` | 独立子进程导入回归 | 需要 NPU/原依赖；CPU 单测不覆盖它 |
| `benchmarks/tp4_gate.py` | 四 rank 精确矩阵、HCCL all_reduce 校验与短计时 | B02-1 历史通过；新机器必须复测 |
| `benchmarks/sdk_tier_probe.py` | 单进程单 NPU MEMORY/SSD 读回、位置审计、SHA、盘计数 | W1 的 1/3/10 MiB 设备证据；不是模型多 buffer 接口 |
| `benchmarks/kv_tier/launch.sh` | 指纹、身份、四卡空闲检查，90 分钟前台执行与清理 | 固定双机，要求 source/destination 地址不同 |
| `benchmarks/kv_tier/node.py` | 容器、Master、辅助 Store、模型启动；真实启动前再检查占用；日志/盘/网/metrics 快照 | 端口、资源限额、容器名写在代码中；新版组合尚未模型设备复测 |
| `benchmarks/kv_tier/run.py` | 两端编排、native micro、worker 预热、T0→prime→各档、三轮/容量门禁 | `micro()` 的 SSD 仅 1/3 MiB；先启动模型再 micro；**没有**独立的 10 MiB×8 key×4 worker 前置验收，也没有单机开关 |
| `store_agent.py`、`segments.py` | 等租约、清本任务 DRAM、副本审计、只对自身 bucket 做 fsync/fadvise；选取四个贡献 worker 控制 hostname | TE endpoint 与 SDK 控制 hostname 不可混用；四 worker 组合改动待设备验证 |
| `instrument.py` | 临时覆盖 backend，记录 `store_ready`、put、get 起止/字节/RPC | 保留原导入顺序；get 是主机 API 计时，非设备分段时间；put 只有开始事件 |
| `stream.py` | SSE、usage、每 token 到达、TTFT/TPOT；无自动重试 | 精确间隔要求 32 个单 token 事件，不能忽略合并事件 |
| `analyze.py`、`tests/test_kv_tier*.py` | 核验位置、重算 tokens、盘读、三轮后导出 CSV/JSON | 15 项 CPU 测试是证据校验逻辑，不是 NPU 功能通过 |
| `benchmarks/prepare_b02_node.sh` | 迁移附件专用准备器 | 依赖附件 `deploy/machines.json`，固定 B02 名称/地址；**不能直接当通用内网部署脚本运行** |
| `reports/mac16/data/` | 正式请求 CSV、summary、压缩脱敏 JSON | `experiment_complete=false`，保留原运行错误；不能只看 `validated=true` |
| `reports/mac16/w1/`、`reports/mac16/failures/` | 单卡 SSD 全部样本、失败摘要、原始归档哈希 | 完整设备日志在任务私有附件，见下节 |

本文及当前使能手册优先说明此次接续范围。`benchmarks/kv_tier/MATRIX.md` 中旧的“native 1/64 MiB”不应套到 SSD；当前代码的 SSD micro 是 1/3 MiB，64 MiB SSD 已知失败。

## 4. 带入内网的最小材料

| 材料 | 获取与校验 | 是否在 Git 中 |
|---|---|---|
| 代码与本文 | 交接分支及历史，或 git bundle；基点 `e35bcd3...` | 是 |
| 容器镜像 | `nds-vllm-source:kv-import-fix-20261012`，image ID `sha256:64155cd1bb77edf3c927a23c83cdad49e9d5394bd4b90113bc02ce017897c270` | 否；从已持有镜像的环境按授权导出/导入 |
| 模型元数据 | `config.json`、`generation_config.json`、`tokenizer.json`、`tokenizer_config.json`、`vocab.json`、`merges.txt`；逐文件校验 [SHA 清单](mac16-handoff/model-assets.sha256) | 否；已在迁移附件中，无真实权重 |
| 原始错误与设备记录 | `MAC16-KV-calibration-20261013.zip`、`MAC16-W1-native-SSD-20261013.zip` | 完整未脱敏日志不在 Git；摘要/脱敏数据在 Git |
| 迁移附件 | `MAC16-B02-MIGRATION-20261014.tar.gz`，其中 `payload/model/Qwen2.5-72B-Instruct/` 可复用 | 否；包含旧节点地址，仅通过用户授权渠道交接，勿整包提交 Git |

迁移附件来自 MAC-16 的 2026-10-14 10:33（北京时间）评论，attachment ID `01a13842-1e94-79d8-b122-1636c37612ed`。已具备原平台权限的外网环境可执行：

```bash
multica attachment download 01a13842-1e94-79d8-b122-1636c37612ed -o ./handoff-assets
```

内网 AI 无需 Multica 凭据、原 simpleHtmlWatch 代理、旧预约或旧 host 映射。由用户/有权环境带入上述材料即可；不要复制平台 token 或 SSH 凭据。原模型 revision 是 `495f39366efef23836d0cfae4fbe635880d2be31`，若从其他来源准备这六个文件，仍须逐文件 SHA 匹配，不能只核对模型名字。

在线拉取，或在可访问 GitHub 的机器创建离线 bundle：

```bash
git clone --branch mac-16-intranet-handoff https://github.com/Kirrito-k423/nds-ascend-mooncake-baseline.git
cd nds-ascend-mooncake-baseline
git rev-parse HEAD
git merge-base --is-ancestor e35bcd3faa4bd24080ef49c3e36b40a033865bf2 HEAD
git bundle create ../MAC16-intranet-handoff.bundle mac-16-intranet-handoff
# 内网：git clone -b mac-16-intranet-handoff MAC16-intranet-handoff.bundle nds-mac16
# 在内网克隆目录中建立自己的实验分支：
git switch -c mac16-intranet-experiments
```

镜像迁移示例（不需要新建真实权重）：

```bash
# 已持有镜像的机器；先确认有足够归档空间。
docker image inspect --format '{{.Id}} {{.Architecture}}' nds-vllm-source:kv-import-fix-20261012
docker save -o MAC16-image.tar nds-vllm-source:kv-import-fix-20261012
sha256sum MAC16-image.tar > MAC16-image.tar.sha256
# 内网每台节点：
sha256sum -c MAC16-image.tar.sha256
docker load -i MAC16-image.tar
docker image inspect --format '{{.Id}} {{.Architecture}}' nds-vllm-source:kv-import-fix-20261012
```

原镜像 Docker 报告大小约 20.2 GB，不是压缩传输量或迁移耗时。Mac 计算 SHA 可用 `shasum -a 256`。仓内 Dockerfile 只是覆盖层，父镜像为已有私有镜像，**仅 git clone 不能在全空白机器重建完整环境**。若拿不到镜像，先报告此具体缺口；替代镜像或不同软件版本属于新环境，需重做导入、Store、模型功能基线，不能沿用旧结果。

## 5. 环境与资源合同

历史版本：Ascend 950DT（A5）、driver `26.2.0.b020`、CANN/NNAL `9.1.0`、Python `3.12.13`、torch `2.10.0`、torch_npu `2.10.0.post4`、vLLM `0.23.0+nds1.empty`、vLLM-Ascend `0.23.0+nds1`、Mooncake NPU `0.3.11.post1`。容器解释器 `/opt/nds-venv/bin/python`；宿主需要 Linux、Docker、Python 3、`timeout`、`findmnt`、`npu-smi`。A2/A3 或不同代际设备应作为重新适配，不声称已按 A5 验证。

模型固定 80 层、64 Q heads、8 KV heads、hidden 8192，BF16/TP4；`max_model_len=25600`、`max_num_batched_tokens=4096`、`max_num_seqs=32`、`gpu_memory_utilization=0.9`，保留当前 graph 路径。配置长度上限不等于 24K 请求已经成功。

容量估算（**由模型参数计算的假设，需设备实测复核**）：

- 权重粗估：72B × 2 B / TP4 ≈ 33.5 GiB/rank，未计运行时、通信、图和 workspace。
- KV/token/rank = 2(K/V) × 80 层 × (8 KV heads / 4) × 128 head_dim × 2 B = 81,920 B。
- 每 128-token key 为 10 MiB/rank；1024 输入为 8 key/rank、80 MiB/rank，四 rank 共 320 MiB；与历史加载量一致。多 buffer 的具体布局仍须记录真实 `sizes`，不能把 8 key 合并成 80 MiB 单对象。
- 源 DRAM 池配置为 4×100 GiB，目标贡献 0；两端容器各 `--memory 800g --cpus 32 --shm-size 16g`。不是所有主机都能按此配置运行。更小机器先估算和显式修改资源参数、记录新基线；不能静默缩模型。
- 后续并发长度用 `C × (L+32) ≤ 实测 KV token capacity × 80%`，仅是当前脚本的容量筛选，不是 OOM 保证；缺容量日志时只做 C1 并报告缺口。

模型端口：两端 API `18196`；辅助 Store `18197`（loopback）；destination 控制 `18198`；source Master `50296`、metrics `50297`。native 单卡使用 `50316/50317`。HIXL/Store 还会分配动态端口，两个数据 IP 必须互通；不能只放行 HTTP 就假定传输可用。控制/API 为实验接口且无认证，只在已授权实验网络开放。

每节点至少四张持续空闲卡，重新验证实际计算与 HCCL；Warning 本身不判死刑，Alarm/占用则停止。宿主挂载与驱动路径见 `node.py`。KV offload 放任务独立本地 NVMe目录，`findmnt` 核对实际盘；四个 rank 目录落一块盘时只算单盘路径。

## 6. 首轮执行顺序与停止条件

每阶段只改变一个关键因素；保存启动前/后的设备占用、版本、完整 stdout/stderr、退出码与耗时。首次失败保留现场证据，禁止无条件自动重试整个大模型矩阵。

| 阶段 | 最小动作 | 通过条件 | 失败后的动作 |
|---|---|---|---|
| A：资产/设备 | 校验 commit、镜像 ID、六文件 SHA；`tp4_gate.py` | 四 rank 精确矩阵和 all_reduce 均通过 | 停在环境/设备层；不启动模型 |
| B：单卡 | 同一镜像 MEMORY→SSD，1/3 MiB，再独立 10 MiB | setup=0、三轮逐字节/SHA/长度一致、SSD-only、实际盘读 | 记录 HIXL 或读回的首个错误，先定位再扩大 |
| C：模型形状 native | 新增四 worker 探针；每 rank 8 个 10 MiB key，并使用 `batch_get_into_multi_buffers` 的真实布局 | 每个 key/buffer 字节校验；各轮重新构造 SSD-only；本机/远端分别通过 | 不进入模型 SSD 性能测量；不能用辅助 agent 单卡通过替代 |
| D：模型启动 | 私有 Master + 辅助 Store + EngineCore + 四 worker；先只做启动/健康验证 | 每个进程正常初始化、四贡献 worker hostname 可审计、重复启动结果可解释 | 保存 CANN/device 日志及 PID/rank/卡映射，定位最小失败拓扑 |
| E：模型 T3 | 本机 T0 + prime + SSD-only 恢复，1024/32、C1、三轮 | 下节全部证据门禁通过 | 失败归档，不写 T3 成功或进入长上下文 |
| F：模型 T4 | 另一物理机器加载 source SSD，配对其本机 T0 | 增加远端盘位置、跨节点加载证据 | 单独保留 T3 已通过部分，T4 仍未通过 |

排期建议（**工作量假设，置信度中，依赖内网资产就绪**）：第 1 天 A/B 与 HIXL 最小复现，第 2 天 C/D 和 T3，第 3 天 T4 与归档；根因难以收敛时在预算内交付最小复现与已证边界，不承诺三天必然全部通过。

### 6.1 单卡探针的可复用调用模板

下面将 W1 已测命令的节点/目录参数化，**本交接仅做语法检查，未在新设备运行**。先选择授权空闲卡、确认容器名与 50316/50317 无占用，设置 `NDS_REPO`（克隆绝对路径）、`NDS_NATIVE_OUT`（新建的独立结果目录）、`NDS_LOCAL_HOST`（本机数据 IP）、`NDS_SINGLE_DEVICE`（一个物理卡号）。结果目录必须为空；旧 offload 不可复用。

```bash
: "${NDS_REPO:?}" "${NDS_NATIVE_OUT:?}" "${NDS_LOCAL_HOST:?}" "${NDS_SINGLE_DEVICE:?}"
export NDS_LOCAL_HOST
# 首次用 [1048576,3145728]；通过后在新的结果目录用 [10485760]。
export NDS_PROBE_SIZES_JSON='[1048576,3145728]'
mkdir "$NDS_NATIVE_OUT"
timeout -k 10 360 docker run --rm --name nds-mac16-native-handoff \
  --privileged --network host --security-opt label=disable --shm-size 2g --cpus 8 --memory 8g \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /etc/hccn.conf:/etc/hccn.conf:ro \
  -v "$NDS_NATIVE_OUT:/evidence" -v "$NDS_REPO/benchmarks:/task:ro" \
  -e ASCEND_RT_VISIBLE_DEVICES="$NDS_SINGLE_DEVICE" \
  -e ASCEND_LOCAL_COMM_RES='{"version":"1.3"}' \
  -e NDS_LOCAL_HOST -e NDS_MASTER="$NDS_LOCAL_HOST:50316" \
  -e NDS_BOUNDARY_PROBE=1 -e NDS_PROBE_SIZES_JSON -e NDS_OWN_FILE_COLD=1 \
  -e MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=134217728 \
  -e MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=536870912 \
  -e MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE=536870912 \
  -e MOONCAKE_OFFLOAD_BUCKET_SIZE_LIMIT_BYTES=268435456 \
  -e MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT=1 \
  -e MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS=1 -e MOONCAKE_OFFLOAD_USE_URING=false \
  nds-vllm-source:kv-import-fix-20261012 bash -lc '
set -e
ulimit -c 0
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/opt/nds-venv/bin:$PATH
python3 - <<"PY"
import socket
for port in [50316,50317]:
    with socket.socket() as s:
        s.bind(("0.0.0.0",port))
PY
mooncake_master --port=50316 --metrics_port=50317 --enable_offload=true --default_kv_lease_ttl=5000 > /evidence/master.log 2>&1 &
master_pid=$!
trap "kill $master_pid 2>/dev/null || true; wait $master_pid 2>/dev/null || true" EXIT
sleep 2
kill -0 "$master_pid"
python3 /task/sdk_tier_probe.py memory > /evidence/memory.log 2>&1
python3 /task/sdk_tier_probe.py disk > /evidence/disk.log 2>&1
'
```

容器前台退出后核验退出码、`native-memory.json`/`native-disk.json`、`functional_pass`、每轮 `exact`/SHA/bytes、副本与盘读；异常超时检查并只停止本次 `nds-mac16-native-handoff`。先归档日志和文件清单，再删除本次结果目录内自己生成的 offload。不要清全局 page cache，也不要把日志/失败与 offload 一起删掉。vendor ATB `set_env.sh` 曾因 `set -u` 引用未定义 `ZSH_VERSION` 失败，因此容器内沿用 `set -e`。

四卡计算/HCCL可复用相同的驱动挂载与镜像，将 `benchmarks/tp4_gate.py` 放在 `/evidence/tp4_gate.py`，选择四卡后执行：

```bash
/opt/nds-venv/bin/python -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29596 \
  /evidence/tp4_gate.py
```

以上命令在已 source CANN/ATB 的容器内执行；先检查 29596 空闲。输出应有 `tp4-rank0.json` 至 `tp4-rank3.json`，两个 correctness 字段为 true。历史镜像未提供 `torchrun` 入口，曾遇 `--standalone` 主机名解析问题，故使用显式地址。

### 6.2 四 worker 探针与单机入口是待补代码

**不可把“下一步要做”误读为“仓内已有命令”。** 接手 AI 应在实验分支补两处最小能力，并用真实设备验证：

1. 四 worker 探针：复用 `sdk_tier_probe.py` 的位置敏感数据、2 MiB 地址对齐、目标清零、同步及逐字节校验；从模型 `NDS_TRACE put/get_begin` 读取每 key 的多 buffer `sizes`，使用模型相同的 `batch_get_into_multi_buffers` 接口。每个 rank 8 key、10 MiB/key；正确映射四个贡献 Store 的 `get_hostname()`，排除零贡献辅助 agent/调度进程。每轮重新清除所有 DRAM 副本并审计，先本机，后跨机。独立 native 成功后仍须验证真实模型进程拓扑。
2. 若只有单机：把 `run.py` 的角色集合、快照、warmup、micro 和 tier 循环改成显式 local-only 模式，并同步调整 `launch.sh` 身份检查与分析覆盖声明；不访问 destination，不生成 T4。保持 T0/T3、三轮、全部位置/盘读/重算门禁。**不能令 source/destination 指向同机来冒充 T4，也不能简单注释掉所有远端失败。** 若双机齐备，可先用现有双机入口完成 T3/T4，无需为形式实现单机模式。

## 7. 双机模型校准：现有入口及参数

以下入口**在阶段 A–D 通过后**使用；当前代码会先启动两端模型，再调用小对象 micro，并不自动代办上述前置门禁。原中台不在内网也可运行：`SHW_RESULTS_DIR` 是本脚本要求的结果路径变量，内网调度器可赋予本次任务专属绝对目录，无需伪造平台预约。

每节点建立独立实验目录，将当前 `benchmarks/kv_tier/` 全目录复制进去，将六个模型文件放到独立 model 目录。`NDS_PROJECT_DIR` 必须指向含 `launch.sh` 的目录，不能指向仓库根目录。模型 SHA 验证：

```bash
export NDS_REPO='/home/t00906153/nds-mac16'
export NDS_MODEL_DIR='/home/t00906153/nds-mac16-assets/Qwen2.5-72B-Instruct'
(cd "$NDS_MODEL_DIR" && sha256sum -c "$NDS_REPO/docs/mac16-handoff/model-assets.sha256")
```

两端分别写自己的环境文件，source 取 `NDS_ROLE=source`、本机 IP=source；destination 取 `destination`、本机 IP=destination。以下占位 IP 必须替换为现场值；选定目录不可覆盖其他任务：

```bash
export NDS_ROLE=source
export NDS_SOURCE_HOST='<源物理节点数据IP>'
export NDS_DESTINATION_HOST='<另一物理节点数据IP>'
export NDS_LOCAL_HOST="$NDS_SOURCE_HOST"  # destination 上改成 "$NDS_DESTINATION_HOST"
export NDS_PROJECT_DIR='/home/t00906153/mac16-intranet-run/kv_tier'
export NDS_MODEL_DIR='/home/t00906153/nds-mac16-assets/Qwen2.5-72B-Instruct'
export SHW_RESULTS_DIR='/home/t00906153/mac16-intranet-run/results-source'
# destination 使用独立 results-destination；两端目录预先创建且为空。
export NDS_IMAGE='nds-vllm-source:kv-import-fix-20261012'
export NDS_VISIBLE_DEVICES='0,1,2,3'
export NDS_MODES_JSON='["disk"]'
export NDS_CONFIGS_JSON='[[1024,1]]'
export NDS_OFFLOAD_BUCKET_KEYS_LIMIT=1
export NDS_OWN_FILE_COLD=1
export NDS_TOOLKIT_SHA256="$(python3 - <<'PY'
import hashlib, os
from pathlib import Path
root=Path(os.environ['NDS_PROJECT_DIR'])
rows=[p.name+':'+hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))]
assert rows
print(hashlib.sha256('\n'.join(rows).encode()).hexdigest())
PY
)"
bash "$NDS_PROJECT_DIR/launch.sh"
```

先运行 destination 的前台控制任务，再在另一个受控任务中运行 source。控制者需收集两端至终态；不以启动 PID 或 API health 代替实验完成。脚本固定容器名 `nds-mac16-source`/`nds-mac16-destination`，开始前确认没有其他任务同名容器：`launch.sh` 的退出 trap 会清理该名称。源端协调正常/异常收尾，目标最长 90 分钟；仍须核验自有容器、端口和 offload 清理结果。不要在两个独立任务之间靠结束 AI 顶层运行“等回调”。

注意区分计时窗口：模型 `batch()` 在请求结束后额外等待 1 秒异步保存，再采集 after 快照；这 1 秒不计入客户端 TTFT/TPOT，但包含在模型盘/网快照窗口中。单卡 `sdk_tier_probe.py` 的盘计数窗口围绕 get+sync，没有这段 1 秒等待。两类原始数值不能不加区分地比较。

两端都核对脚本指纹一致；每次改代码后重新部署并重算，不重用旧 `NDS_TOOLKIT_SHA256`。默认脚本 `.py` 指纹不含 shell，完整交接还应保存 `git rev-parse HEAD` 与所有部署文件的 SHA。

### 必须满足的证据门禁

- 每次正式读取前 HBM 前缀未命中；T0 重算 1024 tokens，恢复请求仅重算 1 token，外部命中 1023；实际完整块加载量按 native 返回值统计，1024 基线历史为 320 MiB/请求。
- T3 读前只保留本机 COMPLETE LOCAL_DISK；T4 只保留 source 物理节点 COMPLETE LOCAL_DISK，destination 不贡献 DRAM。source/destination 不是同一请求的两个阶段。
- 等磁盘副本完成、停止查询并等待 6 秒租约，再按贡献 worker 控制 hostname 清 DRAM；只对自有 bucket `fsync + POSIX_FADV_DONTNEED`；重新审计；每轮重新构造，防止上一轮回填。
- 记录每 rank native 返回字节、get 起止、offload RPC 增量；源盘读量须覆盖恢复载荷。共享主机计数只是组合证据，不能独立归因；T4 另保存物理节点/读取 endpoint 和跨机传输证据。网卡计数不一定覆盖 UB，缺真实链路计数则明确缺口。
- 实际输出 32 tokens，SSE 每 token 事件可分辨；3 轮全部保存，异常和首读不剔除。TTFT/TPOT、吞吐、成功率、排队、各 rank HBM 一并记录。

分析与离线复核：

```bash
python3 -m unittest discover -s tests -p 'test_kv_tier*.py' -v
python3 benchmarks/kv_tier/analyze.py /path/to/new-source-results
# 只复核历史不完整运行的已完成分组：
mkdir -p results/history-recheck
tar -xzf reports/mac16/data/raw-json.tar.gz -C results/history-recheck
python3 benchmarks/kv_tier/analyze.py results/history-recheck --allow-partial
```

新正式运行应检查 `sampling_finished`、无 `error`、目标 tier 及轮次完整；分析器 `experiment_complete=true` 只表示所选 modes/configs 完成，**不表示所有五档、长上下文和 direct 已完成**。`--allow-partial` 不能用于给全任务验收兜底。归档必须含 destination 的日志，不能只交 source CSV。

## 8. 未解决问题与定位路线

| 问题 | 事实 | 最短验证与关闭条件 |
|---|---|---|
| HIXL/AICPU 初始化 | 原 B07 r4：`HixlSyncTransferContext` / `libcann_hixl_kernel.so` 报 `0x2a`，AdxlEngine `507018`，TransferEngine `-1`；发生在模型就绪前 | 固定镜像/节点/设备映射，依次单卡、四 worker、完整 EngineCore+Store 拓扑。采集完整 server/device/CANN日志、PID/rank、动态库版本和占用时间线；必须用原失败拓扑复测修复，不以另一节点单卡成功关闭 |
| 外部进程冲突 | 启动检查后出现外部八卡进程，时间上伴随上述报错 | **假设**：共享占用触发。先用持续隔离窗口复测；隔离下仍失败则外部占用不足以解释。不要故意抢占别人业务制造冲突 |
| 软件/拓扑相关初始化 | 同镜像 B02 单卡成功，完整模型多进程尚未通过新组合验证 | **假设**：初始化顺序、进程设备映射、实际库加载/节点状态差异。只有最小复现支持后才逐因素更改，不能仅凭错误码猜驱动不兼容 |
| 单对象与 batch 边界 | 64 MiB SSD abort；10 MiB 单对象通过 | 保持模型每 key 10 MiB，验证 8-key/multi-buffer；不要重新用 64 MiB 当例行 smoke，也不宣称精确最大值 |
| 位置构造 | K32 小样本未 seal，K1 成功；TE endpoint 不能用于控制副本清除 | 使用当前 K1 与 `store_ready/get_hostname`，四 rank 必须齐全；失败不绕过审计 |
| TTFT 波动与首读 | T0 destination prefill 波动、T2 首读 995.52 ms | 实际 worker 预热后交错重测；补 put 完成、prefill算子/通信/保存等待打点。只记主机 get 起止不足以拆分根因 |
| 端口/清理 | 老版本 TIME_WAIT 误报后加 SO_REUSEADDR；fatal-TE 新增快速退出 | 先核验真实监听与自有容器；SO_REUSEADDR 不意味着可复用他人活动端口。新版退出/清理还需设备验证 |

原错误索引：`reports/mac16/failures/r4-abort-reason.json`；首轮附件内 `raw/ssd-calibration-source-r4.tar.gz → results/disk/server.log`。后续增加的“启动前再检查占用、fatal 错误提前停止、worker 预热”是保护/测量改进，不等于 HIXL 根因修复。

## 9. T3/T4 通过之后再做什么

按优先级推进，不在基础功能未过时盲扫矩阵：

1. 实测容量内逐步增加 8K/16K/24K 与 C1/8/25，固定至少 32 输出；每配置至少三轮，报告跳过原因。C1 的 P95/P99 只描述这三个样本。
2. 分别控制 HBM 压力与外部 DRAM 池压力；当前 eager SSD 持久化+人工清副本不等于自动压力换出已验证。活跃并发和累积前缀量分别记录。
3. 补 SSD盘读、Host staging、跨节点传输、H2D、同步、调度的分段与重叠证据；不要相加重叠阶段，不跨机相减单调时钟。Host DDR/PMU 需校准单位、覆盖范围和背景，不能由盘字节推算 Host DRAM 流量。
4. NDS/direct：先取得在当前 A5 可调用的真实设备内存直达接口，再做等字节、等并发、等缓存状态的 staged/direct 对照及正确性校验。`O_DIRECT`、io_uring、注册 NPU buffer 或 HIXL 跨机成功均不单独证明 SSD→HBM direct。无通路时只报告缺口和显式假设的收益上限。

## 10. 回传与验收

建议每 run 目录包含：`run-manifest.json`（commit/镜像/版本/拓扑/设备/参数/开始结束/退出码）、两端完整结果、native 四 worker 逐 key 校验、日志与设备错误、`requests.csv`/`summary.json`、清理证据及全文件 SHA。新增字段/脚本由内网 AI 实现时注明，不把建议目录误当已有输出。

按以下格式提交阶段结论和 PR（标题带 MAC-16）：

| 验收项 | 目标 | 实测/证据路径 | 结论 |
|---|---|---|---|
| 初始化 | 目标进程拓扑可重复启动，无 HIXL/AICPU 错误 | 必填 | 通过/未通过 |
| native 模型形状 | 四 rank、8 key/rank、10 MiB/key，全量正确 | 必填 | 通过/未通过 |
| T3 | 仅本机 SSD、真实盘读、三轮1024/32/C1、真实加载tokens/bytes | 必填 | 通过/未通过 |
| T4 | 另一物理节点 SSD、跨机加载证据、同上请求要求 | 必填 | 通过/未通过 |
| 未覆盖项 | 长上下文、压力、分段、Host流量、direct | 逐项填写 | 未测不得写通过 |

**总验收只取“通过 / 有条件通过 / 不通过”之一**，并列待办、停止条件和下一次验收点。基础 T3/T4 通过不等于整项五档与 GDS/NDS 收益穿刺完成；dummy 结果不评价业务精度。原始日志通过授权渠道回传，Git 只放不含凭据/真实机器映射的代码、报告与可审计数据；不可删除失败记录。
