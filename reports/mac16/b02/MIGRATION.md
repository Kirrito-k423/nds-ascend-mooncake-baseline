# B02 迁移与现场校验

2026-10-14 已将环境迁移到 B02-1/2/3：三台有相同镜像、最新脚本与原 baseline 的完整配置/tokenizer，CPU 离线校验通过。默认 source 为 B02-1、destination 为 B02-2，B02-3 为备用 destination。本轮没有新增 NPU 模型请求；T3/T4 仍为 0，五档目标未完成。

| 节点 / 数据 IP | 迁移前目标镜像 | 迁移后镜像 | 文件 SHA 校验 | CPU 配置 / tokenizer |
|---|---|---|---:|---|
| a5-b02-1 / 177.5.21.1 | 已有 | 同一 SHA | 19/19 | 80 层 / 往返通过 |
| a5-b02-2 / 177.5.22.1 | 标签及 SHA 均缺 | 导入，同一 SHA | 19/19 | 80 层 / 往返通过 |
| a5-b02-3 / 177.5.23.1 | 标签及 SHA 均缺 | 导入，同一 SHA | 19/19 | 80 层 / 往返通过 |

镜像 `nds-vllm-source:kv-import-fix-20261012` 的 ID 是 `sha256:64155cd1bb77edf3c927a23c83cdad49e9d5394bd4b90113bc02ce017897c270`，Docker Size 20,221,370,011 bytes。从 B02-1 导出的 gzip 文件实际为 7,816,845,122 bytes，SHA `9f152017fd07e283d6ddca97238736a1d749f71450178fa90c5c302dbabf4534`。B02-2/3 均完成全文件大小与 SHA 校验，再导入并核对镜像 ID。没有构建替代镜像或下载真实权重。

控制 Mac 当时仅约17GiB可用，未发现挂载的外置 PSSD，因此大镜像未落到 Mac。最终由 B02-1 的临时文件服务经内网逐台分发，固定绑定 177.5.21.1:50318，只接受两个 destination IP、只提供镜像及SHA两个文件，每路限速128MiB/s、服务限时20分钟。服务完成4次完整请求后退出；该任务已终态归档。上述大小是迁移文件量，不是 SSD native 耗时、TTFT 或 Host DRAM 流量。

六个模型元数据文件来自原 B07-7，合计11,488,527 bytes；config SHA `14ca217334fe0fd10148413592d68c99eeb33431ed89c1afa130fee560be2a29` 与 baseline 一致。每台同时保留元数据清单、9个 toolkit Python 文件、launch.sh、SDK probe 和 launch.env。kit SHA `56cedb68c9ba73d95613238215f0aa41a9c45bb9273098ec6d6fa94e0ce8076b`；probe SHA `e5367b7dcbe5f3105fef6e14b085f420c48988e71a873283627b02436e95009b`。逐文件 SHA、Python AST、实际本机 IP 和镜像 ID 均已核对。

## CPU 验证与执行边界

三台在不挂载NPU设备、网络关闭的临时容器中，实际执行 AutoConfig.from_pretrained 和 AutoTokenizer.from_pretrained 的 local_files_only 路径。验证80层、8192 hidden、64 Q heads/8 KV heads；固定中文文本编码为8个token，解码与输入逐字一致。未把这8个CPU smoke token当成1024输入/32输出的模型样本。

该CPU容器显式设置 TORCH_DEVICE_BACKEND_AUTOLOAD=0；仅避免无设备环境自动加载torch_npu，未写入模型launch.env。读取的分发包版本为 torch 2.10.0、torch-npu 2.10.0.post4、vllm 0.23.0+nds1.empty、transformers 5.5.4。这些是现场元数据字符串，不代表已验证上游对应实现；未找到名为mooncake-transfer-engine的distribution也不能推断SDK缺失。本轮没有调用NPU计算、HCCL、TE/HIXL或SSD native初始化，B02-2/3不能据此宣称计算可用。

三台环境位于 `/home/t00906153/nds-mac16-b02-20261014`；`kv_tier`、`model/Qwen2.5-72B-Instruct`、`sdk_tier_probe.py`、`launch.env`及`staging-manifest.json`保留供预约后启动。prepare helper会拒绝覆盖已有目录，恢复时先核对manifest，不能再次盲跑。source默认对B02-2；选B02-3时在source的任务环境覆盖NDS_DESTINATION_HOST=177.5.23.1，destination使用B02-3的launch.env。可用设备号必须按预约和新鲜门禁覆盖默认0–3。

`benchmarks/prepare_b02_node.sh` 与 `benchmarks/check_b02_staging.py` 是可复核的准备/检查脚本；完整私有传输helper、命令、终态、SHA及日志在issue附件。原STAGING传输包SHA是 `326367fef92916121d4a9ffdbe48707a5f33449cc23086187f56a0b4a189baa7`；后续恢复附件独立封装，不替换该历史指纹。

## 共享存储与清理

先前完整host挂载清单未见已挂载NFS/NFS4、Lustre、Ceph、CIFS、GPFS、BeeGFS或FUSE文件存储，autofs仅是binfmt_misc。三台新环境实际findmnt均落在本地ext4 `/dev/nvme0n1p4`。当时RAM各2261.4GiB，根盘余量分别约3.00/2.34/2.22TiB。结论只覆盖这三台和检查时的挂载，不推断全群没有共享存储；迁移未依赖共享盘。KV SSD offload后续仍放各自本地NVMe，不能用共享盘替代位置对照。

三台本轮CPU容器均自动删除，临时50318服务已退出；完整镜像压缩文件和自有失败传输残片在校验后删除。只保留已导入镜像、专属环境、轻量恢复材料。未停止外部训练，未改驱动/盘结构，也未修改中台排队代码。原B01/B07任务未被本轮中断。

## 失败与完整证据

| 阶段 | 原始结果 | 后续处理 |
|---|---|---|
| 初次SSH流式镜像 | 0bytes，source/destination SSH退出255/255/1 | 查原连接及镜像；无活动save/gzip、目标仍无镜像。连接失败因果未定位，改为先导出固定文件 |
| 控制机SCP -3 | 观察到57,185,280B，约0.6MB/s | 精确结束本轮SCP子PID45252，退出1；转内网，保留回执 |
| B02-2第一次内网接收 | 串行服务搭配并发客户端，curl低速等待超时28；源端该请求0B/BrokenPipe | B02-3先完成，改为逐台传输后B02-2成功 |
| B02-2重试前检查 | 不存在的partial文件stat退出1，未发GET | 按真实文件状态处理后执行成功；原失败保留 |
| B02-1首次CPU检查 | torch自动加载NPU backend失败，task退出1；文件已展开 | CPU容器禁自动加载，查已展开manifest后复核通过，未重放prepare |

本轮6个中台task全部finishedAt/archiveReady，5个退出0、1个CPU前置检查失败；累计中台task由127增至133。失败与修正后样本都保留，见task-terminals.json。文件传输/只读查询的终态见transfer-terminals.json；它们独立于历史20次NPU短诊断计数。没有本轮任务留在后台等待接管。

## 下一步

匹配镜像和模型元数据的缺口已解决。迁移收尾时发现用户安排的中台维护已完成升级：稳定serviceId不变，live版本为dev-forwarded-reservations-20261014.1，新增forwarded-resource-identity-v1。更新后的接口允许不同物理ID/回环端口组成整组，原proxy host去重限制已修复。两次新鲜占用快照中B02-1/2/3均显示8卡无进程、健康Warning；这只说明资源候选，未实际取得整组预约，也未证明健康或独占。用户在03:08/03:10Z已另行安排AIConnector及内网AI接续实测，SE交接文档见PR#5；本轮交付迁移与证据，不重复启动同一计划。本地重复资源复查规则停用，原截止归档规则保留。B02-1只有W1历史四卡计算/HCCL与单卡native证据；三台都需新鲜占用快照和实际启动复查，B02-2/3还需真实计算/HCCL校验。

有效资源窗口内先native TE/HIXL及真实10MiB×8key、四worker同机/跨机SSD-only读回：读前COMPLETE LOCAL_DISK-only、租约过期清内存、逐字节/SHA和真实盘读通过，再1024/C1、实际32输出、至少3轮T3/T4。校准后才能扩长上下文/压力。CPU迁移通过不能代替这些门禁；无可调用SSD→HBM direct仍交具体缺口。硬截止2026-10-16T09:34:44Z，PR#4保持draft，issue保持in_progress，完整目标仍未完成。
