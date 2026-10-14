# B02 迁移核验

2026-10-14 的只读现场检查支持迁移到 B02；尚未部署完整模型环境或取得双机预约。B02-1 已有目标镜像，B02-2/3 缺失同名镜像及精确 SHA 对象。三台原 Qwen 目录均缺配置/tokenizer，不能把资产缺失写成模型初始化失败。

| 节点 | 数据 IP（现场读取） | 目标镜像 | RAM 总量 / 检查时可用（GiB） | 本地根盘可用（TiB） |
|---|---|---|---:|---:|
| a5-b02-1 | 177.5.21.1 | 同 SHA 已存在 | 2261.4 / 2238.8 | 3.00 |
| a5-b02-2 | 177.5.22.1 | 标签和 SHA 均不存在 | 2261.4 / 1934.8 | 2.34 |
| a5-b02-3 | 177.5.23.1 | 标签和 SHA 均不存在 | 2261.4 / 1198.3 | 2.22 |

完整 host mount 清单未见已挂载 NFS/NFS4、Lustre、Ceph、CIFS、GPFS、BeeGFS 或 FUSE 文件存储；autofs 仅是 `/proc/sys/fs/binfmt_misc`，没有据此确认共享存储。三台 `/home` 均落在根盘 `/dev/nvme0n1p4`、ext4。此结论仅覆盖这三台及检查时已挂载的文件系统，不排除集群有尚未挂载的共享存储。

目标镜像 `nds-vllm-source:kv-import-fix-20261012` 的 SHA 为 `64155cd1bb77edf3c927a23c83cdad49e9d5394bd4b90113bc02ce017897c270`，B02-1 的 Docker Size 为 20,221,370,011 bytes。这是镜像报告大小，不是已测网络传输量、压缩归档大小或迁移时长。B02-2/3 应导入同一既有镜像并复核 SHA，不盲构建替代环境。

已从原 a5-b07-7 仅取回六个模型元数据文件：config、tokenizer、tokenizer_config、generation_config、vocab 和 merges。合计 11,488,527 bytes；中台压缩回收包 3,774,934 bytes。config SHA 与原 baseline 的 `14ca217334fe0fd10148413592d68c99eeb33431ed89c1afa130fee560be2a29` 一致。没有读取或复制真实权重；仍使用完整结构的 dummy BF16 TP4。

迁移附件包含上述元数据、最新 kv_tier 脚本、SDK probe、映射、SHA 清单及准备脚本。大 tokenizer 文件只在私有附件中交付，不进入 Git。`benchmarks/prepare_b02_node.sh` 在包内为 `deploy/prepare-node.sh`：校验包哈希、镜像和实际本机 IP 后写入新的专属目录并生成环境变量，不启动模型；该准备脚本仅通过本地语法和包校验，尚未远端执行。

迁移无需共享盘。KV 的 SSD offload 应继续放各节点本地 NVMe，保留本机/远端位置和物理盘读证据；共享文件盘不能代替该对照。

当前中台将不同 SSH proxy 端口的 host 均表示为 127.0.0.1。既有双预约的第二次 409、接口合同的 host 去重及本轮未变的服务身份/版本，仍不支持同时取得两台物理节点预约。等待接口也按 resourceId 和 host 去重，不能用登记等待单声称已经获得双机。用户最新指令已使 B02 成为优先复查范围；尚未提交新的模型任务。

B02-1 的四卡计算/HCCL与单卡 native 1/3/10MiB是 W1 的历史设备证据；B02-2/3 本轮只有只读环境检查，Warning 不等于已通过计算与通信。双机预约成立后仍需重新核验资源，完成 10MiB×8 key / 四 worker 同机与跨机 native 读回，再做 T3/T4 模型校准。T3/T4 模型样本仍为 0。

本轮 6 个 metadata task 均退出 0、finishedAt 与 archiveReady 完整，累计任务数由 121 增至 127；此前 20 个短诊断计数未变。两次镜像 SHA 查询返回缺失，结果完整保留，未重放远端任务。客户端首次将归档流误当 bytes，随后只查询原 ID 并按流回收；没有重复提交该任务。
