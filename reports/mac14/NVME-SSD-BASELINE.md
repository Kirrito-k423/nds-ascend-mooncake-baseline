# A5 NVMe 与 Mooncake SSD 读路径基线

2026-10-13 11:54–12:12（北京时间），MAC-14。B07-3 只读普查，B07-7 完成盘读测量和原生 Ascend SSD 校验；使用管理实例 `netdata-mac-preview` 的独立预约。最终所有远端任务已退出、结果已回收、预约已释放。

**当前可以用于文件级 offload 的是一个系统共享 NVMe 路径。四块 NVMe 没有组成 RAID，也没有四个独立挂载。** 单盘顺序读实测约 7.46 GB/s；四线程仍约 7.48 GB/s。另外三块没有文件系统或挂载，本轮没有改盘结构，因此没有四盘聚合测量值。Mooncake NPU 0.3.11.post1 已完成 NPU put → SSD 副本 → 清除 DRAM 副本 → SSD 读回 NPU 的最小校验；此结果不代表模型 T3/T4 延迟或恢复带宽。

## 物理映射

两台机器的 `lsblk`、`findmnt`、mdstat、LVM 和 PCI 查询给出相同布局。每盘容量 3,840,755,982,336 B（3.8408 TB / 3.4932 TiB），型号 `HWE6AP443T8L00KN`、固件 2025；namespace ID 1，逻辑/物理块均 512 B。四个 NVMe PCI 控制器均在 NUMA 2，PCIe 16 GT/s ×4。

| namespace | PCI 端点 | 分区/文件系统 | 当前角色 | 本轮读测量 |
|---|---|---|---|---|
| nvme0n1 | 0000:91:00.0 | p1=vfat；p2=ext4；p3=swap；p4=ext4 | EFI、boot、swap、根目录；与 Docker 和现有业务共享 | 已测 p4 的专用文件 |
| nvme1n1 | 0000:92:00.0 | 无可见分区、文件系统、挂载 | 是否可重新分配需确认 | 未测：无文件级路径 |
| nvme2n1 | 0000:93:00.0 | 无可见分区、文件系统、挂载 | 是否可重新分配需确认 | 未测：无文件级路径 |
| nvme3n1 | 0000:94:00.0 | 无可见分区、文件系统、挂载 | 是否可重新分配需确认 | 未测：无文件级路径 |

隔离容器 `/offload` → 宿主项目专用目录 → `/`（ext4）→ `/dev/nvme0n1p4` → `nvme0n1` namespace 1 → PCI `0000:91:00.0`。使用 bind mount；数据文件没有落入容器 overlay 或 tmpfs。宿主实际部署目录保存在问题附件，不写入仓库。

`/proc/mdstat` 无活动阵列，pvs/vgs/lvs 查询为空，dmsetup 无映射，PCI 查询显示四个 NVMe 控制器而非 RAID 逻辑盘。根挂载选项是 `rw,relatime,seclabel,stripe=32`。`stripe=32` 是 ext4 分配大小和对齐提示，不能据此认定存在 RAID；下层设备和实测盘计数器都只有 nvme0n1。参见 [内核 ext4 文档](https://www.kernel.org/doc/html/latest/admin-guide/ext4.html)。本结论描述操作系统可见路径，不替代固件内部实现审计。

B07-7 普查时 ext4 可用 2,955,512,803,328 B（2.956 TB / 2.688 TiB）；B07-3 为 3,157,731,254,272 B。可用空间是当时快照，不是独占配额。`/tmp` 为 tmpfs，不能拿来测 SSD。现有业务容器保持运行，未改动共享软件。8 卡均 Warning、无 npu-smi 可见进程；NPU 最小数据计算与最终读回通过，未用 Health=OK 过滤机器。

## fio 实测

fio 3.34（openEuler `fio-3.34-6.oe2403sp3.aarch64.rpm`），官方源下载并验证 RPM 签名/摘要。fio 和缺失的 gperftools 库解包到项目目录，未全局安装。内核 `6.6.0-145.3.33.164.oe2403sp3.aarch64`。CPU 192–195 属于磁盘 NUMA 2；只限制测试线程，不改系统 CPU 配置。

32 GiB 专用文件经一次完整直接写和完整 CRC32C 读校验，退出码 0；文件大小 34,359,738,368 B，实际分配 34,359,742,464 B，filefrag 保留完整 extent 证据。正式测量全为只读，`io_uring / direct=1 / iodepth=32 / thread=1`；每线程一个队列，1 或 4 线程，共享同一个文件。每次预热 3 s，计量 15 s，四种负载交错重复三轮，随机种子 20261013。无全局 page-cache 清理。

| 负载 | 三轮 GB/s（十进制） | 平均 GB/s | 平均 IOPS | BW 样本 CV | fio CPU 合计 |
|---|---|---:|---:|---:|---:|
| 顺序 1 MiB，1 线程 | 7.4715 / 7.4424 / 7.4550 | 7.4563 | 7,108.8 | 0.196% | 24.75% |
| 顺序 1 MiB，4 线程 | 7.4775 / 7.4776 / 7.4780 | 7.4777 | 7,123.0 | 0.004% | 27.88% |
| 随机 4 KiB，1 线程 | 1.5615 / 1.5319 / 1.5689 | 1.5541 | 379,418.1 | 1.259% | 68.74% |
| 随机 4 KiB，4 线程 | 3.9525 / 3.9176 / 3.9323 | 3.9342 | 960,480.6 | 0.446% | 292.52% |

CPU 合计按各 fio worker 的 user+system 百分比求和，100% 相当于一个核心；全主机 `/proc/stat` busy 在各次窗口约 0.71–2.68%，含其他业务。原始 fio JSON 同时保留 worker CPU、队列深度、disk_util 和完整延迟；0.5 s 的盘/CPU 采样、各次退出码在附件。队列深度主要保持在 32。

| 负载 | P50（µs） | P95（µs） | P99（µs） |
|---|---:|---:|---:|
| 顺序 1 MiB，1 线程 | 4,145–4,358 | 8,290–8,716 | 10,289–10,945 |
| 顺序 1 MiB，4 线程 | 16,318–16,908 | 28,180–32,637 | 36,438–38,535 |
| 随机 4 KiB，1 线程 | 79.36–81.41 | 117.25–119.30 | 144.38–148.48 |
| 随机 4 KiB，4 线程 | 85.50–374.78 | 177.15–468.99 | 240.64–536.58 |

范围涵盖三轮各 worker 的百分位，**不是合并后的百分位**。四线程顺序读增加排队延迟，没有增加总带宽；四线程随机读增加吞吐，同时出现明显 worker 延迟差异。

十二个正式文件级读任务全部退出码 0、fio error=0、写数据字节=0。每次 nvme0n1 物理读增量都覆盖 fio 计量读字节；顺序每次约 124.7–125.4 GiB，随机约 25.7–66.3 GiB。物理计数窗口含 3 s 预热，fio 计量不含预热，不能直接把二者差值当成开销。其余三盘读增量全部为零。采样可能包括系统共享盘的少量其他 I/O，但无法用其解释约 7.48 GB/s 的计量读流量。

7.48 GB/s 是**本轮配置测得的单盘文件读上限**。没有测试所有队列/NUMA 参数，也没有跨设备聚合路径，不能把它推广成四盘极限或 Mooncake 恢复带宽。另一台及另三盘没有带宽结果，不用型号相同替代实测。

## Mooncake 原生 SSD 最小校验

固定已有镜像 `nds-vllm-source:kv-import-fix-20261012`，image ID `sha256:64155cd1bb77edf3c927a23c83cdad49e9d5394bd4b90113bc02ce017897c270`；Mooncake NPU 0.3.11.post1、torch 2.10.0、torch_npu 2.10.0.post4、Python 3.12.13、CANN/NNAL 9.1.0、驱动 26.2.0.b020。SDK 必须在 `torch.npu.set_device(0)` 后加载。

已安装 SDK 的 docstring 确認参数 `enable_ssd_offload=True`、`ssd_offload_path`；独立 Master 实际接受 `--enable_offload=true`。使用 64 MiB DRAM segment、32 MiB client buffer 和 32 MiB offload buffer；SSD 总容量和 bucket quota 都显式设为 512 MiB，bucket 8 MiB、每 bucket 1 key，heartbeat 1 s，`MOONCAKE_OFFLOAD_USE_URING=false`。实际 Master 日志显示 SSD 池 512 MiB。完整环境字段保存在 `ssd-offload-result.json`；[官方同版本文档](https://github.com/kvcache-ai/Mooncake/blob/v0.3.11.post1/docs/source/deployment/ssd-offload.md)及参考源码说明了这些配置，运行证据优先于文档假设。

1. NPU 写入确定的 uint8 1 MiB 数据，setup=0、put=0；出现 COMPLETE 的 MEMORY 和 LOCAL_DISK 副本。
2. 等待最后一次副本查询的 5,000 ms 租约过期，实际等待 6 s，再清除指定内存 segment。读取前审计仅有 COMPLETE 的 LOCAL_DISK。
3. Bucket 数据文件 1,048,612 B，分配 1,052,672 B；meta 50 B，分配 4,096 B。每次将目标 NPU buffer 置零，三次读回均返回 1,048,576 B，并逐字节匹配输入。SHA256 `fbbab289f7f94b25736c58be46a994c441fd02552cc6022352e3d86d2fab7c83`。

小样本读取时间 1.482 / 0.515 / 0.414 ms 仅用于复核功能；没有冷盘准备、吞吐统计或业务并发，**不能据此报告 SSD 恢复带宽、TTFT 或 decode 收益**。已验证 eager persistence 和显式清除内存后的读取；没有测 DRAM 压力自动换出、io_uring 对照、模型连接器、跨节点 SSD 或 SSD→HBM 直接 DMA。

首次清除失败的证据完整保留：Master 明确报告 `has active lease, skipping`，脚本未把仍有 DRAM 副本的读取当成 SSD 成功。等待租约后复测通过。单独 `mooncake_master --helpshort` 曾发生 allocator abort；正常 Master 启动和 SSD 流程都通过，未把查询失败等同于 SSD 不支持。

## 给 T3/T4 的配置边界

当前单 client 只有一个 `ssd_offload_path`。多个目录若都在 `/`，仍共享 nvme0n1；该参数和 bucket 文件分组没有证明自动跨四盘条带。**T3 当前按一盘系统共享路径设计，T4 尚需独立目标节点的 SSD owner 与远端读自证。** 不使用 27 GB/s 假设作为验收值。

如需扩展到多盘，先由管理方确认另外三盘的数据归属及使用授权，再准备独立文件系统和挂载。建议每个已挂载盘一个 Real Client、独立目录和显式容量上限，应用按 key 的稳定分片规则路由到对应 owner；每个 owner 单独核对 LOCAL_DISK 描述符，再测多 owner 并行读。单 client 要使用一个跨盘文件系统，则需要额外的 RAID0/LVM stripe 等存储设计和风险评估。两种方案本轮均未实施；多 client 不自动保证均匀条带或线性叠加吞吐。

复现命令和停止方式见 `docs/NVME-SSD-REPRODUCE.md`。未来容量/压力实验必须重新记录共享盘可用空间、磁盘 owner、预留时段和基线条件。当前结果是一个观测基线，尚无项目性能回归阈值或业务 SLO，未宣称达标。

## 执行与清理证据

中台任务：`mac14-audit-b073-20261013`、`mac14-audit-b077-20261013`、`mac14-inspect-b077-20261013`、`mac14-sdk-b077-20261013`、`mac14-sdk-probe-b077-20261013`、`mac14-fio-deps-b077-20261013`、`mac14-fio-reads-b077-20261013`。fio 正式任务最终 `succeeded / exitCode=0 / archiveReady=true`。准备阶段的 GitHub 连接超时和缺少 libtcmalloc 失败保留在历史包；改用官方 OS 包及隔离依赖解决。

管理服务在测量尾段停止响应。根据已有交接授权，严格核验 SSH 主机指纹，用同一目标的既有 Mac key 前台确认原任务 exit=0、回收全部结果，并前台完成两个 SSD 最小运行。没有重复提交 fio，也没有启动、停止或修改管理服务。恢复后核验同一 serviceId、新 runId，重新 probe 原任务、确认归档并释放原预约。

专用 32 GiB 文件及两个 SSD 数据目录已删除，自建容器无残留，50284/50285 端口已关闭，NPU 最终无可见进程；既有业务容器保持运行。独立 fio 工具和自己的日志保留用于复现，没有持续后台实验。两次预约均有 releasedAt。完整原始数据、源脚本、任务退出记录、失败历史和清理记录通过 MAC-14 附件交付；仓库只保存参数化脚本和脱敏结果。

方法依据：[fio 官方手册](https://fio.readthedocs.io/en/latest/fio_doc.html)、[官方 FileStorage 参考源码](https://github.com/kvcache-ai/Mooncake/blob/v0.3.11.post1/mooncake-store/src/file_storage.cpp)、[Master 租约保护源码](https://github.com/kvcache-ai/Mooncake/blob/v0.3.11.post1/mooncake-store/src/master_service.cpp)。
