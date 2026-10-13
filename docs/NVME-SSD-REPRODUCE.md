# NVMe / Ascend SSD 复现

实测条件及结果见 [MAC-14 报告](../reports/mac14/NVME-SSD-BASELINE.md)。先通过管理入口检查实时任务、预约、磁盘占用和可用空间，协调测试窗口。记录自己的任务 ID、预约 ID、设备/CPU 范围、部署目录和 SHW_RESULTS_DIR；ready 不等于资源独占。复测会对共享系统盘产生 I/O，不能省略协调。

## 文件级 fio

原实测从 openEuler 官方仓库下载 AArch64 `fio-3.34-6.oe2403sp3.aarch64.rpm` 和 `gperftools-libs`。逐包用 `rpm -Kv` 验证签名，`rpm2cpio | cpio -id` 解包到项目自己的 fio-root，通过隔离的 LD_LIBRARY_PATH 使用库。记录包 SHA256、`fio --version`、`fio --enghelp` 和 `ldd`，没有全局安装。不同系统需重新核对包兼容性。

将 `benchmarks/nvme_file_reads.py` 放入目标，参数由部署侧提供：

```bash
export LD_LIBRARY_PATH="$NDS_FIO_ROOT/usr/lib64"
python3 nvme_file_reads.py \
  --fio "$NDS_FIO_ROOT/usr/bin/fio" \
  --directory "$NDS_TEST_DIRECTORY" \
  --cpus "$NDS_TEST_CPUS" \
  --output "$SHW_RESULTS_DIR/fio"
```

目录必须是已挂载磁盘上的专用目录，禁止裸盘、tmpfs、Docker overlay。CPU 集合需属于选定磁盘 NUMA 节点且已协调。脚本要求至少 132 GiB 可用空间，只创建一个 32 GiB 文件，拒绝覆盖同名文件。完整 CRC32C 写读校验后执行十二次只读测量，采集盘和 CPU 原始计数器，最后清理自己新建的文件。回收完整目录及 stdout/stderr，并确认远端退出码。

```bash
python3 benchmarks/analyze_nvme_reads.py /path/to/results/fio \
  --output /path/to/fio-summary.json
```

分析器要求十二个成功读样本、无正式数据写、实际文件分配、文件已清理，并核对盘读字节。当前分析器针对基线 nvme0n1；换设备时先更新映射和验证规则，不能默默沿用。延迟范围来自各 worker 的百分位，不能当作合并百分位。

## 原生 SSD 校验

使用已验证的隔离镜像和现有 CANN/NNAL/venv，记录确定版本；新环境重新验证兼容性。仅需一张已协调的空闲 NPU。原容器限制 4 CPU、6 GiB RAM，挂载 driver/dcmi 只读，限定可见设备并设置原基线的 ASCEND_LOCAL_COMM_RES。SDK 必须在 NPU 初始化后加载。

将 `benchmarks/ssd_offload_probe.py` 挂到容器证据目录，将宿主已确认的 NVMe 专用空目录 bind mount 为 `/offload`；不得复用其他 client 的数据目录。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
python3 /evidence/ssd_offload_probe.py /evidence /offload "$NDS_LOCAL_HOST"
```

脚本检查 50284/50285 无占用，启动独立 Master，限定 SSD quota 512 MiB。成功需 setup/put=0、读取前仅有 LOCAL_DISK、三次目标 buffer 置零后的读回均返回 1 MiB 且逐字节一致。最后一次副本查询后等待 6 s，尊重 Master 的 5 s 租约；不等待会被拒绝清除，无法自证独立 SSD 读取。

回收 `ssd-offload-result.json`、`master.log`、完整 stdout/stderr 和宿主 `findmnt -T`。这只是同步 I/O 的功能校验，不提供压力换出、io_uring A/B、长上下文模型、远端 SSD 或稳定吞吐结论。

## 结束和恢复

所有任务必须在同一顶层运行内前台跟踪至退出及归档。脚本销毁自己的 Store 并停止自己的 Master；容器包装应有 timeout 和只针对自己容器的清理 trap。收齐文件后确认自己的容器消失、端口关闭、指定卡无自己的进程，删除专用数据，释放自己的预约。保留工具、日志和失败记录，不改全局 page cache、挂载、RAID、LVM 或共享服务。

管理入口不可达时不要重启、替换或重复提交原任务。若已有交接授权的 SSH 恢复连接，严格核验主机指纹，确认原任务 exit.code并收回结果；没有授权连接则报告缺口。服务恢复后重新核验同一 serviceId/new runId，probe 原任务、确认退出和归档，再通过正式接口释放原预约。不得直接修改任务记录和预留文件。
