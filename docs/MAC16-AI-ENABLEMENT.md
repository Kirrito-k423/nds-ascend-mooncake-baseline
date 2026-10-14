# 五档 KV 实验使能（MAC-16）

本包面向已有授权的两台物理 A5，完整 Qwen2.5-72B-Instruct、dummy BF16、TP4、PD-colocated。保留基线 1b05f19 的原生 SDK 导入顺序修正。实验分为1024/C1校准与容量约束下的长度/并发扫描，前一轮证据通过后才扩大规模。

source是持有KV副本的源机器；destination是另一台物理机器。T1在source处理请求并读本机DRAM，T2在destination处理请求并从source读远端DRAM；T3/T4同样区分本机/远端SSD。两端各有T0冷请求基线，source/destination不是同一请求的两个阶段。

2026-10-14 已核验 B02-1/2/3 的数据地址、RAM、挂载和镜像。B02-1 有精确目标镜像，B02-2/3 按标签及 SHA 均缺失；三台未见已挂载共享文件存储，原 Qwen 路径缺模型元数据。仅配置/tokenizer和最新脚本的私有迁移附件约3.7MB，不含真实权重。`benchmarks/prepare_b02_node.sh` 可在镜像与本机身份匹配后准备独立目录，不启动模型；尚未远端执行。具体映射、证据与缺口见 `reports/mac16/b02/MIGRATION.md`。后续优先复查 B02 的双机窗口，仍受 proxy host 去重预约限制。

## 资源与版本门禁

从同一 simpleHtmlWatch 实例读取 ready 与最新 npu-smi。ready只表示中台可调度，四张卡必须实际空闲，Warning允许通过真实计算/HCCL校验。先预约再立即复查；预约不能阻止其他入口的训练。忙碌时退出，不停止其他进程。两端镜像、模型配置SHA256及设备范围需记录。

本轮已复核镜像 `nds-vllm-source:kv-import-fix-20261012` / `sha256:64155cd1bb77edf3c927a23c83cdad49e9d5394bd4b90113bc02ce017897c270`，torch 2.10.0、torch_npu 2.10.0.post4、vLLM 0.23.0+nds1.empty、vLLM-Ascend 0.23.0+nds1、Mooncake NPU 0.3.11.post1、CANN 9.1.0。模型配置SHA256 `14ca217334fe0fd10148413592d68c99eeb33431ed89c1afa130fee560be2a29`；80层、64 Q heads、8 KV heads、hidden_size8192。没有下载真实权重。

复用 `benchmarks/tp4_gate.py`，在源镜像内使用 `/opt/nds-venv/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=<已核查空闲端口> /evidence/tp4_gate.py`。镜像没有torchrun入口脚本，`--standalone`的主机名解析在本轮失败；不改全局hosts或DNS。

## 部署与前台执行

将 `benchmarks/kv_tier/` 下全部 `.py` / `.sh` 文件放到独立项目目录。两端准备完整模型配置/tokenizer；不缩层缩宽。通过中台文件传输或任务命令分片复制，不复制凭据。使用新的任务ID和自己的预约ID。

在源目录计算脚本版本指纹：

```bash
python3 - <<'PY'
import hashlib,pathlib
root=pathlib.Path('benchmarks/kv_tier')
rows=[p.name+':'+hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))]
print(hashlib.sha256('\n'.join(rows).encode()).hexdigest())
PY
```

每个远端任务显式设置并导出：`NDS_ROLE=source|destination`、`NDS_SOURCE_HOST`、`NDS_DESTINATION_HOST`、`NDS_LOCAL_HOST`、`NDS_PROJECT_DIR`、`NDS_MODEL_DIR`、`NDS_TOOLKIT_SHA256`。地址来自实际节点，不能将跳板的127.0.0.1当成跨节点数据地址。`NDS_VISIBLE_DEVICES`可选，默认0,1,2,3；每端恰好四张卡。`SHW_RESULTS_DIR`由中台提供。

命令：`bash "$NDS_PROJECT_DIR/launch.sh"`。先提交destination控制任务，再提交source协调任务；同一顶层运行前台收集两端终态与归档。source正常或异常都停止两端自有容器。最长90分钟，启动门禁每端1800秒；不得启动后结束顶层运行等待回调。

端口：两端API18196、SDK agent18197、destination控制18198；source Master50296/metrics50297。脚本检查端口冲突，容器名 `nds-mac16-source` / `nds-mac16-destination` 必须无历史活动实例。容器32 CPU、内存上限800GiB，源节点DRAM池每rank100GiB，目标贡献0；提交前检查主机RAM及已有负载。工具目录及结果目录必须在实际文件系统上。

## 校准与扫描

默认仅1024输入、32输出、C1、三轮。MEMORY native做1/64MiB本机及跨节点读回；SSD micro改为已验证的1/3MiB，使用位置敏感数据、完整SHA256和逐字节校验。64MiB SSD在本镜像AllocateBatch中abort，禁止重复。W1单卡10MiB业务key已三轮通过，但未验证8-key/four-worker batch，模型T3/T4仍缺实测。随后memory模式测T0/T1/T2；disk模式单独重启并测T0/T3/T4。最终需要分别分析两个模式的冷对照，不能用历史1024/10的AISBench绝对值代替本轮32输出结果。本包的固定token ID数据与历史AISBench chat不同；均记录真实usage。

校准完成后，复核 `five-tier.json`、每阶段 `.requests.json/.evidence.json/.log`、`offload-cleanup.json` 与两端退出记录，再设置例如：

```bash
export NDS_CONFIGS_JSON='[[8192,1],[16384,1],[24576,1],[8192,8],[16384,8],[24576,8],[8192,25],[16384,25],[24576,25]]'
```

重新用新任务ID执行两端，仍在当前顶层运行收集。脚本按日志KV token capacity的80%预算限制组合；跳过项保留原因，不计为通过。每格3C个正式样本，C1只有3个；P95/P99是描述性样本分位数，不作稳定尾延迟结论。出错保留整个归档，修正后使用新任务ID，不覆盖历史。

分析：`python3 benchmarks/kv_tier/analyze.py <source归档的results目录>`。分析器拒绝未完成运行、短输出、SSE合并事件、不完整或混合副本、缺少加载、没有物理盘读等情况；成功产生requests.csv与summary.json。CPU验证：`python3 -m unittest discover -s tests -p 'test_kv_tier*.py' -v`。这些unit fixture属于模拟输入，不是设备测量。

仓内完整脱敏JSON压缩为`reports/mac16/data/raw-json.tar.gz`；CSV/summary保留为文本。离线复核时解压到空目录，再对该目录显式运行`analyze.py --allow-partial`。原始未脱敏双机归档及失败日志通过Multica附件交付。

从SSD阶段继续可设置`NDS_MODES_JSON='["disk"]'`；默认两个mode。分析已完整完成的部分阶段可显式加`--allow-partial`，仍逐项验证位置、真实重算counter及三轮完整性，保留run_error和experiment_complete=false，不等于整项任务通过。新版端口检查使用SO_REUSEADDR排除TIME_WAIT，活跃监听仍拒绝。新版实际local/remote worker预热未NPU复测，首次执行需重新部署最新指纹；远端r3/r4旧指纹不能直接复用。

r4在source初始化阶段出现HIXL/AICPU错误，未完成SSD native校验或模型请求；共享进程在门禁后出现，尚未确认因果。新版Node.start再次检查四卡占用，并检测已观测的TransferEngine致命错误后通过finally停止自有服务。这两项改动仅语法/CPU验证；取得稳定窗口后先做小规模TE/SSD读回，不直接扩大矩阵。

## SSD与计数边界

source各rank独立offload目录，64GiB显式配额、1GiBstaging buffer、256MiB bucket、默认1 key/bucket（小样本seal校准；可用NDS_OFFLOAD_BUCKET_KEYS_LIMIT显式覆盖并记录）、heartbeat1秒、uring关闭；当前四个目录仍落在同一系统NVMe，不能当成四盘条带。以MAC-14方法等待完成LOCAL_DISK、停止查询6秒，逐segment清除DRAM并重新审计，才发起读请求。 新版记录贡献worker的store.get_hostname控制地址；MEMORY descriptor仅有TE transport_endpoint时，不能当作segment_name。必须凑齐四个贡献worker、排除scheduler零贡献client，不全局猜测。NDS_OWN_FILE_COLD默认1，仅fsync/fadvise本任务/own offload下的bucket文件，不清全局page cache；随后仍要求真实盘计数和native加载字节门禁。每轮重构，排除读后DRAM回填。配置依据为[Mooncake同版本SSD文档](https://github.com/kvcache-ai/Mooncake/blob/v0.3.11.post1/docs/source/deployment/ssd-offload.md)，SDK和运行证据优先。

保留native get各rank开始/结束、正返回字节与offload RPC计数；这是SDK主机调用时长，不能直接等同独立的盘读、H2D或同步时长。跨节点单调时钟不能相减。源码阶段可能重叠，不能相加冒充关键路径。

盘字节同时记录主机/sys/block和可用的容器cgroup io.stat；前者含共享负载，后者仍需区分代码/模型读与KV读。网卡计数未必覆盖UB，不能替代真实链路字节。Host DDRC PMU的flux_rd/flux_wr必须校准单位、覆盖的控制器及背景后再计算流量；不能由盘/网字节推算。缺少分段事件、PMU校准或UB计数时，作为证据缺口保留。

本包通过HBM prefix reset与SSD eager persistence后租约清除构造位置；它没有实现自动DRAM压力换出或HBM容量溢出实验。活跃并发与累积前缀规模需分别报告，后续压力消融不能靠扩大DRAM池代替。

## 结束与恢复

脚本只删除本进程确认过的空目录内生成的offload数据。确认自己的容器不存在、端口关闭、任务真实退出且archiveReady，再释放自己的预约。保留模型配置、镜像与日志。中台不可达不重启服务、不换实例、不按进程名杀守护进程；用原任务ID查询，未知状态不重新提交。

没有SSD→HBM direct的可调用证据时，不运行所谓direct对照。O_DIRECT/io_uring只是存储IO策略；SDK注册NPU buffer或跨节点HIXL不能单独证明SSD直接DMA到HBM。dummy只用于资源/路径/性能，不用于业务精度。

## W1已证与恢复门禁

见`reports/mac16/w1/NATIVE-SSD.md`及data。单卡MEMORY/SSD的1/3/10MiB各三轮正确；SSD读前COMPLETE LOCAL_DISK-only、lease6s，冷文件三轮每次实际盘读payload+4096B。OS缓存保留的1MiB读回虽正确、RPC+1，但盘读0，不计为物理SSD性能。64MiB失败保留；上游参考源码不能当部署库实际大小上限，10MiB通过已推翻4MiB推断。

下一步只在有相同模型资产及可同时预约的双节点窗口时，先验证10MiB×8key、四worker和跨机native加载，再做1024/C1模型SSD三轮。当前所有proxy SSH端口共用预约入口，第二预约409时不能用短执行/旁路SSH替代预约跑模型。Node.start及worker预热、hostname映射/fadvise组合修改尚未四worker设备实测。重新计算并部署最新全部*.py指纹，旧65d89b6不可重用。W1累计121task和20短诊断已终态，自有容器/端口/offload及预约已清。
