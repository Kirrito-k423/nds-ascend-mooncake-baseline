# MAC-17：完整 V4-Flash 在单机八卡 A5 的容量与 KV 验证

2026-10-13 实测，完整原生 V4-Flash fake 权重在一台八卡 Ascend950DT 上，TP1/DP8/EP8、MTP1、全权重留 HBM，完成 8K、32K、100K、200K 和条件性 1M 输入各加32输出。固定镜像与模型 revision 后可复现。初始 TP8 请求失败来自 Q-head 数量超出算子约束，不是 OOM。KV Pool 32K 的16次请求和1,001次全字节恢复校验也通过；该结果适用于单机DRAM路径。

## 模型与量化路线

选定 `deepseek-ai/DeepSeek-V4-Flash@60d8d70770c6776ff598c94bb586a859a38244f1`。主模型43层、hidden4096、256 routed experts；64 Q heads、1 KV head、head_dim512、RoPE64、sliding_window128。compress_ratios 的44项包含最后1个MTP：主层是2个不压缩+21个÷4+20个÷128，MTP不压缩。不能把3个不压缩+21+20算成43主层。[固定官方配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/config.json)

| 路线 | 来源与身份 | 本次结论 |
|---|---|---|
| 原始 MXFP4/MXFP8 | 上述 revision；fp8 config、expert_dtype=fp4、UE8M0 scale | 官方950DT recipe的原生路线，完整结构及真实runtime量化布局实际通过 |
| w8a8-mtp | ModelScope `Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp@2fe07f6ceba026dec274383f56831b128c3ea82a` | 官方recipe面向A2/A3。config/量化描述/index核实，源300,002,377,534 bytes；没有在A5试跑，也未核实全部headers |
| 0731 w4a4c8 | 当前官方教程要求本地msmodelslim量化0731；原始0731 revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`、3 MTP/DSpark | 与本次原始Flash revision及MTP数不同。没有下载真实权重/标定，因此不声称验证了这条路线 |

[固定950DT recipe](https://github.com/vllm-ascend/vllm-ascend-recipes/blob/c9807b27b1b8d3b005c90fe66044bc70ad50ff6f/models/en/DeepSeek/DeepSeek-V4-Flash.yaml) 明确原始MXFP4/MXFP8和TP1 DP4 EP，recipe SHA256保存在证据。我们使用同种原生路线，扩大为DP8 EP8，保留block128；非逐参数复制recipe。[研究时固定的最新教程](https://github.com/vllm-project/vllm-ascend/blob/64f43873948a76de923583be90256d0664c4b138/docs/source/tutorials/models/DeepSeek-V4-Flash.md) 只作为候选路线资料，不是运行时源码版本。

完整46 shard headers逐项核对index、shape×dtype和连续offset：69,187 tensors，159,609,485,896 bytes=148.65 GiB，含MTP 1,575 tensors。仅取HTTP206 header范围，真实tensor payload下载量为0。该值是源checkpoint量，不是运行时HBM。

fake只替换数值：UE8M0=127、packed FP4=0x22、FP8合法小值、scale/norm合法值、hash专家ID在有效范围内；构造shape/dtype、量化重排、NZ packing均走固定loader/runtime。每个rank保留完整43层和MTP，主/草稿CPU parameter bytes均为0。哈希路由为固定少量专家，不代表真实专家负载覆盖、业务准确率或生产吞吐。

## 硬件、软件与启动配置

经simpleHtmlWatch实时机器状态和预约选择B02-1；B07-6/7是A线已有预约，B07-8实际八卡占用，已退出并释放自有预约。B02-1八卡均Warning，64×64矩阵乘与CPU FP64参考一致，4096元素HCCL all-reduce=28，8/8通过。未修改共享驱动或停止他人容器。

八卡Ascend950DT，npu-smi报告98,304 MiB/card；torch可管理91.75 GiB/card。驱动26.2.0.b020，CANN/NNAL9.1.0，Python3.12.13，torch2.10.0，torch_npu2.10.0.post4，vLLM0.23.0+nds1.empty，vLLM-Ascend0.23.0+nds1，transformers5.5.4，tokenizers0.22.2，Mooncake transfer-engine-npu0.3.11.post1。

镜像ID `sha256:64155cd1bb77edf3c927a23c83cdad49e9d5394bd4b90113bc02ce017897c270`；vLLM-Ascend参考tag `v0.23.0` commit `5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`。关键实际安装文件与tag逐文件比较；Mooncake backend使用本仓MAC-3既有ReplicateConfig延迟导入补丁，不假装镜像是未经修改的官方包。实际Pool源码快照/哈希已保留，12文件中可对照的9个匹配tag，backend匹配仓内override，其余没有本地对应文件时记null。版本信息和image ID足以锁定已测资产，但重新构建同名tag不保证同一image ID。

成功核心配置：TP1、DP8、EP8、MTP1、eager、CPU offload=0、block128、batch_tokens2048、max_seqs1、utilization0.9、prefix cache disabled、hybrid KV manager enabled。通过API服务，离线LLM入口不支持DP>1。完整命令保存在各案例server-command.json及复现手册。

## 请求边界

8K/32K取8,192/32,768；100K/200K/1M取100,000/200,000/1,000,000，均是实际input tokens。先tokenize生成合法ID，再构造精确长度ID请求；核验API usage及finish_reason。

| 输入 | 实际输入 | 实际输出 | engine max_model_len | elapsed_s | 结果 |
|---:|---:|---:|---:|---:|---|
| 8,192 | 8,192 | 32 | 200,032 | 26.761 | PASS |
| 32,768 | 32,768 | 32 | 200,032 | 8.459 | PASS |
| 100,000 | 100,000 | 32 | 200,032 | 20.866 | PASS |
| 200,000 | 200,000 | 32 | 200,032 | 47.628 | PASS |
| 1,000,000 | 1,000,000 | 32 | 1,000,032 | 317.694 | PASS |

基础四档在同一次实例内逐级执行；1M是在200K通过后单独启动实例的边界探测。上述时间含首次预热/形状差异，不是统一预热、重复测量的性能基线。只验证单请求；没有验证8个DP同时1M、最大并发、配置maxpos1,048,576的极限输入或更长输出。请求全返回后关闭服务，shutdown日志中的EngineDeadError与SIGTERM不能混作请求失败。

## HBM、KV与capacity

逐rank精确bytes、启动阶段和内存统计见 `experiments/mac17/analysis.json`；原始audit保留。在基线200K配置下，所有rank的model_memory_usage为30,421,404,160 bytes=28.332 GiB，主参数/缓冲去重reported storage 27,415,645,196 bytes=25.533 GiB、MTP 2,722,346,492 bytes=2.535 GiB。两种指标口径不同；不能用参数storage代替模型加载分配或重复除EP/TP。八rank模型加载增量合计226.657 GiB，包含复制和转换后的布局，超过148.65 GiB源checkpoint是可解释的。

| rank/card | 加载增量GiB | profile激活GiB | non-torch GiB | Available KV GiB | KV实际storage GiB | blocks | 宿主采样峰值MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 28.332 | 0.578 | 0.927 | 52.737 | 51.953 | 24915 | 90952 |
| 1 | 28.332 | 0.579 | 0.928 | 52.736 | 51.951 | 24914 | 89317 |
| 2 | 28.332 | 0.589 | 0.927 | 52.726 | 51.940 | 24909 | 89274 |
| 3 | 28.332 | 0.578 | 0.928 | 52.737 | 51.951 | 24914 | 89189 |
| 4 | 28.332 | 0.583 | 0.929 | 52.731 | 51.947 | 24912 | 89188 |
| 5 | 28.332 | 0.591 | 0.929 | 52.723 | 51.938 | 24908 | 89148 |
| 6 | 28.332 | 0.588 | 0.929 | 52.726 | 51.940 | 24909 | 89147 |
| 7 | 28.332 | 0.583 | 0.928 | 52.732 | 51.947 | 24912 | 89190 |


Activation与non-torch为引擎profile记录；非torch含运行时等开销，无法单独拆graph/workspace。eager关闭graph执行，但未独立计量workspace字节，保留null。宿主npu-smi每次采集后间隔5秒（整轮实际中位间隔约6.2秒），记录的是采样峰值下界，不能当连续真实峰值；其口径与torch reserved不同，不能直接相减得到workspace。

运行时有六KV组：

| 组 | 逻辑层cache数 | 真实carrier/状态 | 原始token粒度与约束 |
|---|---:|---|---|
| 0 | 21主latent + 21 indexer | latent FP8 carrier `[B,128,1,640]`；indexer FP8 `[B,128,1,128]`+FP32scale，与132-byte复合view同storage | ÷4，128槽对应512原始tokens |
| 1 | 20主latent | `[B,128,1,640]` | ÷128，128槽对应16,384原始tokens |
| 2/3 | 各22 SWA（含MTP） | `[B,128,1,640]` | 128滑窗；分组/别名复用实际storage |
| 4 | 21主state + 21 indexer state | FP32 `[B,8,1,2048]` 和 `[B,8,1,512]` | 逻辑page 65,536/16,384 bytes，padded 81,920/16,896 |
| 5 | 20主state | FP32 `[B,16,1,1024]` | 逻辑page65,536，padded81,920 |

640是承载混合格式的字节carrier：FP8 latent、BF16 RoPE、scale和对齐区域；整个tensor的float8标签不表示640个语义上等价的FP8元素。indexer复合view不能再叠加到其K与scale字节，状态组也与其它组共享raw pool。按untyped_storage指针去重后，每rank23个主bank+21个indexer bank：`B×(23×128×640 + 21×128×132)`。Planner还分配1个indexer大小的空白bank，其差额不能算成业务cache；若启用Pool，2MiB地址对齐可能增加physical storage，再单列。不能按各层logical view求和当物理HBM，不能把全部KV预分配当200K请求实际占用。

B在基线为24,908–24,915，是各组共享规划下的block capacity。engine展示的token/concurrency capacity是混合组估计，不能乘DP当单请求上限，也不是测得并发能力。KV布局、压缩和SWA共同决定请求驻留量，旧9.7KiB/token与1.9GiB/200K估计弃用。

## KV Pool：实际导出与恢复

`AscendStoreConnector` + Mooncake NPU backend，kv_both，非layerwise、load_async、register_buffer，SWA/压缩hybrid manager启用。Pool8K仅注册和推理，无转移；改32K后16/16请求均返回32输出，13次真实external hit=32,767 tokens，本地prefix命中为0。

八worker注册六组，270 put成功、1,001 get成功；get前将目标buffer首64字节改为0xA5，读取后所有传输buffer全字节SHA256均匹配对应成功put：1,001/1,001匹配，0失配、0未验证。覆盖组0–5、C4/C128 latent/indexer、SWA、FP32 compressor states；独立八rank inventory再次核验完整43层+MTP及CPU参数为0。业务输出相同不作为字节恢复证据。

累计SDK写入582,057,984 bytes、读取2,269,855,744 bytes，包含重复请求和多个视图传输，不能当单请求逻辑KV字节或唯一storage占用。Pool rank0的KV物理storage为55,874,122,752 bytes（block24,914），相同B下无Pool为55,781,848,064 bytes：2MiB对齐新增92,274,688 bytes。DRAM pool配置1 GiB/worker，SSD offload=false；SDK采用实际注册NPU buffers和host DRAM，没有使用A3 fabric-memory专用分支。采样峰值rank0为91,496 MiB，其余89,711–89,859 MiB。

只验证同一台机上的NPU→DRAM→NPU及多个DP worker恢复；没有测试跨机、SSD spill、NDS直读、高并发、200K/1M Pool恢复或业务准确率。审计增加同步/CPU副本，不能据其延迟给生产性能结论。

## 失败证据与后续边界

- 原生dtype探针起初对UE8M0直接`.float()`、NZ直接`.cpu()`报错。实际量化路径按exponent bits和format cast使用；修正探针为CPU解释bits、先NZ→ND后拷贝，通过。探针错误未作为模型不支持证据。
- 首次offline脚本缺main guard，spawn提前失败；保留原日志并修正。
- TP8完整加载并初始化KV，但首8K请求在KvQuantSparseAttnSharedkvMetadata报22007/507018。源码CheckSingleParam限定Q heads=64/128，builder为64/TP；TP8每rank8heads不受支持。64-head针对1/4/128压缩的最小metadata探针通过，然后切换TP1。[固定算子约束](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/csrc/attention/kv_quant_sparse_attn_sharedkv_metadata/README.md)
- DP8 offline入口显式不支持，改API server。
- Pool8K未达到16K传输粒度；32K首启动Gloo rv=-3，停自有容器并回收，不记为KV失败。后续显式GLOO_SOCKET_IFNAME=lo，单机CPU通信走loopback。
- 早期DP日志命名用local device0，八rank阶段记录的dp_rank仍保留但inventory文件覆盖；现已修正，并用独立八rank inventory核验43层、MTP和CPU参数为0。第一次1M宿主SSH sampler断开，只有启动前采样；不能用其低水位宣称1M设备峰值。

MAC-15可复用已成功的原始Flash revision、TP1 DP8 EP8、量化与hybrid配置。扩大并发、长输出、不同revision/W4A4C8、图模式、跨机/SSD/NDS路径或真实权重业务精度，均须独立验证。此次KV Pool只验证最小DRAM路径，不自动证明磁盘或跨节点NDS已完成。

## 交付与资源

完整复现见 `reports/MAC-17-REPRODUCE.md`，脚本见 `experiments/mac17/`。成功与失败的原始日志/JSON经脱敏后随issue附件并在私仓归档；SHA256清单和离线验收器可重新核算headers、请求长度、rank budgets与Pool恢复。删除其他业务容器列表、机器私有连接信息；数据字段未删改成功/失败状态。成本报告以rmb-cost-report技能整理session计数，价格和汇率仅为明确标注的参考估算，不是实际账单。

全部远端任务已前台回收，自有容器/master/sampler均已退出，18218/50217/50218无监听，八卡无NPU进程，原有4个业务容器仍在。B02-1自有预约已释放。没有留下常驻服务或声称持续监控。

保留可复现缓存：镜像20,221,370,011 bytes和本项目metadata/脚本6,711,079 bytes；不占NPU或服务端口。清理句柄是该镜像tag与执行者本地记录的独立项目目录，见cleanup.json；仅在确认其它项目不再依赖后删除。机器地址和私有目录不作为仓库内容。

1M完整采样补测同样返回1,000,000+32，耗时318.555秒、启动98.023秒、70次采样覆盖整轮；逐card峰值MiB为90,861/89,201/89,162/89,438/89,203/89,162/89,182/89,203。八rank独立记录完整，第一轮1M采样缺口仍保留，不用补测覆盖原失败观察。

交付证据ZIP：`artifacts/MAC-17-evidence.zip`，8,901,040 bytes，SHA256 `2850abcc4ef3def9146e8b09d51b09509394055c1b1da419a1285adf0c9d7a90`。离线验收695文件完整性通过，从脱敏证据重算的analysis.json与原始证据计算结果完全一致。
