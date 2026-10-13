# 工作审视报告

## 原定目标

五档KV位置与TTFT/decode、SSD关键路径、可调用direct对照或明确缺口，3天预算内交付原始数据、报告、使能手册及PR。

## 完成情况

- 已完成：固定版本/完整模型配置/双机硬件门禁、12次native DRAM读回、1024/C1的T0/T1/T2三轮完整请求和位置审计；原始数据及初始化尖峰保留。
- 未完成：T3/T4、长上下文/并发扫描、自动压力换出、SSD分段计时、Host PMU校准与direct功能验证。主阻塞为缺少经预约可保持隔离的双机窗口。

## 发现的问题

| 严重程度 | 具体问题 | 方法原因 | 改进 |
|---|---|---|---|
| 必须改正 | node.py端口检查在mode重启时将TIME_WAIT当占用 | 未用重启场景验证生命周期检查 | 增加SO_REUSEADDR及活跃listener/重启CPU回归；SSD续跑尚未进入start，NPU验证待资源 |
| 应当改正 | run.py只预热冷请求，未预热各模型worker的实际远端加载 | 把agent级native探针误当成所有TE endpoint预热 | 新版增加3轮cold/local/remote模型预热；本轮原始尖峰不删除、不据此声称稳态收益 |
| 应当改正 | native reference是每256字节重复模式 | 读回校验覆盖基础搬运，未覆盖整块重排 | 当前校验仅限该模式；后续补按块位置变化的数据，不宣称所有重排风险已排除 |
| 建议改进 | 原默认矩阵先全量memory再进入SSD | 阶段顺序没有优先解决SSD未知量 | 默认只1024/C1校准，支持NDS_MODES_JSON从disk续跑 |
| 应当改正 | 控制任务启动门禁与destination真正起模型之间有时间差；source子引擎失败但API主进程仍等待 | 一次占用快照不覆盖启动周期，主进程存活不等于子引擎有效 | Node.start再次门禁；检测已观测TransferEngine致命日志后finally清理；新保护尚未设备验证 |

## 做得好的地方

没有将ready/Health=OK当资源独占；没有停止其他训练。全量冷/命中结果和失败保留，未拿P50掩盖995.5ms首读。SDK原生导入顺序未回退，没有把AscendDirectTransport或O_DIRECT称作SSD→HBM直通。

## 下次重点关注

先取得能覆盖启动、native SSD校验及三轮模型请求的稳定双机窗口，最小复测HIXL初始化，区分共享负载与SDK/环境问题。重新部署最新指纹，先disk-only续跑，再完成模型worker传输预热和长上下文容量扫描；保留r2结果及r4失败。本报告和goal_process交接将作为后续唤醒上下文。

## W1阶段审视

新增原生SSD物理盘读1/3/10MiB各三轮，10MiB通过推翻参考源码4MiB上限推断；64MiB失败保留。全部51个探索完成读回记录及14份SDK JSON、命令前置错误与abort均归档。模型T3/T4仍0，主阻塞未解除。

本轮可避免的问题：inspect缺换行两次重复、vendor env与nounset不兼容、猜测不存在descriptor API及把TE endpoint误当控制hostname、10MiB shell误带加号。修正后仅用新ID，不把前置脚本错误归为SDK或中台故障。能力检查与真实SDK hostname替代字段假设；CPU测试覆盖控制/数据端口不同、缺rank/错误节点及零贡献client。

OS缓存SSD正确读回而无物理IO说明位置标签不足：只对自有文件fadvise，并与读前位置、RPC、返回值和盘计数共同验收。K1seal及hostnames/位置敏感reference已带回模型helper，MEMORY保留64MiB、SSD默认1/3MiB。15CPU测试通过；模型组合修改仍未NPU验收。下次优先验证四worker/8-key批量与跨节点加载，不重新试64MiB，也不拿native小样本替代T3/T4模型结果。
