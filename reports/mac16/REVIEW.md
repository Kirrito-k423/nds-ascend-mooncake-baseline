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
