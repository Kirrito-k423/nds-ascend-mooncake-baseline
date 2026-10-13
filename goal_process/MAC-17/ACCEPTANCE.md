# 验收账本

| ID | 完成条件 | 状态 | 证据 |
|---|---|---|---|
| AC1 | 固定身份、revision、格式、软件和镜像 | PASS | metadata/header manifests, image ID, installed source hashes |
| AC2 | 单机八卡最小计算与通信校验 | PASS | preflight smoke-rank0..7.json |
| AC3 | 完整结构与量化布局实际启动 | PASS | DP8及Pool逐rank完整43层+MTP tensors，全CPU parameter bytes=0 |
| AC4 | 8K/32K/100K/200K输入与至少32输出 | PASS | dp8-b021/result.json；1M条件探测亦通过 |
| AC5 | 逐rank分配、KV shapes与capacity | PASS | DP8/1M八rank records；1M完整70样本补测已回收 |
| AC6 | 当前版本KV Pool导出恢复验证 | PASS | 32K、13次external hit、1001/1001 poisoned get全字节匹配 |
| AC7 | 脚本、日志JSON、报告、PR及资源释放 | PASS | 695文件脱敏包离线验收通过，私仓交付，预约/容器/端口/采样器均释放 |

1M为200K通过后的条件性探测；已实际通过。dummy不评价业务精度；graph/workspace单项没有独立计量时保留null，不填0。
