# 给MAC-15的交接

成功配置：原始deepseek-ai/DeepSeek-V4-Flash revision60d8d70770c6776ff598c94bb586a859a38244f1；TP1 DP8 EP8，43主层+1MTP，native MXFP4/MXFP8，eager、offload0、block128、hybrid manager enabled；image ID见报告。

单请求：8192/32768/100000/200000/1000000输入各+32全部成功。1M启动参数max_model_len1000032；不把maxpos或8DP容量相加当单请求/并发上限。

每rank模型加载28.332GiB，主+MTP参数reported storage28.068GiB，Available KV约52.72GiB，实际预分配KV约51.94GiB，六组KV按storage别名去重。精确bytes和blocks见analysis.json；不能套常数B/token。

Pool：AscendStoreConnector+Mooncake NPU0.3.11.post1，单机DRAM路径32K复用通过，270put/1001get成功，1001/1001全字节恢复匹配。block128的传输粒度16384原始tokens，8K没有put/get。1GiB DRAM/rank，不走SSD或A3 fabric_memory。连接源码使用MAC-3既有延迟导入补丁，未改本次推理后端。

限制：fake仅验证容量/格式/功能，固定少量专家路由不代表真实负载覆盖；未验证业务精度、生产性能/并发、跨节点或SSD/NDS路径、200K/1M Pool恢复、0731 W4A4C8。

MAC-15由SE决定何时启动；本任务不改派或改状态。下一步可先重放最小32K Pool数据校验，再逐级测试其目标NDS路径与长请求；真实权重精度和稳定性能基线单列。

证据、复现命令与资源释放记录随MAC-17结果交付；无常驻服务或持续监控。镜像与少量metadata留作复现缓存，清理句柄见cleanup.json及复现手册。
