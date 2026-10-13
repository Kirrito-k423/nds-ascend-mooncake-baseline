# 历史诊断入口

这两个脚本保留TP8/EP8和offline LLM的失败复现上下文，不是成功配置。TP8每rank8个Q heads超出固定A5元数据算子约束。成功路径是上一目录的run_case.sh + run_server.py（TP1 DP8 EP8）；复现前读取reports/MAC-17-REPRODUCE.md。
