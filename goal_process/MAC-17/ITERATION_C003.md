# C003：并行策略与容量边界

症状：TP8/EP8完整模型加载成功但8K首请求在KvQuantSparseAttnSharedkvMetadata失败。
新增证据：固定v0.23.0算子源CheckSingleParam限定Q heads=64/128；DSA builder使用64/TP。TP8产生8，超出范围；64-head最小元数据探针对1/4/128压缩均通过。
决策：沿官方950DT原生权重路线改TP1 DP8 EP8，通过API server执行（offline LLM不接受DP>1）。
结果：8K/32K/100K/200K+32成功，条件1M+32亦成功。无缩层缩宽、CPU权重offload、真实权重payload。checkpoint159609485896 bytes不是HBM；每rank model_memory_usage30421404160 bytes。精确预算由analyze.py再核算。
观察器纠偏：DP各worker可显示local NPU0，原文件名冲突导致inventory覆盖；stage记录中的dp_rank保留，预算可恢复。修正全局证据rank并完成独立8rank加载观察。
Pool：8K低于16K传输粒度，无put/get，不能宣称缓存恢复已通过；32K启动第一次遇Gloo解析错误，源栈为CPU communicator而非KV或容量，回收后使用lo隔离。
