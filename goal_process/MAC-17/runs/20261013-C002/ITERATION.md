# C002：完整目标加载、profile、KV分配及长度阶梯

- 开始：2026-10-13T10:15Z；动作 EXPERIMENT。
- 预注册：43层主干、1层MTP、TP8/EP，原生MXFP8/MXFP4路线，固定image ID、原config与revision；仅dummy数值初始化，不改模型构造/形状/量化处理。
- E2门禁：八卡计算/HCCL通过；实际EngineArgs识别主干与MTP；scale原始指数bits校验正确；FP4→NZ转换、显式ND读回及finite校验通过。
- 成功判据：引擎启动，全部主干/草稿parameter驻NPU；每rank有真实分配记录；实际8K/32K/100K/200K输入且至少32输出。
- 唯一主要变量：请求长度；配置固定max_model_len=200032、max_num_seqs=1、batch_tokens=2048、memory_utilization=0.9。
- 停止：首个启动/请求失败即停止增加长度；不改宽度、层数、CPUoffload。Startup超过20分钟人工定点停止自己的容器；整任务硬超时60分钟。
- 回收：前台等至任务终态与归档，仅回收自有容器；保留预约供下一判别动作。
- 下一判断：按最早失败阶段选择动作，不将工具或格式错误直接归因容量。
