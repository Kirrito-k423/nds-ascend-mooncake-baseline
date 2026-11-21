# C001：确认资源、完整源布局与配置加载路径

- 动作：READ / PROBE；2026-10-13 09:34–10:08 UTC。
- 判据：八张空闲 A5；固定模型metadata；实际八卡计算和HCCL；通过实际引擎配置路径确认主干与MTP。
- 回退：仅删除本任务容器并释放预约，不停止其他项目。

结果：A线已预约B07-6/7；B07-8有八卡进程，库存任务归档后释放自有预约。实时监控选取B02-1，库存确认八张Warning卡无进程；固定镜像经流式复制，两端退出0，无本机镜像副本。

原生V4-Flash固定revision 60d8d70770c6776ff598c94bb586a859a38244f1。46个header、69,187个tensor逐一核对索引、shape×dtype、连续data_offsets和总payload，159,609,485,896 bytes；训练payload读取0。W8A8 metadata来自ModelScope固定revision 2fe07f6ceba026dec274383f56831b128c3ea82a，源index为300,002,377,534 bytes；未下载其权重。

八卡矩阵计算与HCCL 8/8通过。第一软件探针调用裸Transformers.AutoConfig失败；修订为实际EngineArgs路径后，DeepseekV4ForCausalLM、DeepSeekV4MTPModel及AscendFp8Config确认。裸AutoConfig不是该运行时的部署前置条件。安装中的9个已比较关键源码文件与官方v0.23.0逐字节相等；镜像sha256固定。

量化观测遇到CANN 9.1的UE8M0→float D2D cast报561103；此cast来自探针，实际量化源码只对scale做reshape/view。已改用原始exponent bits传CPU验证，下一轮验证实际FP4→NZ/FP8转换；不把观测工具失败判为模型不支持。

信息增量：排除Warning统一阻塞、源payload当HBM、裸Transformers识别当vLLM支持判据。尚未启动完整引擎，未测KV容量或任何请求。

下一micro-goal：量化primitive通过后运行完整43层+1MTP、TP8/EP的dummy，分阶段记录真实分配。环境、metadata与工具准备约34分钟；剩余3天窗口内继续。
