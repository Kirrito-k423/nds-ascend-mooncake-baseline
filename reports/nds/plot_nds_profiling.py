# -*- coding: utf-8 -*-
"""NDS 直通 vs 常规 SSD：性能与耗时拆解（B06-1/NPU0，25 对真机样本）"""
import csv, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

F = "/System/Library/Fonts/STHeiti Light.ttc"
fm.fontManager.addfont(F)
plt.rcParams.update({"font.family": fm.FontProperties(fname=F).get_name(),
                     "axes.unicode_minus": False, "figure.facecolor": "white"})
INK, MUTED, GRID = "#243445", "#52687a", "#dde5eb"
NDS_C, SSD_C, WARN = "#b84038", "#176ba8", "#a5630d"

rows = list(csv.DictReader(open("nds_profiling.csv", encoding="utf-8")))
labels = [r["label"] for r in rows]
nds = [float(r["nds_p50_ms"]) for r in rows]
ssd = [float(r["ssd_p50_ms"]) for r in rows]
size = [int(r["bytes"]) for r in rows]
CMD = 131072  # 单条 NVMe 命令上限 128 KiB
cmds = [math.ceil(b / CMD) for b in size]

fig = plt.figure(figsize=(15.0, 10.4), dpi=170)
gs = fig.add_gridspec(2, 2, left=0.068, right=0.982, top=0.815, bottom=0.115,
                      wspace=0.24, hspace=0.40)

# ---- (a) 五档耗时
ax = fig.add_subplot(gs[0, 0])
ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
x = range(len(labels))
ax.plot(x, nds, "-o", color=NDS_C, lw=2.6, ms=9, label="NDS 直通（NPU HBM 直读）")
ax.plot(x, ssd, "-s", color=SSD_C, lw=2.6, ms=8, label="常规 SSD（pread + H2D）")
ax.set_yscale("log")
for i, (n, s) in enumerate(zip(nds, ssd)):
    ax.annotate(f"{n/s:.2f}×", (i, n), textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=9.5, color=NDS_C, weight="bold")
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("P50 端到端耗时（ms，对数轴）", fontsize=10, color=INK)
ax.set_title("(a) 五档规模：NDS 全部慢于常规 SSD", fontsize=12, color=INK, loc="left", weight="bold")
ax.legend(fontsize=9.2, frameon=False, loc="upper left")
ax.tick_params(colors=MUTED, labelsize=9, length=0)

# ---- (b) 有效带宽
ax = fig.add_subplot(gs[0, 1])
ax.grid(color=GRID, lw=0.7, axis="y"); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
bnd = [b / (t / 1000) / 1e9 for b, t in zip(size, nds)]
bsd = [b / (t / 1000) / 1e9 for b, t in zip(size, ssd)]
w = 0.36
ax.bar([i - w/2 for i in x], bnd, w, color=NDS_C, label="NDS 直通")
ax.bar([i + w/2 for i in x], bsd, w, color=SSD_C, label="常规 SSD")
for i, (a, b) in enumerate(zip(bnd, bsd)):
    ax.annotate(f"{a:.2f}", (i - w/2, a), ha="center", va="bottom", fontsize=8.5, color=NDS_C)
    ax.annotate(f"{b:.2f}", (i + w/2, b), ha="center", va="bottom", fontsize=8.5, color=SSD_C)
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("有效带宽（GB/s）", fontsize=10, color=INK)
ax.set_title("(b) 有效带宽：512 MiB 时 0.39 vs 4.27 GB/s", fontsize=12, color=INK, loc="left", weight="bold")
ax.legend(fontsize=9.2, frameon=False)
ax.tick_params(colors=MUTED, labelsize=9, length=0)

# ---- (c) 512 MiB 耗时去向
ax = fig.add_subplot(gs[1, 0])
ax.spines[["top", "right", "left"]].set_visible(False)
total, floor = 1366.6952, 125.78389
wait = total - floor
ax.barh([0], [floor], color=SSD_C, height=0.42, label="数据搬运下限（取常规 SSD 实测）")
ax.barh([0], [wait], left=[floor], color=NDS_C, height=0.42, label="逐条命令串行等待")
ax.annotate(f"搬运 {floor:.1f} ms\n9.2%", (floor/2, 0), ha="center", va="center",
            color="white", fontsize=9.5, weight="bold")
ax.annotate(f"命令等待 {wait:.0f} ms  ·  90.8%", (floor + wait/2, 0), ha="center", va="center",
            color="white", fontsize=10.5, weight="bold")
ax.set_yticks([]); ax.set_xlim(0, total*1.02)
ax.set_xlabel("512 MiB 单次读回耗时（ms）", fontsize=10, color=INK)
ax.set_title("(c) 1367 ms 的去向：九成是等命令，不是搬数据", fontsize=12, color=INK, loc="left", weight="bold")
ax.legend(fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2)
ax.tick_params(colors=MUTED, labelsize=9, length=0)

# ---- (d) 单命令驻留 vs 命令数 + 并发投影
ax = fig.add_subplot(gs[1, 1])
ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
per = [t / c for t, c in zip(nds, cmds)]
ax.plot(cmds, per, "o-", color=WARN, lw=2.2, ms=8, label="实测：总耗时 ÷ 命令数")
ax.axhline(sum(per[1:])/len(per[1:]), color=WARN, ls="--", lw=1.3)
ax.annotate("收敛到约 0.33 ms/条\n→ 一条 128 KiB 命令的固定往返", (cmds[-1], per[-1]),
            textcoords="offset points", xytext=(-14, 26), ha="right", fontsize=9, color=WARN)
ax.set_xscale("log", base=2)
ax.set_xlabel("NVMe 命令条数（按 ≤128 KiB 拆分）", fontsize=10, color=INK)
ax.set_ylabel("单条命令平均驻留（ms）", fontsize=10, color=INK)
ax.set_title("(d) 固定开销 × 命令数，与规模无关", fontsize=12, color=INK, loc="left", weight="bold")
ax.set_ylim(0, 1.25)
ax.legend(fontsize=9, frameon=False, loc="upper right")
ax.tick_params(colors=MUTED, labelsize=9, length=0)

fig.text(0.068, 0.955, "NDS 直通读路径耗时拆解：慢在“一条一条等”，不在带宽",
         fontsize=18.5, color=INK, weight="bold")
fig.text(0.068, 0.918, "B06-1 / NPU0，A5 单机；两臂同源 SSD 范围、同一 HBM 目标缓冲，各 5 对样本取 P50，全部逐字节校验通过；"
                       "NDS 侧为 TE 顺序提交（每次一笔请求、等终态再发下一笔）。", fontsize=9.6, color=MUTED)
fig.text(0.068, 0.893, "每条 NDS_UB_READ 内部把读取拆成 ≤128 KiB 的 NVMe 命令并同步逐条等待：512 MiB 至少要 4,096 条，"
                       "每条约 0.33 ms 固定往返 → 1367 ms；同样字节常规 SSD 只需 125.8 ms。", fontsize=9.6, color=WARN)
fig.text(0.068, 0.045, "设备并行度、在途深度与控制器缓存均未实测；本图为同步顺序提交路径的观测，不代表 NDS 硬件上限。"
                       " 谭邵杰 t00906153 如有问题请联系讨论", fontsize=8.4, color="#8798a6")
fig.savefig("NDS耗时拆解_五档.png")
print("ok")
