# -*- coding: utf-8 -*-
"""图 6 分阶段版：按数据采集顺序出 5 张（1.0 → 0.75 → 0.5 → 0.25 → 0.1 逐档补齐）。
数据同定版：fig6_final.csv（14 已验收点）+ 红侧 0.75 附条件观测。横轴=设置 QPS。
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D

F = "/System/Library/Fonts/STHeiti Light.ttc"
fm.fontManager.addfont(F)
plt.rcParams.update({"font.family": fm.FontProperties(fname=F).get_name(),
                     "axes.unicode_minus": False, "figure.facecolor": "white"})

rows = list(csv.DictReader(open("fig6_final.csv", encoding="utf-8")))
for r in rows:
    if r["series"] == "HBM" and r["target_qps"] == "0.75":
        r.update(dict(ttft_mean_s="40.932652", ttft_p50_s="42.151500",
                      input_tokens_s="12237.914167", hit_rate_pct="7.335455",
                      actual_sent_qps="0.193333", valid_n="224"))

INK, MUTED, GRID = "#243445", "#52687a", "#dde5eb"
STYLE = {
    "HBM": ("纯 HBM 5 GiB/卡", "#b84038", "o"),
    "HBM+DRAM": ("HBM + DRAM 11 GiB", "#176ba8", "s"),
    "HBM+DRAM+SSD": ("HBM + DRAM + SSD 45 GiB（修复态）", "#a5630d", "D"),
}
PANELS = [
    ("(a) 缓存命中率", "响应声明缓存比例（%）", "hit_rate_pct"),
    ("(b) 服务侧 TTFT 均值", "转发后到首个 content token（s）", "ttft_mean_s"),
    ("(c) 服务侧 TTFT P50", "转发后到首个 content token（s）", "ttft_p50_s"),
    ("(d) 输入 token 吞吐", "输入 token / 秒（含缓存）", "input_tokens_s"),
]
STAGES = [
    (1, ["1.00"]),
    (2, ["0.75", "1.00"]),
    (3, ["0.50", "0.75", "1.00"]),
    (4, ["0.25", "0.50", "0.75", "1.00"]),
    (5, ["0.10", "0.25", "0.50", "0.75", "1.00"]),
]


def val(series, t, key):
    for r in rows:
        if r["series"] == series and r["target_qps"] == t:
            v = r.get(key) or "UNKNOWN"
            if v in ("UNKNOWN", ""):
                return None
            try:
                return float(v)
            except ValueError:
                return None
    return None


for stage, targets in STAGES:
    tl = [float(t) for t in targets]
    fig, axs = plt.subplots(2, 2, figsize=(14.6, 9.9), dpi=170)
    fig.subplots_adjust(left=0.068, right=0.985, top=0.815, bottom=0.245, wspace=0.19, hspace=0.33)
    for ax, (title, ylab, key) in zip(axs.ravel(), PANELS):
        ax.grid(color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#bac8d2")
        ax.tick_params(colors=MUTED, labelsize=9, length=0, pad=5)
        ax.set_title(title, fontsize=11.5, color=INK, loc="left", weight="bold")
        ax.set_ylabel(ylab, fontsize=9.5, color=INK)
        ax.set_xlabel("设置 QPS（目标发线速率）", fontsize=9.5, color=INK)
        ax.set_xticks(tl)
        ax.set_xticklabels([f"{t:g}" for t in tl])
        lo = min(tl) - 0.06
        hi = 1.05
        if lo == hi:
            lo, hi = lo - 0.06, hi + 0.06
        ax.set_xlim(lo, hi)
        for series, (lab, c, m) in STYLE.items():
            pts = [(t, val(series, f"{t:.2f}", key)) for t in tl]
            pts = [(x, y) for x, y in pts if y is not None]
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=c, marker=m,
                    ms=8, lw=2.4, zorder=4, label=lab)
            for x, y in pts:
                ax.annotate(f"{y:.2f}" if abs(y) < 10 else f"{y:.0f}", (x, y),
                            textcoords="offset points", xytext=(0, 11), ha="center",
                            fontsize=8.5, color=c, weight="bold")
        if key == "hit_rate_pct":
            ax.set_ylim(0, 104)
        else:
            ax.set_ylim(bottom=0)

    fig.text(0.068, 0.955, f"图 6｜第 {stage} 阶段：已采 {stage}/5 档（设置 QPS {float(targets[0]):g}–1.0）",
             fontsize=17, color=INK, weight="bold")
    fig.text(0.068, 0.921, "A5 单机 DeepSeek-V4-Flash + AgentX 轨迹回放；三条曲线为三级 KV 缓存：纯 HBM 5 GiB/卡 ／ +DRAM 11 GiB ／ +SSD 45 GiB（修复态）。"
                           "输出上限 1 token，每档 120 秒暖机 + 1,200 秒主窗。", fontsize=9.3, color=MUTED)
    fig.text(0.068, 0.897, f"本阶段只画已经采到的 {stage} 档；未采档位不出现、不插值、不占位。TTFT 取服务侧（代理转发后到首个 content token），不含客户端排队。",
             fontsize=9.3, color="#8a5a00")
    h = [Line2D([], [], color=c, marker=m, ls="-", lw=2.4, ms=8, label=lab) for lab, c, m in STYLE.values()]
    if stage == 5:
        h.append(Line2D([], [], color="#b84038", marker="o", ls="none", mfc="none", mew=2.2,
                        ms=11, label="红 0.75：附条件（实际 0.193 QPS，窗尾截断）"))
    fig.legend(handles=h, loc="lower center", ncol=4 if stage == 5 else 3, fontsize=9.6,
               frameon=False, bbox_to_anchor=(0.5, 0.145))
    fig.text(0.068, 0.098, "命中率为响应声明值（缓存输入 token / 输入 token），不是逐键物理 SSD 命中或字节一致；各点机器不同（B06-1 / B06-2 / B06-3 / B02-5），不构成严格配对。",
             fontsize=8.5, color=MUTED)
    if stage == 5:
        fig.text(0.068, 0.072, "红侧 0.75：窗内 224 条四类记录齐全、空内容 0、-400 0；15 条缺口全在窗外（窗尾截断）。该点实际发线 0.193 QPS，与红侧 0.50 档几乎同一工作点。",
                 fontsize=8.5, color="#8a5a00")
        fig.text(0.068, 0.048, "红/蓝两条线在 8 槽并发下封顶 0.19–0.27 QPS，设 0.75/1.0 也打不上去；SSD 修复态可跟到 0.91。",
                 fontsize=8.5, color=MUTED)
    fig.text(0.068, 0.020, "谭邵杰 t00906153 如有问题请联系讨论", fontsize=7.2, color="#8798a6")
    fig.savefig(f"图6_阶段{stage}_设置QPS.png")
    plt.close(fig)
print("ok")
