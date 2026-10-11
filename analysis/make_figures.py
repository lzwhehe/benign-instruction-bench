"""Draw the paper's data figures at their final size (Elsevier 5p: 3.5 in column, 7.2 in page width).

Reads analysis/numbers.json, analysis/numbers_many.json, the stored scores in results/, and bipia_pairs.jsonl (for
the test-half mask of BIPIA). Writes paper/figures/fig_{rank,goal_many,format,roc}.pdf. Figures are included in the
paper at \\columnwidth or \\textwidth without further scaling, so font sizes here are the printed sizes (>= 7 pt).
"""
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def roc_curve(y, p):
    # same points as sklearn.metrics.roc_curve (without dropping collinear ones); avoids the dependency for plotting
    o = np.argsort(-p, kind="mergesort")
    y, p = np.asarray(y)[o], np.asarray(p)[o]
    last = np.r_[np.where(np.diff(p))[0], len(p) - 1]
    tps, fps = np.cumsum(y)[last], np.cumsum(1 - y)[last]
    return np.r_[0, fps / fps[-1]], np.r_[0, tps / tps[-1]], None

COL_W, FULL_W = 3.5, 7.2
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 4,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.grid": True, "grid.linestyle": "--", "grid.linewidth": 0.4, "grid.alpha": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.prop_cycle": mpl.cycler(color=["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]),
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 600,
    "savefig.format": "pdf", "savefig.bbox": "standard", "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02, "pdf.fonttype": 42,
})
FIG = "paper/figures"
N = json.load(open("analysis/numbers.json"))
NM = json.load(open("analysis/numbers_many.json"))
T = NM["table"]

# ------------------------------------------------------------------ rank transfer scatter (7.2 x 2.8 in)
CODE = {"horizon-base": "HL", "wolf-defender": "WD", "prismor-1.5b": "PR", "promptguard2-86m": "P2",
        "promptguard2-22m": "P2s", "promptguard1-86m": "P1", "piguard": "PG", "sheltron": "SH",
        "qualifire-sentinel": "QS", "rogue-sentinel-v2": "S2", "testsavant-large": "TS", "jasperls": "JS",
        "protectai-v1": "A1", "protectai-v2": "A2", "protectai-small-v2": "A2s", "judge-qwen": "JQ", "judge-llama": "JL"}
OFFA = {"wolf-defender": (-4, 2), "prismor-1.5b": (-4, -8), "promptguard2-86m": (4, 0), "qualifire-sentinel": (-4, 1),
        "promptguard2-22m": (4, -1), "judge-qwen": (4, -1), "judge-llama": (-4, -7), "protectai-v1": (4, 2),
        "protectai-v2": (4, -7), "protectai-small-v2": (4, 2), "jasperls": (-4, -7), "rogue-sentinel-v2": (4, -7)}
OFFB = {"protectai-v1": (4, 2), "protectai-v2": (4, -7), "protectai-small-v2": (4, 2), "promptguard2-86m": (-4, 2),
        "promptguard2-22m": (4, -2), "judge-qwen": (-4, -8), "jasperls": (-4, -7), "piguard": (4, 2)}
taus = {k: NM["rank_transfer"][k]["kendall_tau"] for k in NM["rank_transfer"]}
fig, axs = plt.subplots(1, 2, figsize=(FULL_W, 2.8), sharey=True)
panels = [(axs[0], "bipia_tpr1", "BIPIA: TPR at 1% FPR (%)", OFFA,
           "(a) BIPIA vs. AgentDojo, Kendall $\\tau$ = %.2f" % taus["bipia_tpr1_vs_admatched_tpr1"]),
          (axs[1], "tbeval_tpr1", "$\\tau$-bench: TPR at 1% FPR (%)", OFFB,
           "(b) $\\tau$-bench vs. AgentDojo, Kendall $\\tau$ = %.2f" % taus["admatched_tpr1_vs_tbeval_tpr1"])]
for ax, xkey, xlabel, off, title in panels:
    ax.plot([0, 100], [0, 100], color="#999999", lw=0.7, ls="--", zorder=0)
    for n, r in T.items():
        if n not in CODE or "admatched_tpr1" not in r or xkey not in r:
            continue
        judge, pg = n.startswith("judge"), n == "piguard"
        ax.scatter(r[xkey] * 100, r["admatched_tpr1"] * 100, s=20 if judge else 16, zorder=3,
                   marker="D" if judge else ("s" if pg else "o"),
                   color="#009E73" if judge else ("#D55E00" if pg else "#0072B2"))
        dx, dy = off.get(n, (4, 1))
        ax.annotate(CODE[n], (r[xkey] * 100, r["admatched_tpr1"] * 100), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7, ha="right" if dx < 0 else "left")
    ax.set_xlabel(xlabel)
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_title(title)
axs[0].set_ylabel("AgentDojo: TPR at 1% FPR (%)")
handles = [plt.Line2D([], [], ls="", marker="o", color="#0072B2", label="Detector"),
           plt.Line2D([], [], ls="", marker="s", color="#D55E00", label="PIGuard"),
           plt.Line2D([], [], ls="", marker="D", color="#009E73", label="Task-aware judge"),
           plt.Line2D([], [], ls="--", color="#999999", label="Equal performance")]
fig.legend(handles=handles, loc="outside upper center", ncol=4)
fig.savefig(os.path.join(FIG, "fig_rank.pdf"))

# ------------------------------------------------------------------ detection by BIPIA attack group (7.2 x 2.4 in)
PRETTY = {"piguard": "PIGuard", "sheltron": "Sheltron", "horizon-base": "Horizon-Labs", "wolf-defender": "Wolf Defender",
          "prismor-1.5b": "Prismor-1.5B", "judge-qwen": "Judge (Qwen2.5-7B)"}
SHOW = ["piguard", "sheltron", "horizon-base", "wolf-defender", "prismor-1.5b", "judge-qwen"]
COL = ["#D55E00", "#56B4E9", "#0072B2", "#CC79A7", "#E69F00", "#009E73"]
HATCH = ["", "////", "", "\\\\\\\\", "....", "xxxx"]
GG = ["task-irrelevant", "task-relevant", "targeted", "code-passive", "code-active"]
fig, ax = plt.subplots(figsize=(FULL_W, 2.4))
x = np.arange(len(GG))
w = 0.8 / len(SHOW)
for i, n in enumerate(SHOW):
    ax.bar(x + (i - len(SHOW) / 2 + 0.5) * w, [NM["by_group"][n][g]["tpr_at_1fpr"] * 100 for g in GG], w * 0.92,
           color=COL[i], hatch=HATCH[i], edgecolor="white", linewidth=0.3, label=PRETTY[n])
ax.set_xticks(x)
ax.set_xticklabels(["Task-irrelevant", "Task-relevant", "Targeted", "Code (passive)", "Code (active)"])
ax.set_ylabel("TPR at 1% FPR (%)")
ax.set_ylim(0, 105)
ax.grid(axis="x", visible=False)
fig.legend(ncol=6, loc="outside upper center")
fig.savefig(os.path.join(FIG, "fig_goal_many.pdf"))

# ------------------------------------------------------------------ format and false positives (3.5 x 2.0 in)
C2 = {"ProtectAI": "#0072B2", "PIGuard": "#D55E00"}
fig, ax = plt.subplots(1, 2, figsize=(COL_W, 2.1))
ra = N["readme_format_ablation"]
for i, det in enumerate(["ProtectAI", "PIGuard"]):
    v = [ra[det]["original"] * 100, ra[det]["plain"] * 100]
    ax[0].bar(np.arange(2) + (i - 0.5) * 0.38, v, 0.36, color=C2[det], hatch="" if i == 0 else "////",
              edgecolor="white", linewidth=0.3, label="ProtectAI v2" if det == "ProtectAI" else det)
ax[0].set_xticks(range(2))
ax[0].set_xticklabels(["As-is", "Markup\nremoved"])
ax[0].set_ylabel("FPR on READMEs (%)")
ax[0].set_title("(a) README chunks")
ax[0].grid(axis="x", visible=False)
for det in ["ProtectAI", "PIGuard"]:
    a, b = N["AgentDojo"][det], N["AgentDojo"][det + "+norm"]
    ax[1].scatter([a["fpr_at_05"] * 100], [a["tpr_at_05"] * 100], color=C2[det], marker="o", s=16, zorder=3)
    ax[1].scatter([b["fpr_at_05"] * 100], [b["tpr_at_05"] * 100], color=C2[det], marker="s", s=16, zorder=3)
    ax[1].annotate("", xy=(b["fpr_at_05"] * 100, b["tpr_at_05"] * 100), xytext=(a["fpr_at_05"] * 100, a["tpr_at_05"] * 100),
                   arrowprops=dict(arrowstyle="->", color=C2[det], lw=0.8))
ax[1].set_xlabel("FPR (%)")
ax[1].set_ylabel("TPR (%)")
ax[1].set_title("(b) AD-Matched at 0.5")
ax[1].set_xlim(0, 25)
ax[1].set_ylim(40, 90)
fig.legend(*ax[0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
fig.savefig(os.path.join(FIG, "fig_format.pdf"))

# ------------------------------------------------------------------ ROC curves, log FPR (7.2 x 2.4 in)
bip = [json.loads(line) for line in open("bipia_pairs.jsonl")]
keep_b = np.array([r["label"] == 0 or r.get("attack_split") == "test" for r in bip])
SETS = [("AD-Matched", "normalized_probs_main.json", "agentdojo_pairs", None),
        ("AD-Shift", "normalized_probs_shift.json", "agentdojo_pairs_shift", None),
        ("BIPIA (test-half attacks)", "normalized_probs_bipia.json", "bipia_pairs", keep_b)]
METH = [("ProtectAI v2", "#0072B2", "-"), ("PIGuard", "#D55E00", "--"), ("Judge (Qwen2.5-7B)", "#009E73", "-."),
        ("Judge (Llama-3.1-8B)", "#CC79A7", ":")]
fig, axs = plt.subplots(1, 3, figsize=(FULL_W, 2.4), sharey=True)
for k, (ax, (title, det_file, stem, keep)) in enumerate(zip(axs, SETS)):
    D = json.load(open(os.path.join("results", det_file)))["probs"]
    jq = json.load(open(f"results/hfjudge_Qwen2.5-7B-Instruct_{stem}.json"))
    jl = json.load(open(f"results/hfjudge_Llama-3.1-8B-Instruct_{stem}.json"))
    y = np.array(jq["labels"])
    s = {"ProtectAI v2": np.array(D["protectai-deberta-v2/original"]), "PIGuard": np.array(D["piguard/original"]),
         "Judge (Qwen2.5-7B)": np.array(jq["probs"]), "Judge (Llama-3.1-8B)": np.array(jl["probs"])}
    if keep is not None:
        y, s = y[keep], {k: v[keep] for k, v in s.items()}
    for m, c, ls in METH:
        fpr, tpr, _ = roc_curve(y, s[m])
        ax.plot(np.clip(fpr, 1e-3, 1), tpr * 100, color=c, ls=ls, label=m)
    ax.axvline(0.01, color="#888888", lw=0.7, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1)
    ax.set_title(f"({'abc'[k]}) {title}")
    ax.set_xlabel("False-positive rate (log scale)")
axs[0].set_ylabel("True-positive rate (%)")
fig.legend(*axs[0].get_legend_handles_labels(), loc="outside upper center", ncol=4)
fig.savefig(os.path.join(FIG, "fig_roc.pdf"))
print("figures written to", FIG)
