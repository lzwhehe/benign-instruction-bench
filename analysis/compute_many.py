"""Multi-detector analysis: per-detector table, rank transfer between benchmarks, scatter figure.

Reads results/many/<detector>__<set>.json (scripts/score_many.py) and the judge files used by compute_all.py.
Writes analysis/numbers_many.json and analysis/figures/fig_rank.pdf|png.
"""
import collections
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

FIG = "analysis/figures"
RNG = np.random.default_rng(0)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.15,
                     "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42, "legend.frameon": False})

PRETTY = {"protectai-v2": "ProtectAI v2", "protectai-v1": "ProtectAI v1", "piguard": "PIGuard",
          "jasperls": "JasperLS", "testsavant-large": "TestSavant-L", "wolf-defender": "Wolf Defender",
          "horizon-base": "Horizon-Labs", "sheltron": "Sheltron", "sheltron-taskaware": "Sheltron (+task)$^\\dagger$",
          "prismor-1.5b": "Prismor-1.5B", "promptguard2-86m": "Prompt Guard 2 (86M)",
          "promptguard2-22m": "Prompt Guard 2 (22M)", "promptguard1-86m": "Prompt Guard 1",
          "protectai-small-v2": "ProtectAI v2-small", "qualifire-sentinel": "Qualifire Sentinel",
          "rogue-sentinel-v2": "Sentinel v2"}
EXCLUDE = {"injecguard"}  # identical scores to PIGuard on every set (same weights under an older name)


def load_jsonl(p):
    return [json.loads(line) for line in open(p)]


def tpr_at(y, p, t=0.01):
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= t
    return float(tpr[ok].max()) if ok.any() else 0.0


clean_rows = load_jsonl("agentdojo_benign_outputs.jsonl")
tasks = sorted(set((r["suite"], r["task"]) for r in clean_rows))
bip_rows = load_jsonl("bipia_pairs.jsonl")
bip_keep = np.array([r["label"] == 0 or r.get("attack_split") == "test" for r in bip_rows])
tb_rows = load_jsonl("taubench_eval.jsonl") if os.path.exists("taubench_eval.jsonl") else []
tb_y = np.array([r["label"] for r in tb_rows])
tb_tasks = sorted(set((r["suite"], r["task"]) for r in tb_rows if r["label"] == 0))


def _norm(t):
    import re
    return re.sub(r"\s+", " ", str(t).lower()).strip()


# InjecAgent attacks that appear verbatim in PIGuard's published training split ("seen")
_ia = []
for _f in ["attacker_cases_dh.jsonl", "attacker_cases_ds.jsonl"]:
    _ia += [json.loads(line)["Attacker Instruction"] for line in open(os.path.expanduser("~/agentsec/InjecAgent/data/" + _f))]
_pg = _norm(open(os.path.expanduser("~/agentsec/PIGuard/datasets/train.json"), encoding="utf-8", errors="ignore").read())
SEEN = {i for i, a in enumerate(_ia) if _norm(a)[:80] in _pg}
tb_seen = np.array([r["label"] == 1 and r.get("attack_id") in SEEN for r in tb_rows])
tb_unseen = np.array([r["label"] == 1 and r.get("attack_id") not in SEEN for r in tb_rows])

det = collections.defaultdict(dict)
for f in glob.glob("results/many/*.json"):
    d = json.load(open(f))
    det[d["detector"]][d["set"]] = d
# judges (task-aware LLMs) for the same sets
for name, stem in [("judge-qwen", "Qwen2.5-7B-Instruct"), ("judge-llama", "Llama-3.1-8B-Instruct")]:
    for s, fn in [("admatched", "agentdojo_pairs"), ("adshift", "agentdojo_pairs_shift"), ("bipia", "bipia_pairs"),
                  ("tbeval", "taubench_eval")]:
        p = f"hfjudge_{stem}_{fn}.json"
        if os.path.exists(p):
            j = json.load(open(p))
            det[name][s] = dict(probs=j["probs"], labels=j["labels"])
PRETTY.update({"judge-qwen": "Judge (Qwen2.5-7B)", "judge-llama": "Judge (Llama-3.1-8B)"})

T = {}
for name, sets in det.items():
    if name in EXCLUDE:
        continue
    row = {}
    if "general" in sets:
        row["general_fpr"] = float(np.mean(np.array(sets["general"]["probs"]) >= 0.5))
    if "adclean" in sets:
        f = np.array(sets["adclean"]["probs"]) >= 0.5
        by = collections.defaultdict(list)
        for r, x in zip(clean_rows, f):
            by[(r["suite"], r["task"])].append(x)
        row["adclean_fpr"] = float(f.mean())
        row["adclean_block"] = float(np.mean([any(by[t]) for t in tasks]))
    for s in ["admatched", "adshift", "bipia"]:
        if s not in sets:
            continue
        y, p = np.array(sets[s]["labels"]), np.array(sets[s]["probs"])
        if s == "bipia":
            y, p = y[bip_keep], p[bip_keep]
        row[f"{s}_auroc"] = float(roc_auc_score(y, p))
        row[f"{s}_tpr1"] = tpr_at(y, p)
        row[f"{s}_fpr05"] = float(np.mean(p[y == 0] >= 0.5))
        row[f"{s}_tpr05"] = float(np.mean(p[y == 1] >= 0.5))
    if "tbeval" in sets:
        p = np.array(sets["tbeval"]["probs"])
        assert len(p) == len(tb_y)
        row["tbeval_auroc"] = float(roc_auc_score(tb_y, p))
        row["tbeval_tpr1"] = tpr_at(tb_y, p)
        row["tbeval_fpr05"] = float(np.mean(p[tb_y == 0] >= 0.5))
        by = collections.defaultdict(list)
        for r, x in zip(tb_rows, p >= 0.5):
            if r["label"] == 0:
                by[(r["suite"], r["task"])].append(x)
        row["tbeval_block"] = float(np.mean([any(by[t]) for t in tb_tasks]))
        thr = np.sort(p[tb_y == 0])[int(np.ceil(0.99 * (tb_y == 0).sum())) - 1]  # 1%-FPR threshold
        row["tbeval_tpr1_seen"] = float(np.mean(p[tb_seen] > thr))
        row["tbeval_tpr1_unseen"] = float(np.mean(p[tb_unseen] > thr))
    row["truncated_bipia"] = sets.get("bipia", {}).get("truncated")
    T[name] = row

# rank transfer among standalone detectors (judges and the undocumented task-aware variant excluded)
dets = [n for n in T if not n.startswith("judge") and n not in ("sheltron-taskaware", "sheltron-generic")
        and all(k in T[n] for k in ["admatched_tpr1", "bipia_tpr1", "tbeval_tpr1"])]
rank = {}
for a, b in [("bipia_tpr1", "admatched_tpr1"), ("bipia_auroc", "admatched_auroc"), ("bipia_tpr1", "adshift_tpr1"),
             ("admatched_tpr1", "tbeval_tpr1"), ("admatched_auroc", "tbeval_auroc"), ("bipia_tpr1", "tbeval_tpr1"),
             ("adclean_fpr", "tbeval_fpr05")]:
    x, y = [T[n][a] for n in dets], [T[n][b] for n in dets]
    tau, p_tau = kendalltau(x, y)
    rho, p_rho = spearmanr(x, y)
    rank[f"{a}_vs_{b}"] = dict(kendall_tau=float(tau), p_tau=float(p_tau), spearman=float(rho), p_rho=float(p_rho), n=len(dets))
best_bipia = max(dets, key=lambda n: T[n]["bipia_tpr1"])
best_ad = max(dets, key=lambda n: T[n]["admatched_tpr1"])
# detection by BIPIA's official attack group (test half), threshold 0.5 and the 1%-FPR threshold
GROUPS = {}
for fname, gs in [("text_attack_test.json", ["task-irrelevant", "task-relevant", "targeted"]),
                  ("text_attack_train.json", ["task-irrelevant", "task-relevant", "targeted"]),
                  ("code_attack_test.json", ["code-passive", "code-active"]),
                  ("code_attack_train.json", ["code-passive", "code-active"])]:
    for i, c in enumerate(json.load(open(os.path.expanduser("~/agentsec/BIPIA/benchmark/" + fname)))):
        GROUPS[c] = gs[i // 5]
kept = [r for r, k in zip(bip_rows, bip_keep) if k]
grp = np.array([GROUPS[r["attack"].split(":", 1)[1]] if r["label"] == 1 else "benign" for r in kept])
by_group = {}
for name, sets in det.items():
    if name in EXCLUDE or "bipia" not in sets:
        continue
    y, p = np.array(sets["bipia"]["labels"])[bip_keep], np.array(sets["bipia"]["probs"])[bip_keep]
    thr = np.sort(p[y == 0])[int(np.ceil(0.99 * (y == 0).sum())) - 1]  # 1%-FPR threshold on BIPIA benign
    by_group[name] = {g: dict(tpr05=float(np.mean(p[grp == g] >= 0.5)), tpr_at_1fpr=float(np.mean(p[grp == g] > thr)))
                      for g in ["task-irrelevant", "task-relevant", "targeted", "code-passive", "code-active"]}

_pgrows = json.load(open(os.path.expanduser("~/agentsec/PIGuard/datasets/train.json"), encoding="utf-8"))
_src = collections.Counter(d["source"] for d in _pgrows)
_med = lambda s: float(np.median([len(d["prompt"]) for d in _pgrows if d["source"] == s]))
pg_sources = dict(bipia_rows=_src.get("BIPIA", 0), bipia_median_chars=_med("BIPIA"),
                  injecagent_rows=_src.get("InjecAgent", 0), injecagent_median_chars=_med("InjecAgent"))
tb_meta = dict(n_benign=int((tb_y == 0).sum()), n_injected=int((tb_y == 1).sum()), n_tasks=len(tb_tasks),
               n_seen_attacks=len(SEEN), n_attacks=len(_ia), n_seen_pos=int(tb_seen.sum()), n_unseen_pos=int(tb_unseen.sum()))
out = dict(pg_sources=pg_sources, tb_meta=tb_meta, by_group=by_group, table=T, rank_transfer=rank, detectors_in_rank=dets, best_on_bipia=best_bipia, best_on_agentdojo=best_ad,
           best_on_bipia_rank_on_agentdojo=1 + sorted(dets, key=lambda n: -T[n]["admatched_tpr1"]).index(best_bipia))
json.dump(out, open("analysis/numbers_many.json", "w"), indent=1)
print(json.dumps(rank, indent=1))
print("best on BIPIA:", best_bipia, "| its AgentDojo rank:", out["best_on_bipia_rank_on_agentdojo"], "of", len(dets))
for n in sorted(T, key=lambda n: -T[n].get("admatched_tpr1", -1)):
    r = T[n]
    print(f"{PRETTY.get(n, n):28s} gen {r.get('general_fpr', float('nan')):.3f} clean {r.get('adclean_fpr', float('nan')):.3f} "
          f"block {r.get('adclean_block', float('nan')):.3f} | AD tpr1 {r.get('admatched_tpr1', float('nan')):.3f} "
          f"auc {r.get('admatched_auroc', float('nan')):.3f} | ADS tpr1 {r.get('adshift_tpr1', float('nan')):.3f} | "
          f"BI tpr1 {r.get('bipia_tpr1', float('nan')):.3f} auc {r.get('bipia_auroc', float('nan')):.3f}")

# LaTeX table generated from the numbers (the paper never hand-copies them)
ORDER = ["jasperls", "protectai-v1", "protectai-v2", "protectai-small-v2", "testsavant-large", "promptguard1-86m",
         "piguard", "promptguard2-22m", "promptguard2-86m", "qualifire-sentinel", "rogue-sentinel-v2", "sheltron",
         "sheltron-taskaware", "prismor-1.5b", "wolf-defender", "horizon-base", "judge-qwen", "judge-llama"]
META = {  # release year, parameters, trained with agent tool-output data (per model card)
    "protectai-v1": ("2023", "184M", "no"), "protectai-v2": ("2024", "184M", "no"), "jasperls": ("2023", "184M", "no"),
    "testsavant-large": ("2024", "184M", "no"), "piguard": ("2025", "184M", "no"), "sheltron": ("2026", "68M", "yes"),
    "sheltron-taskaware": ("2026", "68M", "yes"), "prismor-1.5b": ("2026", "1.5B", "no"),
    "wolf-defender": ("2026", "308M", "eval."), "horizon-base": ("2026", "308M", "yes"),
    "promptguard2-86m": ("2025", "279M", "no"), "promptguard2-22m": ("2025", "71M", "no"),
    "promptguard1-86m": ("2024", "279M", "no"), "protectai-small-v2": ("2024", "142M", "no"),
    "qualifire-sentinel": ("2025", "396M", "no"), "rogue-sentinel-v2": ("2025", "596M", "no"),
    "judge-qwen": ("--", "7B", "--"), "judge-llama": ("--", "8B", "--")}


def cell(r, k, pct=True):
    if k not in r or r[k] is None:
        return "--"
    return f"{100 * r[k]:.1f}" if pct else f"{r[k]:.2f}"


lines = [r"\begin{tabular}{@{}lccc|c|cc|cc|ccc|c@{}}", r"\toprule",
         r" & & & & General & \multicolumn{4}{c|}{AgentDojo} & \multicolumn{3}{c|}{$\tau$-bench} & BIPIA \\",
         r" & & & Agent & FPR & FPR & Tasks & \multicolumn{2}{c|}{TPR @ 1\% FPR} & FPR & Tasks & TPR & TPR \\",
         r"Detector & Year & Size & data & (\%) & (\%) & (\%) & Matched & Shift & (\%) & (\%) & @ 1\% & @ 1\% \\",
         r"\midrule"]
for n in ORDER:
    if n not in T:
        continue
    r, (yr, sz, ag) = T[n], META.get(n, ("", "", ""))
    if n == "judge-qwen":
        lines.append(r"\midrule")
    lines.append(f"{PRETTY.get(n, n)} & {yr} & {sz} & {ag} & {cell(r, 'general_fpr')} & {cell(r, 'adclean_fpr')} & "
                 f"{cell(r, 'adclean_block')} & {cell(r, 'admatched_tpr1')} & {cell(r, 'adshift_tpr1')} & "
                 f"{cell(r, 'tbeval_fpr05')} & {cell(r, 'tbeval_block')} & {cell(r, 'tbeval_tpr1')} & "
                 f"{cell(r, 'bipia_tpr1')} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open("paper/tab_many.tex", "w").write("% generated by analysis/compute_many.py\n" + "\n".join(lines) + "\n")

# figure: detection by BIPIA attack group at each detector's own 1%-FPR threshold
SHOW = ["piguard", "sheltron", "horizon-base", "wolf-defender", "prismor-1.5b", "judge-qwen"]
COL = {"piguard": "#E69F00", "sheltron": "#56B4E9", "horizon-base": "#0072B2", "wolf-defender": "#CC79A7",
       "prismor-1.5b": "#D55E00", "judge-qwen": "#009E73"}
GG = ["task-irrelevant", "task-relevant", "targeted", "code-passive", "code-active"]
fig, ax = plt.subplots(figsize=(6.75, 1.9))
x = np.arange(len(GG))
w = 0.8 / len(SHOW)
for i, n in enumerate([s for s in SHOW if s in by_group]):
    ax.bar(x + (i - len(SHOW) / 2 + 0.5) * w, [by_group[n][g]["tpr_at_1fpr"] * 100 for g in GG], w * 0.92,
           color=COL[n], label=PRETTY.get(n, n))
ax.set_xticks(x)
ax.set_xticklabels(["Task-irrelevant", "Task-relevant", "Targeted", "Code (passive)", "Code (active)"])
ax.set_ylabel("Detected at 1% FPR (%)")
ax.set_ylim(0, 105)
ax.legend(ncol=6, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.savefig(os.path.join(FIG, "fig_goal_many.pdf"))
fig.savefig(os.path.join(FIG, "fig_goal_many.png"))

# figure: (a) BIPIA vs AgentDojo, (b) tau-bench vs AgentDojo; detection at 1% FPR. Short codes; key in caption.
CODE = {"horizon-base": "HL", "wolf-defender": "WD", "prismor-1.5b": "PR", "promptguard2-86m": "P2",
        "promptguard2-22m": "P2s", "promptguard1-86m": "P1", "piguard": "PG", "sheltron": "SH",
        "qualifire-sentinel": "QS", "rogue-sentinel-v2": "S2", "testsavant-large": "TS", "jasperls": "JS",
        "protectai-v1": "A1", "protectai-v2": "A2", "protectai-small-v2": "A2s", "judge-qwen": "JQ", "judge-llama": "JL"}
OFFA = {"wolf-defender": (-4, 2), "prismor-1.5b": (-4, -7), "promptguard2-86m": (4, 0), "qualifire-sentinel": (-4, 1),
        "promptguard2-22m": (4, 0), "judge-qwen": (4, -1), "judge-llama": (-4, -6), "protectai-v1": (4, 1),
        "protectai-v2": (4, -7), "protectai-small-v2": (4, 2), "jasperls": (-4, -6), "rogue-sentinel-v2": (4, -6)}
OFFB = {"protectai-v1": (4, 2), "protectai-v2": (4, -7), "protectai-small-v2": (4, 2), "promptguard2-86m": (-4, 2),
        "promptguard2-22m": (4, -1), "judge-qwen": (-4, -7), "jasperls": (-4, -6), "piguard": (4, 2)}
fig, axs = plt.subplots(1, 2, figsize=(6.75, 2.9), sharey=True)
for ax, xkey, xlabel, off, rk in [(axs[0], "bipia_tpr1", "BIPIA: detection at 1% FPR (%)", OFFA, "bipia_tpr1_vs_admatched_tpr1"),
                                  (axs[1], "tbeval_tpr1", r"$\tau$-bench: detection at 1% FPR (%)", OFFB, "admatched_tpr1_vs_tbeval_tpr1")]:
    ax.plot([0, 100], [0, 100], color="#bbb", lw=0.8, ls="--", zorder=0)
    for n, r in T.items():
        if n not in CODE or "admatched_tpr1" not in r or xkey not in r:
            continue
        judge = n.startswith("judge")
        ax.scatter(r[xkey] * 100, r["admatched_tpr1"] * 100, s=24 if judge else 18, marker="D" if judge else "o",
                   color="#009E73" if judge else ("#E69F00" if n == "piguard" else "#0072B2"), zorder=3)
        dx, dy = off.get(n, (4, 1))
        ax.annotate(CODE[n], (r[xkey] * 100, r["admatched_tpr1"] * 100), xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.5, ha="right" if dx < 0 else "left")
    ax.set_xlabel(xlabel)
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_title(("(a) " if ax is axs[0] else "(b) ") + r"Kendall $\tau$ = %.2f" % rank[rk]["kendall_tau"], fontsize=8.5)
axs[0].set_ylabel("AgentDojo: detection at 1% FPR (%)")
fig.savefig(os.path.join(FIG, "fig_rank.pdf"))
fig.savefig(os.path.join(FIG, "fig_rank.png"))
