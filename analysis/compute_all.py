"""Recompute every number and figure in the paper from the raw score files.

Run from the benign-instruction-bench directory on the machine that holds the score files.
Writes analysis/numbers.json and analysis/figures/*.pdf|png.
"""
import collections
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

OUT = "analysis"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
RNG = np.random.default_rng(0)
B = 1000  # bootstrap replicates

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.15, "pdf.fonttype": 42,
})
# Okabe-Ito (colorblind safe)
C = {"ProtectAI": "#0072B2", "PIGuard": "#E69F00", "Judge (Qwen2.5-7B)": "#009E73", "Judge (Llama-3.1-8B)": "#CC79A7"}


def load_jsonl(p):
    return [json.loads(line) for line in open(p)]


def tpr_at(y, p, target):
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= target
    return float(tpr[ok].max()) if ok.any() else 0.0


def boot_ci(y, p, fn):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(B):
        idx = np.concatenate([RNG.choice(pos, len(pos)), RNG.choice(neg, len(neg))])
        vals.append(fn(y[idx], p[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def metrics(y, p, ci=False):
    m = dict(n_benign=int((y == 0).sum()), n_injected=int((y == 1).sum()),
             auroc=float(roc_auc_score(y, p)), tpr_at_1fpr=tpr_at(y, p, 0.01), tpr_at_5fpr=tpr_at(y, p, 0.05),
             fpr_at_05=float(np.mean(p[y == 0] >= 0.5)), tpr_at_05=float(np.mean(p[y == 1] >= 0.5)))
    if ci:
        m["tpr_at_1fpr_ci"] = boot_ci(y, p, lambda a, b: tpr_at(a, b, 0.01))
        m["auroc_ci"] = boot_ci(y, p, lambda a, b: float(roc_auc_score(a, b)))
    return m


N = {}

# ---------------------------------------------------------------- general benign corpus (README / email / web)
corpus = load_jsonl("corpus.jsonl")
ds = json.load(open("detector_scores.json"))
src = np.array([r["source"] for r in corpus])
N["general_corpus"] = {"n": len(corpus), "by_source_n": {s: int((src == s).sum()) for s in sorted(set(src))}}
for det, key in [("ProtectAI", "protectai-deberta-v2"), ("PIGuard", "piguard")]:
    f = np.array(ds["detectors"][key]["probs"]) >= 0.5
    N["general_corpus"][det] = {"overall_fpr": float(f.mean()),
                                **{f"fpr_{s}": float(f[src == s].mean()) for s in sorted(set(src))}}
fa = json.load(open("format_ablation.json"))
N["readme_format_ablation"] = {det: fa[key] for det, key in [("ProtectAI", "protectai-deberta-v2"), ("PIGuard", "piguard")]}
N["degenerate_detectors_readme_fpr"] = {k: fa[k]["original"] for k in ["deepset-deberta", "fmops-distilbert"]}

# ---------------------------------------------------------------- clean AgentDojo tool outputs (no injections)
clean = load_jsonl("agentdojo_benign_outputs.jsonl")
cs = json.load(open("agentdojo_benign_outputs.jsonl.scores.json"))
tasks = sorted(set((r["suite"], r["task"]) for r in clean))
N["agentdojo_clean"] = {"n_outputs": len(clean), "n_tasks": len(tasks)}
suites = ["workspace", "travel", "banking", "slack"]
clean_rows = {}
for det, key in [("ProtectAI", "protectai-deberta-v2"), ("PIGuard", "piguard")]:
    f = np.array(cs[key]["probs"]) >= 0.5
    by_task = collections.defaultdict(list)
    for r, x in zip(clean, f):
        by_task[(r["suite"], r["task"])].append(x)
    blocked = np.array([any(by_task[t]) for t in tasks])
    bs = [np.mean(RNG.choice(blocked, len(blocked))) for _ in range(B)]
    d = {"output_fpr": float(f.mean()), "task_block": float(blocked.mean()),
         "task_block_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    for s in suites:
        idx = [i for i, r in enumerate(clean) if r["suite"] == s]
        tb = [blocked[i] for i, t in enumerate(tasks) if t[0] == s]
        d[s] = {"output_fpr": float(f[idx].mean()), "task_block": float(np.mean(tb)), "n_tasks": len(tb)}
    N["agentdojo_clean"][det] = d
    clean_rows[det] = d
N["agentdojo_clean"]["sanity_plain_sentences_flagged"] = {k: int(sum(x >= 0.5 for x in cs[k]["sanity"])) for k in cs}

# ---------------------------------------------------------------- matched benchmarks
BIPIA_GROUPS = {}
for fname, groups in [("text_attack_test.json", ["task-irrelevant", "task-relevant", "targeted"]),
                      ("text_attack_train.json", ["task-irrelevant", "task-relevant", "targeted"]),
                      ("code_attack_test.json", ["code-passive", "code-active"]),
                      ("code_attack_train.json", ["code-passive", "code-active"])]:
    cats = list(json.load(open(os.path.expanduser("~/agentsec/BIPIA/benchmark/" + fname))).keys())
    for i, c in enumerate(cats):  # BIPIA files list categories in blocks of five per group (verified on test split)
        BIPIA_GROUPS[c] = groups[i // 5]

DATASETS = {
    "AgentDojo": ("agentdojo_pairs.jsonl", "normalized_probs_main.json", "agentdojo_pairs"),
    "AgentDojo-shift": ("agentdojo_pairs_shift.jsonl", "normalized_probs_shift.json", "agentdojo_pairs_shift"),
    "BIPIA": ("bipia_pairs.jsonl", "normalized_probs_bipia.json", "bipia_pairs"),
}
METHODS = ["ProtectAI", "PIGuard", "Judge (Qwen2.5-7B)", "Judge (Llama-3.1-8B)"]
scores, labels, rowsets = {}, {}, {}
for dname, (pairs, det_file, stem) in DATASETS.items():
    rows = load_jsonl(pairs)
    y = np.array([r["label"] for r in rows])
    keep = np.ones(len(rows), bool)
    if dname == "BIPIA":  # main analysis: only BIPIA test-split attacks (absent from PIGuard's training split)
        keep = np.array([r["label"] == 0 or r.get("attack_split") == "test" for r in rows])
    D = json.load(open(det_file))["probs"]
    s = {"ProtectAI": np.array(D["protectai-deberta-v2/original"]), "PIGuard": np.array(D["piguard/original"]),
         "ProtectAI+norm": np.array(D["protectai-deberta-v2/values"]), "PIGuard+norm": np.array(D["piguard/values"]),
         "PIGuard+perfield": np.array(D["piguard/per_field"])}
    for m, f in [("Judge (Qwen2.5-7B)", f"hfjudge_Qwen2.5-7B-Instruct_{stem}.json"),
                 ("Judge (Llama-3.1-8B)", f"hfjudge_Llama-3.1-8B-Instruct_{stem}.json")]:
        if os.path.exists(f):
            j = json.load(open(f))
            assert j["labels"] == y.tolist(), f"label order mismatch in {f}"
            s[m] = np.array(j["probs"])
    scores[dname] = {k: v[keep] for k, v in s.items()}
    labels[dname] = y[keep]
    rowsets[dname] = [r for r, k in zip(rows, keep) if k]
    N[dname] = {k: metrics(y[keep], v[keep], ci=(k in METHODS)) for k, v in s.items()}
    N[dname]["_counts"] = {"benign": int((y[keep] == 0).sum()), "injected": int((y[keep] == 1).sum())}

# BIPIA including train-split attacks (contaminated for PIGuard), for the contamination analysis
rows = load_jsonl("bipia_pairs.jsonl")
y = np.array([r["label"] for r in rows])
D = json.load(open("normalized_probs_bipia.json"))["probs"]
for sub in ["test", "train"]:
    m = np.array([r["label"] == 0 or r.get("attack_split") == sub for r in rows])
    N.setdefault("BIPIA_by_attack_split", {})[sub] = {
        "PIGuard": metrics(y[m], np.array(D["piguard/original"])[m]),
        "ProtectAI": metrics(y[m], np.array(D["protectai-deberta-v2/original"])[m])}

# ---------------------------------------------------------------- training-set overlap audit (PIGuard published split)
def norm(t):
    return re.sub(r"\s+", " ", t.lower()).strip()


train_blob = norm(open(os.path.expanduser("~/agentsec/PIGuard/datasets/train.json"), encoding="utf-8", errors="ignore").read())
ov = {}
for fname in ["text_attack_train", "text_attack_test", "code_attack_train", "code_attack_test"]:
    d = json.load(open(os.path.expanduser(f"~/agentsec/BIPIA/benchmark/{fname}.json")))
    strs = [x for v in d.values() for x in v]
    ov[fname] = [int(sum(norm(x)[:80] in train_blob for x in strs)), len(strs)]
for t in ["email", "table", "code"]:
    ctx = [r["text"] for r in rows if r["suite"] == "bipia-" + t and r["label"] == 0]
    ov[f"clean_{t}_contexts"] = [int(sum(norm(c)[:120] in train_blob for c in ctx)), len(ctx)]
ov["agentdojo_template_present"] = "this is an important message from me" in train_blob
N["piguard_train_overlap"] = ov

# ---------------------------------------------------------------- BIPIA by official attack group
grp = {}
r_b, y_b = rowsets["BIPIA"], labels["BIPIA"]
groups = np.array([BIPIA_GROUPS.get(r["attack"].split(":", 1)[1], "?") if r["label"] == 1 else "benign" for r in r_b])
for m in METHODS + ["PIGuard"]:
    if m not in scores["BIPIA"]:
        continue
    p = scores["BIPIA"][m]
    grp[m] = {g: {"tpr_at_05": float(np.mean(p[groups == g] >= 0.5)), "n": int((groups == g).sum())}
              for g in ["task-irrelevant", "task-relevant", "targeted", "code-passive", "code-active"]}
N["BIPIA_by_group"] = grp

json.dump(N, open(os.path.join(OUT, "numbers.json"), "w"), indent=1)
print("wrote", os.path.join(OUT, "numbers.json"))

# ================================================================ FIGURES
avail = [m for m in METHODS if all(m in scores[d] for d in DATASETS)]

# Fig 2: clean tool outputs, per suite
fig, ax = plt.subplots(1, 2, figsize=(6.75, 2.1), sharey=True)
x = np.arange(len(suites))
for k, (metric, title) in enumerate([("output_fpr", "Benign tool outputs flagged"),
                                     ("task_block", "Tasks with $\\geq$1 output flagged")]):
    for i, det in enumerate(["ProtectAI", "PIGuard"]):
        v = [clean_rows[det][s][metric] * 100 for s in suites]
        b = ax[k].bar(x + (i - 0.5) * 0.36, v, 0.34, color=C[det], label=det)
        for bb, vv in zip(b, v):
            ax[k].text(bb.get_x() + bb.get_width() / 2, vv + 1.5, f"{vv:.0f}", ha="center", fontsize=6.5)
    ax[k].set_xticks(x)
    ax[k].set_xticklabels(suites)
    ax[k].set_title(title)
    ax[k].set_ylim(0, 105)
ax[0].set_ylabel("%")
ax[0].legend(loc="upper left")
fig.savefig(os.path.join(FIG, "fig_clean_outputs.pdf"))
fig.savefig(os.path.join(FIG, "fig_clean_outputs.png"))

# Fig 3: TPR at 1% FPR across benchmarks (the "flip")
fig, ax = plt.subplots(figsize=(3.3, 2.2))
dn = list(DATASETS)
x = np.arange(len(dn))
w = 0.8 / len(avail)
for i, m in enumerate(avail):
    v = [N[d][m]["tpr_at_1fpr"] * 100 for d in dn]
    lo = [v[j] - N[d][m]["tpr_at_1fpr_ci"][0] * 100 for j, d in enumerate(dn)]
    hi = [N[d][m]["tpr_at_1fpr_ci"][1] * 100 - v[j] for j, d in enumerate(dn)]
    ax.bar(x + (i - len(avail) / 2 + 0.5) * w, v, w * 0.92, color=C[m], label=m,
           yerr=[lo, hi], error_kw=dict(lw=0.6, capsize=1.5))
ax.set_xticks(x)
ax.set_xticklabels(["AD-Matched", "AD-Shift", "BIPIA"])
ax.set_ylabel("Detection rate at 1% FPR (%)")
ax.set_ylim(0, 105)
ax.legend(fontsize=6.3, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.savefig(os.path.join(FIG, "fig_flip.pdf"))
fig.savefig(os.path.join(FIG, "fig_flip.png"))

# Fig 4: ROC curves, log FPR
fig, axs = plt.subplots(1, 3, figsize=(6.75, 2.1), sharey=True)
for ax, d in zip(axs, dn):
    for m in avail:
        fpr, tpr, _ = roc_curve(labels[d], scores[d][m])
        ax.plot(np.clip(fpr, 1e-3, 1), tpr, color=C[m], lw=1.3, label=("ProtectAI v2" if m == "ProtectAI" else m))
    ax.axvline(0.01, color="#888", lw=0.7, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1)
    ax.set_title({"AgentDojo": "AD-Matched", "AgentDojo-shift": "AD-Shift", "BIPIA": "BIPIA (test-half attacks)"}[d])
    ax.set_xlabel("False positive rate")
axs[0].set_ylabel("True positive rate")
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=7)
fig.savefig(os.path.join(FIG, "fig_roc.pdf"))
fig.savefig(os.path.join(FIG, "fig_roc.png"))

# Fig 5: detection by attack goal group (BIPIA official groups) + AgentDojo for reference
G = ["task-irrelevant", "task-relevant", "targeted", "code-passive", "code-active"]
GL = ["Task-\nirrel.", "Task-\nrelev.", "Tar-\ngeted", "Code\npassive", "Code\nactive", "AD-\nMatched"]
show = ["PIGuard"] + [m for m in avail if m.startswith("Judge")]
fig, ax = plt.subplots(figsize=(3.3, 2.1))
x = np.arange(len(GL))
w = 0.8 / len(show)
for i, m in enumerate(show):
    v = [grp[m][g]["tpr_at_05"] * 100 for g in G] + [N["AgentDojo"][m]["tpr_at_05"] * 100]
    ax.bar(x + (i - len(show) / 2 + 0.5) * w, v, w * 0.92, color=C[m], label=m)
ax.set_xticks(x)
ax.set_xticklabels(GL, fontsize=6.5)
ax.set_ylabel("Detected at threshold 0.5 (%)")
ax.set_ylim(0, 105)
ax.axvline(4.5, color="#888", lw=0.7, ls="--")
ax.legend(fontsize=6.3, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
fig.savefig(os.path.join(FIG, "fig_goal.pdf"))
fig.savefig(os.path.join(FIG, "fig_goal.png"))

# Fig 6: format normalization
fig, ax = plt.subplots(1, 2, figsize=(6.75, 2.0), gridspec_kw=dict(wspace=0.35))
ra = N["readme_format_ablation"]
for i, det in enumerate(["ProtectAI", "PIGuard"]):
    v = [ra[det]["original"] * 100, ra[det]["plain"] * 100]
    ax[0].bar(np.arange(2) + (i - 0.5) * 0.36, v, 0.34, color=C[det], label=det)
ax[0].set_xticks(range(2))
ax[0].set_xticklabels(["README as-is", "markup removed"])
ax[0].set_ylabel("False positive rate (%)")
ax[0].set_title("Benign README chunks")
ax[0].legend()
for i, det in enumerate(["ProtectAI", "PIGuard"]):
    a, b = N["AgentDojo"][det], N["AgentDojo"][det + "+norm"]
    ax[1].scatter([a["fpr_at_05"] * 100], [a["tpr_at_05"] * 100], color=C[det], marker="o", s=22)
    ax[1].scatter([b["fpr_at_05"] * 100], [b["tpr_at_05"] * 100], color=C[det], marker="s", s=22)
    ax[1].annotate("", xy=(b["fpr_at_05"] * 100, b["tpr_at_05"] * 100), xytext=(a["fpr_at_05"] * 100, a["tpr_at_05"] * 100),
                   arrowprops=dict(arrowstyle="->", color=C[det], lw=0.9))
ax[1].set_xlabel("FPR on benign tool outputs (%)")
ax[1].set_ylabel("TPR (%)")
ax[1].set_title("AD-Matched tool outputs")
ax[1].set_xlim(0, 25)
ax[1].set_ylim(40, 90)
fig.savefig(os.path.join(FIG, "fig_format.pdf"))
fig.savefig(os.path.join(FIG, "fig_format.png"))
print("figures written to", FIG)
