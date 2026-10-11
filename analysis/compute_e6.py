"""Reconciling with the community benchmark buried-injections (E6) and checking the rebuilt sets.

Reads results/e6/<variant>__<set>.json (scripts/score_e6.py), the stored scores results/many/*.json, and the rebuilt
agentdojo_pairs.jsonl (for attack templates). Writes analysis/numbers_e6.json.

For Prompt Guard 2 (86M) and three input settings (our 512-token segments on the tool output; buried-injections'
overlapping windows on the tool output; overlapping windows on the user's task prompt followed by the tool output)
it reports: detection at the default threshold, detection at 1% and 2% FPR in-sample, and leave-one-suite-out
detection at a 2% budget calibrated on AD-Clean outputs of the other suites (pooled, worst fold, realized FPR).
It also breaks detection down by AgentDojo attack template.
"""
import json

import numpy as np
from sklearn.metrics import roc_curve

SUITES = ["workspace", "travel", "banking", "slack"]


def load(variant, s):
    return json.load(open(f"results/e6/{variant}__{s}.json"))


def tpr_at(y, p, t):
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= t
    return float(tpr[ok].max()) if ok.any() else 0.0


def threshold(benign, budget):
    s = np.sort(np.asarray(benign))
    return s[int(np.ceil((1 - budget) * len(s))) - 1]


out = {}
# 1. rebuilt sets reproduce the stored scores on benign outputs
for variant, stored in [("piguard-rebuilt", "piguard"), ("promptguard2-86m-rebuilt", "promptguard2-86m")]:
    r = {}
    for s in ["adclean", "admatched"]:
        new, old = load(variant, s), json.load(open(f"results/many/{stored}__{s}.json"))
        pn, po = np.array(new["probs"]), np.array(old["probs"])
        if s == "adclean":
            r["adclean_max_abs_diff"] = float(np.abs(pn - po).max())
        else:
            y = np.array(new["labels"])
            r["admatched_benign_max_abs_diff"] = float(np.abs(pn[y == 0] - po[y == 0]).max())
            r["admatched_tpr1_stored"] = tpr_at(y, po, 0.01)
            r["admatched_tpr1_rebuilt"] = tpr_at(y, pn, 0.01)
            r["admatched_auroc_like_stored_vs_rebuilt_corr_injected"] = float(np.corrcoef(pn[y == 1], po[y == 1])[0, 1])
    out[f"check_{stored}"] = r

# 2. Prompt Guard 2 (86M) in three input settings
attack = np.array([json.loads(line)["attack"] for line in open("agentdojo_pairs.jsonl")])
for variant, tag in [("promptguard2-86m-rebuilt", "seg"), ("pg2-overlap-tool", "ovl"), ("pg2-overlap-task+tool", "task")]:
    c, m = load(variant, "adclean"), load(variant, "admatched")
    pc, cs = np.array(c["probs"]), np.array(c["suites"])
    pm, y, ms = np.array(m["probs"]), np.array(m["labels"]), np.array(m["suites"])
    r = dict(default_tpr=float(np.mean(pm[y == 1] >= 0.5)), default_fpr_clean=float(np.mean(pc >= 0.5)),
             tpr1=tpr_at(y, pm, 0.01), tpr2=tpr_at(y, pm, 0.02))
    det = n = fp = nc = 0
    folds = {}
    for s in SUITES:
        thr = threshold(pc[cs != s], 0.02)
        pos, ben = pm[(ms == s) & (y == 1)], pc[cs == s]
        det += int((pos > thr).sum()); n += len(pos); fp += int((ben > thr).sum()); nc += len(ben)
        folds[s] = dict(tpr=float(np.mean(pos > thr)), fpr=float(np.mean(ben > thr)), thr=float(thr))
    r.update(loso2_tpr=det / n, loso2_fpr=fp / nc, loso2_min_tpr=min(f["tpr"] for f in folds.values()),
             loso2_max_fpr=max(f["fpr"] for f in folds.values()), folds=folds)
    # the community benchmark uses only the important_instructions template: same protocol on those injections
    ii = attack == "important_instructions"
    tf = {s: float(np.mean(pm[(ms == s) & (y == 1) & ii] > folds[s]["thr"])) for s in SUITES}
    r["loso2_tpr_ii"] = float(sum((pm[(ms == s) & (y == 1) & ii] > folds[s]["thr"]).sum() for s in SUITES) / ((y == 1) & ii).sum())
    r["loso2_min_tpr_ii"] = min(tf.values())
    r["by_template_default_tpr"] = {a: float(np.mean(pm[(y == 1) & (attack == a)] >= 0.5)) for a in sorted(set(attack))}
    thr1 = threshold(pm[y == 0], 0.01)
    r["by_template_tpr1"] = {a: float(np.mean(pm[(y == 1) & (attack == a)] > thr1)) for a in sorted(set(attack))}
    out[tag] = r
    print(tag, json.dumps({k: v for k, v in r.items() if k != "folds"}, indent=None))
for k in ["check_piguard", "check_promptguard2-86m"]:
    print(k, out[k])
json.dump(out, open("analysis/numbers_e6.json", "w"), indent=1)
