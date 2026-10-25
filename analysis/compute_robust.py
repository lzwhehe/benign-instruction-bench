"""Robustness of the main results: uncertainty of rank transfer (E1) and thresholds chosen on held-out data (E2).

E1: stratified bootstrap over samples (1,000 resamples, the same resample for every detector) of Kendall's tau between
    two benchmarks' per-detector scores, plus a permutation test over detectors.
E2: the 1%/2%-FPR threshold is chosen on benign outputs the detector is not evaluated on: (a) all of AD-Clean, then
    evaluate on AD-Matched and AD-Shift; (b) leave-one-suite-out: AD-Clean outputs of three AgentDojo suites, then
    evaluate on the fourth suite's AD-Matched outputs and AD-Clean outputs.

Reads results/many/<detector>__<set>.json and the rebuilt evaluation sets in the working directory
(agentdojo_benign_outputs.jsonl, agentdojo_pairs.jsonl, agentdojo_pairs_shift.jsonl, bipia_pairs.jsonl).
Writes analysis/numbers_robust.json and paper/tab_heldout.tex.
"""
import collections
import glob
import json

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import roc_curve

B = 1000
RNG = np.random.default_rng(0)
EXCLUDE = {"injecguard"}  # identical to PIGuard
VARIANTS = {"sheltron-taskaware", "sheltron-generic"}  # not standalone detectors; excluded from rank transfer
SUITES = ["workspace", "travel", "banking", "slack"]


def load_jsonl(p):
    return [json.loads(line) for line in open(p)]


def tpr_at(y, p, t=0.01):
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= t
    return float(tpr[ok].max()) if ok.any() else 0.0


def threshold(benign, budget):
    """Smallest threshold that flags (score > thr) at most `budget` of the given benign scores."""
    s = np.sort(np.asarray(benign))
    return s[int(np.ceil((1 - budget) * len(s))) - 1]


det = collections.defaultdict(dict)
for f in glob.glob("results/many/*.json"):
    d = json.load(open(f))
    det[d["detector"]][d["set"]] = d
det = {k: v for k, v in det.items() if k not in EXCLUDE}

clean_rows = load_jsonl("agentdojo_benign_outputs.jsonl")
pair_rows = load_jsonl("agentdojo_pairs.jsonl")
shift_rows = load_jsonl("agentdojo_pairs_shift.jsonl")
bip_rows = load_jsonl("bipia_pairs.jsonl")
bip_keep = np.array([r["label"] == 0 or r.get("attack_split") == "test" for r in bip_rows])

# The rebuilt sets must line up with the saved scores sample by sample (same order, same labels).
check = {}
for s, rows in [("admatched", pair_rows), ("adshift", shift_rows), ("bipia", bip_rows)]:
    saved = det["piguard"][s]["labels"]
    check[s] = dict(n=len(rows), n_saved=len(saved), labels_equal=[r["label"] for r in rows] == saved)
check["adclean"] = dict(n=len(clean_rows), n_saved=len(det["piguard"]["adclean"]["probs"]))
assert all(c.get("labels_equal", True) and c["n"] == c["n_saved"] for c in check.values()), check
pair_suite = np.array([r["suite"] for r in pair_rows])
shift_suite = np.array([r["suite"] for r in shift_rows])
clean_suite = np.array([r["suite"] for r in clean_rows])
# How many AD-Matched benign outputs are verbatim AD-Clean outputs (calibration and evaluation data share text)?
clean_text = {r["text"] for r in clean_rows}
check["admatched_benign_in_adclean"] = float(np.mean([r["text"] in clean_text for r in pair_rows if r["label"] == 0]))


def arrays(name, s):
    d = det[name][s]
    p = np.asarray(d["probs"], dtype=float)
    y = None if d["labels"] is None else np.asarray(d["labels"])
    if s == "bipia":
        p, y = p[bip_keep], y[bip_keep]
    return y, p


dets = [n for n in det if n not in VARIANTS and all(s in det[n] for s in ["admatched", "bipia", "tbeval", "adclean"])]
dets.sort()

# ---------------- E1: uncertainty of rank transfer ----------------
Y = {s: arrays(dets[0], s)[0] for s in ["admatched", "bipia", "tbeval"]}
P = {(n, s): arrays(n, s)[1] for n in dets for s in ["admatched", "bipia", "tbeval", "adclean"]}


def metric(n, key, idx):
    if key == "adclean_fpr":
        return float(np.mean(P[(n, "adclean")][idx["adclean"]] >= 0.5))
    if key == "tbeval_fpr05":
        p = P[(n, "tbeval")][idx["tbeval"]]
        return float(np.mean(p[Y["tbeval"][idx["tbeval"]] == 0] >= 0.5))
    s = key.replace("_tpr1", "")
    if s == "admatched":  # injected AD-Matched outputs against all AD-Clean outputs (deduplicated, with clean twins)
        pos = P[(n, "admatched")][AD_POS][idx["adpos"]]
        neg = P[(n, "adclean")][idx["adclean"]]
        return tpr_at(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])
    return tpr_at(Y[s][idx[s]], P[(n, s)][idx[s]])


AD_POS = np.where(Y["admatched"] == 1)[0]
# task clusters for the bootstrap: AgentDojo injected outputs and AD-Clean outputs by task; tau-bench outputs by task
_adt = sorted({(pair_rows[i]["suite"], pair_rows[i]["task"]) for i in AD_POS} | {(r["suite"], r["task"]) for r in clean_rows})
_adp = {t: [k for k, i in enumerate(AD_POS) if (pair_rows[i]["suite"], pair_rows[i]["task"]) == t] for t in _adt}
_adc = {t: [k for k, r in enumerate(clean_rows) if (r["suite"], r["task"]) == t] for t in _adt}
_tbr = [json.loads(line) for line in open("taubench_eval.jsonl")]
_tbt = sorted({(r["suite"], r["task"]) for r in _tbr})
_tbi = {t: [k for k, r in enumerate(_tbr) if (r["suite"], r["task"]) == t] for t in _tbt}


def resample():
    ts = [_adt[i] for i in RNG.integers(0, len(_adt), len(_adt))]
    tt = [_tbt[i] for i in RNG.integers(0, len(_tbt), len(_tbt))]
    return dict(adpos=np.array([k for t in ts for k in _adp[t]], dtype=int),
                adclean=np.array([k for t in ts for k in _adc[t]], dtype=int),
                tbeval=np.array([k for t in tt for k in _tbi[t]], dtype=int),
                bipia=stratified(Y["bipia"]))


def stratified(y):
    return np.concatenate([RNG.choice(np.where(y == c)[0], size=(y == c).sum(), replace=True) for c in (0, 1)])


PAIRS = [("bipia_tpr1", "admatched_tpr1"), ("admatched_tpr1", "tbeval_tpr1"), ("bipia_tpr1", "tbeval_tpr1"),
         ("adclean_fpr", "tbeval_fpr05")]
full = {s: np.arange(len(Y[s])) for s in Y}
full["adclean"] = np.arange(len(P[(dets[0], "adclean")]))
full["adpos"] = np.arange(len(AD_POS))
obs = {key: np.array([metric(n, key, full) for n in dets]) for key in {k for pr in PAIRS for k in pr}}
boot = {pr: [] for pr in PAIRS}
for b in range(B):
    idx = resample()
    vals = {key: np.array([metric(n, key, idx) for n in dets]) for key in obs}
    for a, c in PAIRS:
        boot[(a, c)].append(kendalltau(vals[a], vals[c])[0])
e1 = {}
for a, c in PAIRS:
    tau = kendalltau(obs[a], obs[c])[0]
    perm = np.array([kendalltau(obs[a], RNG.permutation(obs[c]))[0] for _ in range(10000)])
    bt = np.array(boot[(a, c)])
    e1[f"{a}_vs_{c}"] = dict(tau=float(tau), ci=[float(np.nanpercentile(bt, 2.5)), float(np.nanpercentile(bt, 97.5))],
                             p_perm=float(np.mean(np.abs(perm) >= abs(tau) - 1e-12)), share_boot_positive=float(np.mean(bt > 0)),
                             n_detectors=len(dets))
    print(f"{a} vs {c}: tau={tau:.2f} 95% CI [{e1[f'{a}_vs_{c}']['ci'][0]:.2f}, {e1[f'{a}_vs_{c}']['ci'][1]:.2f}] "
          f"perm p={e1[f'{a}_vs_{c}']['p_perm']:.3f}")

# ---------------- E2: thresholds chosen on held-out benign outputs ----------------
e2 = {}
for n in sorted(det):
    sets = det[n]
    if not all(s in sets for s in ["adclean", "admatched", "adshift"]):
        continue
    pc = np.asarray(sets["adclean"]["probs"], dtype=float)
    ym, pm = np.asarray(sets["admatched"]["labels"]), np.asarray(sets["admatched"]["probs"], dtype=float)
    ys, ps = np.asarray(sets["adshift"]["labels"]), np.asarray(sets["adshift"]["probs"], dtype=float)
    yf = np.r_[np.ones(int((ym == 1).sum())), np.zeros(len(pc))]
    row = dict(insample_tpr1=tpr_at(yf, np.r_[pm[ym == 1], pc]), insample_tpr2=tpr_at(yf, np.r_[pm[ym == 1], pc], 0.02),
               matchedneg_tpr1=tpr_at(ym, pm), adshift_insample_tpr1=tpr_at(ys, ps))
    for bud, tag in [(0.01, "1"), (0.02, "2")]:
        thr = threshold(pc, bud)
        row[f"clean_tpr{tag}"] = float(np.mean(pm[ym == 1] > thr))
        row[f"clean_fpr{tag}"] = float(np.mean(pm[ym == 0] > thr))
        row[f"clean_shift_tpr{tag}"] = float(np.mean(ps[ys == 1] > thr))
        row[f"clean_shift_fpr{tag}"] = float(np.mean(ps[ys == 0] > thr))
        det_pos = n_pos = fp_m = n_neg_m = fp_c = n_c = 0
        per_suite = {}
        for s in SUITES:
            thr_s = threshold(pc[clean_suite != s], bud)
            pos, neg = pm[(pair_suite == s) & (ym == 1)], pm[(pair_suite == s) & (ym == 0)]
            cs = pc[clean_suite == s]
            det_pos += int((pos > thr_s).sum()); n_pos += len(pos)
            fp_m += int((neg > thr_s).sum()); n_neg_m += len(neg)
            fp_c += int((cs > thr_s).sum()); n_c += len(cs)
            per_suite[s] = dict(tpr=float(np.mean(pos > thr_s)), fpr_clean=float(np.mean(cs > thr_s)))
        row[f"loso_tpr{tag}"] = det_pos / n_pos
        row[f"loso_fpr{tag}"] = fp_m / n_neg_m
        row[f"loso_fpr_clean{tag}"] = fp_c / n_c
        row[f"loso_suites{tag}"] = per_suite
    e2[n] = row
    print(f"{n:22s} in-sample {row['insample_tpr1']:.3f} | AD-Clean thr: TPR {row['clean_tpr1']:.3f} FPR {row['clean_fpr1']:.3f}"
          f" | LOSO 1%: TPR {row['loso_tpr1']:.3f} FPR {row['loso_fpr_clean1']:.3f} | LOSO 2%: TPR {row['loso_tpr2']:.3f}"
          f" FPR {row['loso_fpr_clean2']:.3f}")

json.dump(dict(check=check, e1=e1, e2=e2, detectors_in_rank=dets, bootstrap_resamples=B),
          open("analysis/numbers_robust.json", "w"), indent=1)
print(json.dumps(check, indent=1))  # the paper's table is written by analysis/make_tables.py
