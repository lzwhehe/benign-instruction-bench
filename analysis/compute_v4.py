"""Analyses added after the pre-submission review (journal revision v4).

Reads the stored scores (results/many, results/hfjudge_*), the rebuilt sets in the working directory
(agentdojo_benign_outputs / agentdojo_pairs / agentdojo_pairs_shift / taubench_eval / bipia_pairs / llmail_eval .jsonl),
new scores (results/window, results/hfjudge_noreq_*, results/many/*__llmail.json), PIGuard's training split and
InjecAgent (under ~/agentsec). Writes analysis/numbers_v4.json.

 1. AD-Matched structure: duplicate benign outputs and the clean twins of injected outputs
 2. Detection on AgentDojo with deduplicated negatives that include the clean twins (= AD-Clean), and the rank
    correlations recomputed on it
 3. Task-clustered bootstrap intervals for TPR at 1% FPR (all detectors) and paired differences between leaders
 4. Truncation: where injected text starts, in DeBERTa-v3 tokens
 5. PIGuard seen/unseen InjecAgent attacks: attack-clustered intervals and a difference-in-differences
 6. tau-bench negatives restricted to the tools that carry injections (retail product lookups)
 7. Held-out thresholds: worst suite and task-clustered intervals
 8. Detector-level sensitivity of the rank correlations
 9. AD-Shift by attack template
10. BIPIA stratified by whether the clean context occurs in PIGuard's training split
11. LLMail-Inject phase 2 (adaptive attacks)
12. Judges without the user's request
13. Uniform sliding-window scoring of the detectors that truncate
"""
import collections
import glob
import json
import os
import re

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score, roc_curve

RNG = np.random.default_rng(0)
B = 1000
SUITES = ["workspace", "travel", "banking", "slack"]
EXCLUDE = {"injecguard"}
VARIANTS = {"sheltron-taskaware", "sheltron-generic"}
LEAD = ["horizon-base", "wolf-defender", "prismor-1.5b", "promptguard2-86m", "promptguard2-22m", "qualifire-sentinel",
        "sheltron", "piguard", "protectai-v2"]


def load(p):
    return [json.loads(line) for line in open(p)]


def tpr_at(y, p, t=0.01):
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= t
    return float(tpr[ok].max()) if ok.any() else 0.0


def thr_at(neg, budget):
    s = np.sort(np.asarray(neg))
    return s[int(np.ceil((1 - budget) * len(s))) - 1]


def ci(a):
    a = np.asarray(a, dtype=float)
    return [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]


det = collections.defaultdict(dict)
for f in glob.glob("results/many/*.json"):
    d = json.load(open(f))
    if d["detector"] not in EXCLUDE:
        det[d["detector"]][d["set"]] = d
for name, stem in [("judge-qwen", "Qwen2.5-7B-Instruct"), ("judge-llama", "Llama-3.1-8B-Instruct")]:
    for s, fn in [("admatched", "agentdojo_pairs"), ("tbeval", "taubench_eval"), ("bipia", "bipia_pairs")]:
        if os.path.exists(f"results/hfjudge_{stem}_{fn}.json"):
            det[name][s] = json.load(open(f"results/hfjudge_{stem}_{fn}.json"))
for name, stem in [("judge-qwen", "Qwen2.5-7B-Instruct"), ("judge-llama", "Llama-3.1-8B-Instruct")]:
    for tag, pre in [("", ""), ("-noreq", "noreq_")]:
        for s_, fn in [("admatched", "agentdojo_pairs"), ("adclean", "agentdojo_clean"), ("tbeval", "taubench_eval")]:
            f = f"results/hfjudge_{pre}{stem}_{fn}.json"
            if tag == "" and s_ != "adclean" and os.path.exists(f"results/hfjudge_rb_{stem}_{fn}.json"):
                f = f"results/hfjudge_rb_{stem}_{fn}.json"  # same rebuilt outputs as the no-request runs
            if os.path.exists(f):
                det[name + tag][s_] = json.load(open(f))
JUDGES = ["judge-qwen", "judge-qwen-noreq", "judge-llama", "judge-llama-noreq"]
P = lambda n, s: np.asarray(det[n][s]["probs"], dtype=float)  # noqa: E731
DETS = sorted(n for n in det if n not in VARIANTS and not n.startswith("judge") and "admatched" in det[n])
OUT = {}

clean = load("agentdojo_benign_outputs.jsonl")
pairs = load("agentdojo_pairs.jsonl")
shift = load("agentdojo_pairs_shift.jsonl")
tb = load("taubench_eval.jsonl")
bip = load("bipia_pairs.jsonl")
ym = np.array([r["label"] for r in pairs])
ytb = np.array([r["label"] for r in tb])
assert ym.tolist() == det["piguard"]["admatched"]["labels"] and ytb.tolist() == det["piguard"]["tbeval"]["labels"]

# ---------------------------------------------------------------- 1. structure of AD-Matched
ben_txt = [r["text"] for r in pairs if r["label"] == 0]
pos_keys = {(r["suite"], r["task"], r["step"]) for r in pairs if r["label"] == 1}
clean_key = {(r["suite"], r["task"], r["step"]): r["text"] for r in clean}
twins = {clean_key[k] for k in pos_keys if k in clean_key}
OUT["structure"] = dict(n_benign=len(ben_txt), n_benign_distinct=len(set(ben_txt)), n_injected=int(ym.sum()),
                        n_injection_positions=len(pos_keys), n_adclean=len(clean),
                        n_twins_distinct=len(twins), twins_in_admatched_benign=len(twins & set(ben_txt)),
                        n_twin_rows=sum((r["suite"], r["task"], r["step"]) in pos_keys for r in clean),
                        n_adclean_distinct=len({r["text"] for r in clean}),
                        adclean_per_suite=dict(collections.Counter(r["suite"] for r in clean)))

# ---------------------------------------------------------------- 2. AgentDojo with AD-Clean negatives (dedup + twins)
full = {}
for n in DETS:
    pos, neg = P(n, "admatched")[ym == 1], P(n, "adclean")
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    full[n] = dict(tpr1=tpr_at(y, np.r_[pos, neg]), auroc=float(roc_auc_score(y, np.r_[pos, neg])),
                   orig_tpr1=tpr_at(ym, P(n, "admatched")))
OUT["ad_full_negatives"] = full
bip_keep = np.array([r["label"] == 0 or r.get("attack_split") == "test" for r in bip])
ybi = np.array([r["label"] for r in bip])[bip_keep]


def tau_pair(a, b):
    t, p = kendalltau(a, b)
    return dict(tau=float(t), p=float(p))


RK = [n for n in DETS if "bipia" in det[n] and "tbeval" in det[n]]
bi1 = {n: tpr_at(ybi, P(n, "bipia")[bip_keep]) for n in RK}
tb1 = {n: tpr_at(ytb, P(n, "tbeval")) for n in RK}
OUT["tau_full"] = dict(bipia_ad=tau_pair([bi1[n] for n in RK], [full[n]["tpr1"] for n in RK]),
                       ad_tb=tau_pair([full[n]["tpr1"] for n in RK], [tb1[n] for n in RK]),
                       n=len(RK))

# ---------------------------------------------------------------- 3. task-clustered bootstrap intervals
def cluster_index(rows):
    g = collections.defaultdict(list)
    for i, r in enumerate(rows):
        g[(r["suite"], r["task"])].append(i)
    return list(g.values())


ad_pos_idx = np.where(ym == 1)[0]
ad_neg_rows = clean  # deduplicated negatives with twins
ad_cl_pos = collections.defaultdict(list)
for i in ad_pos_idx:
    ad_cl_pos[(pairs[i]["suite"], pairs[i]["task"])].append(i)
ad_cl_neg = collections.defaultdict(list)
for i, r in enumerate(clean):
    ad_cl_neg[(r["suite"], r["task"])].append(i)
ad_tasks = sorted(set(ad_cl_pos) | set(ad_cl_neg))
tb_cl = collections.defaultdict(list)
for i, r in enumerate(tb):
    tb_cl[(r["suite"], r["task"])].append(i)
tb_tasks = sorted(tb_cl)
boots = collections.defaultdict(lambda: collections.defaultdict(list))
for b in range(B):
    ts = [ad_tasks[i] for i in RNG.integers(0, len(ad_tasks), len(ad_tasks))]
    pi = [j for t in ts for j in ad_cl_pos.get(t, [])]
    ni = [j for t in ts for j in ad_cl_neg.get(t, [])]
    tt = [tb_tasks[i] for i in RNG.integers(0, len(tb_tasks), len(tb_tasks))]
    ti = np.array([j for t in tt for j in tb_cl[t]])
    for n in DETS + [j for j in JUDGES if "adclean" in det[j]]:
        pos, neg = P(n, "admatched")[pi], P(n, "adclean")[ni]
        boots[n]["ad"].append(tpr_at(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))
        if "tbeval" in det[n]:
            boots[n]["tb"].append(tpr_at(ytb[ti], P(n, "tbeval")[ti]))
def ad_full(n):
    pos, neg = P(n, "admatched")[ym == 1], P(n, "adclean")
    return tpr_at(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])


OBS = {n: dict(ad=ad_full(n), tb=tpr_at(ytb, P(n, "tbeval")) if "tbeval" in det[n] else None)
       for n in DETS + [j for j in JUDGES if "adclean" in det[j]]}
OUT["cluster_ci"] = {n: dict(ad=OBS[n]["ad"], ad_ci=ci(boots[n]["ad"]), tb=OBS[n]["tb"],
                             tb_ci=ci(boots[n]["tb"]) if boots[n]["tb"] else None) for n in OBS}
OUT["paired_diff"] = {}
for a, c, s in [("horizon-base", "wolf-defender", "ad"), ("horizon-base", "prismor-1.5b", "ad"),
                ("wolf-defender", "prismor-1.5b", "ad"), ("horizon-base", "sheltron", "tb"),
                ("horizon-base", "wolf-defender", "tb"), ("sheltron", "wolf-defender", "tb"),
                ("judge-qwen-noreq", "judge-qwen", "ad"), ("judge-llama-noreq", "judge-llama", "ad"),
                ("judge-qwen-noreq", "judge-qwen", "tb"), ("judge-llama-noreq", "judge-llama", "tb")]:
    if not boots[a][s] or not boots[c][s]:
        continue
    d = np.array(boots[a][s]) - np.array(boots[c][s])
    # point estimate: the observed difference; interval: percentile bootstrap of the paired difference
    OUT["paired_diff"][f"{a}-{c}-{s}"] = dict(diff=float(OBS[a][s] - OBS[c][s]), boot_mean=float(np.mean(d)),
                                              ci=ci(d), p_le0=float(np.mean(d <= 0)))

# ---------------------------------------------------------------- 4. truncation
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("protectai/deberta-v3-base-prompt-injection-v2")


def start_tokens(text, clean_text):
    k = 0
    for a, c in zip(text, clean_text):
        if a != c:
            break
        k += 1
    return len(tok(text[:k], add_special_tokens=False)["input_ids"])


trunc = {}
for name, rows, ckey in [("admatched", pairs, clean_key), ("adshift", shift, clean_key),
                         ("tbeval", tb, {(r["suite"], r["task"], r["step"]): r["text"] for r in tb if r["label"] == 0
                                         and r["suite"] == "tau-retail"})]:
    st, ln = [], []
    for r in rows:
        if r["label"] != 1:
            continue
        c = ckey.get((r["suite"], r["task"], r["step"]))
        if c is None:
            continue
        st.append(start_tokens(r["text"], c))
        ln.append(len(tok(r["text"], add_special_tokens=False)["input_ids"]))
    st, ln = np.array(st), np.array(ln)
    trunc[name] = dict(n=len(st), start_beyond_510=float(np.mean(st >= 510)), start_beyond_2046=float(np.mean(st >= 2046)),
                       longer_than_510=float(np.mean(ln > 510)), median_start=float(np.median(st)))
OUT["truncation"] = trunc

# ---------------------------------------------------------------- 5. seen/unseen InjecAgent attacks (PIGuard)
def norm(t):
    return re.sub(r"\s+", " ", str(t).lower()).strip()


ia = []
for f in ["attacker_cases_dh.jsonl", "attacker_cases_ds.jsonl"]:
    ia += [json.loads(line)["Attacker Instruction"] for line in open(os.path.expanduser("~/agentsec/InjecAgent/data/" + f))]
pg_blob = norm(open(os.path.expanduser("~/agentsec/PIGuard/datasets/train.json"), encoding="utf-8", errors="ignore").read())
SEEN = {i for i, a in enumerate(ia) if norm(a)[:80] in pg_blob}
neg_tb = ytb == 0
att = np.array([r.get("attack_id", -1) for r in tb])
attacks = sorted(set(att[ytb == 1]))


def per_attack_tpr(n):
    p = P(n, "tbeval")
    t = thr_at(p[neg_tb], 0.01)
    return {a: float(np.mean(p[(att == a) & (ytb == 1)] > t)) for a in attacks}


pa = {n: per_attack_tpr(n) for n in DETS if "tbeval" in det[n]}
pa.update({n: per_attack_tpr(n) for n in ["judge-qwen", "judge-llama"] if "tbeval" in det[n]})
cnt = {a: int(np.sum((att == a) & (ytb == 1))) for a in attacks}
seen_a = [a for a in attacks if a in SEEN]
unseen_a = [a for a in attacks if a not in SEEN]


def gap(n, sa, ua):  # output-weighted seen minus unseen
    w = lambda lst: sum(pa[n][a] * cnt[a] for a in lst) / sum(cnt[a] for a in lst)  # noqa: E731
    return w(sa) - w(ua)


others = [n for n in pa if n != "piguard" and not n.startswith("judge")]
obs_gap = gap("piguard", seen_a, unseen_a)
obs_did = obs_gap - float(np.median([gap(n, seen_a, unseen_a) for n in others]))
bg, bd = [], []
for b in range(B):
    sa = list(RNG.choice(seen_a, len(seen_a)))
    ua = list(RNG.choice(unseen_a, len(unseen_a)))
    g = gap("piguard", sa, ua)
    bg.append(g)
    bd.append(g - float(np.median([gap(n, sa, ua) for n in others])))
OUT["seen_unseen"] = dict(n_seen_attacks=len(seen_a), n_unseen_attacks=len(unseen_a),
                          n_seen_outputs=sum(cnt[a] for a in seen_a), n_unseen_outputs=sum(cnt[a] for a in unseen_a),
                          piguard_gap=obs_gap, piguard_gap_ci=ci(bg),
                          others_median_gap=float(np.median([gap(n, seen_a, unseen_a) for n in others])),
                          did=obs_did, did_ci=ci(bd))

# ---------------------------------------------------------------- 6. tau-bench with matched negatives
pos_fns = {r["function"] for r in tb if r["label"] == 1}
m_neg = np.array([r["label"] == 0 and r["suite"] == "tau-retail" and r["function"] in pos_fns for r in tb])
m_sel = m_neg | (ytb == 1)
OUT["tb_matched_negatives"] = dict(functions=sorted(pos_fns), n_neg=int(m_neg.sum()), detectors={
    n: dict(tpr1=tpr_at(ytb[m_sel], P(n, "tbeval")[m_sel]), auroc=float(roc_auc_score(ytb[m_sel], P(n, "tbeval")[m_sel])),
            orig_tpr1=tb1.get(n))
    for n in DETS if "tbeval" in det[n]})

# ---------------------------------------------------------------- 7. held-out thresholds: worst suite and intervals
csuite = np.array([r["suite"] for r in clean])
psuite = np.array([r["suite"] for r in pairs])
lo = {}
for n in DETS:
    pc, pm = P(n, "adclean"), P(n, "admatched")
    res = {}
    for bud, tag in [(0.01, "1"), (0.02, "2")]:
        folds = {}
        for s in SUITES:
            t = thr_at(pc[csuite != s], bud)
            folds[s] = dict(tpr=float(np.mean(pm[(psuite == s) & (ym == 1)] > t)), fpr=float(np.mean(pc[csuite == s] > t)),
                            n_flag=int(np.sum(pc[csuite == s] > t)), n_neg=int(np.sum(csuite == s)),
                            thr=float(t))
        # task-clustered bootstrap of the pooled held-out TPR and FPR (thresholds fixed per fold)
        bt, bf = [], []
        for b in range(B):
            ts = [ad_tasks[i] for i in RNG.integers(0, len(ad_tasks), len(ad_tasks))]
            pi = [j for t_ in ts for j in ad_cl_pos.get(t_, [])]
            ni = [j for t_ in ts for j in ad_cl_neg.get(t_, [])]
            if not pi or not ni:
                continue
            bt.append(np.mean([pm[j] > folds[pairs[j]["suite"]]["thr"] for j in pi]))
            bf.append(np.mean([pc[j] > folds[clean[j]["suite"]]["thr"] for j in ni]))
        res[tag] = dict(folds=folds, worst_tpr=min(f["tpr"] for f in folds.values()),
                        worst_fpr=max(f["fpr"] for f in folds.values()), tpr_ci=ci(bt), fpr_ci=ci(bf))
    lo[n] = res
OUT["heldout"] = lo

# ---------------------------------------------------------------- 8. detector-level sensitivity of tau
NM = json.load(open("analysis/numbers_many.json"))["table"]
R = [n for n in NM if n in RK]
v = lambda k: [NM[n][k] for n in R]  # noqa: E731
sens = {}
for a, c in [("bipia_tpr1", "admatched_tpr1"), ("admatched_tpr1", "tbeval_tpr1"), ("bipia_tpr1", "tbeval_tpr1")]:
    full_t = kendalltau(v(a), v(c))[0]
    jk = [kendalltau([NM[n][a] for n in R if n != m], [NM[n][c] for n in R if n != m])[0] for m in R]
    nopg = kendalltau([NM[n][a] for n in R if n != "piguard"], [NM[n][c] for n in R if n != "piguard"])[0]
    fam = ["protectai-v2", "jasperls", "testsavant-large", "piguard", "promptguard2-86m", "qualifire-sentinel",
           "rogue-sentinel-v2", "sheltron", "horizon-base", "wolf-defender", "prismor-1.5b"]
    fam_t = kendalltau([NM[n][a] for n in fam], [NM[n][c] for n in fam])
    sens[f"{a}_vs_{c}"] = dict(tau=float(full_t), jackknife=[float(min(jk)), float(max(jk))], without_piguard=float(nopg),
                               one_per_family=float(fam_t[0]), one_per_family_p=float(fam_t[1]), n_family=len(fam))
for a, c in [("bipia_auroc", "admatched_auroc"), ("admatched_auroc", "tbeval_auroc"), ("general_fpr", "adclean_fpr"),
             ("general_fpr", "tbeval_fpr05")]:
    t, p = kendalltau(v(a), v(c))
    sens[f"{a}_vs_{c}"] = dict(tau=float(t), p=float(p))
OUT["tau_sensitivity"] = sens

# ---------------------------------------------------------------- 9. AD-Shift by attack template
ys = np.array([r["label"] for r in shift])
tpl = np.array([r["attack"] for r in shift])
by_t = {}
pm_t = np.array([r["attack"] for r in pairs])
for n in LEAD:
    t = thr_at(P(n, "adclean"), 0.01)  # same AD-Clean threshold for every template
    p = P(n, "adshift")
    by_t[n] = {a: float(np.mean(p[(tpl == a) & (ys == 1)] > t)) for a in sorted(set(tpl[ys == 1]))}
    p = P(n, "admatched")
    by_t[n].update({a: float(np.mean(p[(pm_t == a) & (ym == 1)] > t)) for a in sorted(set(pm_t[ym == 1]))})
OUT["by_template"] = by_t

# ---------------------------------------------------------------- 10. BIPIA by context-in-training
ctx_seen = {}
for r in bip:
    if r["label"] == 0:
        ctx_seen[r["task"]] = norm(r["text"])[:120] in pg_blob
yb = np.array([r["label"] for r in bip])
cs = np.array([ctx_seen.get(r["task"], False) for r in bip])
p = P("piguard", "bipia")
t = thr_at(p[(yb == 0)], 0.01)
OUT["bipia_context"] = dict(
    n_benign_seen=int(np.sum(cs & (yb == 0))), n_benign_unseen=int(np.sum(~cs & (yb == 0))),
    n_inj_seen_test=int(np.sum(cs & (yb == 1) & bip_keep)), n_inj_unseen_test=int(np.sum(~cs & (yb == 1) & bip_keep)),
    tpr_seen_test=float(np.mean(p[cs & (yb == 1) & bip_keep] > t)), tpr_unseen_test=float(np.mean(p[~cs & (yb == 1) & bip_keep] > t)),
    fpr_seen=float(np.mean(p[cs & (yb == 0)] > t)), fpr_unseen=float(np.mean(p[~cs & (yb == 0)] > t)),
    benign_median_seen=float(np.median(p[cs & (yb == 0)])), benign_median_unseen=float(np.median(p[~cs & (yb == 0)])))

# ---------------------------------------------------------------- 11. LLMail-Inject phase 2
if os.path.exists("llmail_eval.jsonl"):
    ll = load("llmail_eval.jsonl")
    yl = np.array([r["label"] for r in ll])
    grp = np.array([r["group"] for r in ll])
    res = {}
    for n in DETS:
        if "llmail" not in det[n]:
            continue
        p = P(n, "llmail")
        t1 = thr_at(P(n, "adclean"), 0.01)  # threshold calibrated on AgentDojo tool outputs (deployment-style)
        res[n] = dict(default_evaded=float(np.mean(p[grp == "evaded"] >= 0.5)),
                      default_attempt=float(np.mean(p[grp == "attempt"] >= 0.5)),
                      default_fpr=float(np.mean(p[grp == "benign"] >= 0.5)),
                      cal_evaded=float(np.mean(p[grp == "evaded"] > t1)), cal_attempt=float(np.mean(p[grp == "attempt"] > t1)),
                      cal_fpr=float(np.mean(p[grp == "benign"] > t1)),
                      auroc_evaded=float(roc_auc_score(np.r_[np.zeros(np.sum(grp == "benign")), np.ones(np.sum(grp == "evaded"))],
                                                       np.r_[p[grp == "benign"], p[grp == "evaded"]])))
    OUT["llmail"] = dict(n={g: int(np.sum(grp == g)) for g in ["benign", "evaded", "attempt"]}, detectors=res)

# ---------------------------------------------------------------- 12. judges without the user's request
# AgentDojo: injected AD-Matched outputs against AD-Clean, as for every other method; tau-bench: TB-Matched.
def judge_scores(prefix, stem, fn):
    f = f"results/hfjudge_{prefix}{stem}_{fn}.json"
    return np.array(json.load(open(f))["probs"]) if os.path.exists(f) else None


nr = {}
for name, stem in [("judge-qwen", "Qwen2.5-7B-Instruct"), ("judge-llama", "Llama-3.1-8B-Instruct")]:
    rb = os.path.exists(f"results/hfjudge_rb_{stem}_agentdojo_pairs.json")
    for tag, prefix in [("req", "rb_" if rb else ""), ("noreq", "noreq_")]:
        pos = judge_scores(prefix, stem, "agentdojo_pairs")
        neg = judge_scores("noreq_" if tag == "noreq" else "", stem, "agentdojo_clean")
        if pos is not None and neg is not None:
            y = np.r_[np.ones(int(ym.sum())), np.zeros(len(neg))]
            sc = np.r_[pos[ym == 1], neg]
            nr.setdefault(name, {}).setdefault("admatched", {}).update(
                {f"tpr1_{tag}": tpr_at(y, sc), f"auroc_{tag}": float(roc_auc_score(y, sc)),
                 f"fpr05_{tag}": float(np.mean(neg >= 0.5))})
        tb_s = judge_scores(prefix, stem, "taubench_eval")
        if tb_s is not None:
            nr.setdefault(name, {}).setdefault("tbeval", {}).update(
                {f"tpr1_{tag}": tpr_at(ytb, tb_s), f"auroc_{tag}": float(roc_auc_score(ytb, tb_s)),
                 f"fpr05_{tag}": float(np.mean(tb_s[ytb == 0] >= 0.5))})
OUT["judge_no_request"] = nr

# ---------------------------------------------------------------- 13. uniform sliding-window scoring
win = {}
for f in glob.glob("results/window/*.json"):
    d = json.load(open(f))
    win.setdefault(d["detector"], {})[d["set"]] = np.asarray(d["probs"], dtype=float)
wres = {}
for n, sets in win.items():
    r = {}
    if "admatched" in sets and "adclean" in sets:
        pos, neg = sets["admatched"][ym == 1], sets["adclean"]
        r["ad_full_tpr1"] = tpr_at(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])
        r["ad_tpr1"] = tpr_at(ym, sets["admatched"])
        r["adclean_fpr05"] = float(np.mean(neg >= 0.5))
    if "tbeval" in sets:
        r["tb_tpr1"] = tpr_at(ytb, sets["tbeval"])
    if "bipia" in sets:
        r["bipia_tpr1"] = tpr_at(ybi, sets["bipia"][bip_keep])
    if "admatched_rb" in det[n]:  # truncated scores of the same rebuilt outputs the windows saw
        pos, neg = P(n, "admatched_rb")[ym == 1], P(n, "adclean")
        r["rb_trunc_ad_full_tpr1"] = tpr_at(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])
    r["orig"] = dict(ad_tpr1=tpr_at(ym, P(n, "admatched")), ad_full_tpr1=full[n]["tpr1"], tb_tpr1=tb1.get(n),
                     bipia_tpr1=bi1.get(n), adclean_fpr05=float(np.mean(P(n, "adclean") >= 0.5)))
    wres[n] = r
OUT["window"] = wres

# ---------------------------------------------------------------- 14. format normalization with AD-Clean negatives
if os.path.exists("results/normalized_probs_full.json"):
    nf = json.load(open("results/normalized_probs_full.json"))
    yn = np.array(nf["labels"])
    nres = {}
    for k, v in nf["probs"].items():
        q = np.array(v)
        nres[k] = dict(fpr05=float(np.mean(q[yn == 0] >= 0.5)), tpr05=float(np.mean(q[yn == 1] >= 0.5)), tpr1=tpr_at(yn, q))
    OUT["normalized_full"] = nres

json.dump(OUT, open("analysis/numbers_v4.json", "w"), indent=1)
print(json.dumps({k: OUT[k] for k in ["structure", "tau_full", "truncation", "seen_unseen", "bipia_context"]}, indent=1))
