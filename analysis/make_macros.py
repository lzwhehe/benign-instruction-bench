"""Turn analysis/numbers.json into paper/numbers.tex (one \\newcommand per number used in the text)."""
import json

N = json.load(open("analysis/numbers.json"))
M = {}


def pct(v, d=1):
    return f"{100 * v:.{d}f}\\%"


def pp(v, d=1):  # percentage without the % sign (for tables)
    return f"{100 * v:.{d}f}"


# general corpus
g = N["general_corpus"]
M["NGeneral"] = f"{g['n']:,}".replace(",", "{,}")
for det, k in [("ProtectAI", "PA"), ("PIGuard", "PG")]:
    M[f"Gen{k}ReadmeFPR"] = pct(g[det]["fpr_readme"])
    M[f"Gen{k}EmailFPR"] = pct(g[det]["fpr_email"])
    M[f"Gen{k}WebFPR"] = pct(g[det]["fpr_web"])
    M[f"Fmt{k}ReadmeOrig"] = pct(N["readme_format_ablation"][det]["original"])
    M[f"Fmt{k}ReadmePlain"] = pct(N["readme_format_ablation"][det]["plain"])

M["DegDeepsetReadme"] = pct(N["degenerate_detectors_readme_fpr"]["deepset-deberta"])
M["DegFmopsReadme"] = pct(N["degenerate_detectors_readme_fpr"]["fmops-distilbert"])

# clean tool outputs
c = N["agentdojo_clean"]
M["NCleanOutputs"] = str(c["n_outputs"])
M["NCleanTasks"] = str(c["n_tasks"])
for det, k in [("ProtectAI", "PA"), ("PIGuard", "PG")]:
    M[f"Clean{k}FPR"] = pct(c[det]["output_fpr"])
    M[f"Clean{k}Block"] = pct(c[det]["task_block"])
    lo, hi = c[det]["task_block_ci"]
    M[f"Clean{k}BlockCI"] = f"{100 * lo:.0f}--{100 * hi:.0f}\\%"
    M[f"Clean{k}BankingFPR"] = pct(c[det]["banking"]["output_fpr"])

# matched benchmarks
names = {"AgentDojo": "AD", "AgentDojo-shift": "ADS", "BIPIA": "BI"}
meth = {"ProtectAI": "PA", "PIGuard": "PG", "Judge (Qwen2.5-7B)": "JQ", "Judge (Llama-3.1-8B)": "JL",
        "ProtectAI+norm": "PAN", "PIGuard+norm": "PGN", "PIGuard+perfield": "PGF"}
for d, dk in names.items():
    M[f"N{dk}Benign"] = f"{N[d]['_counts']['benign']:,}".replace(",", "{,}")
    M[f"N{dk}Inj"] = f"{N[d]['_counts']['injected']:,}".replace(",", "{,}")
    for m, mk in meth.items():
        if m not in N[d]:
            continue
        r = N[d][m]
        M[f"{dk}{mk}Auc"] = f"{r['auroc']:.2f}"
        M[f"{dk}{mk}Tpr"] = pct(r["tpr_at_1fpr"])
        M[f"{dk}{mk}TprT"] = pp(r["tpr_at_1fpr"])
        M[f"{dk}{mk}FprHalf"] = pct(r["fpr_at_05"])
        M[f"{dk}{mk}TprHalf"] = pct(r["tpr_at_05"])
        M[f"{dk}{mk}FprHalfT"] = pp(r["fpr_at_05"])
        M[f"{dk}{mk}TprHalfT"] = pp(r["tpr_at_05"])
        if "tpr_at_1fpr_ci" in r:
            lo, hi = r["tpr_at_1fpr_ci"]
            M[f"{dk}{mk}TprCI"] = f"{100 * lo:.1f}--{100 * hi:.1f}"

# contamination
o = N["piguard_train_overlap"]
M["OvTextTrain"] = f"{o['text_attack_train'][0]} of {o['text_attack_train'][1]}"
M["OvTextTest"] = f"{o['text_attack_test'][0]} of {o['text_attack_test'][1]}"
M["OvCodeTrain"] = f"{o['code_attack_train'][0]} of {o['code_attack_train'][1]}"
M["OvCodeTest"] = f"{o['code_attack_test'][0]} of {o['code_attack_test'][1]}"
for t in ["email", "code", "table"]:
    a, b = o[f"clean_{t}_contexts"]
    M[f"OvClean{t.capitalize()}"] = pct(a / b, 0 if a / b >= 0.05 else 1)
s = N["BIPIA_by_attack_split"]
M["BISplitPGTestTpr"] = pct(s["test"]["PIGuard"]["tpr_at_1fpr"])
M["BISplitPGTrainTpr"] = pct(s["train"]["PIGuard"]["tpr_at_1fpr"])

# goal groups
gr = N["BIPIA_by_group"]
for m, mk in [("PIGuard", "PG"), ("Judge (Qwen2.5-7B)", "JQ"), ("Judge (Llama-3.1-8B)", "JL")]:
    if m not in gr:
        continue
    for g, gk in [("task-irrelevant", "Irr"), ("task-relevant", "Rel"), ("targeted", "Tgt"),
                  ("code-passive", "Pas"), ("code-active", "Act")]:
        M[f"Goal{mk}{gk}"] = pct(gr[m][g]["tpr_at_05"], 0)

# multi-detector study (analysis/compute_many.py)
GROUPTAG = {"task-irrelevant": "Irr", "task-relevant": "Rel", "targeted": "Tgt", "code-passive": "Pas", "code-active": "Act"}
import os
if os.path.exists("analysis/numbers_many.json"):
    NM = json.load(open("analysis/numbers_many.json"))
    TT = NM["table"]
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
             "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen"]
    M["NDetectors"] = words[len(NM["detectors_in_rank"])]
    rk = NM["rank_transfer"]["bipia_tpr1_vs_admatched_tpr1"]
    M["RankTau"] = f"{rk['kendall_tau']:.2f}"
    M["RankTauP"] = f"{rk['p_tau']:.3f}"
    M["RankRho"] = f"{rk['spearman']:.2f}"
    rk2 = NM["rank_transfer"]["bipia_auroc_vs_admatched_auroc"]
    M["RankTauAuc"] = f"{rk2['kendall_tau']:.2f}"
    for key, tag in [("admatched_tpr1_vs_tbeval_tpr1", "ADTB"), ("admatched_auroc_vs_tbeval_auroc", "ADTBAuc"),
                     ("bipia_tpr1_vs_tbeval_tpr1", "BITB"), ("adclean_fpr_vs_tbeval_fpr05", "FPRADTB")]:
        if key in NM["rank_transfer"]:
            r = NM["rank_transfer"][key]
            M[f"Tau{tag}"] = f"{r['kendall_tau']:.2f}"
            M[f"Tau{tag}P"] = f"{r['p_tau']:.4f}" if r["p_tau"] < 0.001 else f"{r['p_tau']:.3f}"
    ps = NM.get("pg_sources", {})
    if ps:
        M["PGBipiaRows"] = f"{ps['bipia_rows']:,}".replace(",", "{,}")
        M["PGBipiaChars"] = f"{ps['bipia_median_chars']:.0f}"
        M["PGIARows"] = str(ps["injecagent_rows"])
        M["PGIAChars"] = f"{ps['injecagent_median_chars']:.0f}"
    tm = NM.get("tb_meta", {})
    if tm:
        M["NTBBenign"] = f"{tm['n_benign']:,}".replace(",", "{,}")
        M["NTBInj"] = str(tm["n_injected"])
        M["NTBTasks"] = str(tm["n_tasks"])
        M["NIASeen"] = f"{tm['n_seen_attacks']} of {tm['n_attacks']}"
        M["NIAUnseen"] = str(tm["n_attacks"] - tm["n_seen_attacks"])
        M["NTBSeenPos"] = str(tm["n_seen_pos"])
        M["NTBUnseenPos"] = str(tm["n_unseen_pos"])
    for key, tag in [("protectai-v1", "PAo"), ("protectai-v2", "PA"), ("piguard", "PG"), ("jasperls", "JL"),
                     ("testsavant-large", "TS"), ("sheltron", "SH"), ("prismor-1.5b", "PR"), ("wolf-defender", "WF"),
                     ("horizon-base", "HZ"), ("judge-qwen", "JQ"), ("judge-llama", "JLl"),
                     ("sheltron-generic", "SHG"), ("sheltron-taskaware", "SHT"), ("promptguard2-86m", "PGt"),
                     ("promptguard2-22m", "PGts"), ("promptguard1-86m", "PGo"), ("qualifire-sentinel", "QS"),
                     ("rogue-sentinel-v2", "RS"), ("protectai-small-v2", "PAs")]:
        r = TT.get(key, {})
        for k, nm in [("tbeval_fpr05", "TBFPR"), ("tbeval_block", "TBBlock"), ("tbeval_tpr1", "TBTpr"),
                      ("tbeval_tpr1_seen", "TBSeen"), ("tbeval_tpr1_unseen", "TBUnseen")]:
            if r.get(k) is not None:
                M[f"{nm}{tag}"] = pct(r[k])
        if r.get("tbeval_auroc") is not None:
            M[f"TBAuc{tag}"] = f"{r['tbeval_auroc']:.2f}"
    M["BestBipiaRankOnAD"] = words[NM["best_on_bipia_rank_on_agentdojo"]]
    for key, tag in [("horizon-base", "HZ"), ("wolf-defender", "WF"), ("prismor-1.5b", "PR"), ("sheltron", "SH"), ("piguard", "PG"),
                     ("judge-qwen", "JQ"), ("promptguard1-86m", "PGo"), ("promptguard2-86m", "PGt")]:
        if key in NM.get("by_group", {}):
            for g, gt in GROUPTAG.items():
                M[f"G{tag}{gt}"] = pct(NM["by_group"][key][g]["tpr_at_1fpr"], 0)
    for key, tag in [("horizon-base", "HZ"), ("wolf-defender", "WF"), ("prismor-1.5b", "PR"), ("sheltron", "SH"),
                     ("sheltron-taskaware", "SHT"), ("sheltron-generic", "SHG"), ("testsavant-large", "TS"), ("protectai-v1", "PAo"), ("jasperls", "JL"),
                     ("promptguard2-86m", "PGt"), ("promptguard2-22m", "PGts"), ("promptguard1-86m", "PGo"),
                     ("qualifire-sentinel", "QS"), ("rogue-sentinel-v2", "RS"), ("protectai-small-v2", "PAs")]:
        if key not in TT:
            continue
        r = TT[key]
        for k, nm in [("general_fpr", "Gen"), ("adclean_fpr", "Clean"), ("adclean_block", "Block"), ("admatched_tpr1", "AD"),
                      ("adshift_tpr1", "ADS"), ("bipia_tpr1", "BI"), ("admatched_auroc", "ADAuc"), ("bipia_auroc", "BIAuc")]:
            if k in r and r[k] is not None:
                M[f"M{tag}{nm}"] = f"{r[k]:.2f}" if k.endswith("auroc") else pct(r[k])

# robustness analyses (analysis/compute_robust.py): rank-transfer uncertainty and held-out thresholds
if os.path.exists("analysis/numbers_robust.json"):
    NR = json.load(open("analysis/numbers_robust.json"))
    M["NBoot"] = f"{NR['bootstrap_resamples']:,}".replace(",", "{,}")
    for key, tag in [("bipia_tpr1_vs_admatched_tpr1", "Rank"), ("admatched_tpr1_vs_tbeval_tpr1", "ADTB"),
                     ("bipia_tpr1_vs_tbeval_tpr1", "BITB"), ("adclean_fpr_vs_tbeval_fpr05", "FPRADTB")]:
        r = NR["e1"][key]
        M[f"Tau{tag}CI"] = f"{r['ci'][0]:.2f} to {r['ci'][1]:.2f}".replace("-", "$-$")
        M[f"Tau{tag}Perm"] = "<0.001" if r["p_perm"] < 0.001 else f"{r['p_perm']:.3f}"
    M["AdMatchedBenignInClean"] = pct(NR["check"]["admatched_benign_in_adclean"], 0)
    for key, tag in [("horizon-base", "HZ"), ("wolf-defender", "WF"), ("prismor-1.5b", "PR"), ("promptguard2-86m", "PGt"),
                     ("promptguard2-22m", "PGts"), ("qualifire-sentinel", "QS"), ("piguard", "PG"), ("protectai-v2", "PA"),
                     ("testsavant-large", "TS"), ("rogue-sentinel-v2", "RS"), ("promptguard1-86m", "PGo"), ("jasperls", "JL")]:
        r = NR["e2"][key]
        M[f"Cal{tag}Tpr"] = pct(r["clean_tpr1"])
        M[f"Lo{tag}Tpr"] = pct(r["loso_tpr1"])
        M[f"Lo{tag}Fpr"] = pct(r["loso_fpr_clean1"])
        M[f"Lo{tag}TprTwo"] = pct(r["loso_tpr2"])
        M[f"Lo{tag}FprTwo"] = pct(r["loso_fpr_clean2"])
        M[f"Lo{tag}MinTprTwo"] = pct(min(s["tpr"] for s in r["loso_suites2"].values()))
        M[f"Lo{tag}MaxFprTwo"] = pct(max(s["fpr_clean"] for s in r["loso_suites2"].values()))
    M["LoPGtWsFprTwo"] = pct(NR["e2"]["promptguard2-86m"]["loso_suites2"]["workspace"]["fpr_clean"])

# community-benchmark setting and rebuild check (analysis/compute_e6.py)
if os.path.exists("analysis/numbers_e6.json"):
    NE = json.load(open("analysis/numbers_e6.json"))
    M["ReproMaxDiff"] = f"{max(NE[k][d] for k in ['check_piguard', 'check_promptguard2-86m'] for d in ['adclean_max_abs_diff', 'admatched_benign_max_abs_diff']):.3f}"
    M["PGtRebuiltTpr"] = pct(NE["check_promptguard2-86m"]["admatched_tpr1_rebuilt"])
    M["PGRebuiltTpr"] = pct(NE["check_piguard"]["admatched_tpr1_rebuilt"])
    for tag, k in [("Seg", "seg"), ("Ovl", "ovl"), ("Task", "task")]:
        r = NE[k]
        M[f"Comm{tag}Default"] = pct(r["default_tpr"])
        M[f"Comm{tag}TprOne"] = pct(r["tpr1"])
        M[f"Comm{tag}LoTpr"] = pct(r["loso2_tpr"])
        M[f"Comm{tag}LoMin"] = pct(r["loso2_min_tpr"])
        M[f"Comm{tag}LoFpr"] = pct(r["loso2_fpr"])
        M[f"Comm{tag}LoMaxFpr"] = pct(r["loso2_max_fpr"])
        M[f"Comm{tag}LoII"] = pct(r["loso2_tpr_ii"])
        M[f"Comm{tag}LoIIMin"] = pct(r["loso2_min_tpr_ii"])
        for a, at in [("important_instructions", "II"), ("direct", "Dir"), ("ignore_previous", "Ign")]:
            M[f"Comm{tag}Default{at}"] = pct(r["by_template_default_tpr"][a])
            M[f"Comm{tag}One{at}"] = pct(r["by_template_tpr1"][a])

# replay validity on a real agent run (scripts/e5_compare.py, scripts/e5_score.py)
if os.path.exists("results/e5_summary.json") and os.path.exists("results/e5_detectors.json"):
    S5 = json.load(open("results/e5_summary.json"))
    M["EFiveTasks"] = str(S5["tasks"])
    M["EFiveCalls"] = f"{S5['calls']:,}".replace(",", "{,}")
    M["EFiveIdentical"] = pct(S5["identical"] / S5["calls"])
    M["EFiveIdenticalN"] = f"{S5['identical']:,}".replace(",", "{,}")
    M["EFiveInGT"] = pct(S5["in_ground_truth"] / S5["calls"])
    M["EFiveErrors"] = pct(S5["errors"] / S5["calls"])
    M["EFiveUtility"] = pct(S5["utility"] / S5["tasks"])
    M["EFiveDiffN"] = str(S5["calls"] - S5["identical"])
    D5 = json.load(open("results/e5_detectors.json"))
    M["EFiveObsOutputs"] = str(D5["protectai-v2"]["observed"]["n_outputs"])
    M["EFiveObsTasks"] = str(D5["protectai-v2"]["observed"]["n_tasks"])
    for key, tag in [("protectai-v2", "PA"), ("piguard", "PG"), ("promptguard2-86m", "PGt"), ("horizon-base", "HZ"),
                     ("qualifire-sentinel", "QS"), ("testsavant-large", "TS")]:
        if key in D5:
            M[f"EFive{tag}ObsFpr"] = pct(D5[key]["observed"]["output_fpr"])
            M[f"EFive{tag}ObsBlock"] = pct(D5[key]["observed"]["task_block"])
            M[f"EFive{tag}RepFpr"] = pct(D5[key]["replay"]["output_fpr"])
            M[f"EFive{tag}RepBlock"] = pct(D5[key]["replay"]["task_block"])

# inference cost (scripts/latency.py)
if os.path.exists("results/latency.json"):
    L = json.load(open("results/latency.json"))
    enc = [k for k in L if not k.startswith("judge")]
    M["CostGPU"] = L[enc[0]]["gpu_name"].replace("NVIDIA ", "")
    M["NCostOutputs"] = str(L[enc[0]]["gpu"]["n"])
    encs = [k for k in enc if k != "prismor-1.5b"]
    M["CostEncMaxCpu"] = f"{max(L[k]['cpu']['median_ms'] for k in encs):.0f}"
    M["CostEncMaxMem"] = f"{max(L[k]['gpu']['peak_mem_mb'] for k in encs) / 1024:.1f}"
    M["CostEncMinMem"] = f"{min(L[k]['gpu']['peak_mem_mb'] for k in encs):.0f}"
    for key, tag in [("judge-qwen", "JQ"), ("judge-llama", "JL")]:
        if key in L:
            M[f"Cost{tag}Mem"] = f"{L[key]['gpu']['peak_mem_mb'] / 1024:.1f}"
    M["CostEncMaxMs"] = f"{max(L[k]['gpu']['median_ms'] for k in enc if k != 'prismor-1.5b'):.0f}"
    M["CostEncMinMs"] = f"{min(L[k]['gpu']['median_ms'] for k in enc):.0f}"
    for key, tag in [("horizon-base", "HZ"), ("wolf-defender", "WF"), ("promptguard2-86m", "PGt"),
                     ("promptguard2-22m", "PGts"), ("prismor-1.5b", "PR"), ("judge-qwen", "JQ"), ("judge-llama", "JL"),
                     ("piguard", "PG"), ("protectai-v2", "PA")]:
        if key in L:
            M[f"Cost{tag}Gpu"] = f"{L[key]['gpu']['median_ms']:.0f}"
            if L[key].get("cpu"):
                M[f"Cost{tag}Cpu"] = f"{L[key]['cpu']['median_ms']:,.0f}".replace(",", "{,}")

if os.path.exists("results/horizon_revision_check.json"):
    HR = json.load(open("results/horizon_revision_check.json"))["revisions"]
    M["HZvTwoFlagged"] = str(HR["e4fbfc6f603e24c675cdf310c3b8d7abaa825666"]["flagged_at_05"])

# ---------------------------------------------------------------- revision v4 (analysis/compute_v4.py)
TAGS = {"horizon-base": "HZ", "wolf-defender": "WF", "prismor-1.5b": "PR", "promptguard2-86m": "PGt",
        "promptguard2-22m": "PGts", "qualifire-sentinel": "QS", "sheltron": "SH", "piguard": "PG", "protectai-v2": "PA",
        "testsavant-large": "TS", "rogue-sentinel-v2": "RS", "jasperls": "JL", "promptguard1-86m": "PGo",
        "protectai-v1": "PAo", "protectai-small-v2": "PAs", "judge-qwen": "JQ", "judge-llama": "JLl"}
ORD = ["zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
       "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth"]
pp1 = lambda v: f"{100 * v:.1f}"  # noqa: E731
if os.path.exists("analysis/numbers_many.json"):
    NM = json.load(open("analysis/numbers_many.json"))
    TT = NM["table"]
    M["BestBipiaRankOnADOrd"] = ORD[NM["best_on_bipia_rank_on_agentdojo"]]
    for key, tag in TAGS.items():
        r = TT.get(key, {})
        for k, nm in [("admatched_tpr1", "AD"), ("adshift_tpr1", "ADS"), ("admatched_tpr1_matchedneg", "ADm"),
                      ("adshift_tpr1_matchedneg", "ADSm"), ("admatched_auroc", "ADAuc"), ("adclean_fpr", "Clean"),
                      ("adclean_block", "Block"), ("bipia_tpr1", "BI"), ("tbeval_tpr1", "TB")]:
            if r.get(k) is not None:
                M[f"V{tag}{nm}"] = f"{r[k]:.2f}" if k.endswith("auroc") else pct(r[k])
    for key, tag in [("bipia_tpr1_vs_tbeval_tpr1", "BITB"), ("bipia_auroc_vs_admatched_auroc", "AucBIAD"),
                     ("admatched_auroc_vs_tbeval_auroc", "AucADTB")]:
        r = NM["rank_transfer"].get(key)
        if r:
            M[f"Tau{tag}V"] = f"{r['kendall_tau']:.2f}"
if os.path.exists("analysis/numbers_v4.json"):
    V = json.load(open("analysis/numbers_v4.json"))
    st = V["structure"]
    M["NADBenignDistinct"] = str(st["n_benign_distinct"])
    M["NInjPositions"] = str(st["n_injection_positions"])
    M["NTwins"] = str(st["n_twins_distinct"])
    M["NTwinRows"] = str(st["n_twin_rows"])
    M["NADCleanDistinct"] = str(st["n_adclean_distinct"])
    h = V["heldout"]["horizon-base"]["1"]["folds"]
    w = max(h, key=lambda k: h[k]["fpr"])
    M["LoHZWorstCount"] = f"{h[w]['n_flag']} of {h[w]['n_neg']}"
    M["LoHZWorstSuite"] = w
    for key, tag in TAGS.items():
        c = V["cluster_ci"].get(key)
        if c:
            M[f"CI{tag}AD"] = f"{pp1(c['ad_ci'][0])}--{pp1(c['ad_ci'][1])}"
            if c.get("tb_ci"):
                M[f"CI{tag}TB"] = f"{pp1(c['tb_ci'][0])}--{pp1(c['tb_ci'][1])}"
    PD = V["paired_diff"]
    for k, nm in [("horizon-base-wolf-defender-ad", "HZWFad"), ("horizon-base-prismor-1.5b-ad", "HZPRad"),
                  ("horizon-base-sheltron-tb", "HZSHtb"), ("sheltron-wolf-defender-tb", "SHWFtb"),
                  ("horizon-base-wolf-defender-tb", "HZWFtb"), ("judge-qwen-noreq-judge-qwen-ad", "JQad"),
                  ("judge-llama-noreq-judge-llama-ad", "JLlad"), ("judge-qwen-noreq-judge-qwen-tb", "JQtb"),
                  ("judge-llama-noreq-judge-llama-tb", "JLltb")]:
        if k in PD:
            M[f"PD{nm}"] = f"{100 * PD[k]['diff']:.1f}"
            M[f"PD{nm}CI"] = f"{pp1(PD[k]['ci'][0])} to {pp1(PD[k]['ci'][1])}".replace("-", "$-$")
    tr = V["truncation"]
    M["TruncAD"] = pct(tr["admatched"]["start_beyond_510"])
    M["TruncADS"] = pct(tr["adshift"]["start_beyond_510"])
    M["TruncTB"] = pct(tr["tbeval"]["start_beyond_510"])
    M["LongAD"] = pct(tr["admatched"]["longer_than_510"])
    M["MedStartAD"] = f"{tr['admatched']['median_start']:.0f}"
    su = V["seen_unseen"]
    M["SUGap"] = f"{100 * su['piguard_gap']:.1f}".replace("-", "$-$")
    M["SUGapCI"] = f"{pp1(su['piguard_gap_ci'][0])} to {pp1(su['piguard_gap_ci'][1])}".replace("-", "$-$")
    M["SUOthers"] = f"{100 * su['others_median_gap']:.1f}".replace("-", "$-$")
    M["SUDiD"] = f"{100 * su['did']:.1f}".replace("-", "$-$")
    M["SUDiDCI"] = f"{pp1(su['did_ci'][0])} to {pp1(su['did_ci'][1])}".replace("-", "$-$")
    M["SUNSeenOut"] = str(su["n_seen_outputs"])
    M["SUNUnseenOut"] = str(su["n_unseen_outputs"])
    tm = V["tb_matched_negatives"]
    M["TBmN"] = str(tm["n_neg"])
    for key, tag in TAGS.items():
        r = tm["detectors"].get(key)
        if r:
            M[f"TBm{tag}"] = pct(r["tpr1"])
    for key, tag in TAGS.items():
        h = V["heldout"].get(key)
        if h:
            M[f"Lo{tag}WorstFpr"] = pct(h["1"]["worst_fpr"])
            M[f"Lo{tag}WorstTpr"] = pct(h["1"]["worst_tpr"])
            M[f"Lo{tag}WorstFprTwo"] = pct(h["2"]["worst_fpr"])
            M[f"Lo{tag}TprCI"] = f"{pp1(h['1']['tpr_ci'][0])}--{pp1(h['1']['tpr_ci'][1])}"
            M[f"Lo{tag}FprCI"] = f"{pp1(h['1']['fpr_ci'][0])}--{pp1(h['1']['fpr_ci'][1])}"
    ts = V["tau_sensitivity"]
    r = ts["bipia_tpr1_vs_admatched_tpr1"]
    M["TauBIADnoPG"] = f"{r['without_piguard']:.2f}"
    M["TauBIADJack"] = f"{r['jackknife'][0]:.2f} to {r['jackknife'][1]:.2f}".replace("-", "$-$")
    M["TauBIADFamP"] = f"{r['one_per_family_p']:.3f}"
    M["TauBIADFam"] = f"{r['one_per_family']:.2f}"
    r = ts["admatched_tpr1_vs_tbeval_tpr1"]
    M["TauADTBJack"] = f"{r['jackknife'][0]:.2f} to {r['jackknife'][1]:.2f}".replace("-", "$-$")
    M["TauADTBFam"] = f"{r['one_per_family']:.2f}"
    M["TauADTBFamP"] = f"{r['one_per_family_p']:.3f}"
    M["NFamily"] = str(ts["bipia_tpr1_vs_admatched_tpr1"]["n_family"])
    for k, nm in [("general_fpr_vs_adclean_fpr", "GenAD"), ("general_fpr_vs_tbeval_fpr05", "GenTB")]:
        M[f"Tau{nm}"] = f"{ts[k]['tau']:.2f}"
        M[f"Tau{nm}P"] = f"{ts[k]['p']:.3f}"
    bt = V["by_template"]
    for key, tag in [("sheltron", "SH"), ("prismor-1.5b", "PR"), ("horizon-base", "HZ"), ("wolf-defender", "WF"),
                     ("promptguard2-86m", "PGt")]:
        for a, at in [("injecagent", "IA"), ("tool_knowledge", "TK"), ("system_message", "SM"),
                      ("important_instructions", "II"), ("ignore_previous", "IP"), ("direct", "Dir")]:
            if a in bt.get(key, {}):
                M[f"Tpl{tag}{at}"] = pct(bt[key][a])
    bc = V["bipia_context"]
    M["BICtxSeenN"] = str(bc["n_benign_seen"])
    M["BICtxSeenTpr"] = pct(bc["tpr_seen_test"])
    M["BICtxUnseenTpr"] = pct(bc["tpr_unseen_test"])
    if "llmail" in V:
        L4 = V["llmail"]
        M["NLLBenign"] = str(L4["n"]["benign"])
        M["NLLEvaded"] = str(L4["n"]["evaded"])
        M["NLLAttempt"] = f"{L4['n']['attempt']:,}".replace(",", "{,}")
        for key, tag in TAGS.items():
            r = L4["detectors"].get(key)
            if r:
                for k, nm in [("cal_evaded", "CalEv"), ("cal_attempt", "CalAt"), ("cal_fpr", "CalFpr"),
                              ("default_evaded", "DefEv"), ("default_attempt", "DefAt"), ("default_fpr", "DefFpr")]:
                    M[f"LL{tag}{nm}"] = pct(r[k])
    for key, tag in [("judge-qwen", "JQ"), ("judge-llama", "JLl")]:
        for s_, sn in [("admatched", "AD"), ("tbeval", "TB")]:
            r = V["judge_no_request"].get(key, {}).get(s_)
            if r:
                M[f"NoReq{tag}{sn}"] = pct(r["tpr1_noreq"])
                M[f"Req{tag}{sn}"] = pct(r["tpr1_req"])
                M[f"NoReq{tag}{sn}Auc"] = f"{r['auroc_noreq']:.2f}"
                M[f"Req{tag}{sn}Auc"] = f"{r['auroc_req']:.2f}"
                M[f"NoReq{tag}{sn}Fpr"] = pct(r["fpr05_noreq"])
                M[f"Req{tag}{sn}Fpr"] = pct(r["fpr05_req"])
    for key, tag in TAGS.items():
        w = V["window"].get(key)
        if w:
            for k, nm in [("ad_full_tpr1", "AD"), ("tb_tpr1", "TB"), ("bipia_tpr1", "BI"), ("adclean_fpr05", "Clean"),
                          ("rb_trunc_ad_full_tpr1", "RbTruncAD")]:
                if k in w:
                    M[f"Win{tag}{nm}"] = pct(w[k])
    # windowed versus truncated scoring of the same (rebuilt) AgentDojo outputs
    others = [abs(w["ad_full_tpr1"] - w["rb_trunc_ad_full_tpr1"]) for k, w in V["window"].items()
              if k not in ("piguard", "protectai-v2") and "rb_trunc_ad_full_tpr1" in w]
    if others:
        M["WinMaxOther"] = f"{100 * max(others):.1f}"
    if "normalized_full" in V:
        for k, nm in [("piguard/original", "PG"), ("piguard/values", "PGN"), ("protectai-deberta-v2/original", "PA"),
                      ("protectai-deberta-v2/values", "PAN")]:
            r = V["normalized_full"][k]
            M[f"Nf{nm}Fpr"] = pct(r["fpr05"])
            M[f"Nf{nm}TprHalf"] = pct(r["tpr05"])
            M[f"Nf{nm}Tpr"] = pct(r["tpr1"])

for fn in ["results/llmail_format_a.json", "results/llmail_format_b.json"]:  # scripts/llmail_format.py
    if os.path.exists(fn):
        for key, r in json.load(open(fn)).items():
            tag = TAGS[key]
            for form, nm in [("wrapped", "Wrap"), ("plain", "Plain"), ("body", "Body")]:
                M[f"LLFmt{tag}{nm}"] = pct(r[form]["benign_fpr05"])
                M[f"LLFmt{tag}{nm}Ev"] = pct(r[form]["evaded_tpr05"])

with open("paper/numbers.tex", "w") as f:
    f.write("% Generated by analysis/make_macros.py from analysis/numbers.json. Do not edit.\n")
    # the benchmark's name; Kendall's coefficient is written $\tau_K$ in the text
    f.write("\\newcommand{\\taubench}{$\\tau$-bench\\xspace}\n")
    # layout for the two-column 5p format: allow some extra stretch, and let URLs break at hyphens and slashes
    f.write("\\setlength{\\emergencystretch}{1.5em}\n")
    f.write(r"\def\UrlBreaks{\do\.\do\/\do\-\do\_\do\=\do\?\do\&}" + "\n")
    for k, v in sorted(M.items()):
        # interval macros are used inside brackets/parentheses, where \xspace would insert a stray space
        tail = "" if k.endswith("CI") else "\\xspace"
        f.write(f"\\newcommand{{\\{k}}}{{{v}{tail}}}\n")
print(len(M), "macros written")
