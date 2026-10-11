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
    M["RankTauP"] = f"{rk['p_tau']:.2f}"
    M["RankRho"] = f"{rk['spearman']:.2f}"
    rk2 = NM["rank_transfer"]["bipia_auroc_vs_admatched_auroc"]
    M["RankTauAuc"] = f"{rk2['kendall_tau']:.2f}"
    for key, tag in [("admatched_tpr1_vs_tbeval_tpr1", "ADTB"), ("admatched_auroc_vs_tbeval_auroc", "ADTBAuc"),
                     ("bipia_tpr1_vs_tbeval_tpr1", "BITB"), ("adclean_fpr_vs_tbeval_fpr05", "FPRADTB")]:
        if key in NM["rank_transfer"]:
            r = NM["rank_transfer"][key]
            M[f"Tau{tag}"] = f"{r['kendall_tau']:.2f}"
            M[f"Tau{tag}P"] = f"{r['p_tau']:.3f}" if r["p_tau"] < 0.01 else f"{r['p_tau']:.2f}"
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
        M[f"Tau{tag}Perm"] = "<0.001" if r["p_perm"] < 0.001 else f"{r['p_perm']:.2f}"
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

with open("paper/numbers.tex", "w") as f:
    f.write("% Generated by analysis/make_macros.py from analysis/numbers.json. Do not edit.\n")
    for k, v in sorted(M.items()):
        # interval macros are used inside brackets/parentheses, where \xspace would insert a stray space
        tail = "" if k.endswith("CI") else "\\xspace"
        f.write(f"\\newcommand{{\\{k}}}{{{v}{tail}}}\n")
print(len(M), "macros written")
