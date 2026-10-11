#!/usr/bin/env bash
# Rebuild every evaluation set from public sources, score them, and regenerate all paper numbers and figures.
# Needs: one CUDA GPU (>=20 GB for the 4-bit judges), ~20 GB disk, Python 3.12 env from requirements.txt,
# a separate Python env with agentdojo==0.1.34 (AGENTDOJO_PY), and local copies of the two judge models.
set -euo pipefail
# collect_agentdojo_pairs.py assigns injection tasks with hash(); fix the seed so injected texts are reproducible
export PYTHONHASHSEED=0
AGENTDOJO_PY=${AGENTDOJO_PY:?path to a python that has agentdojo installed}
QWEN=${QWEN:?path to Qwen2.5-7B-Instruct}
LLAMA=${LLAMA:?path to Llama-3.1-8B-Instruct}
export GPU_MEM=${GPU_MEM:-18GiB}

# external sources (not redistributed here)
[ -d ~/agentsec/BIPIA ]   || git clone --depth 1 https://github.com/microsoft/BIPIA.git ~/agentsec/BIPIA
[ -d ~/agentsec/PIGuard ] || git clone --depth 1 https://github.com/leolee99/PIGuard.git ~/agentsec/PIGuard

# 1. general benign corpus (README / email / web) and detector scores, markup ablation
python scripts/build_corpus.py --per_source 2500 --out corpus.jsonl
python scripts/run_detectors.py --corpus corpus.jsonl --out detector_scores.json
python scripts/run_format_ablation.py

# 2. AgentDojo: clean replay and matched sets (differential labels)
$AGENTDOJO_PY scripts/collect_agentdojo_outputs.py
python scripts/score_texts.py agentdojo_benign_outputs.jsonl
$AGENTDOJO_PY scripts/collect_agentdojo_pairs.py important_instructions,ignore_previous,direct agentdojo_pairs.jsonl
$AGENTDOJO_PY scripts/collect_agentdojo_pairs.py tool_knowledge,injecagent,system_message,important_instructions_no_names,important_instructions_wrong_model_name agentdojo_pairs_shift.jsonl
$AGENTDOJO_PY scripts/attach_requests.py agentdojo_pairs.jsonl agentdojo_pairs_shift.jsonl

# 3. BIPIA matched set; tau-bench clean replay + InjecAgent attacks in product names
python scripts/build_bipia.py
[ -d ~/agentsec/tau-bench ] || git clone --depth 1 https://github.com/sierra-research/tau-bench.git ~/agentsec/tau-bench
[ -d ~/agentsec/InjecAgent ] || git clone --depth 1 https://github.com/uiuc-kang-lab/InjecAgent.git ~/agentsec/InjecAgent
python scripts/collect_taubench.py && python scripts/build_taubench_eval.py

# 4. detectors (original / YAML-keys-removed / per-field) on the three matched sets
for d in agentdojo_pairs:main agentdojo_pairs_shift:shift bipia_pairs:bipia; do
  python scripts/score_normalized.py ${d%%:*}.jsonl && cp normalized_probs.json normalized_probs_${d##*:}.json
done

# 5. task-aware judges (4-bit)
for m in "$QWEN" "$LLAMA"; do
  for d in agentdojo_pairs agentdojo_pairs_shift bipia_pairs taubench_eval; do
    python scripts/hf_judge.py "$m" $d.jsonl "hfjudge_$(basename "$m")_${d}.json"
  done
done

# 6. nine open detectors (+ Sheltron variants), each run as its model card documents.
#    Wolf Defender needs transformers>=5.6 (TRANSFORMERS5_PY); the others run in the main env.
$AGENTDOJO_PY scripts/attach_requests.py agentdojo_benign_outputs.jsonl
python scripts/score_many.py protectai-v1 protectai-v2 piguard injecguard jasperls testsavant-large horizon-base sheltron sheltron-taskaware sheltron-generic prismor-1.5b
${TRANSFORMERS5_PY:-python} scripts/score_many.py wolf-defender
python scripts/audit_horizon.py   # needs ~/agentsec/BIPIA
python scripts/audit_injecagent.py

# 7. robustness and cost (JISA revision): community-benchmark setting for Prompt Guard 2, inference cost,
#    and a real-agent run (needs vLLM serving Qwen2.5-7B-Instruct on localhost:8000) for the replay check
python scripts/score_e6.py
python scripts/latency.py results/latency.json protectai-v1 protectai-v2 protectai-small-v2 jasperls testsavant-large \
  piguard promptguard1-86m promptguard2-22m promptguard2-86m qualifire-sentinel rogue-sentinel-v2 sheltron horizon-base \
  prismor-1.5b judge-qwen judge-llama
${TRANSFORMERS5_PY:-python} scripts/latency.py results/latency.json wolf-defender
if [ -n "${RUN_AGENT:-}" ]; then
  $AGENTDOJO_PY scripts/e5_run.py --model LOCAL --benchmark-version v1.2 --logdir e5_runs \
    -s workspace -s travel -s banking -s slack
  $AGENTDOJO_PY scripts/e5_compare.py e5_runs e5_observed_outputs.jsonl
  python scripts/e5_score.py e5_observed_outputs.jsonl results/e5_detectors.json
fi

# 8. every number, table and figure in the paper
python analysis/compute_all.py
python analysis/compute_many.py
python analysis/compute_robust.py
python analysis/make_macros.py
echo "done: analysis/numbers.json, analysis/figures/, paper/numbers.tex"
