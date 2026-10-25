# Passing the Test You Trained On

Re-evaluating prompt-injection detectors where LLM agents actually use them: on the tool outputs an agent reads.

Teams pick injection detectors by their benchmark scores. We check whether those scores predict behavior inside an agent, and find that they mostly measure how close the benchmark is to the detector's training data.

## Findings

Fifteen detectors (nine open, six license-gated including Meta's Prompt Guard 2) and two task-aware LLM judges, on two agent benchmarks (AgentDojo, tau-bench) and on BIPIA. Benign tool outputs come from replaying each benchmark's ground-truth tool calls without an LLM. Detection is TPR at the threshold that gives 1% FPR on the benign outputs of the same benchmark.

| Detector | Released | FPR on benign tool outputs (AgentDojo / tau-bench) | Detection AgentDojo | Detection tau-bench | Detection BIPIA |
|---|---|---|---|---|---|
| Horizon-Labs guard-base | 2026 | 0.0% / 0.7% | 83.1% | 100.0% | 37.7% |
| Wolf Defender | 2026 | 0.9% / 0.0% | 70.1% | 90.8% | 3.5% |
| Prompt Guard 2 (22M) | 2025 | 0.0% / 0.0% | 60.9% | 60.8% | 31.7% |
| Prompt Guard 2 (86M) | 2025 | 0.6% / 0.0% | 57.4% | 57.9% | 10.4% |
| Judge (Llama-3.1-8B) | – | – | 51.9% | 78.3% | 8.3% |
| Prismor-1.5B | 2026 | 6.5% / 46.2% | 29.2% | 15.2% | 4.0% |
| Sheltron-68M | 2026 | 10.6% / 4.4% | 20.1% | 95.2% | 57.2% |
| PIGuard | 2025 | 29.5% / 11.0% | 1.6% | 52.7% | **95.1%** |
| ProtectAI v2 | 2024 | 30.1% / 66.9% | 0.5% | 10.1% | 0.7% |

(All 15 detectors and both judges, with AUROC and 95% task-clustered bootstrap intervals: Table 2 of the paper.)

1. BIPIA rankings are at most weakly related to AgentDojo rankings (Kendall τ over 15 detectors: 0.12, p = 0.559, not significant; bootstrap interval 0.05 to 0.23; leaving out single detectors gives 0.01 to 0.25, and 0.25 without PIGuard). The largest reversal comes from a detector trained on BIPIA's distribution: the BIPIA leader (PIGuard) ranks thirteenth of 15 on AgentDojo. The two agent benchmarks agree more (τ = 0.39, p = 0.046); BIPIA vs tau-bench gives τ = 0.31.
2. False-positive rates on benign tool outputs range from 0.0% to 93.1% and are consistent across the two agent benchmarks (τ = 0.67, p = 0.0007).
3. PIGuard's BIPIA score does not depend on memorized attack strings and is consistent with training on BIPIA's distribution: its training split contains 1,116 full BIPIA inputs. It also saw 53 of 62 InjecAgent attacks, but only as short prompts; inside tau-bench tool outputs it detects seen and unseen InjecAgent attacks alike (52.1% vs 55.8%). Horizon-Labs, trained on agent-style inputs, leads both agent benchmarks; neither it nor the second-best detector can be shown to be independent of AgentDojo.
4. Task-aware 7–8B judges reach 51.9% (Llama) and 59.0% (Qwen) on AgentDojo and 78.3% and 54.3% on tau-bench at 1% FPR, below the best dedicated detectors. Apart from PIGuard, no approach catches more than 39% of BIPIA's task-irrelevant injections.
5. Removing serialization markup or declaring the input type lowers default-threshold false positives by up to about twentyfold, but does not make PIGuard or ProtectAI v2 useful at 1% FPR on AgentDojo.

All numbers are regenerated from the score files by the scripts in `analysis/` (`compute_all.py`, `compute_many.py`, `compute_robust.py`, `compute_v4.py`); `analysis/make_macros.py` writes them to the LaTeX macros the paper reads.

## Layout

```
scripts/     build evaluation sets, score detectors and judges
analysis/    compute_all.py, compute_many.py, compute_robust.py, compute_e6.py, compute_cost.py (numbers),
             make_tables.py, make_figures.py (tables and figures), make_macros.py (-> LaTeX macros)
results/     detector and judge scores used in the paper
paper/       LaTeX source, figures, compiled main.pdf
```

## Reproduce

```bash
pip install -r requirements.txt
AGENTDOJO_PY=/path/to/agentdojo-env/bin/python QWEN=/models/Qwen2.5-7B-Instruct LLAMA=/models/Llama-3.1-8B-Instruct bash reproduce.sh
```

To regenerate only the numbers and figures from the released scores, copy `results/*.json` into the working directory, rebuild the `.jsonl` sets (steps 1–3 of `reproduce.sh`, CPU only, with `PYTHONHASHSEED=0`), then run the analysis scripts of step 8 and `python analysis/make_tables.py && python analysis/make_figures.py`.

Robustness and cost experiments (journal version):

- `analysis/compute_robust.py`: bootstrap intervals and permutation tests for rank transfer; thresholds calibrated on held-out AgentDojo suites.
- `scripts/score_e6.py`, `analysis/compute_e6.py`: Prompt Guard 2 in the setting of the community benchmark buried-injections, and a check that the rebuilt sets reproduce the stored scores.
- `scripts/latency.py`, `analysis/compute_cost.py`: latency and memory per tool output on GPU and CPU.
- `scripts/e5_run.py`, `scripts/e5_compare.py`, `scripts/e5_score.py`: a real agent (Qwen2.5-7B served by vLLM) runs AgentDojo without attacks; its tool calls are replayed without a model and compared output by output.
- `scripts/check_revision.py`: which Hugging Face revision produced the stored scores (Horizon-Labs changed its weights on 2026-10-02; the stored scores are revision `af3c52e6a9d8`).

Evaluation data is rebuilt from the public sources, not redistributed: AgentDojo v1.2, tau-bench (MIT), InjecAgent attacks (MIT), BIPIA (MIT), GitHub READMEs (`h1alexbel/github-readmes`), AESLC, FineWeb-Edu. Some scripts assume BIPIA and PIGuard are cloned under `~/agentsec/`.

## Building the paper

`paper/usenix-2020-09-xetex.sty` is a build copy of the USENIX style that compiles with tectonic/XeTeX. For submission, switch `main.tex` to the official `usenix-2020-09.sty` and compile with pdflatex.

## Status

Preprint on arXiv; journal version in preparation. Not evaluated against adaptive attacks; both agent benchmarks are simulations and tau-bench's injection point is ours; commercial API detectors not included. See the limitations section of the paper.

## License

Code, analysis scripts, and score files: MIT (see `LICENSE`). Third-party files keep their own terms: the USENIX LaTeX style (`paper/usenix-2020-09*.sty`) and the benchmarks and detectors the scripts download (AgentDojo, tau-bench, InjecAgent, BIPIA, and each detector's model license).
