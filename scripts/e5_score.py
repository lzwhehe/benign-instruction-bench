"""Detectors on the tool outputs a real agent received (E5): output-level FPR and task-level block rate at the
default threshold, next to the same detectors on the LLM-free replay of the ground-truth calls (AD-Clean).

usage: python scripts/e5_score.py e5_observed_outputs.jsonl results/e5_detectors.json
"""
import collections
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import score_many as sm  # noqa: E402

obs_path, out_path = sys.argv[1:3]
obs = sm.load(obs_path)
obs = [r for r in obs if not r["error"]]  # error messages are not tool data
clean = sm.load("agentdojo_benign_outputs.jsonl")
NAMES = ["protectai-v2", "piguard", "promptguard2-86m", "horizon-base", "qualifire-sentinel", "testsavant-large"]


def score(name, rows):
    repo, kind, max_len, idx, remote = sm.DETECTORS[name]
    texts = [r["text"] for r in rows]
    if kind == "cls":
        return sm.run_cls(repo, max_len, idx, remote, texts)[0]
    return sm.run_segmented(repo, max_len, texts)[0]


def rates(rows, p):
    f = np.array(p) >= 0.5
    by = collections.defaultdict(list)
    for r, x in zip(rows, f):
        by[(r["suite"], r["task"])].append(x)
    return dict(output_fpr=float(f.mean()), task_block=float(np.mean([any(v) for v in by.values()])),
                n_outputs=len(rows), n_tasks=len(by))


res = {}
for name in NAMES:
    res[name] = dict(observed=rates(obs, score(name, obs)), replay=rates(clean, score(name, clean)))
    print(name, json.dumps(res[name]), flush=True)
    torch.cuda.empty_cache()
json.dump(res, open(out_path, "w"), indent=1)
