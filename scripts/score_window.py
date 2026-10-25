"""Uniform input handling (truncation check): score the detectors that truncate at 512 tokens with overlapping windows
(510 tokens, stride 384, maximum over windows) instead, on the AgentDojo, tau-bench, and BIPIA sets.

usage: python scripts/score_window.py      writes results/window/<detector>__<set>.json
"""
import json
import os
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEV = os.environ.get("DEVICE", "cuda")  # DEVICE=cpu runs the encoders on CPU

sys.path.insert(0, os.path.dirname(__file__))
import score_many as sm  # noqa: E402

OUT = "results/window"
os.makedirs(OUT, exist_ok=True)
NAMES = ["protectai-v1", "protectai-v2", "protectai-small-v2", "jasperls", "testsavant-large", "piguard"]
SETS = {"adclean": "agentdojo_benign_outputs.jsonl", "admatched": "agentdojo_pairs.jsonl",
        "tbeval": "taubench_eval.jsonl", "bipia": "bipia_pairs.jsonl"}


def run_windows(repo, idx, remote, texts, window=510, stride=384):
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSequenceClassification.from_pretrained(repo, trust_remote_code=remote).to(DEV).eval()
    probs = []
    for t in texts:
        ids = tok(t or ".", add_special_tokens=False)["input_ids"]
        starts = range(0, max(len(ids) - window, 0) + stride, stride)
        wins = [[tok.cls_token_id, *ids[s:s + window], tok.sep_token_id] for s in starts]
        batch = tok.pad({"input_ids": wins}, return_tensors="pt").to(DEV)
        batch.pop("token_type_ids", None)
        with torch.no_grad():
            probs.append(model(**batch).logits.float().softmax(-1)[:, idx].max().item())
    return probs


for name in NAMES:
    repo, kind, max_len, idx, remote = sm.DETECTORS[name]
    for sname, path in SETS.items():
        out = f"{OUT}/{name}__{sname}.json"
        if os.path.exists(out):
            continue
        rows = sm.load(path)
        p = run_windows(repo, idx, remote, [r["text"] for r in rows])
        json.dump(dict(detector=name, set=sname, probs=p, labels=[r.get("label") for r in rows]), open(out, "w"))
        print(name, sname, len(p), flush=True)
        torch.cuda.empty_cache() if DEV == "cuda" else None
