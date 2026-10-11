"""Reproduction check and the community-benchmark setting (E6) for Prompt Guard 2, on rebuilt AgentDojo sets.

1. Rescore the rebuilt AD-Clean and AD-Matched sets with PIGuard and Prompt Guard 2 (86M) exactly as score_many.py
   does, so the rebuilt sets can be compared with the saved scores.
2. Prompt Guard 2 (86M) in the setting of the community benchmark buried-injections: overlapping 510-token windows
   (stride 384, maximum over windows), with and without the user's task prompt placed before the tool output.

usage: python scripts/score_e6.py   (in a directory with the rebuilt agentdojo_*.jsonl)
writes results/e6/<variant>__<set>.json
"""
import json
import os
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
import score_many as sm  # noqa: E402

OUT = "results/e6"
os.makedirs(OUT, exist_ok=True)
SETS = {"adclean": "agentdojo_benign_outputs.jsonl", "admatched": "agentdojo_pairs.jsonl"}
PG2 = "meta-llama/Llama-Prompt-Guard-2-86M"


def run_windows(repo, texts, window=510, stride=384):
    """Overlapping windows with maximum pooling, as in buried-injections (bench/detectors/__init__.py)."""
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSequenceClassification.from_pretrained(repo).cuda().eval()
    probs = []
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        starts = range(0, max(len(ids) - window, 0) + stride, stride)
        wins = [[tok.cls_token_id, *ids[s:s + window], tok.sep_token_id] for s in starts]
        batch = tok.pad({"input_ids": wins}, return_tensors="pt").to("cuda")
        with torch.no_grad():
            probs.append(model(**batch).logits.float().softmax(-1)[:, 1].max().item())
    return probs


def save(name, sname, rows, probs):
    labels = [r["label"] for r in rows] if "label" in rows[0] else None
    json.dump(dict(variant=name, set=sname, probs=probs, labels=labels,
                   suites=[r["suite"] for r in rows], tasks=[r["task"] for r in rows]),
              open(f"{OUT}/{name}__{sname}.json", "w"))
    print(name, sname, len(probs), flush=True)


for sname, path in SETS.items():
    rows = sm.load(path)
    texts = [r["text"] for r in rows]
    with_task = [f"{r['user_request']}\n{r['text']}" for r in rows]
    save("piguard-rebuilt", sname, rows, sm.run_cls("leolee99/PIGuard", 512, 1, True, texts)[0])
    save("promptguard2-86m-rebuilt", sname, rows, sm.run_segmented(PG2, 512, texts)[0])
    save("pg2-overlap-tool", sname, rows, run_windows(PG2, texts))
    save("pg2-overlap-task+tool", sname, rows, run_windows(PG2, with_task))
