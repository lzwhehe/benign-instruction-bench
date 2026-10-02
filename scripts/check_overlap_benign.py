"""Are BIPIA's clean contexts (benign side) present in PIGuard's training split?"""
import json
import os
import re


def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


blobn = norm(open(os.path.expanduser("~/agentsec/PIGuard/datasets/train.json"), encoding="utf-8", errors="ignore").read())
rows = [json.loads(line) for line in open("bipia_pairs.jsonl")]
for t in ["bipia-email", "bipia-table", "bipia-code"]:
    ctx = [r["text"] for r in rows if r["suite"] == t and r["label"] == 0]
    hit = sum(norm(c)[:120] in blobn for c in ctx)
    print(f"{t}: {hit}/{len(ctx)} clean contexts found verbatim (first 120 chars) in PIGuard train")
