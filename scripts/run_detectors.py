"""Score every corpus chunk with open prompt-injection detectors and report false-positive rates by imperative density."""
import argparse
import json
import sys

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DETECTORS = {
    "protectai-deberta-v2": "protectai/deberta-v3-base-prompt-injection-v2",
    "piguard": "leolee99/PIGuard",
}

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="corpus.jsonl")
ap.add_argument("--out", default="detector_scores.json")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--max_len", type=int, default=512)
a = ap.parse_args()

rows = [json.loads(line) for line in open(a.corpus)]
if a.limit:
    rows = rows[:a.limit]
texts = [r["text"] for r in rows]
result = {"n": len(rows), "detectors": {}}

for name, repo in DETECTORS.items():
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSequenceClassification.from_pretrained(repo, trust_remote_code=True).cuda().eval()
    id2label = model.config.id2label
    inj = [i for i, l in id2label.items() if "inj" in l.lower() or l.lower() in ("label_1", "unsafe", "malicious")]
    inj_idx = inj[0] if inj else 1
    print(name, "labels:", id2label, "-> injection index", inj_idx, file=sys.stderr)
    probs = []
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], padding=True, truncation=True, max_length=a.max_len, return_tensors="pt").to("cuda")
        with torch.no_grad():
            p = torch.softmax(model(**enc).logits.float(), -1)[:, inj_idx].cpu().numpy()
        probs.extend(p.tolist())
    result["detectors"][name] = dict(repo=repo, inj_label=id2label[inj_idx], probs=probs)
    del model
    torch.cuda.empty_cache()

for r_i, r in enumerate(rows):
    r.pop("text", None)
result["meta"] = rows
json.dump(result, open(a.out, "w"))


def report():
    dens = np.array([r["density"] for r in rows])
    src = np.array([r["source"] for r in rows])
    bins = [("d=0", dens == 0), ("0<d<0.3", (dens > 0) & (dens < 0.3)), ("d>=0.3", dens >= 0.3), ("d>=0.5", dens >= 0.5)]
    for name, d in result["detectors"].items():
        p = np.array(d["probs"])
        flag = p >= 0.5
        print(f"\n== {name}: overall FPR {flag.mean():.4f} (n={len(p)})")
        for b, m in bins:
            if m.sum():
                print(f"  {b:8s} n={m.sum():5d}  FPR={flag[m].mean():.4f}")
        for s in sorted(set(src)):
            m = src == s
            print(f"  source={s:7s} n={m.sum():5d}  FPR={flag[m].mean():.4f}")


report()
