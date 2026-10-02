"""Does normalizing structured tool outputs cut false positives without losing detection?

Variants per tool output:
  original  : the YAML-formatted tool result the agent sees
  values    : YAML keys / list markers / quotes stripped, values joined as prose
  per_field : each value line scored separately; flag if any line is flagged
Reports FPR on benign outputs, TPR on outputs containing a benchmark injection, and task-level block rate.
"""
import collections
import json
import re
import sys

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DETECTORS = {"protectai-deberta-v2": "protectai/deberta-v3-base-prompt-injection-v2", "piguard": "leolee99/PIGuard"}
KEY = re.compile(r"^\s*(-\s+)?([A-Za-z_][\w ]{0,40}):\s?", re.M)


def values(t):
    t = KEY.sub(" ", t)
    t = re.sub(r"^\s*-\s+", " ", t, flags=re.M)
    t = t.replace("'", " ").replace('"', " ").replace("{", " ").replace("}", " ")
    return re.sub(r"\s+", " ", t).strip() or "."


def fields(t):
    lines = [values(x) for x in t.splitlines()]
    return [x for x in lines if len(x) > 1] or ["."]


rows = [json.loads(line) for line in open(sys.argv[1] if len(sys.argv) > 1 else "agentdojo_pairs.jsonl")]
y = np.array([r["label"] for r in rows])
res = {}
PROBS = {}
for name, repo in DETECTORS.items():
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSequenceClassification.from_pretrained(repo, trust_remote_code=(name == "piguard")).cuda().eval()

    def score(ts):
        ps = []
        for i in range(0, len(ts), 16):
            enc = tok(ts[i:i + 16], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            with torch.no_grad():
                ps.extend(torch.softmax(model(**enc).logits.float(), -1)[:, 1].cpu().tolist())
        return np.array(ps)

    var = {"original": score([r["text"] or "." for r in rows]), "values": score([values(r["text"]) for r in rows])}
    flat, owner = [], []
    for i, r in enumerate(rows):
        for f in fields(r["text"]):
            flat.append(f)
            owner.append(i)
    pf = score(flat)
    mx = np.zeros(len(rows))
    for o, p in zip(owner, pf):
        mx[o] = max(mx[o], p)
    var["per_field"] = mx
    from sklearn.metrics import roc_auc_score, roc_curve
    for v, p in var.items():
        PROBS[f"{name}/{v}"] = p.tolist()
        fpr, tpr, _ = roc_curve(y, p)
        t1 = tpr[fpr <= 0.01].max() if (fpr <= 0.01).any() else 0.0
        t5 = tpr[fpr <= 0.05].max() if (fpr <= 0.05).any() else 0.0
        print(f"  [{name}/{v}] AUROC={roc_auc_score(y, p):.3f}  TPR@1%FPR={t1:.3f}  TPR@5%FPR={t5:.3f}")
    print(f"\n=== {name}  (benign n={int((y == 0).sum())}, injected n={int((y == 1).sum())})")
    for v, p in var.items():
        f = p >= 0.5
        tasks = collections.defaultdict(list)
        for r, x, lab in zip(rows, f, y):
            if r["attack"] == "important_instructions":
                tasks[(r["suite"], r["task"])].append((x, lab))
        benign_block = np.mean([any(x for x, lab in v_ if lab == 0) for v_ in tasks.values()])
        print(f"  {v:10s} FPR={f[y == 0].mean():.3f}  TPR={f[y == 1].mean():.3f}  "
              f"tasks w/ benign output flagged={benign_block:.3f}")
        res[f"{name}/{v}"] = dict(fpr=float(f[y == 0].mean()), tpr=float(f[y == 1].mean()), task_block=float(benign_block))
    del model
    torch.cuda.empty_cache()
json.dump(res, open("normalized_results.json", "w"), indent=1)
json.dump({"labels": y.tolist(), "probs": PROBS}, open("normalized_probs.json", "w"))
