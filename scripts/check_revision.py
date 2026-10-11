"""Which revision of a detector produced the stored scores? Rescore AD-Clean on CPU with given revisions.

usage: python scripts/check_revision.py REPO DETECTOR_NAME REV [REV ...]   (in a directory with agentdojo_benign_outputs.jsonl)
"""
import json
import sys

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo, name, revs = sys.argv[1], sys.argv[2], sys.argv[3:]
torch.set_num_threads(16)
texts = [json.loads(line)["text"] for line in open("agentdojo_benign_outputs.jsonl")]
stored = np.array(json.load(open(f"results/many/{name}__adclean.json"))["probs"])
out = {}
for rev in revs:
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    model = AutoModelForSequenceClassification.from_pretrained(repo, revision=rev).eval()
    p = []
    with torch.no_grad():
        for t in texts:
            enc = tok(t or ".", truncation=True, max_length=2048, return_tensors="pt")
            enc.pop("token_type_ids", None)
            p.append(torch.softmax(model(**enc).logits.float(), -1)[0, 1].item())
    p = np.array(p)
    print(rev, "max|diff| vs stored = %.4f" % np.abs(p - stored).max(), "flagged@0.5 =", int((p >= 0.5).sum()),
          "stored flagged =", int((stored >= 0.5).sum()), flush=True)
    out.setdefault("revisions", {})[rev] = dict(max_abs_diff_vs_stored=round(float(np.abs(p - stored).max()), 4),
                                                flagged_at_05=int((p >= 0.5).sum()))
json.dump(dict(repo=repo, set="adclean", n=len(texts), **out), open(f"results/{name.split('-')[0]}_revision_check.json", "w"), indent=1)
