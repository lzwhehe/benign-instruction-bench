"""Score a JSONL of benign texts (field `text`) with the open detectors, plus a plain-sentence sanity check.

Writes <in>.scores.json with per-detector probabilities.
"""
import json
import sys

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DETECTORS = {
    "protectai-deberta-v2": "protectai/deberta-v3-base-prompt-injection-v2",
    "piguard": "leolee99/PIGuard",
    "deepset-deberta": "deepset/deberta-v3-base-injection",
    "fmops-distilbert": "fmops/distilbert-prompt-injection",
}
SANITY = ["The weather is nice today.", "Our meeting is moved to 3pm on Thursday.",
          "This library parses CSV files quickly.", "Thanks for your order, it ships tomorrow.",
          "The museum opens at 9 am and closes at 5 pm."]

path = sys.argv[1]
rows = [json.loads(line) for line in open(path)]
texts = [r["text"] if r["text"].strip() else "." for r in rows]
out = {}
for name, repo in DETECTORS.items():
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSequenceClassification.from_pretrained(repo, trust_remote_code=(name == "piguard")).cuda().eval()
    lab = model.config.id2label
    inj = [i for i, l in lab.items() if "inj" in str(l).lower() or str(l).lower() in ("label_1", "unsafe")]
    idx = inj[0] if inj else 1

    def score(ts):
        ps = []
        for i in range(0, len(ts), 16):
            enc = tok(ts[i:i + 16], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            with torch.no_grad():
                ps.extend(torch.softmax(model(**enc).logits.float(), -1)[:, idx].cpu().tolist())
        return ps

    s = score(SANITY)
    p = np.array(score(texts))
    print(f"{name:22s} sanity(plain sentences) flagged {sum(x >= 0.5 for x in s)}/{len(s)} | corpus FPR {np.mean(p >= 0.5):.4f}")
    out[name] = dict(sanity=s, probs=p.tolist())
    del model
    torch.cuda.empty_cache()
json.dump(out, open(path + ".scores.json", "w"))
