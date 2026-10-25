"""Does the LLMail email wrapper ("Subject of the email: ... Body: ...") drive false positives on benign emails?

Scores the benign and evaded LLMail rows in two forms, the challenge's wrapper and the body alone, with detectors from
score_many.py. Writes results/llmail_format.json.
Usage: DEVICE=cpu python scripts/llmail_format.py horizon-base qualifire-sentinel
"""
import json
import sys

import numpy as np

sys.argv, dets = sys.argv[:1], sys.argv[1:]
import score_many as sm  # noqa: E402

rows = [json.loads(line) for line in open("llmail_eval.jsonl")]
rows = [r for r in rows if r["group"] in ("benign", "evaded")]
wrapped = [r["text"] for r in rows]
body = [r["text"].split("Body: ", 1)[1] if "Body: " in r["text"] else r["text"] for r in rows]
subj = [r["text"].split("Body: ", 1)[0].replace("Subject of the email: ", "").strip().rstrip(".") for r in rows]
plain = [f"Subject: {s}\n\n{b}" for s, b in zip(subj, body)]
y = np.array([r["label"] for r in rows])
out = {}
for d in dets:
    repo, kind, max_len, idx, remote = sm.DETECTORS[d]
    res = {}
    for name, texts in [("wrapped", wrapped), ("plain", plain), ("body", body)]:
        p = np.array(sm.run_cls(repo, max_len, idx, remote, texts, sm.REVISIONS.get(d))[0])
        res[name] = dict(benign_fpr05=float(np.mean(p[y == 0] >= 0.5)), evaded_tpr05=float(np.mean(p[y == 1] >= 0.5)),
                         benign_median=float(np.median(p[y == 0])))
        print(d, name, res[name], flush=True)
    out[d] = res
json.dump(out, open("results/llmail_format.json", "w"), indent=1)
