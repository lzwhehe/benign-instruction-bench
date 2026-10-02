"""Detection rate by insertion position and context length, to expose 512-token truncation artifacts."""
import json

import numpy as np
from transformers import AutoTokenizer

rows = [json.loads(line) for line in open("bipia_pairs.jsonl")]
tok = AutoTokenizer.from_pretrained("protectai/deberta-v3-base-prompt-injection-v2")
ntok = np.array([len(tok(r["text"], add_special_tokens=False)["input_ids"]) for r in rows])
J = np.array(json.load(open("judge_bipia_qwen.json"))["probs"])
D = json.load(open("normalized_probs_bipia.json"))["probs"]
S = {"judge": J, "piguard": np.array(D["piguard/original"]), "protectai": np.array(D["protectai-deberta-v2/original"])}
y = np.array([r["label"] for r in rows])
pos = np.array([r["position"] for r in rows])
task = np.array([r["suite"].split("-")[1] for r in rows])
print("fraction of texts longer than 512 tokens:", {t: round(float(np.mean(ntok[task == t] > 512)), 3) for t in ["email", "table", "code"]})
for t in ["email", "table", "code"]:
    for p in ["start", "middle", "end"]:
        m = (y == 1) & (task == t) & (pos == p)
        print(f"{t:5s} {p:6s} n={m.sum():4d} >512tok={np.mean(ntok[m] > 512):.2f} | " +
              " ".join(f"{k}={np.mean(v[m] >= .5):.2f}" for k, v in S.items()))
