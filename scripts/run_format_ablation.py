"""Is the README false-positive rate caused by formatting (HTML/markdown/URLs/code) or by instruction-like language?

For each README chunk we score: original text, a "plain" version (HTML/markdown/URLs/badges stripped),
prose-only (code blocks removed) and code-only. We use four open detectors (benign text only).
"""
import json
import re
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
rows = [json.loads(line) for line in open("corpus.jsonl")]
readme = [r for r in rows if r["source"] == "readme"]


def plain(t):
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"[#*_`|>~-]{1,}", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def prose_only(t):
    return re.sub(r"```.*?```", " ", t, flags=re.S)


def code_only(t):
    parts = re.findall(r"```.*?```", t, flags=re.S)
    return " ".join(parts)


variants = {
    "original": [r["text"] for r in readme],
    "plain": [plain(r["text"]) for r in readme],
    "prose_only": [plain(prose_only(r["text"])) for r in readme],
}
has_code = np.array(["```" in r["text"] for r in readme])
code_texts = [code_only(r["text"]) for r in readme]
out = {}
for name, repo in DETECTORS.items():
    try:
        tok = AutoTokenizer.from_pretrained(repo)
        model = AutoModelForSequenceClassification.from_pretrained(repo, trust_remote_code=(name == "piguard")).cuda().eval()
    except Exception as e:
        print(name, "LOAD FAIL", str(e)[:120], file=sys.stderr)
        continue
    lab = model.config.id2label
    inj = [i for i, l in lab.items() if "inj" in str(l).lower() or str(l).lower() in ("label_1", "unsafe")]
    idx = inj[0] if inj else 1

    def score(texts):
        ps = []
        for i in range(0, len(texts), 16):
            enc = tok([t if t.strip() else "." for t in texts[i:i + 16]], padding=True, truncation=True,
                      max_length=512, return_tensors="pt").to("cuda")
            with torch.no_grad():
                ps.extend(torch.softmax(model(**enc).logits.float(), -1)[:, idx].cpu().tolist())
        return np.array(ps)

    res = {k: score(v) for k, v in variants.items()}
    pc = score([c for c, h in zip(code_texts, has_code) if h])
    print(f"\n== {name} labels={lab}")
    for k, p in res.items():
        print(f"  {k:11s} FPR={np.mean(p >= 0.5):.4f}")
    print(f"  code_only   FPR={np.mean(pc >= 0.5):.4f} (n={len(pc)})")
    out[name] = {k: float(np.mean(p >= 0.5)) for k, p in res.items()}
    out[name]["code_only"] = float(np.mean(pc >= 0.5))
    del model
    torch.cuda.empty_cache()
json.dump(out, open("format_ablation.json", "w"), indent=1)
