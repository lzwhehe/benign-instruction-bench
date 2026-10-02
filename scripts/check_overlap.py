"""Verbatim overlap between benchmark attack strings and a detector's published training split."""
import json
import os
import re


def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


P = os.path.expanduser("~/agentsec/PIGuard/datasets/")
B = os.path.expanduser("~/agentsec/BIPIA/benchmark/")
for tf in ["train.json", "valid.json"]:
    raw = open(P + tf, encoding="utf-8", errors="ignore").read()
    blobn = norm(raw)
    try:
        n = len(json.loads(raw))
    except Exception:
        n = "?"
    parts = []
    for name in ["text_attack_test", "text_attack_train", "code_attack_test", "code_attack_train"]:
        d = json.load(open(B + name + ".json"))
        strs = [s for v in d.values() for s in v]
        hit = sum(norm(s)[:80] in blobn for s in strs)
        parts.append("%s=%d/%d" % (name, hit, len(strs)))
    print(tf, "rows:", n, "|", " ".join(parts), "| AgentDojo template:",
          "this is an important message from me" in blobn)
