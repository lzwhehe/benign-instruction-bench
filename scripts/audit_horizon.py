"""Search Horizon-Labs' agentic training sources for AgentDojo attack templates and BIPIA attack strings.

Sources are those listed in horizon-ai-labs/agent-io-guards data/build_v0.py. Verbatim search after
whitespace/case normalization, so the result is a lower bound.
"""
import json
import os
import re

from datasets import load_dataset


def norm(t):
    return re.sub(r"\s+", " ", str(t).lower()).strip()


B = os.path.expanduser("~/agentsec/BIPIA/benchmark/")
bipia = [norm(s)[:80] for f in ["text_attack_train", "text_attack_test", "code_attack_train", "code_attack_test"]
         for v in json.load(open(B + f + ".json")).values() for s in v]
AGENTDOJO_MARKERS = ["this is an important message from me", "<information>", "before you can solve the task that i gave you in the beginning",
                     "ignore your previous iunstructions", "todo: "]
SOURCES = [("rgeada/tool-response-injections", None, "train"),
           ("nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1", None, "train"),
           ("3nesdeniz/agentic-prompt-injection-5k", None, "train"),
           ("3nesdeniz/agentic-prompt-injection-boundary-pairs", None, "train"),
           ("hendzh/PromptShield", None, "train")]
report = {}
for name, cfg, split in SOURCES:
    try:
        ds = load_dataset(name, cfg, split=split)
    except Exception as e:
        report[name] = f"load failed: {type(e).__name__}: {str(e)[:120]}"
        print(name, report[name], flush=True)
        continue
    blob = norm(" ".join(json.dumps(r, ensure_ascii=False) for r in ds))
    hits = {m: blob.count(m) for m in AGENTDOJO_MARKERS}
    bip = sum(s in blob for s in bipia)
    report[name] = dict(rows=len(ds), agentdojo_marker_counts=hits, bipia_attack_strings_found=f"{bip} of {len(bipia)}")
    print(name, json.dumps(report[name]), flush=True)
json.dump(report, open("results/horizon_training_audit.json", "w"), indent=1)
