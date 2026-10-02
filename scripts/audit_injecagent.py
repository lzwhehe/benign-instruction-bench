"""Are InjecAgent's attacker instructions in PIGuard's training split or Horizon-Labs' agent-oriented sources?"""
import json
import os
import re

from datasets import load_dataset


def norm(t):
    return re.sub(r"\s+", " ", str(t).lower()).strip()


IA = os.path.expanduser("~/agentsec/InjecAgent/data/")
attacks = [norm(json.loads(l)["Attacker Instruction"])[:80] for f in ["attacker_cases_dh.jsonl", "attacker_cases_ds.jsonl"]
           for l in open(IA + f)]
out = {}
pg = norm(open(os.path.expanduser("~/agentsec/PIGuard/datasets/train.json"), encoding="utf-8", errors="ignore").read())
out["piguard_train"] = f"{sum(a in pg for a in attacks)} of {len(attacks)}"
for name in ["rgeada/tool-response-injections", "nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1",
             "3nesdeniz/agentic-prompt-injection-5k", "3nesdeniz/agentic-prompt-injection-boundary-pairs", "hendzh/PromptShield"]:
    blob = norm(" ".join(json.dumps(r, ensure_ascii=False) for r in load_dataset(name, split="train")))
    out[name] = f"{sum(a in blob for a in attacks)} of {len(attacks)}"
print(json.dumps(out, indent=1))
json.dump(out, open("results/injecagent_overlap_audit.json", "w"), indent=1)
