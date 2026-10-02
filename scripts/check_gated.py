"""Which gated detectors are accessible with the logged-in account, and what are their labels?"""
import json
import re

from huggingface_hub import hf_hub_download

GATED = ["meta-llama/Llama-Prompt-Guard-2-86M", "meta-llama/Llama-Prompt-Guard-2-22M", "meta-llama/Prompt-Guard-86M",
         "qualifire/prompt-injection-sentinel", "rogue-security/prompt-injection-jailbreak-sentinel-v2",
         "protectai/deberta-v3-small-prompt-injection-v2", "codeintegrity-ai/promptguard"]
for g in GATED:
    try:
        cfg = json.load(open(hf_hub_download(g, "config.json")))
        print(g, "OK", cfg.get("architectures"), cfg.get("id2label"), "maxpos", cfg.get("max_position_embeddings"))
    except Exception as e:
        print(g, "not accessible:", type(e).__name__)
card = open(hf_hub_download("meta-llama/Llama-Prompt-Guard-2-86M", "README.md")).read()
keep = [line for line in card.splitlines() if re.search(r"(?i)label|512|threshold|window|malicious|benign", line)]
print("\n".join(keep)[:2500])
