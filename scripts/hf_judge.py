"""Same task-aware judge as llm_judge.py, but with local HF models in 4-bit (fits next to a training job).

Score = P(Yes) / (P(Yes) + P(No)) from the first-token distribution, summing over Yes/No token variants.
usage: python hf_judge.py MODEL_DIR pairs.jsonl out.json
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, __file__.rsplit("/", 1)[0])
model_dir, pairs, out = sys.argv[1:4]
src = open(__file__.rsplit("/", 1)[0] + "/llm_judge.py").read()
SYS = eval(src.split("SYS = ", 1)[1].split("\n\n", 1)[0])  # identical prompt to the vLLM judge

rows = [json.loads(line) for line in open(pairs)]
need = [r for r in rows if not r.get("user_request")]
if need:
    from agentdojo.task_suite import get_suite  # noqa
    prompts = {}
    for s in ["workspace", "travel", "banking", "slack"]:
        for tid, t in get_suite("v1.2", s).user_tasks.items():
            prompts[(s, tid)] = t.PROMPT
else:
    prompts = {}

tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                                      bnb_4bit_quant_type="nf4",
                                                      llm_int8_enable_fp32_cpu_offload=True),
    device_map="auto", max_memory={0: os.environ.get("GPU_MEM", "6GiB"), "cpu": "64GiB"}).eval()
vocab = tok.get_vocab()
YES = [i for t, i in vocab.items() if tok.convert_tokens_to_string([t]).strip(" -_>.,").lower() == "yes"]
NO = [i for t, i in vocab.items() if tok.convert_tokens_to_string([t]).strip(" -_>.,").lower() == "no"]
print("yes/no token variants:", len(YES), len(NO))

p = []
with torch.no_grad():
    for r in rows:
        request = r.get("user_request") or prompts[(r["suite"], r["task"])]
        user = f"User request:\n{request}\n\nTool `{r['function']}` returned:\n{r['text'][:6000]}"
        ids = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                                      add_generation_prompt=True, return_tensors="pt").to(0)
        probs = torch.softmax(model(ids).logits[0, -1].float(), -1)
        py, pn = probs[YES].sum().item(), probs[NO].sum().item()
        p.append(py / (py + pn + 1e-12))
p = np.array(p)
y = np.array([r["label"] for r in rows])
fpr, tpr, _ = roc_curve(y, p)
print(f"{model_dir.rstrip('/').split('/')[-1]} (4-bit) on {pairs}: AUROC={roc_auc_score(y, p):.3f} "
      f"TPR@1%FPR={tpr[fpr <= 0.01].max():.3f} TPR@5%FPR={tpr[fpr <= 0.05].max():.3f} "
      f"@0.5 FPR={np.mean(p[y == 0] >= .5):.3f} TPR={np.mean(p[y == 1] >= .5):.3f}")
json.dump(dict(model=model_dir, probs=p.tolist(), labels=y.tolist()), open(out, "w"))
