"""tau-bench evaluation set: all clean tool outputs (benign) + injected outputs from the matched replay."""
import json

rows = [dict(json.loads(line), label=0, attack="none") for line in open("taubench_clean.jsonl")]
rows += [json.loads(line) for line in open("taubench_pairs.jsonl") if json.loads(line)["label"] == 1]
with open("taubench_eval.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("benign", sum(r["label"] == 0 for r in rows), "injected", sum(r["label"] == 1 for r in rows))
