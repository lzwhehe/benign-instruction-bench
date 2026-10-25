"""Adaptive attacks from the LLMail-Inject challenge (Microsoft, MIT license), phase 2 only.

In the challenge, participants wrote emails that an LLM email assistant retrieves as tool output, and iterated against
defended systems (Prompt Shields v2, TaskTracker, an LLM judge, and combinations). Phase 1 is excluded because
Horizon-Labs lists it among its training sources.

llmail_eval.jsonl:
  benign   : the challenge's own emails for false-positive tests (label 0)
  adaptive : unique phase-2 submissions that were not detected by the challenge defense and made the assistant send
             the exfiltration email with the right recipient and body (label 1, group "evaded")
  attempts : a random sample of unique phase-2 submissions labeled as attack attempts (label 1, group "attempt")
Texts follow the challenge's own formatting of a retrieved email ("Subject of the email: ... Body: ...").
"""
import json
import random

from huggingface_hub import hf_hub_download

REPO, N = "microsoft/llmail-inject-challenge", 1000
get = lambda f: hf_hub_download(REPO, f, repo_type="dataset")  # noqa: E731
rng = random.Random(0)


def fmt(subject, body):
    return f"Subject of the email: {subject}.   Body: {body}"


benign = json.load(open(get("data/emails_for_fp_tests.json")))
evaded, seen = [], set()
for line in open(get("data/raw_submissions_phase2.jsonl")):
    r = json.loads(line)
    o = json.loads(r["objectives"]) if isinstance(r.get("objectives"), str) else (r.get("objectives") or {})
    if all(o.get(k) for k in ["defense.undetected", "exfil.sent", "exfil.destination", "exfil.content"]):
        t = fmt(r.get("subject", ""), r.get("body", ""))
        if t not in seen:
            seen.add(t)
            evaded.append((t, r.get("scenario", "")))
labelled = json.load(open(get("data/labelled_unique_submissions_phase2.json")))
attempts = sorted(t for t, v in labelled.items() if str(v.get("attack_attempt")) == "True")

rows = [dict(suite="llmail", task="fp", text=t, label=0, group="benign", source="email") for t in benign]
for t, sc in rng.sample(evaded, min(N, len(evaded))):
    rows.append(dict(suite="llmail", task=sc, text=t, label=1, group="evaded", source="email"))
for t in rng.sample(attempts, N):
    rows.append(dict(suite="llmail", task="attempt", text=t, label=1, group="attempt", source="email"))
with open("llmail_eval.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(dict(benign=len(benign), evaded_unique=len(evaded), attempts_unique=len(attempts),
           written={g: sum(r["group"] == g for r in rows) for g in ["benign", "evaded", "attempt"]}))
