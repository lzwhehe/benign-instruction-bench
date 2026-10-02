"""Build a small benign corpus (READMEs, emails, general web text) and score each chunk by *imperative density*.

Only benign public text is used. Nothing here contains injection payloads.
"""
import argparse
import json
import re
import random

from datasets import load_dataset

IMPERATIVES = set("""run install open click use set add create make ensure please send write copy delete remove replace
enter select go check call import download type press save update restart configure add edit change put take
follow note try keep start stop build clone navigate move provide include avoid consider read verify test
launch execute fill upload attach forward reply let contact submit review sign log insert append""".split())
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
MARKERS = re.compile(r"\b(you (should|must|need to|can|will)|make sure|be sure to|don't forget|please)\b", re.I)


def density(text: str) -> float:
    sents = [s.strip(" -*#>`\t") for s in SENT_SPLIT.split(text) if len(s.strip()) > 8]
    if not sents:
        return 0.0
    hit = 0
    for s in sents:
        w = s.split()
        first = re.sub(r"[^a-z']", "", w[0].lower()) if w else ""
        if first in IMPERATIVES or MARKERS.search(s):
            hit += 1
    return hit / len(sents)


def chunks(text: str, lo=80, hi=220):
    words = text.split()
    out, i = [], 0
    while i < len(words):
        n = random.randint(lo, hi)
        piece = " ".join(words[i:i + n])
        if len(piece.split()) >= lo // 2:
            out.append(piece)
        i += n
    return out


def stream(name, split, field, n_docs, config=None):
    ds = load_dataset(name, config, split=split, streaming=True)
    k = 0
    for r in ds:
        t = r.get(field) if isinstance(r.get(field), str) else None
        if t and len(t.split()) > 60:
            yield t
            k += 1
            if k >= n_docs:
                break


ap = argparse.ArgumentParser()
ap.add_argument("--per_source", type=int, default=2500, help="max chunks per source")
ap.add_argument("--out", default="corpus.jsonl")
a = ap.parse_args()
random.seed(0)

sources = [
    ("readme", dict(name="h1alexbel/github-readmes", split="train", field=None)),
    ("email", dict(name="Yale-LILY/aeslc", split="train", field="email_body")),
    ("web", dict(name="HuggingFaceFW/fineweb-edu", split="train", field="text", config="sample-10BT")),
]

rows = []
for src, spec in sources:
    ds = load_dataset(spec["name"], spec.get("config"), split=spec["split"], streaming=True)
    first = next(iter(ds))
    field = spec["field"]
    if field is None or field not in first:
        # pick the longest string-valued field
        field = max((k for k, v in first.items() if isinstance(v, str)), key=lambda k: len(first[k]))
    print(src, "field:", field)
    got = 0
    for r in ds:
        t = r.get(field)
        if not isinstance(t, str) or len(t.split()) < 60:
            continue
        for c in chunks(t):
            rows.append(dict(source=src, text=c, density=density(c)))
            got += 1
        if got >= a.per_source:
            break
    print(src, "chunks:", got)

with open(a.out, "w") as f:
    for i, r in enumerate(rows):
        r["id"] = i
        f.write(json.dumps(r) + "\n")
print("total", len(rows))
