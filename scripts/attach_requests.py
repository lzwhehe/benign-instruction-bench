"""Attach AgentDojo user-task prompts to pair rows as `user_request` (so judges need no agentdojo import)."""
import json
import sys

from agentdojo.task_suite import get_suite

P = {(s, tid): t.PROMPT for s in ["workspace", "travel", "banking", "slack"] for tid, t in get_suite("v1.2", s).user_tasks.items()}
for path in sys.argv[1:]:
    rows = [json.loads(line) for line in open(path)]
    for r in rows:
        r["user_request"] = P[(r["suite"], r["task"])]
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(path, len(rows))
