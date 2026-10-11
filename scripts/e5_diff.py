"""Show how the non-identical outputs of the replay check (E5) differ: replay each task's calls and diff line by line."""
import difflib
import glob
import json
import os
import sys

from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite import get_suite

logdir = sys.argv[1]
kinds = {}
for f in sorted(glob.glob(os.path.join(logdir, "*", "*", "user_task_*", "none", "none.json"))):
    log = json.load(open(f))
    suite = get_suite("v1.2", log["suite_name"])
    env = suite.load_and_inject_default_environment({})
    rt = FunctionsRuntime(suite.tools)
    for m in log["messages"]:
        if m["role"] != "tool" or not m.get("tool_call"):
            continue
        fn, args = m["tool_call"]["function"], dict(m["tool_call"].get("args") or {})
        if fn not in rt.functions:
            continue
        try:
            res, err = rt.run_function(env, fn, args)
        except Exception as e:  # noqa: BLE001
            res, err = None, str(e)
        obs = "".join(c.get("content", "") for c in m["content"]) if isinstance(m["content"], list) else m["content"]
        rep = tool_result_to_str(res) if err is None else None
        if m.get("error") or rep is None or obs == rep:
            continue
        d = [l for l in difflib.unified_diff(obs.splitlines(), rep.splitlines(), lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
        key = tuple(sorted({l[1:].split(":")[0].strip() for l in d}))
        kinds[key] = kinds.get(key, 0) + 1
        if kinds[key] == 1:
            print(fn, d[:4])
print(kinds)
