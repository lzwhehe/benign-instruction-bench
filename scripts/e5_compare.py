"""Replay validity check (E5): do LLM-free replays reproduce the tool outputs a real agent sees?

Reads AgentDojo logs of a real agent run without attacks (python -m agentdojo.scripts.benchmark --model LOCAL ...).
For every task, the agent's own sequence of tool calls is executed again, without an LLM, in a fresh default
environment, and each output is compared with the one the agent received. Also records whether each call matches a
ground-truth call of the task (same function and arguments), and writes the observed outputs so detectors can be
run on them.

usage: python scripts/e5_compare.py LOGDIR e5_observed_outputs.jsonl     (env with agentdojo==0.1.34)
"""
import glob
import json
import os
import sys

from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite import get_suite

logdir, out_path = sys.argv[1:3]


def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(c.get("content", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
    return ""


def norm_args(a):
    return json.dumps(a, sort_keys=True, default=str)


out = open(out_path, "w")
summary = dict(tasks=0, tasks_with_calls=0, calls=0, identical=0, errors=0, in_ground_truth=0, utility=0)
for suite_name in ["workspace", "travel", "banking", "slack"]:
    suite = get_suite("v1.2", suite_name)
    files = sorted(glob.glob(os.path.join(logdir, "*", suite_name, "user_task_*", "none", "none.json")))
    for f in files:
        log = json.load(open(f))
        task_id = log["user_task_id"]
        task = suite.user_tasks[task_id]
        summary["tasks"] += 1
        summary["utility"] += bool(log.get("utility"))
        gt_env = suite.load_and_inject_default_environment({})
        gt = {(c.function, norm_args(dict(c.args))) for c in task.ground_truth(gt_env.model_copy(deep=True))}
        env = suite.load_and_inject_default_environment({})
        runtime = FunctionsRuntime(suite.tools)
        step = 0
        for m in log["messages"]:
            if m["role"] != "tool" or not m.get("tool_call"):
                continue
            call = m["tool_call"]
            fn, args = call["function"], dict(call.get("args") or {})
            observed_error = m.get("error")
            observed = f"Error: {observed_error}" if observed_error else text_of(m["content"])
            if not fn or fn == "<empty-function-name>":  # the executor's messages for calls it refuses to run (agent_pipeline/tool_execution.py)
                result, err = None, "Empty function name provided. Provide a valid function name."
            elif fn not in runtime.functions:
                result, err = None, f"Invalid tool {fn} provided."
            else:
                try:
                    result, err = runtime.run_function(env, fn, args)
                except Exception as e:  # noqa: BLE001
                    result, err = None, str(e)
            replayed = f"Error: {err}" if err is not None else tool_result_to_str(result)
            same = observed == replayed
            in_gt = (fn, norm_args(args)) in gt
            summary["calls"] += 1
            summary["identical"] += same
            summary["errors"] += observed_error is not None
            summary["in_ground_truth"] += in_gt
            out.write(json.dumps(dict(suite=suite_name, task=task_id, step=step, function=fn, text=observed,
                                      replay_identical=same, error=observed_error is not None, in_ground_truth=in_gt,
                                      user_request=task.PROMPT)) + "\n")
            step += 1
        summary["tasks_with_calls"] += step > 0
print(json.dumps(summary, indent=1))
json.dump(summary, open(out_path.replace(".jsonl", "_summary.json"), "w"), indent=1)
