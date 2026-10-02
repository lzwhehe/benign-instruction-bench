"""Replay AgentDojo ground-truth tool calls (no LLM, no injections) and save the benign tool outputs.

Run inside an env with agentdojo installed. Output: one JSON line per tool call.
"""
import json

from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.task_suite import get_suite

out = open("agentdojo_benign_outputs.jsonl", "w")
n = 0
for suite_name in ["workspace", "travel", "banking", "slack"]:
    suite = get_suite("v1.2", suite_name)
    for task_id, task in suite.user_tasks.items():
        env = suite.load_and_inject_default_environment({})  # no injections: default (benign) placeholders
        runtime = FunctionsRuntime(suite.tools)
        try:
            calls = task.ground_truth(env.model_copy(deep=True))
        except Exception as e:
            print("skip", suite_name, task_id, e)
            continue
        for k, call in enumerate(calls):
            try:
                result, err = runtime.run_function(env, call.function, dict(call.args))
            except Exception as e:
                result, err = None, str(e)
            text = tool_result_to_str(result) if err is None else f"Error: {err}"
            out.write(json.dumps(dict(suite=suite_name, task=task_id, step=k, function=call.function,
                                      text=text, error=err is not None)) + "\n")
            n += 1
print("wrote", n, "tool outputs")
