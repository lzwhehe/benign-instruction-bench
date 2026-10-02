"""Replay AgentDojo ground-truth calls with AgentDojo's *built-in* benchmark attacks placed in injection vectors.

Each tool output is labelled positive if it contains the injected text, else benign. This gives positives and
benign outputs from the same tasks, so detector TPR/FPR can be measured on matched data.
"""
import json

from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite import get_suite


class _Target:  # attacks only read the pipeline name (to fill the model name in templates)
    name = "gpt-4o-2024-05-13"


import sys
ATTACKS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["important_instructions", "ignore_previous", "direct"]
OUT = sys.argv[2] if len(sys.argv) > 2 else "agentdojo_pairs.jsonl"
out = open(OUT, "w")
n_pos = n_neg = 0
for suite_name in ["workspace", "travel", "banking", "slack"]:
    suite = get_suite("v1.2", suite_name)
    inj_tasks = sorted(suite.injection_tasks.items())
    for attack_name in ATTACKS:
        attack = load_attack(attack_name, suite, _Target())
        for task_id, task in suite.user_tasks.items():
            try:
                attack.get_injection_candidates(task)
            except ValueError:
                continue
            inj_id, inj_task = inj_tasks[hash(task_id) % len(inj_tasks)]
            injections = attack.attack(task, inj_task)
            # Label by differential replay: the same ground-truth call in the clean environment.
            # (Substring matching fails because YAML dumping reformats multi-line injected text.)
            clean_env = suite.load_and_inject_default_environment({})
            env = suite.load_and_inject_default_environment(injections)
            runtime = FunctionsRuntime(suite.tools)
            calls = task.ground_truth(env.model_copy(deep=True))
            clean_calls = task.ground_truth(clean_env.model_copy(deep=True))
            for k, call in enumerate(calls):
                try:
                    result, err = runtime.run_function(env, call.function, dict(call.args))
                except Exception as e:
                    result, err = None, str(e)
                text = tool_result_to_str(result) if err is None else f"Error: {err}"
                if k < len(clean_calls) and clean_calls[k].function == call.function:
                    try:
                        cres, cerr = runtime.run_function(clean_env, clean_calls[k].function, dict(clean_calls[k].args))
                    except Exception as e:
                        cres, cerr = None, str(e)
                    clean_text = tool_result_to_str(cres) if cerr is None else f"Error: {cerr}"
                else:
                    clean_text = None
                if clean_text is None:
                    continue  # cannot establish the label; drop rather than guess
                label = int(text != clean_text)
                n_pos += label
                n_neg += 1 - label
                out.write(json.dumps(dict(suite=suite_name, task=task_id, attack=attack_name, inj=inj_id, step=k,
                                          function=call.function, label=label, text=text)) + "\n")
print("positives", n_pos, "benign", n_neg)
