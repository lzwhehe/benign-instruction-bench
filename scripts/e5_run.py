"""Run AgentDojo's benchmark with a local model served by vLLM (for the replay check, E5).

AgentDojo 0.1.34's LocalLLM sends its internal content blocks ({"type": "text", "content": ...}) to the server, which
an OpenAI-compatible server rejects, and it passes tool results as a JSON dump of those blocks. This runner flattens
every message to plain text before sending and returns tool results as text in a user turn; the prompt format and
the parser of LocalLLM are unchanged. Tool outputs logged by AgentDojo are not affected by this change.

usage: python scripts/e5_run.py --model LOCAL --benchmark-version v1.2 --logdir e5_runs -s workspace ...
"""
import json

from agentdojo.agent_pipeline.llms import local_llm
from agentdojo.types import get_text_content_as_str


def query(self, query, runtime, env=local_llm.EmptyEnv(), messages=[], extra_args={}):
    msgs = []
    for m in messages:
        role = m["role"]
        text = get_text_content_as_str(m["content"]) if m["content"] is not None else ""
        if role == "system":
            msgs.append({"role": "system", "content": local_llm._make_system_prompt(text, runtime.functions.values())})
        elif role == "tool":
            result = {"error": m["error"]} if m.get("error") else {"result": text if text != "None" else "Success"}
            msgs.append({"role": "user", "content": "Function result: " + json.dumps(result)})
        else:
            msgs.append({"role": role, "content": text})
    completion = local_llm.chat_completion_request(self.client, model=self.model, messages=msgs,
                                                   temperature=self.temperature, top_p=self.top_p)
    return query, runtime, env, [*messages, local_llm._parse_model_output(completion)], extra_args


local_llm.LocalLLM.query = query

if __name__ == "__main__":
    from agentdojo.scripts.benchmark import main
    main()
