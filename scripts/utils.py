from datasets import (
    Dataset,
    load_dataset,
    get_dataset_config_names,
    concatenate_datasets,
)
import scenarios

import re
import string
from typing import List, Union
from message import Message

REASONING_SYSTEM_PROMPT = """\
First think through your response within <think> </think> tags. Then provide your final response. Follow the following format:
<think>
...
</think>
...
"""

CORRECTNESS_REWARD = 5.0


def normalize_params(params):
    if not params:
        return None

    return {
        "secret": str(params["secret"]) if "secret" in params else None,
        "animals": params["animals"] if "animals" in params else None,
        "password": str(params["password"]) if "password" in params else None,
        "sounds": params["sounds"] if "sounds" in params else None,
        "user1": str(params["user1"]) if "user1" in params else None,
        "user2": str(params["user2"]) if "user2" in params else None,
        "word": str(params["word"]) if "word" in params else None,
        "users": params["users"] if "users" in params else None,
        "key": str(params["key"]) if "key" in params else None,
        "number": params["number"] if "number" in params else None,
        "secrets": params["secrets"] if "secrets" in params else None,
        "balance1": params["balance1"] if "balance1" in params else None,
        "balance2": params["balance2"] if "balance2" in params else None,
    }


def get_redteam(shuffle: bool) -> Dataset:
    dataset_repo = "jonluj/rules_redteam_qwen2.5-7b"
    datasets = [
        load_dataset(dataset_repo, config)["train"]
        for config in get_dataset_config_names(dataset_repo) if config not in ["AccessControl", "DiningCryptographers"]
    ]
    formatted_data = [
        data.map(
            lambda x: {
                "prompt": [
                    {
                        "role": "system",
                        "content": x["messages"][0]["content"]
                        + "\n\n"
                        + REASONING_SYSTEM_PROMPT,
                    },
                    *x["messages"][1:],
                ],
                "id": x["id"],
                "scenario_args": normalize_params(x["params"]),
                "full_scenario": data.config_name,
            },
            remove_columns=data.column_names,
            num_proc=2,
        )
        for data in datasets
    ]

    full_dataset = concatenate_datasets(formatted_data)
    if shuffle:
        return full_dataset.shuffle(seed=42)
    return full_dataset


def extract_answer(r) -> str:
    if "</think>" not in r:
        return ""
    return r.split("</think>")[-1]


def format_messages(messages):
    return "\n\n".join([f"{m['role']}: {m['content']}" for m in messages])


def correctness_reward_func(
    prompts, completions, scenario_args, full_scenario, **kwargs
) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]
    rewards = []
    for i in range(len(extracted_responses)):
        scenario_cls = scenarios.SCENARIOS[full_scenario[i].split("_")[0]]
        scenario = scenario_cls(scenario_args[i])
        testcase_messages = prompts[i]
        full_messages = testcase_messages + [
            {"role": "assistant", "content": extracted_responses[i]}
        ]
        rewards.append(
            CORRECTNESS_REWARD
            * scenario.evaluate(Message.unserialize(full_messages), True),
        )
    print(
        "-" * 20,
        f"Messages:\n{format_messages(prompts[0])}",
        f"\nResponse:\n{responses[0]}",
        f"\nReward:\n{rewards[0]}",
    )
    return rewards


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^.*?\n</think>\n.*$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def loose_format_reward_func(completions, **kwargs) -> list[float]:
    return [0.5 if "</think>" in completion[0]['content'] else 0.0 for completion in completions]


def normalize(response: str):
    return response.lower().strip(string.punctuation + string.whitespace)


def contains(
    text: Union[str, List[str]], query: Union[str, List[str]], ignore_case: bool = False
):
    if isinstance(query, str):
        query = [query]
    if isinstance(text, str):
        text = [text]

    for q in query:
        for t in text:
            if bool(re.search(q, t, flags=re.IGNORECASE if ignore_case else 0)):
                return True
    return False
