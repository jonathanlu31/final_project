from datasets import Dataset
import json
from reasoning.ifeval.instructions_registry import INSTRUCTION_DICT
import inspect

def get_ifeval(shuffle: bool = True) -> Dataset:
    data = []
    dataset_path = "datagen/ifeval/ifeval.jsonl"
    with open(dataset_path, "r") as f:
        for line in f:
            ex = json.loads(line)
            data.append({
                "prompt": ex["messages"],
                "instruction_ids": ex["instruction_id_list"],
                "kwargs": ex["kwargs"]
            })

    dataset = Dataset.from_list(data)
    if shuffle:
        return dataset.shuffle(seed=42)
    return dataset

def extract_response_text(completions):
    return [completion[0]['content'] for completion in completions]

def reward_instruction_following(
    completions,
    instruction_ids: list[list[str]],
    kwargs: list[list[dict]],
    **_
) -> list[float]:
    responses = extract_response_text(completions)
    rewards = []

    for i, response in enumerate(responses):
        follow_flags = []

        for j, instruction_id in enumerate(instruction_ids[i]):
            instruction_cls = INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)

            # Filter kwargs to only accepted arguments
            valid_keys = inspect.signature(instruction.build_description).parameters
            safe_kwargs = {
                k: v for k, v in kwargs[i][j].items() if k in valid_keys
            }

            try:
                instruction.build_description(**safe_kwargs)
                followed = instruction.check_following(response)
                follow_flags.append(bool(followed))
            except Exception:
                follow_flags.append(False)

        reward = sum(follow_flags) / len(follow_flags) if follow_flags else 0.0
        rewards.append(reward)

    return rewards
