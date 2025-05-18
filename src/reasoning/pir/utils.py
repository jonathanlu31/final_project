# https://huggingface.co/docs/trl/main/en/dataset_formats#prompt-only
def format_for_grpo(example: dict) -> dict:
    """change messages to prompts and remove the assistant response for generation"""
    prompt = []
    for el in example["messages"][:2]:
        prompt.append({
            "role": el["role"],
            "content": el["content"]
        })
    
    assert len(prompt) == 2 and prompt[0]["role"] == "system" and prompt[1]["role"] == "user", f"prompt format error: len: {len(prompt)}, [{prompt[0]['role']}, {prompt[1]['role']}]"
    
    return {
        "prompt": prompt,
        "ground_truth": {
            "benign": example["benign"],
            "witness": example["witness"],
        }
    }

def pir_reward_func(completions: list[list[dict[str: str]]], ground_truth: dict[str: str], **kwargs) -> list[int]:
    """reward function for GRPO"""
    rewards = []
    for gen, truth in zip(completions, ground_truth):
        gen_str = gen[0]["content"].strip()
        
        if truth["benign"]:
            rewards.append(1) if truth["witness"] in gen_str else rewards.append(0)
        else:
            rewards.append(0) if truth["witness"] in gen_str else rewards.append(1)
    return rewards
        
