from reasoning.data import tokenize_secalign_data
from reasoning.tokenizer import get_tokenizer_and_collator

tokenizer, collator = get_tokenizer_and_collator("Qwen/Qwen3-4B")

example = {
    "prompt": [{"role": "system", "content": "What is the capital of France?"}, {"role": "user", "content": "blah"}],
    "chosen": [{"role": "assistant", "content": "Paris", "reasoning_content": "Paris is the capital of France."}],
    "rejected": [{"role": "assistant", "content": "London", "reasoning_content": "London is the capital of England."}],
}

formatted = tokenize_secalign_data(tokenizer, example)

print(formatted["prompt"], end='')
print("-" * 100)
print(formatted["chosen"], end='')
print("-" * 100)
print(formatted["rejected"], end='')
