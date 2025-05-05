from reasoning.data import format_pir_data
from reasoning import get_tokenizer_and_collator

EXPECTED_TEXT = """\
<|im_start|>system
you are a helpful assistant<|im_end|>
<|im_start|>user
this is a test<|im_end|>
<|im_start|>assistant
<think>
Okay, the user said, "this is a test." I need to respond appropriately. Since they mentioned it's a test, maybe they're checking if I'm functioning correctly. My role is to be helpful and provide assistance.

First, I should acknowledge their message. A simple "Hello!" or "Hi there!" would be good. Then, I can ask how I can assist them. But I should also make sure to keep the tone friendly and open-ended.

Wait, the user might be testing if I can handle different types of queries. So, I should be ready to help with any task they have. Maybe I can offer examples of what I can do, like answering questions, providing information, or helping with tasks.

I should also consider if there's any specific context they're referring to when they say "test." Maybe they're testing my knowledge or my ability to follow instructions. In that case, I should confirm that I'm ready to assist and ask them to provide more details if needed.

I need to make sure my response is clear and not too technical. The user might not be familiar with the capabilities of an AI assistant, so a friendly and approachable tone is important. Let me put that all together into a concise response.
</think>

Hello! How can I assist you today? 😊<|im_end|>"""

example = dict(
    messages=[
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": "this is a test"},
        {
            "role": "assistant",
            "reasoning_content": "Okay, the user said, \"this is a test.\" I need to respond appropriately. Since they mentioned it's a test, maybe they're checking if I'm functioning correctly. My role is to be helpful and provide assistance.\n\nFirst, I should acknowledge their message. A simple \"Hello!\" or \"Hi there!\" would be good. Then, I can ask how I can assist them. But I should also make sure to keep the tone friendly and open-ended.\n\nWait, the user might be testing if I can handle different types of queries. So, I should be ready to help with any task they have. Maybe I can offer examples of what I can do, like answering questions, providing information, or helping with tasks.\n\nI should also consider if there's any specific context they're referring to when they say \"test.\" Maybe they're testing my knowledge or my ability to follow instructions. In that case, I should confirm that I'm ready to assist and ask them to provide more details if needed.\n\nI need to make sure my response is clear and not too technical. The user might not be familiar with the capabilities of an AI assistant, so a friendly and approachable tone is important. Let me put that all together into a concise response.",
            "content": "Hello! How can I assist you today? 😊",
        },
    ]
)
tokenizer, collator = get_tokenizer_and_collator("Qwen/Qwen3-4B")
formatted = format_pir_data(tokenizer, example)
assert formatted["text"] == EXPECTED_TEXT, print(formatted["text"])
print("✅ Tokenization passed")

example2 = dict(messages=[{"role": "user", "content": "This is another test"}, {"role": "assistant", "reasoning_content": "this is a real test", "content": "this is not a real response"}])
example3 = dict(messages=[{"role": "system", "content": "just checking"}, {"role": "user", "content": "This is yet another test"}, {"role": "assistant", "reasoning_content": "this is a real test", "content": "this is another not a real response"}])

# TODO: check this for the actual trainer. Add special tokens needs to be false
batch = [tokenizer(format_pir_data(tokenizer, ex)['text'], add_special_tokens=False) for ex in [example, example2, example3]]
result = collator(batch)
for i, (input_ids, labels) in enumerate(zip(result['input_ids'], result['labels'])):
    print(f"\nBatch {i}:")
    for input_id, label, mask in zip(input_ids, labels, result['attention_mask'][i]):
        token = tokenizer.decode([input_id])
        if mask == 0:
            print(f"\033[94m{token}\033[0m", end="")  # Blue for masked tokens
        elif label == -100:
            print(f"\033[91m{token}\033[0m", end="")  # Red for -100
        else:
            print(f"\033[92m{token}\033[0m", end="")  # Green for other labels
    print()
