from reasoning.data import format_pir_data
from reasoning import get_tokenizer_and_collator

EXPECTED_TEXT = """\
<｜begin▁of▁sentence｜><｜User｜>This is a test<｜Assistant｜><think>
Alright, so I'm trying to figure out what this test is about. The user just sent me "This is a test." It seems pretty straightforward, but I need to dig deeper to understand exactly what they're looking for. Maybe they're testing if I can recognize when they're just sharing a simple message, or perhaps they're trying to see how I handle it. 

First, I should consider the context. The message is very short and doesn't contain any specific instructions or questions. It could be that the user is trying to assess my ability to handle minimalist inputs. Alternatively, they might be checking if I can identify that the message doesn't require a detailed response. 

I wonder if there's more to it. Perhaps the user is experimenting with how I respond to different types of messages. They might be interested in seeing if I can flag it as a test or if I can provide additional information based on the lack of a question. 

Another angle is whether the user is testing my language model capabilities. Maybe they want to see if I can recognize the simplicity of the task or if I can generate a response that acknowledges the test nature of the message. 

I should also think about the possibility of the user being a developer or someone working on an interface, trying to see how the AI handles user inputs. They might be evaluating the system's ability to differentiate between meaningful queries and trivial messages. 

In terms of my response, I need to be clear and helpful. I should confirm that I recognize the test and perhaps offer assistance if they have a follow-up question or need something else. It's important to be polite and open-ended to encourage further interaction.

Wait, maybe the user is actually testing the system's ability to handle ambiguous or incomplete inputs. They might be assessing how well the AI can infer the user's intent from a simple message. In that case, my response should reflect an understanding of the ambiguity and offer a helpful reply.

I should also consider if the user is trying to see if I can generate a response without additional information. Since the message is so short, it's a bit challenging, but I can still provide a structured answer that acknowledges the test and invites further interaction.

Another thought: perhaps the user is using this as a way to see if I can recognize when a message is a test or a placeholder. They might be evaluating the AI's ability to maintain context and provide relevant responses even with minimal input.

In summary, my approach should be to acknowledge the test, confirm understanding, and offer assistance. This way, I cover all bases, whether the user is testing the system or simply looking for a confirmation. I'll make sure my response is clear, helpful, and opens the door for them to provide more details if needed.
</think>

The message "This is a test." is a straightforward test, likely checking if the system can recognize minimalist inputs. My response is acknowledging this and inviting further interaction:

---

Thank you for reaching out! It seems like you're testing the system. If you have any specific questions or need assistance, feel free to ask. I'm here to help! 😊

--- 

This response confirms understanding, offers assistance, and maintains an open and friendly tone.<｜end▁of▁sentence｜>"""

example = dict(
    messages=[
        {"role": "user", "content": "This is a test"},
        {
            "role": "assistant",
            "reasoning_content": "Alright, so I'm trying to figure out what this test is about. The user just sent me \"This is a test.\" It seems pretty straightforward, but I need to dig deeper to understand exactly what they're looking for. Maybe they're testing if I can recognize when they're just sharing a simple message, or perhaps they're trying to see how I handle it. \n\nFirst, I should consider the context. The message is very short and doesn't contain any specific instructions or questions. It could be that the user is trying to assess my ability to handle minimalist inputs. Alternatively, they might be checking if I can identify that the message doesn't require a detailed response. \n\nI wonder if there's more to it. Perhaps the user is experimenting with how I respond to different types of messages. They might be interested in seeing if I can flag it as a test or if I can provide additional information based on the lack of a question. \n\nAnother angle is whether the user is testing my language model capabilities. Maybe they want to see if I can recognize the simplicity of the task or if I can generate a response that acknowledges the test nature of the message. \n\nI should also think about the possibility of the user being a developer or someone working on an interface, trying to see how the AI handles user inputs. They might be evaluating the system's ability to differentiate between meaningful queries and trivial messages. \n\nIn terms of my response, I need to be clear and helpful. I should confirm that I recognize the test and perhaps offer assistance if they have a follow-up question or need something else. It's important to be polite and open-ended to encourage further interaction.\n\nWait, maybe the user is actually testing the system's ability to handle ambiguous or incomplete inputs. They might be assessing how well the AI can infer the user's intent from a simple message. In that case, my response should reflect an understanding of the ambiguity and offer a helpful reply.\n\nI should also consider if the user is trying to see if I can generate a response without additional information. Since the message is so short, it's a bit challenging, but I can still provide a structured answer that acknowledges the test and invites further interaction.\n\nAnother thought: perhaps the user is using this as a way to see if I can recognize when a message is a test or a placeholder. They might be evaluating the AI's ability to maintain context and provide relevant responses even with minimal input.\n\nIn summary, my approach should be to acknowledge the test, confirm understanding, and offer assistance. This way, I cover all bases, whether the user is testing the system or simply looking for a confirmation. I'll make sure my response is clear, helpful, and opens the door for them to provide more details if needed.",
            "content": "The message \"This is a test.\" is a straightforward test, likely checking if the system can recognize minimalist inputs. My response is acknowledging this and inviting further interaction:\n\n---\n\nThank you for reaching out! It seems like you're testing the system. If you have any specific questions or need assistance, feel free to ask. I'm here to help! 😊\n\n--- \n\nThis response confirms understanding, offers assistance, and maintains an open and friendly tone.",
        },
    ]
)
tokenizer, collator = get_tokenizer_and_collator("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
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
