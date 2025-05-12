from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM


def get_tokenizer_and_collator(tokenizer_path: str, padding_free: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, padding_side="left", use_fast=True
    )
    if "R1" in tokenizer_path:
        response_template = "<｜Assistant｜><think>"
        # This is a hack to use a pad token that is not eos token
        # I believe DataCollator probably calculates mask by checking token == pad_token_id
        # so if pad_token_id is eos_token_id, it will cause the model to not predict eos
        tokenizer.pad_token = "<|video_pad|>"
    elif "Qwen3" in tokenizer_path:
        response_template = "<|im_start|>assistant\n"
        # No need to set pad token because Qwen sets it correctly
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer_path}")

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        padding_free=padding_free,
    )
    return tokenizer, collator
