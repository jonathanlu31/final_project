# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser
from colorama import init, Fore


def main(args: dict):
    # Pop arguments not used by LLM
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")

    # Create an LLM
    llm = LLM(**args)

    # Create sampling params object
    sampling_params = llm.get_default_sampling_params()
    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k

    init()

    system_message = input(f"{Fore.BLUE}System: {Fore.RESET}")
    conversation = [
        {"role": "system", "content": system_message},
    ]

    while True:
        user_message = input(f"{Fore.GREEN}User: {Fore.RESET}")
        conversation.append({"role": "user", "content": user_message})
        outputs = llm.chat(conversation, sampling_params, use_tqdm=False)
        asst_message = "<think>\n" + outputs[0].outputs[0].text
        print(f"{Fore.YELLOW}Assistant: {Fore.RESET}", asst_message)
        conversation.append({"role": "assistant", "content": asst_message})


if __name__ == "__main__":
    parser = FlexibleArgumentParser()

    engine_group = parser.add_argument_group("Engine arguments")
    EngineArgs.add_cli_args(engine_group)
    engine_group.set_defaults(
        model="/scratch/jonathan/research/reasoning_project/final_project/scripts/results/qwen_rules/checkpoint-400"
    )

    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int, default=2048)
    sampling_group.add_argument("--temperature", type=float, default=0.9)
    sampling_group.add_argument("--top-p", type=float, default=0.9)
    sampling_group.add_argument("--top-k", type=int, default=None)
    args: dict = vars(parser.parse_args())
    main(args)
