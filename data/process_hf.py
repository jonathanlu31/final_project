from collections import defaultdict
import json
from llm_rules import Message, Role, scenarios
from importlib import resources
import os

from datasets import Dataset, DatasetDict
def build_initial_messages(
    scenario: scenarios.BaseScenario,
    test_messages: list[Message],
    use_system_instructions: bool,
    remove_precedence_reminders: bool,
):
    """
    Build the initial conversation (system/user messages) for one test case.
    Handle removing precedence reminders, skipping pre-filled assistant messages, etc.

    Returns:
      - The initial list of messages
      - The index of the next user message (in test_messages) that needs an assistant response
    """
    full_prompt = scenario.prompt
    if remove_precedence_reminders:
        full_prompt = scenarios.utils.remove_precedence_reminders(full_prompt)

    messages_so_far = [Message(Role.SYSTEM, full_prompt)]

    # Skip over prefilled assistant messages
    assistant_indices = [
        i for i, m in enumerate(test_messages) if m.role == Role.ASSISTANT
    ]
    next_user_idx = 0
    if assistant_indices:
        print('asst')
        last_assistant_idx = max(assistant_indices)
        # Add all messages up to that assistant index
        messages_so_far += test_messages[: last_assistant_idx + 1]
        next_user_idx = last_assistant_idx + 1

    return messages_so_far, next_user_idx

def load_dataset():
    """Load all *.jsonl test files from the specified test_suite package."""
    dataset = defaultdict(dict)
    testsuite = "redteam"
    dir_path = resources.files("llm_rules").joinpath("data").joinpath(testsuite)

    files = [
        f.name
        for f in dir_path.iterdir()
        if f.name.endswith(".jsonl")
    ]
    files = sorted(files)

    for file in files:
        print("Loading: {}".format(file))
        scenario_name = os.path.splitext(file)[0]
        behavior_name = ""
        if "_" in scenario_name:
            scenario_name, behavior_name = scenario_name.split("_")

        with dir_path.joinpath(file).open() as f:
            testcases = [json.loads(line) for line in f.readlines()]

            for t in testcases:
                if "category" not in t:
                    t["category"] = "default"
                    print('cat')
                if "id" not in t:
                    t["id"] = None
                    print('id')

            dataset[scenario_name][behavior_name] = testcases

    return dataset

dataset = load_dataset()
counts = defaultdict(int)
full_dataset = defaultdict()
printed_set = set()
for scenario_name in scenarios.SCENARIOS:
    for behavior_name in dataset[scenario_name]:
        for testcase in dataset[scenario_name][behavior_name]:
            scenario_cls = scenarios.SCENARIOS[scenario_name]
            scenario = scenario_cls(testcase["params"])
            test_messages = Message.unserialize(testcase["messages"])
            counts[len(test_messages)] += 1
            if len(test_messages) > 1 and (scenario_name, behavior_name) not in printed_set:
                printed_set.add((scenario_name, behavior_name))
                print(scenario_name, behavior_name)
            # Build initial conversation and figure out next user index
            # messages_so_far, next_user_idx = build_initial_messages(
            #     scenario,
            #     test_messages,
            #     use_system_instructions=True,
            #     remove_precedence_reminders=False,
            # )
        
print(counts)
# hf_dataset = DatasetDict({
#     task_name: Dataset.from_dict(full_dataset[task_name]) for task_name in full_dataset
# })

# hf_dataset.push_to_hub("jonluj/llm_rules_redteam", private=True)
