import json

with open("tokenizer_config.json", 'r') as config_file:
    config = json.load(config_file)

with open("chat_template.jinja", "r") as chat_file:
    chat_template = chat_file.read()

config['chat_template'] = chat_template

with open("tokenizer_config.json", 'w') as config_file:
    json.dump(config, config_file, indent=2)
