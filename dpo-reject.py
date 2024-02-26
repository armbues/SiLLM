import sys
import json

import sillm

TEMPLATES = {
    "llama-2": {
        "system":       "[INST] <<SYS>>\n{0}\n<</SYS>>\n\n",
        "user":         "[INST]{0}[/INST] ",
        "initial":      "{0}[/INST] ",
        "assistant":    ""
    },
    "chatml": {
        "system":       "<|im_start|>system\n{0}<|im_end|>\n",
        "user":         "<|im_start|>user\n{0}<|im_end|>\n",
        "assistant":    "<|im_start|>assistant\n"
    },
    "alpaca": {
        "system":       "### System Prompt\n{0}\n\n",
        "user":         "### User Message\n{0}\n\n",
        "assistant":    "### Assistant\n"
    },
    "gemma": {
        "system":       "",
        "user":         "<start_of_turn>user\n{0}<end_of_turn>\n",
        "assistant":    "<start_of_turn>model\n"
    }
}

template = TEMPLATES["gemma"]

entries = []
with open(sys.argv[1], "r") as f:
    for line in f:
        entry = json.loads(line)

        result = {
            "_prompt": entry["prompt"],
            "prompt": template["user"].format(entry["prompt"]) + template["assistant"],
            "chosen": entry["chosen"],
            "rejected": ""
        }

        entries.append(result)

model = sillm.load(sys.argv[2])

with open(sys.argv[3], "w") as f:
    for entry in entries:
        print("##################################################")
        prompt = entry["prompt"]
        print(prompt)

        rejected = model.completion(prompt, num_tokens=512)
        entry["rejected"] = rejected
        print(rejected)

        if rejected.startswith("I "):
            f.write(json.dumps(entry) + "\n")