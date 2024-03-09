import pathlib
import json

import datasets
import huggingface_hub

# Create directories
pathlib.Path('model').mkdir(exist_ok=True)
pathlib.Path('data/sft').mkdir(exist_ok=True, parents=True)
pathlib.Path('data/dpo').mkdir(exist_ok=True, parents=True)
pathlib.Path('adapters/sft').mkdir(exist_ok=True, parents=True)
pathlib.Path('adapters/dpo').mkdir(exist_ok=True, parents=True)

# Download dataset
train_sft, train_dpo, valid_sft, valid_dpo, test_sft, test_dpo = datasets.load_dataset("argilla/dpo-mix-7k", split=['train[:45%]', 'train[45%:90%]', 'train[90%:95%]', 'train[95%:]', 'test[:50%]', 'test[50%:]'])

def store_dataset(dataset, fpath, sft=True):
    fpath = pathlib.Path(fpath)

    if not fpath.exists():
        with open(fpath, 'w') as f:
            for entry in dataset:
                if len(entry['chosen']) != 2 or len(entry['rejected']) != 2:
                    continue

                if sft:
                    result = {
                        "prompt": entry["chosen"][0]["content"],
                        "response": entry["chosen"][1]["content"]
                    }
                else:
                    result = {
                        "prompt": entry["chosen"][0]["content"],
                        "chosen": entry["chosen"][1]["content"],
                        "rejected": entry["rejected"][1]["content"]
                    }

                f.write(json.dumps(result) + "\n")
    else:
        print(f"Dataset file {fpath} already exists, skipping")

# Store SFT dataset splits
store_dataset(train_sft, 'data/sft/train.jsonl', sft=True)
store_dataset(valid_sft, 'data/sft/valid.jsonl', sft=True)
store_dataset(test_sft, 'data/sft/test.jsonl', sft=True)

# Store DPO dataset splits
store_dataset(train_dpo, 'data/dpo/train.jsonl', sft=False)
store_dataset(valid_dpo, 'data/dpo/valid.jsonl', sft=False)
store_dataset(test_dpo, 'data/dpo/test.jsonl', sft=False)

print("Downloaded dataset to data/")

# Download model files
def download_hf_file(repo_id, filename):
    huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename, local_dir="model/", local_dir_use_symlinks=False)
    
model_files = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors"
]

for filename in model_files:
    fpath = pathlib.Path("model") / filename
    if not fpath.exists():
        download_hf_file("Qwen/Qwen1.5-7B-Chat", filename)
    else:
        print(f"Model file {fpath} already exists, skipping")

print("Downloaded model files to model/")