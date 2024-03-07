import sys
import pathlib
import logging
import json

import datasets
import huggingface_hub

logging.basicConfig(level=20, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")

# Create directories
pathlib.Path('data/').mkdir(exist_ok=True)
pathlib.Path('model/').mkdir(exist_ok=True)
pathlib.Path('adapters/').mkdir(exist_ok=True)

# Download dataset
train, valid, test = datasets.load_dataset("nvidia/HelpSteer", split=['train', 'validation[:50%]', 'validation[-50%:]'])

def store_dataset(dataset, fpath):
    fpath = pathlib.Path(fpath)

    if not fpath.exists():
        with open(fpath, 'w') as f:
            for entry in dataset:
                f.write(json.dumps(entry) + "\n")
    else:
        logging.info(f"Dataset file {fpath} already exists, skipping")

# Store dataset splits
store_dataset(train, 'data/train.jsonl')
store_dataset(valid, 'data/valid.jsonl')
store_dataset(test, 'data/test.jsonl')

logging.info("Downloaded dataset to data/")

# Download model files
def download_hf_file(repo_id, filename):
    huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename, local_dir="model/", local_dir_use_symlinks=False)
    
model_files = [
    "config.json",
    "tokenizer.model",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors"
]

for filename in model_files:
    fpath = pathlib.Path("model") / filename
    if not fpath.exists():
        download_hf_file("mistralai/Mistral-7B-Instruct-v0.2", filename)
    else:
        logging.info(f"Model file {fpath} already exists, skipping")

logging.info("Downloaded model files to model/")