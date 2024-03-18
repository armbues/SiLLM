import pathlib
import json

import datasets

# Create directories
pathlib.Path('data/').mkdir(exist_ok=True)

fpath = pathlib.Path("data") / "wikipedia.jsonl"
if fpath.exists():
    print(f"Dataset file {fpath} already exists, skipping")
else:
    # Initialize dataset stream
    wikipedia = datasets.load_dataset("Cohere/wikipedia-2023-11-embed-multilingual-v3", "en", split="train", streaming=True)

    with open(fpath, 'w') as f:
        num_entries = 0
        for doc in wikipedia:
            doc_id = doc['_id']

            if doc_id.endswith('_0'):
                entry = {
                    "text": doc['text']
                }
                f.write(json.dumps(entry) + "\n")
                
                num_entries += 1
                if num_entries >= 1000:
                    break

    print(f"Downloaded dataset to {fpath}")