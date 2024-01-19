import pathlib
import json
import logging

class Dataset:
    """
    Dataset wrapper.
    """
    def __init__(self,
                 tokenizer,
                 dataset_path,
                 key: str = "text",
                 max_length: int = 4096
                 ):
        """
        Args:
            tokenizer: Tokenizer to use.
            dataset_path: Path to dataset file.
            key: Key to use for text.
            max_length: Max token length per training entry.
        """
        self._key = key
        self._data = []

        if pathlib.Path(dataset_path).exists():
            with open(dataset_path, "r") as f:
                for line in f:
                    entry = json.loads(line)
                    text = entry[key]
                    tokens = tokenizer.encode(text, eos=True)

                    if len(tokens) < max_length:
                        self._data.append(tokens)

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __len__(self):
        return len(self._data)
    
    @staticmethod
    def load(tokenizer, dataset_dir, key="text", max_length=4096):
        """
        Load dataset from JSONL file.
        Args:
            tokenizer: Tokenizer to use.
            dataset_dir: Directory with dataset files.
            key: Key to use for text.
            max_length: Max token length of text.
        Returns:
            Training, validation, and test datasets.
        """
        datasets = []
        for name in ("train", "valid", "test"):
            dataset_path = pathlib.Path(dataset_dir) / f"{name}.jsonl"
            dataset = Dataset(tokenizer, dataset_path, key, max_length)
            datasets.append(dataset)

            logging.info(f"Loaded {name} dataset with {len(dataset)} entries from {dataset_path}")

        return datasets