import pathlib
import json
import logging

import numpy as np

import mlx.core as mx

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
                    # TODO warning if text is too long
        else:
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __len__(self):
        return len(self._data)
    
    ########
    # Based on mlx-examples:
    # https://github.com/ml-explore/mlx-examples/blob/e74889d0fa0fb49d95bfdf6a1dcad907713eb50e/lora/lora.py#L166
    ########
    def iterate_batches(self,
                        batch_size: int,
                        train: bool = False):
        """
        Iterate over batches.
        Args:
            batch_size: Batch size.
            train: Whether to train.
        """
        # Shuffle indices
        while True:
            indices = np.arange(len(self._data))
            if train:
                indices = np.random.permutation(indices)

            # Collect batches from dataset
            for i in range(0, len(indices) - batch_size + 1, batch_size):
                batch = [self._data[indices[i+j]] for j in range(batch_size)]
                lengths = [len(x) for x in batch]

                # Pad to the max length
                batch_arr = np.zeros((batch_size, max(lengths)), np.int32)
                for j in range(batch_size):
                    batch_arr[j, : lengths[j]] = batch[j]
                batch = mx.array(batch_arr)

                yield batch[:, :-1], batch[:, 1:], mx.array(lengths)

            if not train:
                break
    
    @staticmethod
    def load(tokenizer,
             dataset_dir,
             key="text",
             max_length=4096):
        """
        Load datasets from a directory with JSONL files.
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
    
class DatasetDPO(Dataset):
    """
    DPO dataset wrapper.
    """
    def __init__(self,
                 tokenizer,
                 dataset_path,
                 prompt_key: str = "prompt",
                 chosen_key: str = "chosen",
                 rejected_key: str = "rejected",
                 max_length: int = 4096
                 ):
        """
        Args:
            tokenizer: Tokenizer to use.
            dataset_path: Path to dataset file.
            key: Key to use for text.
            max_length: Max token length per training entry.
        """
        self._keys = {
            "chosen": chosen_key,
            "rejected": rejected_key
        }
        self._data = []

        if pathlib.Path(dataset_path).exists():
            with open(dataset_path, "r") as f:
                for line in f:
                    entry = json.loads(line)

                    tokens_prompt = tokenizer.encode(entry[prompt_key])
                    tokens_chosen = tokens_prompt + tokenizer.encode(entry[chosen_key], bos=False, eos=True)
                    tokens_rejected = tokens_prompt + tokenizer.encode(entry[rejected_key], bos=False, eos=True)

                    if len(tokens_chosen) < max_length and len(tokens_rejected) < max_length:
                        self._data.append((tokens_prompt, tokens_chosen, tokens_rejected))
                    # TODO warning if text is too long
        else:
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        self.pad_id = tokenizer.pad_id if tokenizer.pad_id >= 0 else tokenizer.eos_id

    def iterate_batches(self,
                        batch_size: int,
                        train: bool = False):
        """
        Iterate over batches.
        Args:
            batch_size: Batch size (unused).
            train: Whether to train.
        """
        # Shuffle indices
        while True:
            indices = np.arange(len(self._data))
            if train:
                indices = np.random.permutation(indices)

            # Collect batches from dataset
            for i in range(0, len(indices) - batch_size + 1, batch_size):
                batch = [self._data[indices[i+j]] for j in range(batch_size)]
                prompt_lengths = [len(x[0]) for x in batch]
                chosen_lengths = [len(x[1]) for x in batch]
                rejected_lengths = [len(x[2]) for x in batch]

                # Pad to the max length
                chosen_arr = np.full((batch_size, max(chosen_lengths)), self.pad_id, np.int32)
                rejected_arr = np.full((batch_size, max(rejected_lengths)), self.pad_id, np.int32)
                for j in range(batch_size):
                    chosen_arr[j, : chosen_lengths[j]] = batch[j][1]
                    rejected_arr[j, : rejected_lengths[j]] = batch[j][2]
                
                yield mx.array(chosen_arr), mx.array(rejected_arr), mx.array(prompt_lengths), mx.array(chosen_lengths), mx.array(rejected_lengths)

            if not train:
                break

    @staticmethod
    def load(tokenizer,
             dataset_dir,
             prompt_key: str = "prompt",
             chosen_key: str = "chosen",
             rejected_key: str = "rejected",
             max_length=4096):
        """
        Load datasets from a directory with JSONL files.
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
            dataset = DatasetDPO(tokenizer, dataset_path, prompt_key, chosen_key, rejected_key, max_length)
            datasets.append(dataset)

            logging.info(f"Loaded {name} dataset with {len(dataset)} entries from {dataset_path}")

        return datasets