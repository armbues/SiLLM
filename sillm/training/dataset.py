import pathlib
import json
import logging

import numpy as np

import mlx.core as mx

logger = logging.getLogger("sillm")

class Dataset:
    """
    Dataset wrapper.
    """
    def __init__(self,
                 entries,
                 tokenizer,
                 template = None,
                 max_length: int = 4096
                 ):
        """
        Args:
            tokenizer: Tokenizer to use.
            dataset_path: Path to dataset file.
            key: Key to use for text.
            max_length: Max token length per training entry.
        """
        raise NotImplementedError("Class Dataset is used for inheritance only")
                 
    def __getitem__(self, idx: int):
        return self._data[idx]

    def __len__(self):
        return len(self._data)
    
class DatasetCompletion(Dataset):
    """
    Completion dataset wrapper.
    """
    def __init__(self,
                 entries,
                 tokenizer,
                 template = None,
                 max_length: int = 4096
                 ):
        """
        Args:
            tokenizer: Tokenizer to use.
            dataset_path: Path to dataset file.
            key: Key to use for text.
            max_length: Max token length per training entry.
        """
        self._type_id = "Completion"
        self._data = []
        if tokenizer.pad_id is not None and tokenizer.pad_id >= 0:
            self.pad_id = tokenizer.pad_id
        else:
            self.pad_id = tokenizer.eos_id

        key = "text"

        num_outsized = 0
        for entry in entries:
            tokens = tokenizer.encode(entry[key], eos=True)

            if len(tokens) < max_length:
                self._data.append(tokens)
            else:
                num_outsized += 1

        if num_outsized > 0:
            logger.debug(f"Removed {num_outsized} entries from dataset due to max. length {max_length}")
    
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
                batch_arr = np.full((batch_size, max(lengths)), self.pad_id, np.int32)
                for j in range(batch_size):
                    batch_arr[j, : lengths[j]] = batch[j]
                batch = mx.array(batch_arr)

                inputs = batch[:, :-1]
                targets = batch[:, 1:]

                lengths = mx.array(lengths)
                loss_masks = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

                yield inputs, targets, loss_masks

            if not train:
                break
    
class DatasetMessages(Dataset):
    """
    Messages dataset wrapper.
    """
    def __init__(self,
                 entries,
                 tokenizer,
                 template,
                 max_length: int = 4096
                 ):
        """
        Args:
            tokenizer: Tokenizer to use.
            dataset_path: Path to dataset file.
            key: Key to use for text.
            max_length: Max token length per training entry.
        """
        if template is None:
            raise ValueError("Template must be specified for Messages dataset")

        self._type_id = "Messages"
        self._data = []
        if tokenizer.pad_id is not None and tokenizer.pad_id >= 0:
            self.pad_id = tokenizer.pad_id
        else:
            self.pad_id = tokenizer.eos_id

        messages_key = "messages"

        num_outsized = 0
        for entry in entries:
            messages = entry[messages_key]

            text_prompt = template.apply_chat_template(messages[:-1])
            tokens_prompt = tokenizer.encode(text_prompt)
            
            text = template.apply_chat_template(messages)
            tokens = tokenizer.encode(text, eos=True)

            if len(tokens) < max_length:
                self._data.append((tokens_prompt, tokens))
            else:
                num_outsized += 1

        if num_outsized > 0:
            logger.debug(f"Removed {num_outsized} entries from dataset due to max. length {max_length}")

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
                prompts = [self._data[indices[i+j]][0] for j in range(batch_size)]
                batch = [self._data[indices[i+j]][1] for j in range(batch_size)]

                # Pad to the max length
                lengths = [len(x) for x in batch]
                batch_arr = np.full((batch_size, max(lengths)), self.pad_id, np.int32)
                for j in range(batch_size):
                    batch_arr[j, : lengths[j]] = batch[j]
                batch = mx.array(batch_arr)

                inputs = batch[:, :-1]
                targets = batch[:, 1:]

                lengths = mx.array(lengths)
                prompt_lengths = mx.array([len(x) for x in prompts])
                loss_masks = mx.logical_and(
                    mx.arange(inputs.shape[1])[None, :] < lengths[:, None],
                    mx.arange(inputs.shape[1])[None, :] > prompt_lengths[:, None]
                )

                yield inputs, targets, loss_masks

            if not train:
                break

class DatasetInstruct(DatasetMessages):
    """
    Q&A dataset wrapper.
    """
    def __init__(self,
                 entries,
                 tokenizer,
                 template = None,
                 max_length: int = 4096
                 ):
        """
        Args:
            tokenizer: Tokenizer to use.
            dataset_path: Path to dataset file.
            key: Key to use for text.
            max_length: Max token length per training entry.
        """
        self._type_id = "Instruct"
        self._data = []
        if tokenizer.pad_id is not None and tokenizer.pad_id >= 0:
            self.pad_id = tokenizer.pad_id
        else:
            self.pad_id = tokenizer.eos_id

        prompt_key = "prompt"
        response_key = "response"

        num_outsized = 0
        for entry in entries:
            prompt = entry[prompt_key]
            response = entry[response_key]
            if template is None:
                tokens_prompt = tokenizer.encode(prompt)
                tokens = tokenizer.encode(prompt + response, eos=True)
            else:
                messages_prompt = [
                    { "role": "user", "content": prompt}
                ]
                tokens_prompt = tokenizer.encode(template.apply_chat_template(messages_prompt))

                messages_response = [
                    { "role": "user", "content": prompt},
                    { "role": "assistant", "content": response }
                ]
                tokens = tokenizer.encode(template.apply_chat_template(messages_response), eos=True)

            if len(tokens) < max_length:
                self._data.append((tokens_prompt, tokens))
            else:
                num_outsized += 1

        if num_outsized > 0:
            logger.debug(f"Removed {num_outsized} entries from dataset due to max. length {max_length}")

class DatasetDPO(Dataset):
    """
    DPO dataset wrapper.
    """
    def __init__(self,
                 entries,
                 tokenizer,
                 template = None,
                 max_length: int = 4096
                 ):
        """
        Args:
            tokenizer: Tokenizer to use.
            dataset_path: Path to dataset file.
            key: Key to use for text.
            max_length: Max token length per training entry.
        """
        self._type_id = "DPO"
        self._data = []
        if tokenizer.pad_id is not None and tokenizer.pad_id >= 0:
            self.pad_id = tokenizer.pad_id
        else:
            self.pad_id = tokenizer.eos_id

        prompt_key = "prompt"
        chosen_key = "chosen"
        rejected_key = "rejected"

        num_outsized = 0
        for entry in entries:
            prompt = entry[prompt_key]
            chosen = entry[chosen_key]
            rejected = entry[rejected_key]
            if template is None:
                tokens_prompt = tokenizer.encode(prompt)
                tokens_chosen = tokenizer.encode(prompt+chosen, eos=True)
                tokens_rejected = tokenizer.encode(prompt+rejected, eos=True)
            else:
                messages_prompt = [
                    { "role": "user", "content": prompt}
                ]
                tokens_prompt = tokenizer.encode(template.apply_chat_template(messages_prompt))

                messages_chosen = [
                    { "role": "user", "content": prompt},
                    { "role": "assistant", "content": chosen}
                ]
                tokens_chosen = tokenizer.encode(template.apply_chat_template(messages_chosen), eos=True)

                messages_rejected = [
                    { "role": "user", "content": prompt},
                    { "role": "assistant", "content": rejected}
                ]
                tokens_rejected = tokenizer.encode(template.apply_chat_template(messages_rejected), eos=True)

            if len(tokens_chosen) < max_length and len(tokens_rejected) < max_length:
                self._data.append((tokens_prompt, tokens_chosen, tokens_rejected))
            else:
                num_outsized += 1
        
        if num_outsized > 0:
            logger.debug(f"Removed {num_outsized} entries from dataset due to max. length {max_length}")

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
                chosen_lengths = [len(x[1]) for x in batch]
                rejected_lengths = [len(x[2]) for x in batch]

                # Pad to the max length
                chosen = np.full((batch_size, max(chosen_lengths)), self.pad_id, np.int32)
                rejected = np.full((batch_size, max(rejected_lengths)), self.pad_id, np.int32)
                for j in range(batch_size):
                    chosen[j, : chosen_lengths[j]] = batch[j][1]
                    rejected[j, : rejected_lengths[j]] = batch[j][2]

                # Mask prompt and padding tokens
                chosen_lengths = mx.array(chosen_lengths)
                rejected_lengths = mx.array(rejected_lengths)
                prompt_lengths = mx.array([len(x[0]) for x in batch])
                chosen_masks = mx.logical_and(
                    mx.arange(chosen.shape[1] - 1)[None, :] < chosen_lengths[:, None],
                    mx.arange(chosen.shape[1] - 1)[None, :] > prompt_lengths[:, None]
                )
                rejected_masks = mx.logical_and(
                    mx.arange(rejected.shape[1] - 1)[None, :] < rejected_lengths[:, None],
                    mx.arange(rejected.shape[1] - 1)[None, :] > prompt_lengths[:, None]
                )
                
                yield mx.array(chosen), mx.array(rejected), chosen_masks, rejected_masks

            if not train:
                break
    
def load_jsonl(fpath,
               shuffle: bool = True
               ):
    entries = []
    with open(fpath, "r") as f:
        for line in f:
            entry = json.loads(line)
            entries.append(entry)

    if shuffle:
        np.random.shuffle(entries)

    return entries
    
def load_dataset(tokenizer,
                 dataset_path,
                 train_split: float = 0.9,
                 valid_split: float = 0.05,
                 test_split: float = 0.05,
                 template = None,
                 max_length: int = 4096,
                 shuffle: bool = True
                 ):
    dataset_path = pathlib.Path(dataset_path)

    def guess_type(entry):
        if "text" in entry:
            return DatasetCompletion
        elif "messages" in entry:
            return DatasetMessages
        elif "prompt" in entry and "chosen" in entry and "rejected" in entry:
            return DatasetDPO
        elif "prompt" in entry and "response" in entry:
            return DatasetInstruct
        else:
            entry_keys = list(entry.keys())
            raise ValueError(f"Unknown dataset type with keys: {', '.join(entry_keys)}")

    if dataset_path.is_file():
        entries = load_jsonl(dataset_path, shuffle=shuffle)

        assert train_split + valid_split + test_split == 1.0, "Dataset splits must sum to 1.0"
        ix_train = int(len(entries) * train_split)
        ix_valid = int(len(entries) * (train_split + valid_split))
        ix_test = int(len(entries) * (train_split + valid_split + test_split))

        entries_split = {
            "train": entries[:ix_train],
            "valid": entries[ix_train:ix_valid],
            "test": entries[ix_valid:ix_test]
        }
    elif dataset_path.is_dir():
        entries_split = {}

        for name in ("train", "valid", "test"):
            dataset_file = dataset_path / f"{name}.jsonl"
            if dataset_file.exists():
                entries_split[name] = load_jsonl(dataset_file)
            else:
                raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    else:
        raise FileNotFoundError(f"Dataset file or directory not found: {dataset_path}")

    # Guess dataset type
    dataset_class = guess_type(entries_split["train"][0])

    # Load datasets
    datasets = []
    for name in ("train", "valid", "test"):
        dataset = dataset_class(entries_split[name], tokenizer, template=template, max_length=max_length)
        datasets.append(dataset)

        logger.info(f"Loaded {name} split for {dataset._type_id} dataset with {len(dataset)} entries")

    return datasets