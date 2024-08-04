import logging
import random

import tqdm

import numpy as np
from sklearn.decomposition import PCA

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from sillm.core.llm import LLM

logger = logging.getLogger("sillm")

def save_vectors(fpath: str,
                 vectors: dict,
                 metadata: dict = {}):
    """
    Save control vectors to a file.
    Args:
        fpath: File path.
        vectors: Control vectors.
        metadata: Metadata.
    """
    mx.save_safetensors(fpath, vectors, metadata=metadata)

def load_vectors(fpath: str):
    """
    Load control vectors from a file.
    Args:
        fpath: File path.
    Returns:
        Control vectors.
    """
    return mx.load(fpath, return_metadata=True)

class ContrastDataset:
    """
    Contrast dataset for control vector training.
    """
    def __init__(self,
                 tokenizer,
                 positive: list,
                 negative: list
                 ):
        self.tokenizer = tokenizer
        self._positive = positive
        self._negative = negative
    
    def iterate(self):
        """
        Iterate over contrast pairs.
        """
        pad_id = self.tokenizer.pad_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_id

        while True:
            pos = random.choice(self._positive)
            neg = random.choice(self._negative)

            lengths = [len(pos), len(neg)]
            max_length = max(lengths)

            batch_arr = np.full((2, max(lengths)), pad_id, np.int32)
            batch_arr[0, max_length-lengths[0]:] = pos
            batch_arr[1, max_length-lengths[1]:] = neg

            yield mx.array(batch_arr)
    
class ConversationContrastDataset(ContrastDataset):
    """
    Conversation dataset with positive and negative prompts for control vector training.
    """
    def __init__(self,
                 tokenizer,
                 template,
                 positive: list,
                 negative: list,
                 responses: list
                 ):
        self.tokenizer = tokenizer

        def get_entry(prompt, response):
            # Apply chat template and tokenize
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            text = template.apply_chat_template(messages)

            return self.tokenizer.encode(text)

        self._positive = []
        self._negative = []
        for response in responses:
            for prompt in positive:
                pos = get_entry(prompt, response)
                self._positive.append(pos)
            for prompt in negative:
                neg = get_entry(prompt, response)
                self._negative.append(neg)

class SystemContrastDataset(ContrastDataset):
    """
    Conversation dataset with a system message for control vector training.
    """
    def __init__(self,
                 tokenizer,
                 template,
                 positive: list,
                 negative: list,
                 prompts: list,
                 responses: list
                 ):
        self.tokenizer = tokenizer

        if len(positive) == 0:
            positive = [None]
        if len(negative) == 0:
            negative = [None]

        def get_entry(messages):
            # Apply chat template and tokenize
            text = template.apply_chat_template(messages)

            return self.tokenizer.encode(text)

        self._positive = []
        self._negative = []
        for response in responses:
            for prompt in prompts:
                messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
                for system in positive:
                    messages[0]["content"] = system
                    if system is None:
                        pos = get_entry(messages[1:])
                    else:
                        pos = get_entry(messages)
                    self._positive.append(pos)
                for system in negative:
                    messages[0]["content"] = system
                    if system is None:
                        neg = get_entry(messages[1:])
                    else:
                        neg = get_entry(messages)
                    self._negative.append(neg)

class ControlModule(nn.Module):
    """
    Control module wrapper.
    """
    @staticmethod
    def from_module(module: nn.Module):
        """
        Create control module wrapper for a given module.
        Args:
            module: Module to wrap.
        Returns:
            Control module wrapper.
        """
        control = ControlModule()
        control.module = module
        control.name = module.name

        return control

    def __init__(self):
        super().__init__()

        self.hidden_states = None

        self.capture = False
        self.mode = 'output'
        self.vector = None
        self.alpha = 1.0
        self.beta = -1.0

    def hook(self, func, x, *args, **kwargs):
        """
        Hook function for forward pass.
        Stores hidden states during training and applies control vector during inference.
        """
        output = func(x, *args, **kwargs)

        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output

        # Store hidden states
        if self.capture:
            if self.mode == "input":
                mx.eval(x)
                self.hidden_states = x
            elif self.mode == "output":
                mx.eval(h)
                self.hidden_states = h

        # Apply control vector
        if self.vector is not None:
            if self.vector.dtype != h.dtype:
                self.vector = self.vector.astype(h.dtype)

            if self.mode == 'output':
                norm_pre = mx.linalg.norm(h, axis=-1, keepdims=True)

                h += self.alpha * self.vector

                norm_post = mx.linalg.norm(h, axis=-1, keepdims=True)
                norm = norm_pre / norm_post
                norm = mx.where(mx.isnan(norm), 1.0, norm)
                
                h *= norm
            elif self.mode == 'input':
                vector = self.vector * self.alpha
                vector_reshaped = mx.reshape(vector, (-1, 1)).astype(h.dtype)
                proj = (h @ vector_reshaped) * vector

                h += proj * self.beta
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
        else:
            return output
            
        if isinstance(output, tuple):
            return (h,) + output[1:]
        else:
            return h
    
    def __call__(self, x, *args, **kwargs):
        return self.hook(self.module, x, *args, **kwargs)
    
    def forward(self, x, *args, **kwargs):
        return self.hook(self.module.forward, x, *args, **kwargs)
    
class ControlledLLM(LLM):
    """
    LLM model control vector wrapper.
    """
    @staticmethod
    def from_model(llm: LLM):
        return ControlledLLM(llm)
    
    def __init__(self, llm: LLM):
        self.model = llm.model
        self.tokenizer = llm.tokenizer
        self.args = llm.args
        self._quantization = llm._quantization
        self._mapping = llm._mapping

        # Set module names
        for key, module in self.model.named_modules():
            module.name = key

        self._mode = None
        self._control_modules = []

    def get_module_index(self,
                         min_layer: int = 0,
                         max_layer: int = 0,
                         segment: str = None,
                         transformer: bool = False,
                         attention: bool = False,
                         feed_forward: bool = False,
                         attention_norm: bool = False,
                         ffn_norm: bool = False,
                         attention_output: bool = False,
                         ffn_down: bool = False
                         ):
        """
        Get the module index for the given layers and module types.
        Args:
            min_layer: Minimum layer index.
            max_layer: Maximum layer index.
            segment: Segment of layers (all/intermediate/core).
            transformer: Include transformer modules.
            attention: Include attention modules.
            feed_forward: Include feed forward modules.
            attention_norm: Include attention normalization modules.
            ffn_norm: Include feed forward normalization modules.
            attention_output: Include attention output modules.
            ffn_down: Include feed forward down modules.
        Returns:
            List of control module indices.
        """
        control_index = []

        if segment == "all":
            min_layer = 0
            max_layer = len(self.model.layers) - 1
        elif segment == "intermediate":
            min_layer = int(len(self.model.layers) * 0.25)
            max_layer = int(len(self.model.layers) * 0.75)
        elif segment == "core":
            min_layer = int(len(self.model.layers) * 0.33)
            max_layer = int(len(self.model.layers) * 0.66)

        if max_layer < 1:
            max_layer = len(self.model.layers) + max_layer

        assert min_layer >= 0 and min_layer <= len(self.model.layers)
        assert max_layer >= 0 and max_layer < len(self.model.layers)

        for i in range(min_layer, max_layer+1):
            if transformer:
                control_index.append(f"layers.{i}")
            if attention:
                control_index.append(f"layers.{i}.attention")
            if feed_forward:
                control_index.append(f"layers.{i}.feed_forward")
            if attention_norm and hasattr(self.model.layers[i], "attention_norm"):
                control_index.append(f"layers.{i}.attention_norm")
            if ffn_norm and hasattr(self.model.layers[i], "ffn_norm"):
                control_index.append(f"layers.{i}.ffn_norm")
            if attention_output and hasattr(self.model.layers[i].attention, "wo"):
                control_index.append(f"layers.{i}.attention.wo")
            if ffn_down and hasattr(self.model.layers[i].feed_forward, "w2"):
                control_index.append(f"layers.{i}.feed_forward.w2")

        return control_index
    
    def init_control(self,
                     control_index: list = [],
                     mode: str = 'output'
                     ):
        """
        Initialize control modules.
        Args:
            control_index: Index of control modules.
        """
        self._mode = mode

        if len(self._control_modules) > 0:
            logger.warning("Control modules already initialized")

        control_modules = {}
        control_tree = []

        for name, module in self.model.named_modules():
            if name in control_index:
                control_module = ControlModule.from_module(module)
                control_module.mode = mode
                control_modules[name] = control_module

                control_tree.append((name, control_module))

        self._control_modules = control_modules

        control_tree = tree_unflatten(control_tree)
        if 'layers' in control_tree and len(control_tree['layers']) != len(self.model.layers):
            control_tree['layers'] += [{} for _ in range(len(self.model.layers) - len(control_tree['layers']))]

        self.model.update_modules(control_tree)
        logger.debug(f"Initialized control modules: {', '.join(control_modules.keys())}")

    def reset(self):
        """
        Unwrap all control modules.
        """
        control_tree = []
        for name, module in self._control_modules.items():
            control_tree.append((name, module.module))
        
        control_tree = tree_unflatten(control_tree)
        if 'layers' in control_tree and len(control_tree['layers']) != len(self.model.layers):
            control_tree['layers'] += [{} for _ in range(len(self.model.layers) - len(control_tree['layers']))]

        self.model.update_modules(control_tree)

        self._control_modules = {}

    def capture(self,
                mode: bool = True
                ):
        """
        Enable or disable capturing module states.
        Args:
            mode: Enable or disable capturing.
        """
        for module in self._control_modules.values():
            module.capture = mode

            if mode is False:
                module.hidden_states = None
    
    def set_control_vectors(self,
                            control_vectors: dict,
                            alpha: float = 1.0,
                            beta: float = -1.0
                            ):
        """
        Set control vectors for control modules.
        Args:
            control_vectors: Dictionary of control vectors.
        """
        if len(self._control_modules) == 0:
            control_index = list(control_vectors.keys())
            self.init_control(control_index)

        for name, vector in control_vectors.items():
            if name in self._control_modules:
                module = self._control_modules[name]

                module.vector = vector
                module.alpha = alpha
                module.beta = beta
            else:
                logger.warning(f"Attempting to set control vector for missing module: {name}")

    def set_control_vector(self,
                           control_vector: mx.array,
                           alpha: float = 1.0,
                           beta: float = -1.0
                           ):
        """
        Set a single control vector for all control modules.
        Args:
            control_vector: Control vector.
            alpha: Control vector coefficient.
        """      
        for module in self._control_modules.values():
            module.vector = control_vector
            module.alpha = alpha
            module.beta = beta

    def save_control_vectors(self,
                             fpath: str
                             ):
        """
        Save control vectors to a file.
        Args:
            fpath: File path.
        """
        vectors = {}
        for name, module in self._control_modules.items():
            vectors[name] = module.vector

        metadata = {
            "mode": self._mode
        }

        save_vectors(fpath, vectors, metadata)

        logger.debug(f"Saved control vectors to: {fpath}")

    def load_control_vectors(self,
                             fpath: str
                             ):
        """
        Load control vectors from a file.
        Args:
            fpath: File path.
        """
        vectors, metadata = load_vectors(fpath)
        control_index = list(vectors.keys())

        self.init_control(control_index=control_index, mode=metadata["mode"])
        self.set_control_vectors(vectors)

        logger.debug(f"Loaded control vectors from: {fpath}")

    def set_coeff(self,
                  alpha: float = 1.0,
                  beta: float = -1.0
                  ):
        """
        Set control vector coefficients for all control modules.
        Args:
            alpha: Vector coefficient.
            beta: Projection coefficient.
        """
        for module in self._control_modules.values():
            module.alpha = alpha
            module.beta = beta

        logger.debug(f"Set control vector coefficients: alpha={alpha}, beta={beta}")

    def modify_weights(self,
                       vectors: dict,
                       alpha: float = 1.0,
                       beta: float = -1.0
                       ):
        """
        Modify model weights with the given direction.
        Args:
            vectors: Modification vectors.
            alpha: Vector coefficient.
            beta: Projection coefficient.
        """
        for name, module in tqdm.tqdm(self.model.named_modules()):
            if name in vectors:
                vector = vectors[name].astype(mx.float32) * alpha
                vector_reshaped = mx.reshape(vector, (-1, 1))

                weight = module.weight.astype(mx.float32)
                proj = (vector @ weight) * vector_reshaped

                module.weight += proj.astype(module.weight.dtype) * beta

        logger.debug(f"Modified weights for modules: {', '.join(vectors.keys())}")

    def train(self,
              dataset: ContrastDataset,
              method: str = "pca_center",
              steps: int = 100,
              stack: bool = False
              ):
        """
        Train control vectors.
        Args:
            dataset: Contrast dataset.
            method: Training method (pca_center, pca_diff, cluster_mean).
            steps: Training steps.
        Returns:
            Dictionary of directions.
        """
        hidden_states_pos = {}
        hidden_states_neg = {}
        if stack:
            hidden_states_pos["stack"] = []
            hidden_states_neg["stack"] = []
        else:
            for name in self._control_modules.keys():
                hidden_states_pos[name] = []
                hidden_states_neg[name] = []

        # Enable training mode to store hidden states
        self.capture(mode=True)

        logger.info("Evaluating forward pass of training data to capture hidden states")
        for _ in tqdm.tqdm(range(steps)):
            batch = next(dataset.iterate())
            self.model(batch)

            for name, module in self._control_modules.items():
                if stack:
                    name = "stack"
                
                hidden_states_pos[name] += module.hidden_states[0].tolist()
                hidden_states_neg[name] += module.hidden_states[1].tolist()

        # Disable training mode
        self.capture(mode=None)

        def project_onto_direction(hidden_states, direction):
            mag = mx.linalg.norm(direction)
            projection = (hidden_states @ direction) / mag

            return projection

        logger.info("Training directions")
        directions = {}
        for name in tqdm.tqdm(hidden_states_pos.keys()):
            h_pos = np.array(hidden_states_pos[name]).astype(np.float32)
            h_neg = np.array(hidden_states_neg[name]).astype(np.float32)

            if method.startswith("pca_"):
                if method == "pca_center":
                    h_center = (h_pos + h_neg) / 2
                    h_pos = h_pos - h_center
                    h_neg = h_neg - h_center
                    h_train = np.vstack([h_pos, h_neg])
                elif method == "pca_diff":
                    h_train = h_pos - h_neg
                    
                pca_model = PCA(n_components=1, whiten=False).fit(h_train)
                dir = pca_model.components_.astype(np.float32).squeeze()
            elif method == "cluster_mean":
                # Simple differences of positive and negative means
                h_pos_mean = h_pos.mean(axis=0, keepdims=True)
                h_neg_mean = h_neg.mean(axis=0, keepdims=True)
                dir = (h_pos_mean - h_neg_mean).squeeze()
                dir = dir / np.linalg.norm(dir)
            dir = mx.array(dir)

            projected_pos = project_onto_direction(h_pos, dir)
            projected_neg = project_onto_direction(h_neg, dir)

            projected_pos_mean = (projected_pos > projected_neg).mean().item()
            projected_neg_mean = (projected_neg > projected_pos).mean().item()
            
            logger.debug(f"Extracted direction for {name}")
            logger.debug(f"Projected positive mean: {projected_pos_mean}")
            logger.debug(f"Projected negative mean: {projected_neg_mean}")
            if projected_neg_mean > projected_pos_mean:
                dir *= -1
                logger.debug(f"Flipped direction for module: {name}")

            directions[name] = dir

        return directions