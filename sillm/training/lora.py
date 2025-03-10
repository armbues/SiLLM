import pathlib
import logging
import math
import re
import json

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from sillm.core.llm import LLM
from sillm.models.args import ModelArgs
from sillm.training.trainer import TrainableLLM

logger = logging.getLogger("sillm")

########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/e74889d0fa0fb49d95bfdf6a1dcad907713eb50e/lora/models.py#L55
# https://github.com/ml-explore/mlx-examples/blob/854ad8747a9c703773adf8866602b114f68aa54a/llms/mlx_lm/tuner/lora.py#L7
########
class LoRALinear(nn.Module):
    """
    Linear layer with LoRA weights.
    """
    @staticmethod
    def from_linear(
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16,
        stabilize: bool = False,
        dropout: float = 0.0,
        dtype: mx.Dtype = None
        ):
        """
        Convert linear layer to LoRA linear layer.
        Args:
            linear: Linear layer to convert.
            rank: Rank to use for LoRA.
            alpha: Alpha to use for LoRA.
            dropout: Dropout to use for LoRA.
            scale: Scale to use for LoRA.
        Returns:
            LoRA linear layer.
        """
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
        bias = "bias" in linear

        if dtype is None:
            dtype = mx.bfloat16 if isinstance(linear, nn.QuantizedLinear) else linear.weight.dtype

        lora_lin = LoRALinear(input_dims, output_dims, rank=rank, alpha=alpha, stabilize=stabilize, dropout=dropout, bias=bias, dtype=dtype)
        lora_lin.linear = linear

        return lora_lin

    def __init__(self,
                 input_dims: int,
                 output_dims: int,
                 rank: int = 8,
                 alpha: float = 16,
                 stabilize: bool = False,
                 dropout: float = 0.0,
                 bias: bool = False,
                 dtype: mx.Dtype = mx.bfloat16
                 ):
        """
        Args:
            input_dims: Input dimensions.
            output_dims: Output dimensions.
            rank: Rank to use for LoRA.
            alpha: Alpha to use for LoRA.
            dropout: Dropout to use for LoRA.
            scale: Scale to use for LoRA.
            bias: Whether to use bias.
        """
        super().__init__()

        # Initialize linear layer weights
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)

        # Initialize LoRA dropout
        self.lora_dropout = nn.Dropout(p=dropout)

        # Initialize LoRA weights
        if stabilize is True:
            ########
            # References:
            # Damjan Kalajdzievski. A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA https://arxiv.org/pdf/2312.03732
            self.scale = alpha / math.sqrt(rank)
        else:
            self.scale = alpha / rank
        bound = 1 / math.sqrt(input_dims)
        input_shape = (input_dims, rank)
        output_shape = (rank, output_dims)
        self.lora_a = mx.random.uniform(low=-bound, high=bound, shape=input_shape, dtype=dtype)
        self.lora_b = mx.zeros(shape=output_shape, dtype=dtype)

    @property
    def lora_size(self):
        """
        Returns:
            Number of LoRA parameters.
        """
        return self.lora_a.size + self.lora_b.size

    def __call__(self, x):
        """
        Args:
            x: Input tensor.
        Returns:
            Output tensor.
        """
        # Determine dtype
        dtype = self.linear.weight.dtype
        if isinstance(self.linear, nn.QuantizedLinear):
            dtype = self.linear.scales.dtype

        # Apply linear layer and LoRA
        y = self.linear(x.astype(dtype))
        z = (self.lora_dropout(x) @ self.lora_a) @ self.lora_b

        return y + self.scale * z
    
    def merge(self):
        """
        Merge LoRA weights into linear weights.
        Returns:
            Linear layer with merged weights.
        """
        linear = self.linear
        weight = linear.weight
        dtype = linear.weight.dtype

        quantized = isinstance(linear, nn.QuantizedLinear)
        if quantized:
            dtype = mx.float16
            group_size = linear.group_size
            bits = linear.bits
            weight = mx.dequantize(weight, linear.scales, linear.biases, group_size, bits)

        # Merge LoRA weights into linear weights
        update = (self.lora_a @ self.lora_b).transpose()
        weight = (weight + (self.scale * update)).astype(dtype)

        if quantized:
            output_dims, input_dims = weight.shape
            bias = "bias" in linear
            linear = nn.Linear(input_dims, output_dims, bias=bias)

            return nn.QuantizedLinear.from_linear(linear, group_size, bits)
        else:
            linear.weight = weight

            return linear

class TrainableLoRA(TrainableLLM):
    """
    Trainable LoRA model wrapper.
    """
    @staticmethod
    def from_model(llm: LLM):
        """
        Convert LLM to trainable LLM.
        Args:
            llm: LLM to convert.
        Returns:
            Trainable LLM.
        """
        model = TrainableLoRA(llm.model, llm.tokenizer, llm.args)
        model._quantization = llm._quantization
        model._mapping = llm._mapping

        return model
    
    def __init__(self,
                 model,
                 tokenizer,
                 args: ModelArgs
                 ):
        """
        Args:
            tokenizer: Tokenizer instance.
            args: Model arguments.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        self._lora = None
        self._lora_modules = []

    def init_lora(self,
                  num_layers: int = 0,
                  target_modules: str = "query_value",
                  rank: int = 8,
                  alpha: float = 16,
                  stabilize: bool = False,
                  dropout: float = 0.0
                  ):
        """
        Initialize LoRA for model.
        Args:
            num_layers: Number of layers to apply LoRA to.
            target_modules: Modules to apply LoRA to.
            rank: Rank to use for LoRA.
            alpha: Alpha to use for LoRA.
            dropout: Dropout to use for LoRA.
            scale: Scale to use for LoRA.
        """
        if self._lora is None:
            self.model.freeze()

            def get_modules(module):
                # Initialize target modules
                for key, child in module.named_modules():
                    if isinstance(child, nn.Linear) or isinstance(child, nn.QuantizedLinear):
                        if module._name != "":
                            key = module._name + "." + key

                        if target_modules == "all_linear":
                            yield key, LoRALinear.from_linear(child, rank=rank, alpha=alpha, stabilize=stabilize, dropout=dropout)
                        elif target_modules == "query_value" and re.search(r"\.attention\.(wq|wv|wqkv)$", key):
                            yield key, LoRALinear.from_linear(child, rank=rank, alpha=alpha,  stabilize=stabilize, dropout=dropout)

            if num_layers > 0:
                # Apply LoRA to last n layers
                self._lora_modules = []
                for layer in self.model.layers[-num_layers:]:
                    self._lora_modules.extend(get_modules(layer))
            else:
                # Apply LoRA to all layers
                num_layers = 0
                self._lora_modules = list(get_modules(self.model))

            if len(self._lora_modules) == 0:
                logger.error(f"No target modules found for LoRA: {target_modules}")

            # Update model with LoRA layers
            self.model.update_modules(tree_unflatten(self._lora_modules))

            # Initialize LoRA configuration
            self._lora = {
                "num_layers": num_layers,
                "target_modules": target_modules,
                "rank": rank,
                "alpha": alpha,
                "stabilize": stabilize,
                "dropout": dropout
            }

            # Enable training mode
            self.model.train(mode=True)

            logger.info(f"Initialized LoRA with rank {rank} for {'all' if num_layers == 0 else num_layers} layers")
            logger.debug(f"LoRA target modules: {target_modules}")
            logger.debug(f"LoRA parameters: Alpha {alpha}, rsLoRA {stabilize}, Dropout {dropout}")

            trainable_params = 0
            for _, module in self._lora_modules:
                trainable_params += module.lora_size
            logger.debug(f"LoRA trainable parameters: {trainable_params/ 10**6:.2f}M")
        else:
            logger.warning(f"LoRA already initialized")

    def merge_and_unload_lora(self):
        """
        Merge LoRA layers back into model.
        """
        if self._lora is not None:
            merged_modules = [
                (key, module.merge())
                for key, module in self._lora_modules
            ]
            self.model.update_modules(tree_unflatten(merged_modules))

            logger.info(f"Merged LoRA layers back into model")

        self._lora = None
        self._lora_modules = []

        # Disable training mode
        self.model.train(mode=False)

    def save_lora_config(self,
                         config_path: str
                         ):
        """
        Save LoRA configuration.
        Args:
            config_path: Folder to save LoRA configuration to.
        """
        assert self._lora is not None

        config_path = pathlib.Path(config_path) / "lora.json"

        with open(config_path, "w") as f:
            f.write(json.dumps(self._lora))

    def load_lora_config(self,
                         config_path: str
                         ):
        """
        Load LoRA configuration.
        Args:
            config_path: Path to load LoRA configuration from.
        """
        config_path = pathlib.Path(config_path)
        if config_path.is_file():
            if config_path.suffix == ".safetensors":
                config_path = config_path.parent
        if config_path.is_dir():
            config_path = config_path / "lora.json"

        if config_path.exists():
            with open(config_path, "r") as f:
                logger.info(f"Loaded LoRA configuration from {config_path}")

                return json.loads(f.read())
            
        return {}

    def save_adapters(self,
                      adapter_path: str,
                      ):
        """
        Save adapter weights.
        Args:
            adapter_path: Path to save adapter weights to.
        """
        assert self._lora is not None

        state = dict(tree_flatten(self.model.trainable_parameters()))
        metadata = {
            "format": "mlx"
        }
        mx.save_safetensors(adapter_path, state, metadata=metadata)

    def load_adapters(self,
                      adapter_path: str
                      ):
        """
        Load adapter weights.
        Args:
            adapter_path: Path to adapter weights.
        """
        assert pathlib.Path(adapter_path).exists(), adapter_path
        if self._lora is not None:
            self.model.load_weights(adapter_path, strict=False)
        
            logger.info(f"Loaded adapter weights from {adapter_path}")
        else:
            logger.error(f"LoRA not initialized, cannot load adapter weights")

    def save_checkpoint(self,
                        checkpoint_path: str,
                        steps: int = -1
                        ):
        """
        Save model checkpoint.
        Args:
            checkpoint_path: Directory to save checkpoints to.
            steps: Number of steps.
        """
        checkpoint_path = pathlib.Path(checkpoint_path)
        if steps >= 0:
            adapter_path = checkpoint_path / f"ckpt-{steps}.safetensors"
        else:
            adapter_path = checkpoint_path / f"ckpt-final.safetensors"

        self.save_adapters(str(adapter_path))
        
        return str(adapter_path)