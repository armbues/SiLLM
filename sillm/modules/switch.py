import math

import mlx.core as mx
import mlx.nn as nn

########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/f530f56df2738a54982c4541189a8c8d7cd94c44/llms/mlx_lm/models/switch_layers.py#L9
########
class QuantizedSwitchLinear(nn.Module):
    def __init__(self,
                 input_dims: int,
                 output_dims: int,
                 num_experts: int,
                 bias: bool = True,
                 group_size: int = 64,
                 bits: int = 4,
                 ):
        super().__init__()

        scale = math.sqrt(1 / input_dims)
        self.weight, self.scales, self.biases = mx.quantize(
            mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(num_experts, output_dims, input_dims),
            ),
            group_size=group_size,
            bits=bits,
        )

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

        self.group_size = group_size
        self.bits = bits

        # Freeze this model's parameters
        self.freeze()

    def unfreeze(self, *args, **kwargs):
        super().unfreeze(*args, **kwargs)
        self.freeze(recurse=False)

    @property
    def input_dims(self):
        return self.scales.shape[2] * self.group_size

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices):
        x = mx.gather_qmm(x, self["weight"], self["scales"], self["biases"], rhs_indices=indices, transpose=True, group_size=self.group_size, bits=self.bits)
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/f530f56df2738a54982c4541189a8c8d7cd94c44/llms/mlx_lm/models/switch_layers.py#L75
########
class SwitchLinear(nn.Module):
    def __init__(self,
                 input_dims: int,
                 output_dims: int,
                 num_experts: int,
                 bias: bool = True
                 ):
        super().__init__()

        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(low=-scale, high=scale, shape=(num_experts, output_dims, input_dims))

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self):
        return self.weight.shape[2]

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices):
        x = mx.gather_mm(x, self["weight"].swapaxes(-1, -2), rhs_indices=indices)
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)

        return x
    
    def to_quantized(self,
                     group_size: int = 64,
                     bits: int = 4
                     ) -> QuantizedSwitchLinear:
        num_experts, output_dims, input_dims = self.weight.shape
        
        ql = QuantizedSwitchLinear(input_dims, output_dims, num_experts, False, group_size, bits)
        ql.weight, ql.scales, ql.biases = mx.quantize(self.weight, group_size, bits)
        if "bias" in self:
            ql.bias = self.bias

        return ql

########
# Based on mlx-examples:
# https://github.com/ml-explore/mlx-examples/blob/f530f56df2738a54982c4541189a8c8d7cd94c44/llms/mlx_lm/models/switch_layers.py#L119
########
class SwitchGLU(nn.Module):
    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 num_experts: int,
                 bias: bool = False,
                 activation = nn.silu,
                 ):
        super().__init__()

        self.w1 = SwitchLinear(dim, hidden_dim, num_experts, bias=bias)
        self.w2 = SwitchLinear(hidden_dim, dim, num_experts, bias=bias)
        self.w3 = SwitchLinear(dim, hidden_dim, num_experts, bias=bias)

        self.act = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        x_up = self.w3(x, indices)
        x_gate = self.w1(x, indices)
        x = self.w2(self.act(x_gate) * x_up, indices)

        return x.squeeze(-2)