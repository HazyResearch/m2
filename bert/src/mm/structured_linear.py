# Adapted from https://github.com/HazyResearch/fly/tree/master/src/models/layers

import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class StructuredLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        """Subclasses should call reset_parameters
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Subclasses may override {in,out}_features_extended
        if not hasattr(self, 'in_features_extended'):
            self.in_features_extended = in_features
        if not hasattr(self, 'out_features_extended'):
            self.out_features_extended = out_features
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self) -> None:
        self.set_weights_from_dense_init(dense_init_fn_=partial(init.kaiming_uniform_, a=math.sqrt(5)))
        self.reset_parameters_bias()

    def set_weights_from_dense_init(self, dense_init_fn_):
        raise NotImplementedError

    def reset_parameters_bias(self):
        if self.bias is not None:
            fan_in = self.bias.shape[-1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    @property
    def saving(self):
        raise NotImplementedError

    def convert_to_dense_weight(self):
        factory_kwargs = {'device': self.weight.device, 'dtype': self.weight.dtype}
        dense_weight = self.forward_matmul(torch.eye(self.in_features, **factory_kwargs)).T
        return dense_weight

    def preprocess(self, x):
        in_features = x.shape[-1]
        if in_features < self.in_features_extended:
            x = F.pad(x, (0, self.in_features_extended - in_features))
        return x

    def postprocess(self, output):
        out_features_extended = output.shape[-1]
        if out_features_extended > self.out_features:
            output = output[..., :self.out_features]
        return output

    def forward_matmul(self, x):
        raise NotImplementedError

    def forward(self, x):
        output = self.forward_matmul(x)
        # Convert bias to output.dtype in case of AMP, otherwise bias and activation will be in FP32
        return (output + self.bias.to(dtype=output.dtype)) if self.bias is not None else output