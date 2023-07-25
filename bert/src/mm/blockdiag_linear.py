# Adapted from https://github.com/HazyResearch/fly/tree/master/src/models/layers

import math
import torch
import torch.nn as nn
from einops import rearrange

from src.mm.structured_linear import StructuredLinear
from src.mm.blockdiag_multiply import blockdiag_multiply


class BlockdiagLinear(StructuredLinear):

    def __init__(self, *args, nblocks=4, shuffle=False, **kwargs):
        """shuffle: apply channel_shuffle operation before the matmul as in ShuffleNet
        """
        super().__init__(*args, **kwargs)
        in_blksz = int(math.ceil(self.in_features / nblocks))
        out_blksz = int(math.ceil(self.out_features / nblocks))
        self.in_features_extended = in_blksz * nblocks
        self.out_features_extended = out_blksz * nblocks
        self.shuffle = shuffle
        self.weight = nn.Parameter(torch.empty(nblocks, out_blksz, in_blksz))
        self.reset_parameters()

    def set_weights_from_dense_init(self, dense_init_fn_):
        dense_weight = torch.empty(self.out_features_extended, self.in_features_extended,
                                   device=self.weight.device, dtype=self.weight.dtype)
        dense_init_fn_(dense_weight)
        # Scale by sqrt because the weight is sparse
        scaling = math.sqrt(dense_weight.numel() / self.weight.numel())
        dense_weight *= scaling
        with torch.no_grad():
            nblocks = self.weight.shape[0]
            self.weight.copy_(rearrange(dense_weight, '(b o) (b1 i) -> b b1 o i',
                                        b=nblocks, b1=nblocks)[0])

    @property
    def saving(self):
        return self.weight.numel() / (self.in_features * self.out_features)

    def forward_matmul(self, x):
        x = self.preprocess(x)
        if self.shuffle:
            x = rearrange(x, '... (group c_per_group) -> ... (c_per_group group)',
                          group=self.weight.shape[0])  # group=nblocks
        output = blockdiag_multiply(x, self.weight)
        return self.postprocess(output)


class BlockdiagSparsityConfig:

    def __init__(self, nblocks, block=32, global_size=0):
        """shuffle: apply channel_shuffle operation before the matmul as in ShuffleNet
        """
        self.nblocks = nblocks
        self.block = block
        self.global_size = global_size

    def make_layout(self, out_features, in_features):
        assert out_features % self.block == 0 and in_features % self.block == 0
        assert out_features % self.nblocks == 0 and in_features % self.nblocks == 0
        layout = torch.block_diag(*[torch.ones(out_features // self.nblocks,
                                               in_features // self.nblocks,
                                               dtype=torch.int32)] * self.nblocks)
        if self.global_size > 0:
            layout[:self.global_size] = 1
            layout[:, :self.global_size] = 1
        # Convert from (out_features, in_features) mask to
        # (out_features // block, in_features // block) mask
        layout = rearrange(layout, '(p blksz) (r blksz1) -> p r (blksz blksz1)',
                           blksz=self.block, blksz1=self.block)
        return (layout > 0).any(dim=-1).int()