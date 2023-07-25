# Adapted from https://github.com/HazyResearch/fly/tree/master/src/models/layers

import numpy as np
import torch
from torch.nn import functional as F
from einops import rearrange


def blockdiag_weight_to_dense_weight(weight):
    """
    Argumments:
        weight: (nblocks, out / nblocks, in / blocks)
    Return:
        dense_weight: (out / in)
    """
    return torch.block_diag(*torch.unbind(weight, dim=0))


def blockdiag_multiply_reference(x, weight):
    """
    This implementation is slow but more likely to be correct.
    Arguments:
        x: (..., n)
        weight: (nblocks, q, n / nblocks)
    Outputs:
        out: (..., nblocks * q)
    """
    n = x.shape[-1]
    nblocks, q, p = weight.shape
    assert nblocks * p == n

    x_reshaped = rearrange(x, '... (nblocks p) -> ... nblocks p', nblocks=nblocks)
    return rearrange(torch.einsum('...kp, kqp -> ...kq', x_reshaped, weight),
                     '... nblocks q -> ... (nblocks q)')


class BlockdiagMultiply(torch.autograd.Function):

    """This is a faster implementation, with careful memory copies for the fastest
    bmm performance.
    The backward pass is also written manually with careful memory copies.
    Arguments:
        x: (..., n)
        weight: (nblocks, q, n / nblocks)
    Outputs:
        out: (..., nblocks * q)
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.bfloat16)
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        nblocks, q, p = weight.shape
        assert nblocks * p == n
        x_reshaped = x.reshape(batch_dim, nblocks, p).transpose(0, 1)
        out = torch.empty(batch_dim, nblocks, q, device=x.device, dtype=x.dtype).transpose(0, 1)
        out = torch.bmm(x_reshaped, weight.transpose(-1, -2), out=out).transpose(0, 1)
        return out.reshape(*batch_shape, nblocks * q)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout):
        x, weight = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        nblocks, q, p = weight.shape
        assert nblocks * p == n
        dx, dweight = None, None
        dout_reshaped = dout.reshape(batch_dim, nblocks, q).transpose(0, 1)
        if ctx.needs_input_grad[0]:
            dx = torch.empty(batch_dim, nblocks, p, device=x.device, dtype=x.dtype)
            dx = torch.bmm(dout_reshaped, weight.conj(),
                           out=dx.transpose(0, 1)).transpose(0, 1).reshape(*batch_shape, n)
        if ctx.needs_input_grad[1]:
            x_reshaped = x.reshape(batch_dim, nblocks, p).transpose(0, 1)
            dweight = torch.bmm(dout_reshaped.transpose(-1, -2), x_reshaped.conj())
        return dx, dweight


blockdiag_multiply = BlockdiagMultiply.apply