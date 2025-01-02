# Adapted from https://github.com/HazyResearch/fly/tree/master/src/models/layers

import math
import numpy as np

import torch
from torch.nn import functional as F

from einops import rearrange


def blockdiag_butterfly_multiply_reference(x, w1_bfly, w2_bfly, version=2):
    """
    This implementation is slow but more likely to be correct.
    There are 3 implementations, which should all yield the same answer
    Arguments:
        x: (batch, n)
        w1_bfly: (k, q, p), where k = n / p
        w2_bfly: (l, s, r), where l = k * q / r = n * q / (p * r)
    Outputs:
        out: (batch, m), where m = l * s = n * s * q / (p * r)
    """
    if version not in [1, 2, 3]:
        raise NotImplementedError('version must be either 1, 2, or 3')
    batch, n = x.shape
    k, q, p = w1_bfly.shape
    l, s, r = w2_bfly.shape
    assert k * p == n
    assert l * r == k * q

    x_reshaped = rearrange(x, 'b (k p) -> b k p', k=k)
    if version == 1:  # Implementation 1 (only works for when k = q = p = l = s = r = sqrt(n))
        assert k == q == p == l == s == r == int(math.sqrt(n))
        return torch.einsum('bkp,kqp,qlk->blq', x_reshaped, w1_bfly, w2_bfly).reshape(batch, n)
    elif version == 2:  # Implementation 2
        out1 = torch.einsum('kqp,bkp->bkq', w1_bfly, x_reshaped)
        out1 = rearrange(rearrange(out1, 'b k q -> b (k q)'), 'b (r l) -> b l r', l=l)
        return torch.einsum('lsr,blr->bsl', w2_bfly, out1).reshape(batch, s * l)
    # Implementation 3: most likely to be correct, but it's the slowest
    elif version == 3:
        w1_dense = torch.block_diag(*torch.unbind(w1_bfly, dim=0))
        out1 = F.linear(x, w1_dense)
        out1 = rearrange(out1, 'b (r l) -> b (l r)', l=l)
        w2_dense = torch.block_diag(*torch.unbind(w2_bfly, dim=0))
        out2 = F.linear(out1, w2_dense)
        out2 = rearrange(out2, 'b (l s) -> b (s l)', l=l)
        return out2

class BlockdiagButterflyMultiply(torch.autograd.Function):
    """This is a faster implementation, with careful memory copies for the fastest
    bmm performance.
    The backward pass is also written manually with careful memory copies.
    Arguments:
        x: (batch, n)
        w1_bfly: (nblocks, blk_blk2_in, blk_sz)
        w2_bfly: (nblocks, blk_sz, blk_r)
    Outputs:
        out: (batch, m), where m = l * s = n * s * q / (p * r)
    """

    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.bfloat16)
    def forward(ctx, x, w1_bfly, w2_bfly, debug_out1=False):
        batch_shape, n = x.shape[:-1], x.shape[-1]
        seq_dim = np.prod(batch_shape)

        w1_bfly = w1_bfly.to(x.dtype)
        w2_bfly = w2_bfly.to(x.dtype)

        # Typically blk1_out = blk2_in and nblocks1 = nblocks2
        # e.g. (4, 4, 1024)
        nblocks1, blk1_out, blk1_in = w1_bfly.shape
        nblocks2, blk2_out, blk2_in = w2_bfly.shape
        assert nblocks1 * blk1_in == n
        assert nblocks2 * blk2_in == nblocks1 * blk1_out

        # Typical shape for Llama 7B on Math reasoning: (4, 666, 1024)
        x_reshaped = x.reshape(seq_dim, nblocks1, blk1_in).transpose(0, 1)
        out1 = torch.empty(nblocks1, seq_dim, blk1_out, device=x.device, dtype=x.dtype)

        # (nblocks1, seq_dim, blk1_in) @ (nblocks1, blk1_in, blk1_out)
        out1 = torch.bmm(x_reshaped, w1_bfly.transpose(-1, -2), out=out1)  # -> (nblocks1, seq_dim, blk1_out)
        del x_reshaped

        # Feature shuffling
        out1 = (
            out1.transpose(0, 1).reshape(seq_dim, blk2_in, nblocks2).permute(2, 0, 1)
        )  # (seq_dim, nblocks2, blk1_out) -> (.., blk2_in, nblocks2) -> (nblocks2, seq_dim, blk2_in)

        out2 = torch.empty(nblocks2, seq_dim, blk2_out, device=x.device, dtype=x.dtype)
        out2 = torch.bmm(
            out1, w2_bfly.transpose(-1, -2), out=out2
        )  # (nblocks2, seq_dim, blk2_in) @ (nblocks2, blk2_in, blk2_out) -> (nblocks2, seq_dim, blk2_out)

        out2 = out2.permute(1, 2, 0).reshape(
            *batch_shape, blk2_out * nblocks2
        )  # (nblocks2, seq_dim, blk2_out) -> (seq_dim, nblocks2 * blk2_out )

        ctx.save_for_backward(x, w1_bfly, w2_bfly, out1, None, None)
        if debug_out1:
            return out2, out1
        return out2

    @staticmethod
    @torch.amp.custom_bwd
    def backward(ctx, dout):
        x, w1_bfly, w2_bfly, out1, *_ = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        seq_dim = np.prod(batch_shape)
        nblocks1, blk1_out, blk1_in = w1_bfly.shape
        nblocks2, blk2_out, blk2_in = w2_bfly.shape

        dx, dw1_bfly, dw2_bfly = None, None, None

        dout_reshaped = dout.reshape(seq_dim, blk2_out, nblocks2).transpose(-1, -2)
        dout_reshaped = dout_reshaped.transpose(0, 1).contiguous()  # (nblocks2, seq_dim, blk2_out)
        if ctx.needs_input_grad[2]:
            # dw2_bfly = torch.empty(nblocks2, blk2_out, blk2_in, device=w2_bfly.device, dtype=w2_bfly.dtype)
            # dw2_bfly = torch.bmm(dout_reshaped.transpose(-1, -2), out1, out=dw2_bfly)

            # (nblocks2, blk2_out, seq_dim) @ (nblocks2, seq_dim, blk1_out) -> (nblocks2, blk2_out, blk1_out)
            dw2_bfly = torch.bmm(dout_reshaped.transpose(-1, -2), out1.conj())
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[0]:
            dout1 = torch.empty(nblocks2, seq_dim, blk2_in, device=x.device, dtype=x.dtype)
            dout1 = torch.bmm(dout_reshaped, w2_bfly.conj(), out=dout1)  # -> (nblocks2, seq_dim, blk2_in)
            del dout_reshaped
            # dout1 = dout1.transpose(0, 1).transpose(-1, -2).contiguous().reshape(seq_dim, nblocks1, blk1_out).transpose(0, 1)
            # NOTE: We do NOT need contiguous in between? This should save memory & time
            dout1 = (
                dout1.permute(1, 2, 0).reshape(seq_dim, nblocks1, blk1_out).transpose(0, 1)
            )  # -> (nblocks1, seq_dim, blk2_in)
            if ctx.needs_input_grad[0]:
                dx = torch.empty(seq_dim, nblocks1, blk1_in, device=x.device, dtype=x.dtype)
                # (nblocks1, seq_dim, blk1_out) @ (nblocks1, blk1_out, blk1_in) -> (nblocks1, seq_dim, blk1_in)
                dx = torch.bmm(dout1, w1_bfly.conj(), out=dx.transpose(0, 1)).transpose(0, 1).reshape(*batch_shape, n)
            if ctx.needs_input_grad[1]:
                x_reshaped = x.reshape(seq_dim, nblocks1, blk1_in).transpose(0, 1)
                # （nblocks2, blk2_in, seq_dim) @ (nblocks2, seq_dim, blk1_out) -> (nblocks2, blk2_in, blk1_in)
                dw1_bfly = torch.bmm(dout1.transpose(-1, -2), x_reshaped.conj())
        return dx, dw1_bfly, dw2_bfly, None, None


blockdiag_butterfly_multiply = BlockdiagButterflyMultiply.apply