import torch

from softmaxlib import additive_masked_softmax_dropout_forward
from softmaxlib import masked_scale_softmax_backward_recompute

from src.ops.triton.softmax_dropout import softmax_dropout


class _fused_softmax_dropout(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, p, mask, return_dropout_mask=False):
        """
        x: (batch_size, nheads, q_seqlen, k_seqlen)
        p: float
        mask: (batch_size, 1, 1, k_seqlen)
        """
        assert x.dtype == torch.float16
        assert x.ndim == 4
        assert mask is not None

        x = x.contiguous()
        dropout_results, dropout_mask = additive_masked_softmax_dropout_forward(x, mask, p)

        ctx.save_for_backward(x, mask, dropout_mask)
        ctx.dropout_prob = p

        return dropout_results, (None if not return_dropout_mask else dropout_mask)

    @staticmethod
    def backward(ctx, grad_out, grad_dropout_mask):
        x, mask, dropout_mask = ctx.saved_tensors
        p = ctx.dropout_prob
        grad_in = masked_scale_softmax_backward_recompute(grad_out, x, mask, dropout_mask, p)
        return grad_in, None, None, None


def fused_softmax_dropout(x, p, mask):
    if x.is_cuda and x.dtype == torch.float16 and mask is not None and p != 0.0:
        return _fused_softmax_dropout.apply(x, p, mask)[0]
    else:
        return softmax_dropout(x, p, mask, mask_type='bk')
