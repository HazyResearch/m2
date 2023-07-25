# Copyright (c) 2023, Dan Fu and Simran Arora.

import torch
import torch.nn as nn
import math
from einops import rearrange
import opt_einsum as oe

contract = oe.contract
from flashmm import mm_block_fwd, hyena_filter_fwd, exp_mod_in_place_fwd
from src.utils.train import OptimModule

def fast_mm_block(
    u,
    linear, out_linear,
    x1_s, x2_s, v_s,
    x1_s_bias, x2_s_bias, v_s_bias,
    k, k_resid, D, Du,
    dropout_mask,
    gelu, fft_size
):
    # u.shape: B L H
    x1x2v = linear(u)
    H = x1x2v.shape[-1] // 3
    x1, x2, v = x1x2v.split(H, dim=-1)
    x1 = x1.transpose(1, 2).contiguous()
    x2 = x2.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()

    k_f = torch.fft.rfft(k.to(torch.float32), n=fft_size)
    k_residual_f = torch.fft.rfft(k_resid.to(torch.float32), n=fft_size)
    out = mm_block_fwd(
        x1, x2, v,
        x1_s, x2_s, v_s,
        x1_s_bias, x2_s_bias, v_s_bias,
        k_f, None, u.transpose(1, 2).to(x1.dtype).contiguous(), k_residual_f, Du, D, dropout_mask, gelu, fft_size,
        False, False
    )

    out = out.transpose(-1, -2)

    return out_linear(out)

def pos_emb_init(seq_len, emb_dim):
    t = torch.linspace(0, 1, seq_len)[None, :, None]  # 1, L, 1

    if emb_dim > 1:
        bands = (emb_dim - 1) // 2
    # To compute the right embeddings we use the "proper" linspace
    t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
    w = 2 * math.pi * t_rescaled / seq_len  # 1, L, 1

    f = torch.linspace(1e-4, bands - 1, bands)[None, None]
    z = torch.exp(-1j * f * w)
    z = torch.cat([t, z.real, z.imag], dim=-1)
    return z

class FastFilter(OptimModule):
    def __init__(
        self,
        d_model,
        channels,
        bidirectional=True,
        order=16,
        seq_len=128,
        lr=1e-3,
        lr_pos_emb=1e-5,
        w=1,  # frequency of periodic activations
        wd=0,  # weight decay of kernel parameters
        emb_dim=5,
    ):
        # create positional embeddings
        super().__init__()

        self.bidirectional = bidirectional
        if self.bidirectional:
            channels *= 2
        self.channels = channels

        z = pos_emb_init(seq_len, emb_dim).repeat(self.channels, 1, 1)

        sin_freq = w * torch.ones(self.channels, order)

        # create parameters for eo_mat, eo_bias
        eo_linears = [
            nn.Linear(emb_dim, order)
            for _ in range(self.channels)
        ]
        eo_mat = torch.stack([l.weight for l in eo_linears], dim=0).transpose(-1, -2).contiguous()
        eo_bias = torch.stack([l.bias for l in eo_linears], dim=0)

        # create parameters for oo1_mat, oo1_bias
        oo1_linears = [
            nn.Linear(order, order)
            for _ in range(self.channels)
        ]
        oo1_mat = torch.stack([l.weight for l in oo1_linears], dim=0).transpose(-1, -2).contiguous()
        oo1_bias = torch.stack([l.bias for l in oo1_linears], dim=0)

        # create parameters for oo2_mat, oo2_bias
        oo2_linears = [
            nn.Linear(order, order)
            for _ in range(self.channels)
        ]
        oo2_mat = torch.stack([l.weight for l in oo2_linears], dim=0).transpose(-1, -2).contiguous()
        oo2_bias = torch.stack([l.bias for l in oo2_linears], dim=0)

        # create parameters for oh_mat
        oh_linears = [
            nn.Linear(order, d_model, bias=False)
            for _ in range(self.channels)
        ]
        oh_mat = torch.stack([l.weight for l in oh_linears], dim=0).transpose(-1, -2)

        # create reverse parameter
        if self.bidirectional:
            reverse = torch.Tensor([
                [0, 1] for _ in range(self.channels // 2)
            ]).flatten().int()
        else:
            reverse = torch.Tensor([0 for _ in range(self.channels)]).int()

        self.register("z", z, lr=lr_pos_emb)
        self.register('sin_freq', sin_freq, lr=lr, wd=wd)
        self.register('eo_mat', eo_mat, lr=lr, wd=wd)
        self.register('eo_bias', eo_bias, lr=lr, wd=wd)
        self.register('oo1_mat', oo1_mat, lr=lr, wd=wd)
        self.register('oo1_bias', oo1_bias, lr=lr, wd=wd)
        self.register('oo2_mat', oo2_mat, lr=lr, wd=wd)
        self.register('oo2_bias', oo2_bias, lr=lr, wd=wd)
        self.register('oh_mat', oh_mat, lr=lr, wd=wd)
        self.register('reverse', reverse, lr=0)

        target=1e-2
        fast_decay_pct=0.3
        slow_decay_pct=1.5
        self.min_decay = math.log(target) / slow_decay_pct
        self.max_decay = math.log(target) / fast_decay_pct
        self.shift = 0.

    def forward(self):
        k = hyena_filter_fwd(
            self.z, self.sin_freq, self.eo_mat, self.eo_bias,
            self.oo1_mat, self.oo1_bias, self.oo2_mat, self.oo2_bias,
            self.reverse, None
        )
        k = torch.bmm(k, self.oh_mat)
        k = exp_mod_in_place_fwd(k, self.reverse, self.min_decay, self.max_decay, self.shift)
        return k

class FlashMMSequenceMixing(nn.Module):
    def __init__(
        self,
        d_model,
        l_max=128,
        hyena_kernel_lr=None,
        bidirectional=False,
        hyena_lr_pos_emb=1e-5,
        hyena_w=10,
        hyena_w_mod=1,
        hyena_wd=0.1,
        hyena_emb_dim=5,
        hyena_filter_order=128,
        residual_long_conv=False,
        **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.l_max = l_max
        self.kernel_lr = hyena_kernel_lr
        self.channels = 1
        self.bidirectional = bidirectional
        self.residual_long_conv = residual_long_conv

        print('Using Flash MM Sequence Mixing (no bwd pass!)')
        print('-- Bidirectional:', self.bidirectional)
        print("-- Using Long Conv Residual:", self.residual_long_conv)
        print('-- Hyena w:', hyena_w)
        print('-- Hyena w mod:', hyena_w_mod)

        channels = 1
        if self.residual_long_conv:
            channels *= 2

        self.fast_filter = FastFilter(
            self.d_model,
            channels=channels,
            bidirectional=self.bidirectional,
            order=hyena_filter_order,
            seq_len=self.l_max,
            lr=hyena_kernel_lr,
            lr_pos_emb=hyena_lr_pos_emb,
            w=hyena_w,  # frequency of periodic activations
            wd=hyena_wd,  # weight decay of kernel parameters
            emb_dim=hyena_emb_dim,
        )

        # setup projections
        self.in_linear = nn.Linear(d_model, 3 * d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        # to use inits from Conv1d
        short_filter = nn.Conv1d(
            in_channels=3 * d_model,
            out_channels=3 * d_model,
            kernel_size=4,
            groups=3 * d_model,
            padding=3
        )
        self.x1_s = nn.Parameter(short_filter.weight[:d_model, 0, :].clone())
        self.x2_s = nn.Parameter(short_filter.weight[d_model:2 * d_model, 0, :].clone())
        self.v_s = nn.Parameter(short_filter.weight[2 * d_model:, 0, :].clone())
        self.x1_s_bias = nn.Parameter(short_filter.bias[:d_model].clone())
        self.x2_s_bias = nn.Parameter(short_filter.bias[d_model:2 * d_model].clone())
        self.v_s_bias = nn.Parameter(short_filter.bias[2 * d_model:].clone())

        self.bias = nn.Parameter(torch.randn(self.d_model))
        if self.residual_long_conv:
            self.residual_bias = nn.Parameter(torch.randn(self.d_model))
        else:
            self.residual_bias = None


    def forward(self, u, **kwargs):
        fft_size = 2 * self.l_max

        all_kernels = self.fast_filter() # C L H
        C, L, H = all_kernels.shape
        if self.residual_long_conv:
            k = all_kernels[:C // 2].reshape((C // 2) * L, H).transpose(0, 1).contiguous()
            k_resid = all_kernels[C // 2:].reshape((C // 2) * L, H).transpose(0, 1).contiguous()
        else:
            k = all_kernels.reshape(C * L, H).transpose(0, 1).contiguous()
            k_resid = None

        return fast_mm_block(
            u,
            self.in_linear, self.out_linear,
            self.x1_s, self.x2_s, self.v_s,
            self.x1_s_bias, self.x2_s_bias, self.v_s_bias,
            k, k_resid, self.bias, self.residual_bias,
            None,
            False, fft_size
        ), None

 