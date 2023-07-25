# Copyright (c) 2023, Dan Fu and Simran Arora.
# Adapted from https://github.com/HazyResearch/safari/blob/main/src/models/sequence/hyena.py

import torch.nn as nn
from einops import rearrange
import opt_einsum as oe

contract = oe.contract
from src.mm.hyena_utils import HyenaFilter


class MonarchMixerSequenceMixing(nn.Module):
    def __init__(
        self,
        d_model,
        l_max=128,
        dropout=0.0,
        hyena_kernel_lr=None,
        bidirectional=False,
        hyena_lr_pos_emb=1e-5,
        hyena_w=10,
        hyena_w_mod=1,
        hyena_wd=0.1,
        hyena_emb_dim=3,
        hyena_filter_dropout=0.0,
        hyena_filter_order=16,
        residual_long_conv=False,
        hyena_training_additions=False,
    ):
        super().__init__()

        self.d_model = d_model
        self.l_max = l_max
        self.kernel_lr = hyena_kernel_lr
        self.channels = 1
        self.bidirectional = bidirectional
        self.residual_long_conv = residual_long_conv
        self.NUM_PROJECTIONS = 3

        print('-- Bidirectional:', self.bidirectional)
        print("-- Using Long Conv Residual:", self.residual_long_conv)
        print('-- Hyena w:', hyena_w)
        print('-- Hyena w mod:', hyena_w_mod)
        print(f"-- Hyena filter order: {hyena_filter_order}")
        print(f"-- Hyena filter dropout: {hyena_filter_dropout}")
        print(f"-- Hyena filter wd: {hyena_wd}")
        print(f"-- Hyena filter emb dim: {hyena_emb_dim}")
        print(f"-- Hyena filter lr: {hyena_kernel_lr}")
        print(f"-- Hyena filter lr pos emb: {hyena_lr_pos_emb}")

        self.filter_fn = HyenaFilter(
            self.d_model,
            order=hyena_filter_order,
            seq_len=self.l_max,
            dropout=hyena_filter_dropout,
            bidirectional=self.bidirectional,
            lr=hyena_kernel_lr,
            lr_pos_emb=hyena_lr_pos_emb,
            w=hyena_w,  # frequency of periodic activations
            w_mod=hyena_w_mod,
            wd=hyena_wd,  # weight decay of kernel parameters
            emb_dim=hyena_emb_dim,
        )
        
        if self.residual_long_conv:
            self.filter_fn2 = HyenaFilter(
                self.d_model,
                order=hyena_filter_order,
                seq_len=self.l_max,
                dropout=hyena_filter_dropout,
                bidirectional=self.bidirectional,
                lr=hyena_kernel_lr,
                lr_pos_emb=hyena_lr_pos_emb,
                w=hyena_w,  # frequency of periodic activations
                w_mod=hyena_w_mod,
                wd=hyena_wd,  # weight decay of kernel parameters
                emb_dim=hyena_emb_dim,
            )
        
        # setup projections
        self.in_linear = nn.Linear(d_model, 3 * d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.hyena_training_additions = hyena_training_additions
        if self.hyena_training_additions:
            self.act = nn.Identity()
            self.drop = nn.Dropout(dropout)
            self.layernorm = nn.LayerNorm(d_model)
        
        # setup short conv
        total_width = self.d_model * self.NUM_PROJECTIONS
        self.short_filter = nn.Conv1d(
            in_channels=total_width,
            out_channels=total_width,
            kernel_size=3,
            groups=total_width,
            padding=2,
        )


    def forward(self, u, **kwargs):
        # u is B L H
        if self.hyena_training_additions:
            u = self.layernorm(u)
        L = u.size(-2)

        # in projection
        u_orig = u
        u = self.in_linear(u)
        u = rearrange(u, "b l d -> b d l")
        
        # short filter
        uc = self.short_filter(u)[..., :L]

        x1, x2, v = uc.split(self.d_model, dim=1)
        
        v = v * x1
        if self.hyena_training_additions:
            v = self.drop(v)

        k = self.filter_fn.filter(L, device=u.device)
        k = rearrange(k, "c l d -> c d l")[0] # `c` is always 1 by default

        if self.bidirectional:
            k_rev = self.filter_fn.filter_rev(L, device=u.device)
            k_rev = rearrange(k_rev, "c l d -> c d l")[0] # `c` is always 1 by default
        else:
            k_rev = None

        y = self.filter_fn(v, L, k_fwd=k, k_rev=k_rev, bias= self.filter_fn.bias[None, :, None])

        if self.residual_long_conv:
            k2 = self.filter_fn2.filter(L, device=u.device)
            k2 = rearrange(k2, "c l d -> c d l")[0]

            if self.bidirectional:
                k2_rev = self.filter_fn2.filter_rev(L, device=u.device)
                k2_rev = rearrange(k2_rev, "c l d -> c d l")[0] # `c` is always 1 by default
            else:
                k2_rev = None                

            yu = self.filter_fn2(u_orig.transpose(-1, -2), L, k_fwd=k2, k_rev=k2_rev, bias= self.filter_fn2.bias[None, :, None])
        
        # post gating
        y = y * x2

        if self.residual_long_conv:
            y = y + yu

        y = y.transpose(-1, -2)
        if self.hyena_training_additions:
            y = self.drop(self.act(y))
        y = self.out_linear(y)

        return y, None

 