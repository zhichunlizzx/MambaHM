#!/usr/bin/env python
# Copyright 2023 Z Zhang

# BioSeq2Seq, Version 1.0;
# you may not use this file except in compliance with the License.
# Use of this code requires following originality guidelines
# and declaring the source of the code.
# email:zhichunli@mail.dlut.edu.cn
# =========================================================================
import inspect
import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
from mamba_ssm.modules.mamba_simple import Mamba
try:
    from mamba_ssm.modules.mamba_simple import Block
except ImportError:
    from mamba_ssm.modules.block import Block
from torch import nn
from .modeling_rcps import RCPSMambaBlock

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm
except ImportError:
    try:
        from mamba_ssm.ops.triton.layer_norm import RMSNorm
    except ImportError:
        RMSNorm = None


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        bidirectional=True,
        bidirectional_strategy="add",
        bidirectional_weight_tie=True,
        rcps=False,
        device=None,
        dtype=None,
):
    """Create Caduceus block.

    Adapted from: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
    """

    # 打印所有参数及其值
    # print("Function create_block initialized with parameters:")
    # print(f"d_model: {d_model}")
    # print(f"ssm_cfg: {ssm_cfg}")
    # print(f"norm_epsilon: {norm_epsilon}")
    # print(f"rms_norm: {rms_norm}")
    # print(f"residual_in_fp32: {residual_in_fp32}")
    # print(f"fused_add_norm: {fused_add_norm}")
    # print(f"layer_idx: {layer_idx}")
    # print(f"bidirectional: {bidirectional}")
    # print(f"bidirectional_strategy: {bidirectional_strategy}")
    # print(f"bidirectional_weight_tie: {bidirectional_weight_tie}")
    # print(f"rcps: {rcps}")
    # print(f"device: {device}")
    # print(f"dtype: {dtype}")

    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    bidirectional_kwargs = {
        "bidirectional": bidirectional,
        "bidirectional_strategy": bidirectional_strategy,
        "bidirectional_weight_tie": bidirectional_weight_tie,
    }
    mixer_cls = partial(BiMamba, layer_idx=layer_idx, **ssm_cfg, **bidirectional_kwargs, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    # print('rcps: ', rcps)
    # 用的RCPSMambaBlock, 使不使用rc，加上rc的内容，channel变双倍
    block_cls = RCPSMambaBlock if rcps else Block
    # mambav2 compatibility
    if "mlp_cls" in inspect.signature(block_cls.__init__).parameters:
        block = block_cls(
            d_model,
            mixer_cls,
            mlp_cls=nn.Identity,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
    else:
        # 走的这边
        block = block_cls(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
    block.layer_idx = layer_idx
    return block


class BiMamba(nn.Module):
    """bi-directionality Mamba."""

    def __init__(
            self,
            d_model: int,
            bidirectional: bool = True,
            bidirectional_strategy: Optional[str] = "add",
            bidirectional_weight_tie: bool = True,
            **mamba_kwargs,
    ):
        super().__init__()
        # 打印所有参数及其值
        # print("Initialized with parameters of BiMamba")
        # print(f"d_model: {d_model}")
        # print(f"bidirectional: {bidirectional}")
        # print(f"bidirectional_strategy: {bidirectional_strategy}")
        # print(f"bidirectional_weight_tie: {bidirectional_weight_tie}")
        # print("mamba_kwargs:", mamba_kwargs)

        # Initialized with parameters of BiMamba:
        # d_model: 128
        # bidirectional: True
        # bidirectional_strategy: add
        # bidirectional_weight_tie: True
        # mamba_kwargs: {'layer_idx': 0, 'd_state': 16, 'd_conv': 4, 'expand': 2, 'dt_rank': 'auto', 'dt_min': 0.001, 'dt_max': 0.1, 'dt_init': 'random', 'dt_scale': 1.0, 'dt_init_floor': 0.0001, 'conv_bias': True, 'bias': False, 'use_fast_path': True, 'device': device(type='cuda', index=0), 'dtype': None}

        if bidirectional and bidirectional_strategy is None:
            bidirectional_strategy = "add"  # Default strategy: `add`
        if bidirectional and bidirectional_strategy not in ["add", "ew_multiply"]:
            raise NotImplementedError(f"`{bidirectional_strategy}` strategy for bi-directionality is not implemented!")
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.mamba_fwd = Mamba(
            d_model=d_model,
            **mamba_kwargs
        )
        if bidirectional:
            self.mamba_rev = Mamba(
                d_model=d_model,
                **mamba_kwargs
            )
            if bidirectional_weight_tie:  # Tie in and out projections (where most of param count lies)
                self.mamba_rev.in_proj.weight = self.mamba_fwd.in_proj.weight
                self.mamba_rev.in_proj.bias = self.mamba_fwd.in_proj.bias
                self.mamba_rev.out_proj.weight = self.mamba_fwd.out_proj.weight
                self.mamba_rev.out_proj.bias = self.mamba_fwd.out_proj.bias
        else:
            self.mamba_rev = None

    def forward(self, hidden_states, inference_params=None):
        """Bidirectional-enabled forward pass

        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        # print('hidden_states: ', hidden_states.shape)
        out = self.mamba_fwd(hidden_states, inference_params=inference_params)
        # print(out.shape)
        if self.bidirectional:
            out_rev = self.mamba_rev(
                hidden_states.flip(dims=(1,)),  # Flip along the sequence length dimension
                inference_params=inference_params
            ).flip(dims=(1,))  # Flip back for combining with forward hidden states
            if self.bidirectional_strategy == "add":
                out = out + out_rev
            elif self.bidirectional_strategy == "ew_multiply":
                out = out * out_rev
            else:
                raise NotImplementedError(f"`{self.bidirectional_strategy}` for bi-directionality not implemented!")
        # print(out.shape)
        return out

