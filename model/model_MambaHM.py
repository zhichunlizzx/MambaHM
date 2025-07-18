#!/usr/bin/env python
# Copyright 2023 Z Zhang

# BioSeq2Seq, Version 1.0;
# you may not use this file except in compliance with the License.
# Use of this code requires following originality guidelines
# and declaring the source of the code.
# email:zhichunli@mail.dlut.edu.cn
# =========================================================================
import torch
from einops.layers.torch import Rearrange
from .BiMamba_block import create_block

torch.manual_seed(0)

class Residual(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self._module = module

    def forward(self, inputs: torch.Tensor):
        return inputs + self._module(inputs)
    

def pooling_module(kind, pool_size):
    """Pooling module wrapper."""
    if kind == "max":
        return torch.nn.MaxPool1d(kernel_size=pool_size)
    else:
        raise ValueError(f"Invalid pooling kind: {kind}.")
    

class TargetLengthCrop1D(torch.nn.Module):
    def __init__(self, target_length: int, name="target_length_crop", **kwargs):
        super().__init__()
        self._target_length = target_length

    def forward(self, inputs):
        trim = (inputs.shape[-2] - self._target_length) // 2
        if trim < 0:
            raise ValueError("inputs longer than target length")

        return inputs[..., trim:-trim, :]


def gelu(x: torch.Tensor, approximate: bool = True) -> torch.Tensor:
    return torch.sigmoid(1.702 * x) * x


class GELU(torch.nn.Module):
    def __init__(self, approximate: bool = True):
        super().__init__()
        self.approximate = approximate
        self.supports_masking = True
    
    def forward(self, input:torch.Tensor) -> torch.Tensor:
        return gelu(input, approximate=self.approximate)

def conv_block(dim_in, dim_out, width=1, w_init=None):
    bn = torch.nn.BatchNorm1d(dim_in, eps=1e-3, momentum=0.9)
    gl = GELU()
    conv = torch.nn.Conv1d(dim_in, dim_out, kernel_size=width, padding='same')

    if w_init is not None:
        w_init(conv.weight)
 
    conv_block_module = torch.nn.Sequential(bn, gl, conv)
    # conv_block_module = torch.nn.Sequential(gl, conv)
    return conv_block_module


class MambaHM(torch.nn.Module):
    """Main model."""
    def __init__(self,
               channels: int = 768,
               num_transformer_layers: int = 11,
               num_heads: int = 8,
               output_channels = 1,
               target_length=896,
               pooling_type: str = 'max',
               device = torch.device("cuda:0"),
               ):
        super().__init__()
        self.window_size = 128
        self.device = device
        dropout_rate = 0.4
        # 和attention相关, 里边有想关除法计算
        assert channels % num_heads == 0, ('channels needs to be divisible '
                                       f'by {num_heads}')
        
        whole_attention_kwargs = {
                'channels': channels,
                'attention_dropout_rate': 0.05,
                'initializer': None,
                'key_size': 64,
                'num_heads': num_heads,
                'num_relative_position_features': channels // num_heads,
                'positional_dropout_rate': 0.01,
                'relative_position_functions': [
                    'positional_features_exponential',
                    'positional_features_central_mask',
                    'positional_features_gamma'
                ],
                'relative_positions': True,
                'scaling': True,
                'value_size': channels // num_heads,
                'zero_initialize': True,
                'device': self.device,
            }
        
        stem = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=4, out_channels=channels, kernel_size=15, padding='same'),
            pooling_module(pooling_type, pool_size=16)
        )

        stem_seq = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=channels, kernel_size=15, padding='same'),
            pooling_module(pooling_type, pool_size=16)
        )
        
        self.concat_x_proseq = torch.nn.Linear(channels * 2, channels * 2)

        factory_kwargs = {"device": device, "dtype": None}

        self.mamba = torch.nn.ModuleList(
            [
                create_block(
                    channels,
                    ssm_cfg={'d_state': 16, 'd_conv': 4, 'expand': 2, 'dt_rank': 'auto', 'dt_min': 0.001, 'dt_max': 0.1, 'dt_init': 'random', 'dt_scale': 1.0, 'dt_init_floor': 0.0001, 'conv_bias': True, 'bias': False, 'use_fast_path': True},
                    norm_epsilon=1e-05,
                    rms_norm=True,
                    residual_in_fp32=False,
                    fused_add_norm=True,
                    layer_idx=i,
                    bidirectional=True,
                    bidirectional_strategy='add',
                    bidirectional_weight_tie=True,
                    rcps=True,
                    **factory_kwargs,
                )
                for i in range(2)
            ]
        )

        crop_final = TargetLengthCrop1D(target_length, name='target_input')

        final_pointwise = torch.nn.Sequential(
            conv_block(dim_in=channels * 2, dim_out=channels * 2, width=1),
            torch.nn.Dropout(p=dropout_rate / 8),
            GELU()
        )

        self.conv = torch.nn.Sequential(
                            Rearrange('b n d -> b d n'),
                            stem,
                            Rearrange('b d n -> b n d'),
                            )
        
        self.conv_seq = torch.nn.Sequential(
                            Rearrange('b n d -> b d n'),
                            stem_seq,
                            Rearrange('b d n -> b n d'),
                            )

        self.heads = torch.nn.Sequential(
                            crop_final,
                            Rearrange('b n d -> b d n'),
                            final_pointwise,
                            Rearrange('b d n -> b n d'),
                            torch.nn.Linear(2*channels, output_channels),
                            torch.nn.Softplus()
                            )
        
    def forward(self, dna, seq):
        input_dna_encoding = dna
        input_seq_feature = seq

        out_dna = self.conv(input_dna_encoding)
        out_seq = self.conv_seq(input_seq_feature)

        outputs = torch.cat([out_dna, out_seq], dim=-1)

        outputs = self.concat_x_proseq(outputs)

        residual = None
        for layer in self.mamba:
            # TODO: Add support for gradient checkpointing
            outputs, residual = layer(
                outputs, residual, inference_params=None
            )

        outputs = self.heads(outputs)

        return outputs
    