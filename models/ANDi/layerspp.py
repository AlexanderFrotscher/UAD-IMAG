# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Layers for defining DDPM++."""

import math
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from models.ANDi.layers import (
    NIN,
    ddpm_conv1x1,
    ddpm_conv3x3,
    dense,
    naive_downsample_2d,
    naive_upsample_2d,
    zero_conv3x3,
    zero_module,
)

conv1x1 = ddpm_conv1x1
conv3x3 = ddpm_conv3x3


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class AttnBlockpp(nn.Module):
    """Channel-wise self-attention block. Modified from DDPM."""

    def __init__(self, channels, rescale=False, init_scale=0.0):
        super().__init__()
        self.GroupNorm_0 = nn.GroupNorm(
            num_groups=min(channels // 4, 32), num_channels=channels, eps=1e-6
        )
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
        self.rescale = rescale

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)

        w = torch.einsum("bchw,bcij->bhwij", q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum("bhwij,bcij->bchw", w, v)
        h = self.NIN_3(h)
        if self.rescale:
            return (x + h) / np.sqrt(2.0)
        else:
            return x + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        rescale=True,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.rescale = rescale
        self.norm = nn.GroupNorm(
            num_groups=min(channels // 4, 32), num_channels=channels, eps=1e-6
        )
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttentionLegacy(self.num_heads)
        # The zero module kills the self attention in the first forward. Over time it gets introduced to the model
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        if self.rescale:
            return (x + h).reshape(b, c, *spatial) / np.sqrt(2.0)
        else:
            return (x + h).reshape(b, c, *spatial)


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        embed_dim: int = 64,
        rescale: bool = True,
        is_causal: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert channels % embed_dim == 0
        self.channels = channels
        num_heads = channels // embed_dim
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        inner_dim = num_heads * embed_dim
        self.rescale = rescale
        # Perform causal masking
        self.is_causal = is_causal

        self.norm = nn.GroupNorm(
            num_groups=min(channels // 4, 32), num_channels=channels, eps=1e-6
        )
        # First reduce the dimension to the number of heads times the embedding dimension
        self.proj_in = nn.Linear(channels, inner_dim)
        # key, query, value projections for all heads in one batch
        self.qkv = nn.Linear(inner_dim, inner_dim * 3, bias=False)
        self.proj_out = zero_module(nn.Linear(inner_dim, channels))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, c, *spatial = x.shape
        h = self.norm(x)
        h = h.permute(0, 2, 3, 1).contiguous()
        h = h.view(b, -1, c)
        h = self.proj_in(h)
        qkv = self.qkv(h)
        query, key, value = qkv.chunk(3, -1)
        query = query.view(b, -1, self.num_heads, self.embed_dim).transpose(1, 2)
        key = key.view(b, -1, self.num_heads, self.embed_dim).transpose(1, 2)
        value = value.view(b, -1, self.num_heads, self.embed_dim).transpose(1, 2)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            a = F.scaled_dot_product_attention(
                query, key, value, attn_mask=None, is_causal=self.is_causal
            )
        a = a.transpose(1, 2).view(b, -1, self.num_heads * self.embed_dim)
        a = self.dropout(self.proj_out(a))
        a = a.view(b, *spatial, c)
        a = a.permute(0, 3, 1, 2).contiguous()
        if self.rescale:
            return (x + a) / np.sqrt(2.0)
        else:
            return x + a


class ResnetBlockBigGANpp(nn.Module):
    def __init__(
        self,
        act,
        in_ch: int,
        out_ch: Union[None, int] = None,
        temb_dim: Union[None, int] = 128,
        up: bool = False,
        down: bool = False,
        dropout: float = 0.1,
        rescale: bool = True,
    ):
        super().__init__()

        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = nn.GroupNorm(
            num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
        )
        self.up = up
        self.down = down
        self.rescale = rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.updown = up or down

        self.Conv_0 = conv3x3(in_ch, out_ch)

        if temb_dim:
            self.emb_layers = nn.Sequential(self.act, dense(temb_dim, 2 * out_ch))

        self.out_layers = nn.Sequential(
            nn.GroupNorm(
                num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6
            ),
            nn.Dropout(dropout),
            zero_conv3x3(out_ch, out_ch),
        )
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch)

    def forward(self, x, temb=None):
        h = self.act(self.GroupNorm_0(x))

        if self.updown:
            if self.up:
                h = self.Conv_0(h)
                h = naive_upsample_2d(h, factor=2)
                x = naive_upsample_2d(x, factor=2)
            elif self.down:
                h = naive_downsample_2d(h, factor=2)
                x = naive_downsample_2d(x, factor=2)
                h = self.Conv_0(h)
        else:
            h = self.Conv_0(h)

        if temb is not None:
            emb_out = self.emb_layers(temb)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]

            # adaptive group norm form OpenAI
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)

        else:
            h = self.out_layers(h)

        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)

        if self.rescale:
            return (x + h) / np.sqrt(2.0)
        else:
            return x + h
