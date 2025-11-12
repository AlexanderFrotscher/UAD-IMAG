import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from models.ANDi.layers import (
    ddpm_conv1x1,
    ddpm_conv3x3,
    naive_downsample_2d,
    naive_upsample_2d,
    zero_module,
)

conv1x1 = ddpm_conv1x1
conv3x3 = ddpm_conv3x3


class BatchNorm32(nn.BatchNorm2d):
    def __init__(self, num_channels, swish, eps=1e-5):
        super().__init__(num_features=num_channels, eps=eps)
        self.swish = swish

    def forward(self, x):
        y = super().forward(x.float()).to(x.dtype)
        if self.swish == 1.0:
            y = F.silu(y)
        elif self.swish:
            y = y * F.sigmoid(y * float(self.swish))
        return y


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        embed_dim: int = 64,
        rescale: bool = True,
        is_causal: bool = True,
        dropout=0.1,
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

        self.norm = BatchNorm32(channels, swish=0.0)
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
        self, act, in_ch, out_ch=None, up=False, down=False, dropout=0.1, rescale=True
    ):
        super().__init__()

        out_ch = out_ch if out_ch else in_ch
        self.up = up
        self.down = down
        self.rescale = rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.updown = up or down

        self.norm = BatchNorm32(in_ch, swish=1.0)
        self.Conv_1 = conv3x3(in_ch, out_ch)

        self.out_layers = nn.Sequential(
            BatchNorm32(out_ch, swish=0.0),
            self.act,
            nn.Dropout(dropout),
            zero_module(conv3x3(out_ch, out_ch)),
        )
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch)

    def forward(self, x):
        h = self.norm(x)
        if self.updown:
            if self.up:
                h = self.Conv_1(h)
                h = naive_upsample_2d(h, factor=2)
                x = naive_upsample_2d(x, factor=2)
            elif self.down:
                h = naive_downsample_2d(h, factor=2)
                x = naive_downsample_2d(x, factor=2)
                h = self.Conv_1(h)
        else:
            h = self.Conv_1(h)

        h = self.out_layers(h)

        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)

        if self.rescale:
            return (x + h) / np.sqrt(2.0)
        else:
            return x + h
