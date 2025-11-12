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

import functools

import torch
import torch.nn as nn

import models.ANDi.layers as layers
import models.ANDi.layerspp as layerspp

ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
default_initializer = layers.default_init


class DDPMpp(nn.Module):
    """DDPM++ model"""

    def __init__(self, config):
        super().__init__()
        self.register_buffer("step", torch.ones(1) * 0)
        self.act = act = get_act(config)
        self.nf = nf = config.model.nf
        ch_mult = config.model.ch_mult
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        self.num_resolutions = num_resolutions = len(ch_mult)
        all_resolutions = [
            config.data.image_size // (2**i) for i in range(num_resolutions)
        ]

        self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
        attn_emb_dim = config.model.attn_emb_dim
        assert nf % attn_emb_dim == 0
        for resolution in attn_resolutions:
            assert resolution in all_resolutions

        dropout = config.model.dropout
        dropout_at = config.model.dropout_at
        assert dropout_at in all_resolutions

        self.rescale = rescale = config.model.rescale
        self.embedding_type = embedding_type = config.model.embedding_type
        assert embedding_type in ["fourier", "positional", None]

        modules = []
        # timestep/noise_level embedding; only for continuous training
        if embedding_type:
            if embedding_type == "fourier":
                # Gaussian Fourier features embeddings.
                assert config.training.continuous, (
                    "Fourier features are only used for continuous training."
                )

                modules.append(
                    layerspp.GaussianFourierProjection(
                        embedding_size=nf, scale=config.model.fourier_scale
                    )
                )
                embed_dim = 2 * nf

            elif embedding_type == "positional":
                embed_dim = nf

            else:
                raise ValueError(f"embedding type {embedding_type} unknown.")

            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

            ResnetBlock = functools.partial(
                ResnetBlockBigGAN, act=act, rescale=rescale, temb_dim=nf * 4
            )

        else:
            ResnetBlock = functools.partial(
                ResnetBlockBigGAN, act=act, rescale=rescale, temb_dim=None
            )

        AttnBlock = functools.partial(
            layerspp.CausalSelfAttention,
            embed_dim=attn_emb_dim,
            rescale=rescale,
            dropout=dropout,
        )

        # Downsampling block
        channels = config.data.num_channels
        modules.append(conv3x3(channels, nf))
        hs_c = [nf]
        in_ch = nf

        for i_level in range(num_resolutions):
            if all_resolutions[i_level] > dropout_at:
                my_drop = 0
            else:
                my_drop = dropout
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks[i_level]):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch, dropout=my_drop))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                modules.append(ResnetBlock(down=True, in_ch=in_ch, dropout=my_drop))
                hs_c.append(in_ch)

        in_ch = hs_c[-1]

        # bottle neck
        modules.append(ResnetBlock(in_ch=in_ch, dropout=dropout))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch, dropout=dropout))

        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            if all_resolutions[i_level] > dropout_at:
                my_drop = 0
            else:
                my_drop = dropout
            for i_block in range(num_res_blocks[i_level] + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(
                    ResnetBlock(
                        in_ch=in_ch + hs_c.pop(), out_ch=out_ch, dropout=my_drop
                    )
                )
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if i_level != 0:
                modules.append(ResnetBlock(in_ch=in_ch, up=True, dropout=my_drop))

        assert not hs_c
        modules.append(
            nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
        )
        modules.append(layers.zero_conv3x3(in_ch, channels))
        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, time_cond=None):
        modules = self.all_modules
        m_idx = 0
        temb = None
        if self.embedding_type:
            if self.embedding_type == "fourier":
                # Gaussian Fourier features embeddings.
                temb = modules[m_idx](torch.log(time_cond))
                m_idx += 1

            elif self.embedding_type == "positional":
                # Sinusoidal positional embeddings.
                timesteps = time_cond
                temb = layers.get_timestep_embedding(timesteps, self.nf)

            else:
                raise ValueError(f"embedding type {self.embedding_type} unknown.")

            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1

        # Downsampling block
        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks[i_level]):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                hs.append(h)

        # Bottle Neck
        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks[i_level] + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if i_level != 0:
                h = modules[m_idx](h, temb)
                m_idx += 1

        assert not hs

        h = self.act(modules[m_idx](h))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1

        assert m_idx == len(modules)
        return h

    def update_step(self, step, device):
        self.step = torch.ones(1, device=device) * step
