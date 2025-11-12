import functools

import layerspp
import torch
import torch.nn as nn

from models.ANDi.layers import get_act, zero_module

ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
batchnorm = layerspp.BatchNorm32


class UNet(nn.Module):
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
        self.attn_emb_dim = attn_emb_dim = config.model.attn_emb_dim
        assert nf % attn_emb_dim == 0
        for resolution in attn_resolutions:
            assert resolution in all_resolutions

        dropout = config.model.dropout
        dropout_at = config.model.dropout_at
        assert dropout_at in all_resolutions

        self.rescale = rescale = config.model.rescale
        self.condition = condition = config.model.condition
        modules = []

        AttnBlock = functools.partial(
            layerspp.CausalSelfAttention,
            embed_dim=attn_emb_dim,
            rescale=rescale,
            dropout=dropout,
        )

        ResnetBlock = functools.partial(
            ResnetBlockBigGAN,
            act=act,
            rescale=rescale,
        )

        channels = config.data.num_channels
        if condition:
            channels += 1

        modules.append(conv3x3(channels, nf))
        hs_c = [nf]
        in_ch = nf

        # Downsampling block
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
        modules.append(batchnorm(in_ch, swish=1.0))
        modules.append(zero_module(conv3x3(in_ch, config.data.num_channels)))

        self.all_modules = nn.ModuleList(modules)

    def forward(self, x):
        modules = self.all_modules
        m_idx = 0

        hs = [modules[m_idx](x)]
        m_idx += 1

        # Downsampling block
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks[i_level]):
                h = modules[m_idx](hs[-1])
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                h = modules[m_idx](hs[-1])
                m_idx += 1
                hs.append(h)

        # Bottle Neck
        h = hs[-1]
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks[i_level] + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1))
                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if i_level != 0:
                h = modules[m_idx](h)
                m_idx += 1

        assert not hs

        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1


        assert m_idx == len(modules)
        return h

    def update_step(self, step, device):
        self.step = torch.ones(1, device=device) * step
