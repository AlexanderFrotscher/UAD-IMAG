from functools import partial
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor
from torchvision import models as tv_models

from utils.feature_extractor import NetworkFeatureExtractor


def vanilla_feature_encoder(
    in_channels: int,
    hidden_dims: List[int],
    norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
    dropout: float = 0.0,
    bias: bool = False,
):
    """
    Vanilla feature encoder.
    Args:
        in_channels (int): Number of input channels
        hidden_dims (List[int]): List of hidden channel dimensions
        norm_layer (Optional[Callable[..., nn.Module]] = nn.BatchNorm2d): Normalization layer to use
        dropout (float): Dropout rate
        bias (bool): Whether to use bias
    Returns:
        encoder (nn.Module): The encoder
    """
    ks = 6  # Kernel size
    pad = 2  # Padding

    # Build encoder
    enc = nn.Sequential()
    # for i, hidden_dim in enumerate(hidden_dims):
    for i in range(len(hidden_dims)):
        # Add a new layer
        layer = nn.Sequential()

        # Convolution
        layer.add_module(
            f"encoder_conv_{i}",
            nn.Conv2d(
                in_channels, hidden_dims[i], ks, stride=2, padding=pad, bias=bias
            ),
        )

        # If not last layer
        # if i < len(hidden_dims) - 1:
        # Normalization
        if norm_layer is not None:
            layer.add_module(f"encoder_norm_{i}", norm_layer(hidden_dims[i]))

        # LeakyReLU
        layer.add_module(f"encoder_relu_{i}", nn.LeakyReLU())

        # Dropout
        if dropout > 0:
            layer.add_module(f"encoder_dropout_{i}", nn.Dropout2d(dropout))

        # Add the layer to the encoder
        enc.add_module(f"encoder_layer_{i}", layer)

        in_channels = hidden_dims[i]

    # Final layer
    enc.add_module(
        "encoder_conv_final",
        nn.Conv2d(in_channels, in_channels, 5, stride=1, padding=pad, bias=bias),
    )

    return enc


def vanilla_feature_decoder(
    out_channels: int,
    hidden_dims: List[int],
    norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
    dropout: float = 0.0,
    bias: bool = False,
):
    """
    Vanilla feature decoder.
    Args:
        out_channels (int): Number of output channels
        hidden_dims (List[int]): List of hidden channel dimensions
        norm_layer (Optional[Callable[..., nn.Module]] = nn.BatchNorm2d): Normalization layer to use
        dropout (float): Dropout rate
        bias (bool): Whether to use bias
    Returns:
        decoder (nn.Module): The decoder
    """
    ks = 6  # Kernel size
    pad = 2  # Padding

    hidden_dims = [out_channels] + hidden_dims

    # Build decoder
    dec = nn.Sequential()
    for i in range(len(hidden_dims) - 1, 0, -1):
        # Add a new layer
        layer = nn.Sequential()

        if i == len(hidden_dims) - 1:
            layer.add_module(
                f"decoder_tconv_{i}",
                nn.ConvTranspose2d(
                    hidden_dims[i],
                    hidden_dims[i - 1],
                    7,
                    stride=2,
                    padding=pad,
                    bias=bias,
                ),
            )
        else:
            # Transposed convolution
            layer.add_module(
                f"decoder_tconv_{i}",
                nn.ConvTranspose2d(
                    hidden_dims[i],
                    hidden_dims[i - 1],
                    ks,
                    stride=2,
                    padding=pad,
                    bias=bias,
                ),
            )

        # Normalization
        if norm_layer is not None:
            layer.add_module(f"decoder_norm_{i}", norm_layer(hidden_dims[i - 1]))

        # LeakyReLU
        layer.add_module(f"decoder_relu_{i}", nn.LeakyReLU())

        # Dropout
        if dropout > 0:
            layer.add_module(f"decoder_dropout_{i}", nn.Dropout2d(dropout))

        # Add the layer to the decoder
        dec.add_module(f"decoder_layer_{i}", layer)

    # Final layer
    dec.add_module(
        "decoder_conv_final", nn.Conv2d(hidden_dims[0], out_channels, 1, bias=False)
    )

    return dec


class FeatureAutoencoder(nn.Module):
    def __init__(self, config, first_channels):
        super().__init__()

        assert config.model.version in ("vanilla", "v2"), (
            "The model version needs to be vanilla or v2."
        )
        if config.model.version == "vanilla":
            self.enc = vanilla_feature_encoder(
                first_channels,
                config.model.channels,
                norm_layer=partial(nn.BatchNorm2d),
                dropout=config.model.dropout,
            )
            self.dec = vanilla_feature_decoder(
                first_channels,
                config.model.channels,
                norm_layer=partial(nn.BatchNorm2d),
                dropout=config.model.dropout,
            )
        elif config.model.version == "v2":
            self.enc = feature_encoder(
                config.model.blocks_per_channel,
                config.model.channels,
                config.model.dropout,
                first_channels,
                config.model.kernel_size,
            )
            self.dec = feature_decoder(
                config.model.blocks_per_channel,
                config.model.channels,
                config.model.dropout,
                first_channels,
                config.model.kernel_size,
            )

    def forward(self, x: Tensor) -> Tensor:
        z = self.enc(x)
        rec = self.dec(z)
        return rec


class FeatureReconstructor(nn.Module):
    backbones = {
        "ResNet18": (partial(tv_models.resnet18), "IMAGENET1K_V1"),
        "EfficientNet-B4": (partial(tv_models.efficientnet_b4), "IMAGENET1K_V1"),
    }

    def __init__(self, config):
        super().__init__()
        self.register_buffer("step", torch.ones(1) * 0)
        self.featmap_size = config.model.featmap_size
        self.backbone = config.model.backbone
        self.normalize = config.model.normalize

        if self.backbone not in self.backbones:
            raise KeyError(
                f"Invalid backbone {self.backbone}. "
                "You can use the string ResNet18 or EfficientNet-B4."
            )
        
        model, weights = self.backbones[self.backbone]
        self.extractor = NetworkFeatureExtractor(
            model(weights = weights),
            config.model.layers,
        )
        self.extractor.eval()
        feature_dimensions, _ = self.extractor.feature_dimensions(
            config.model.input_shape
        )
        self.ae = FeatureAutoencoder(config, sum(feature_dimensions))

    def forward(self, x: Tensor):
        with torch.no_grad():
            normalize = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
                if self.normalize:
                    x = normalize(x)
            feats = self.extractor(x)
            features = []
            for feat_map in feats.values():
                feat_map = F.interpolate(
                    feat_map,
                    size=self.featmap_size,
                    mode="bilinear",
                    align_corners=True,
                )
                features.append(feat_map)

            # Concatenate to tensor
            features = torch.cat(features, dim=1)
        return features, self.ae(features)

    def update_step(self, step, device):
        self.step = torch.ones(1, device=device) * step


class FeatureBlock(nn.Module):
    def __init__(
        self, input_dim, output_dim, kernel_size, dropout, up, down, ex_kernel=None
    ):
        super().__init__()

        self.in_ch = input_dim
        self.out_ch = output_dim
        assert kernel_size % 2
        self.kernel = kernel_size
        self.padding = (kernel_size - 1) // 2

        img_kernel = 0  # needed for last dim
        if ex_kernel:  # last dimension does not match up with img_size 224
            img_kernel = 1
        if ex_kernel is None:
            ex_kernel = kernel_size

        # residual behaviour
        self.up = up
        self.down = down
        self.updown = down or up
        self.resid = down or up
        if input_dim == output_dim:
            self.resid = True

        self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(self.in_ch)
        if self.up:
            self.conv1 = nn.ConvTranspose2d(
                self.in_ch, self.out_ch, ex_kernel + 1, stride=2, padding=self.padding
            )
        elif self.down:
            self.conv1 = nn.Conv2d(
                self.in_ch, self.out_ch, self.kernel, stride=2, padding=self.padding
            )
        else:
            self.conv1 = nn.Conv2d(
                self.in_ch, self.out_ch, self.kernel, padding=self.padding
            )

        self.out_layers = nn.Sequential(
            nn.BatchNorm2d(self.out_ch),
            self.act,
            nn.Dropout(dropout),
            nn.Conv2d(self.out_ch, self.out_ch, self.kernel, padding=self.padding),
        )

        ### Taking care of the img dimension for residual connection
        if self.down:
            self.conv2 = nn.Conv2d(
                self.in_ch, self.out_ch, kernel_size=1, stride=2, bias=False
            )
        elif self.up:
            if (
                img_kernel
            ):  # this is only for the last dimension needing to go from size 4x4 to 7x7
                self.conv2 = nn.ConvTranspose2d(
                    self.in_ch,
                    self.out_ch,
                    kernel_size=img_kernel,
                    stride=2,
                    bias=False,
                )
            else:
                self.conv2 = nn.ConvTranspose2d(
                    self.in_ch, self.out_ch, kernel_size=2, stride=2, bias=False
                )

    def forward(self, x):
        h = self.act(self.norm(x))
        h = self.conv1(h)
        result = self.out_layers(h)
        if self.updown:
            x = self.conv2(x)
        if self.resid:
            result += x
        return result


def feature_encoder(blocks_per_ch, channels, dropout, input_channels, kernel_size):
    enc = []
    enc.append(nn.Conv2d(input_channels, channels[0], kernel_size=3, padding=1))
    for i in range(len(channels)):
        down, up = True, False
        in_ch = channels[i]
        out_ch = in_ch
        for i_block in range(blocks_per_ch[i]):
            if i_block == blocks_per_ch[i] - 1:
                try:
                    out_ch = channels[i + 1]
                except IndexError:
                    out_ch = in_ch
            enc.append(FeatureBlock(in_ch, out_ch, kernel_size, dropout, up, down))
            down = False
    for p in enc[0].parameters():
        p.detach().zero_()
    enc = nn.Sequential(*enc)
    return enc


def feature_decoder(blocks_per_ch, channels, dropout, output_channels, kernel_size):
    dec = []
    for i in range(len(channels) - 1, -1, -1):
        down, up = False, False
        in_ch = channels[i]
        out_ch = channels[i - 1]
        ex_kernel = None
        if i == len(channels) - 1:
            ex_kernel = kernel_size - 1
        if i == 0:
            out_ch = in_ch
        for i_block in range(blocks_per_ch[i]):
            if i_block == blocks_per_ch[i] - 1:
                up = True
            dec.append(
                FeatureBlock(in_ch, out_ch, kernel_size, dropout, up, down, ex_kernel)
            )
            ex_kernel = None
            in_ch = out_ch
    dec.append(nn.Conv2d(channels[0], output_channels, kernel_size=3, padding=1))
    for p in dec[-1].parameters():
        p.detach().zero_()
    dec = nn.Sequential(*dec)
    return dec
