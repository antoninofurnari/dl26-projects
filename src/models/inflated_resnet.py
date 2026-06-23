"""
I3D-style inflation of an ImageNet-pretrained ResNet-50.

Why this exists
---------------
Our previous backbone (R3D-18) was pretrained on Kinetics-400, and our target
is a Kinetics subset. That caused information leakage: features were already
aligned to the target, inflating zero-shot accuracy and leaving no real domain
shift for the adaptation to correct. To get an honest setting we instead use a
backbone whose weights come from **ImageNet** (images, never Kinetics).

Following I3D (Carreira & Zisserman, "Quo Vadis", 2017) we take a 2D ResNet-50
pretrained on ImageNet and "inflate" its 2D kernels (k x k) into 3D kernels
(t x k x k), transferring the ImageNet weights. We use the centered
initialization (Detect-and-Track, 2018): the temporal-center slice gets the 2D
weights, the rest is zero, so the 3D net initially reproduces the 2D behavior.

This module is self-contained (adapted from hassony2/inflated_convnets_pytorch,
modernized for current torchvision) so nothing extra needs to be installed on
the offline cluster.

Pooled output dimension for ResNet-50 = 2048.
"""

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Inflation primitives: 2D layer -> 3D layer with weight transfer
# --------------------------------------------------------------------------- #
def inflate_conv(conv2d, time_dim=1, time_padding=0, center=True):
    """Inflate a Conv2d into a Conv3d, transferring weights.

    center=True: put the 2D kernel on the temporal-center slice, zeros elsewhere.
    center=False: replicate across time and divide by time_dim (mean).
    """
    conv3d = nn.Conv3d(
        conv2d.in_channels, conv2d.out_channels,
        kernel_size=(time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1]),
        stride=(1, conv2d.stride[0], conv2d.stride[1]),
        padding=(time_padding, conv2d.padding[0], conv2d.padding[1]),
        bias=conv2d.bias is not None,
    )
    w2d = conv2d.weight.data  # (out, in, kh, kw)
    w3d = torch.zeros(conv2d.out_channels, conv2d.in_channels,
                      time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1])
    if center:
        mid = time_dim // 2
        w3d[:, :, mid, :, :] = w2d
    else:
        for t in range(time_dim):
            w3d[:, :, t, :, :] = w2d / time_dim
    conv3d.weight = nn.Parameter(w3d)
    if conv2d.bias is not None:
        conv3d.bias = nn.Parameter(conv2d.bias.data.clone())
    return conv3d


def inflate_batch_norm(bn2d):
    bn3d = nn.BatchNorm3d(bn2d.num_features)
    bn3d.weight = nn.Parameter(bn2d.weight.data.clone())
    bn3d.bias = nn.Parameter(bn2d.bias.data.clone())
    bn3d.running_mean = bn2d.running_mean.clone()
    bn3d.running_var = bn2d.running_var.clone()
    return bn3d


def inflate_pool(pool2d, time_dim=1, time_stride=1, time_padding=0):
    if isinstance(pool2d, nn.MaxPool2d):
        return nn.MaxPool3d(
            kernel_size=(time_dim, pool2d.kernel_size, pool2d.kernel_size),
            stride=(time_stride, pool2d.stride, pool2d.stride),
            padding=(time_padding, pool2d.padding, pool2d.padding),
        )
    raise ValueError(f"Unsupported pool type: {type(pool2d)}")


# --------------------------------------------------------------------------- #
# Inflated Bottleneck block (ResNet-50 uses Bottleneck)
# --------------------------------------------------------------------------- #
class Bottleneck3d(nn.Module):
    def __init__(self, bottleneck2d, inflate_time=True):
        super().__init__()
        # conv1: 1x1 -> inflate temporally (3) on alternating blocks for some temporal
        # receptive field; here we inflate conv2 (the 3x3) center-wise as in the repo.
        self.conv1 = inflate_conv(bottleneck2d.conv1, time_dim=1)
        self.bn1 = inflate_batch_norm(bottleneck2d.bn1)
        self.conv2 = inflate_conv(bottleneck2d.conv2, time_dim=1)
        self.bn2 = inflate_batch_norm(bottleneck2d.bn2)
        self.conv3 = inflate_conv(bottleneck2d.conv3, time_dim=1)
        self.bn3 = inflate_batch_norm(bottleneck2d.bn3)
        self.relu = nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = nn.Sequential(
                inflate_conv(bottleneck2d.downsample[0], time_dim=1),
                inflate_batch_norm(bottleneck2d.downsample[1]),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class InflatedResNet50(nn.Module):
    """ResNet-50 inflated to 3D, ImageNet weights transferred. Outputs pooled
    2048-dim features (fc removed)."""

    def __init__(self, resnet2d):
        super().__init__()
        self.conv1 = inflate_conv(resnet2d.conv1, time_dim=3, time_padding=1, center=True)
        self.bn1 = inflate_batch_norm(resnet2d.bn1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = inflate_pool(resnet2d.maxpool, time_dim=1, time_stride=1)

        self.layer1 = self._inflate_layer(resnet2d.layer1)
        self.layer2 = self._inflate_layer(resnet2d.layer2)
        self.layer3 = self._inflate_layer(resnet2d.layer3)
        self.layer4 = self._inflate_layer(resnet2d.layer4)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.feature_dim = 2048

    @staticmethod
    def _inflate_layer(layer2d):
        return nn.Sequential(*[Bottleneck3d(b) for b in layer2d])

    def forward(self, x):  # x: (B, C, T, H, W)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)  # (B, 2048)


def build_inflated_resnet50(weights_path=None):
    """Build an inflated ResNet-50 from an ImageNet-pretrained 2D ResNet-50.

    weights_path: local path to the 2D ResNet-50 ImageNet state_dict (.pth),
                  required offline. If None, torchvision downloads them.
    """
    from torchvision.models import resnet50, ResNet50_Weights

    if weights_path is None:
        resnet2d = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    else:
        resnet2d = resnet50(weights=None)
        state = torch.load(weights_path, map_location="cpu")
        resnet2d.load_state_dict(state)
        print(f"[backbone] loaded local ImageNet ResNet-50 weights from {weights_path}")

    model = InflatedResNet50(resnet2d)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
