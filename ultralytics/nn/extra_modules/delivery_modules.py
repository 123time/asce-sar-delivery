# Ultralytics YOLO, AGPL-3.0 license
"""Focused custom modules required by the delivery model YAML."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import Bottleneck, C3k, C3k2
from ultralytics.nn.modules.conv import Conv

from .shiftwise_conv import ReparamLargeKernelConv

__all__ = (
    "EMAAttention",
    "SPDConv",
    "DySample",
    "EMA",
    "SWC",
    "FMKernel",
)


class EMAAttention(nn.Module):
    """Efficient multi-scale attention block."""

    def __init__(self, channels, factor=8):
        super().__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class SPDConv(nn.Module):
    """Space-to-depth convolution."""

    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        return self.conv(x)


class Bottleneck_EMA(nn.Module):
    """YOLO bottleneck followed by EMA attention."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.attention = EMAAttention(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.attention(self.cv2(self.cv1(x)))
        return x + y if self.add else y


class C3k_EMA(C3k):
    """C3k block using EMA bottlenecks."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_EMA(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class EMA(C3k2):
    """C3k2 block using EMA bottlenecks."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_EMA(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_EMA(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class DySample(nn.Module):
    """Dynamic upsampling module."""

    def __init__(self, in_channels, scale=2, style="lp", groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ["lp", "pl"]
        if style == "pl":
            assert in_channels >= scale**2 and in_channels % scale**2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == "pl":
            in_channels = in_channels // scale**2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale**2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            self.constant_init(self.scope, val=0.0)

        self.register_buffer("init_pos", self._init_pos())

    @staticmethod
    def normal_init(module, mean=0, std=1, bias=0):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def constant_init(module, val, bias=0):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h], indexing="ij")).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        b, _, h, w = offset.shape
        offset = offset.view(b, 2, -1, h, w)
        coords_h = torch.arange(h) + 0.5
        coords_w = torch.arange(w) + 0.5
        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
            .to(x.device)
        )
        normalizer = torch.tensor([w, h], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = (
            F.pixel_shuffle(coords.view(b, -1, h, w), self.scale)
            .view(b, 2, -1, self.scale * h, self.scale * w)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        return F.grid_sample(
            x.reshape(b * self.groups, -1, h, w),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).reshape((b, -1, self.scale * h, self.scale * w))

    def forward_lp(self, x):
        if hasattr(self, "scope"):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, "scope"):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == "pl":
            return self.forward_pl(x)
        return self.forward_lp(x)


class Bottleneck_SWC(Bottleneck):
    """Standard bottleneck with shift-wise convolution."""

    def __init__(self, c1, c2, kernel_size, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        self.cv2 = ReparamLargeKernelConv(c2, c2, kernel_size, groups=(c2 // 16))


class C3k_SWC(C3k):
    """C3k block using shift-wise convolution bottlenecks."""

    def __init__(self, c1, c2, n=1, kernel_size=13, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_SWC(c_, c_, kernel_size, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class SWC(C3k2):
    """C3k2 block using shift-wise convolution bottlenecks."""

    def __init__(self, c1, c2, n=1, kernel_size=13, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_SWC(self.c, self.c, 2, kernel_size, shortcut, g)
            if c3k
            else Bottleneck_SWC(self.c, self.c, kernel_size, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )


class FGM(nn.Module):
    """Frequency gated module used by OmniKernel."""

    def __init__(self, dim):
        super().__init__()
        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)
        x2_fft = torch.fft.fft2(x2, norm="backward")
        out = torch.fft.ifft2(x1 * x2_fft, dim=(-2, -1), norm="backward")
        out = torch.abs(out)
        return out * self.alpha + x * self.beta


class OmniKernel(nn.Module):
    """Omni-kernel spatial/frequency mixing block."""

    def __init__(self, dim):
        super().__init__()
        ker = 31
        pad = ker // 2
        self.in_conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), nn.GELU())
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1, ker), padding=(0, pad), groups=dim)
        self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker, 1), padding=(pad, 0), groups=dim)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, groups=dim)
        self.act = nn.ReLU()
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fgm = FGM(dim)

    def forward(self, x):
        out = self.in_conv(x)
        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out, norm="backward")
        x_fca = torch.fft.ifft2(x_att * x_fft, dim=(-2, -1), norm="backward")
        x_fca = torch.abs(x_fca)
        x_att = self.conv(self.pool(x_fca))
        x_sca = self.fgm(x_att * x_fca)
        out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out) + x_sca
        return self.out_conv(self.act(out))


class FMKernel(nn.Module):
    """CSP wrapper for OmniKernel."""

    def __init__(self, dim, e=0.25):
        super().__init__()
        self.e = e
        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(dim, dim, 1)
        self.m = OmniKernel(int(dim * self.e))

    def forward(self, x):
        c_ok = int(self.cv1.conv.out_channels * self.e)
        c_identity = int(self.cv1.conv.out_channels * (1 - self.e))
        ok_branch, identity = torch.split(self.cv1(x), [c_ok, c_identity], dim=1)
        return self.cv2(torch.cat((self.m(ok_branch), identity), 1))
