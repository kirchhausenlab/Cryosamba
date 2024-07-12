import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils import softsplat

def backwarp(tenInput, tenFlow): 
    
    tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]),
            tenFlow.shape[3], device=tenFlow.device).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
    tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]),
            tenFlow.shape[2], device=tenFlow.device).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

    backwarp_tenGrid = torch.cat([ tenHor, tenVer ], 1)

    backwarp_tenPartial = tenFlow.new_ones([ tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3] ])

    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
    tenInput = torch.cat([ tenInput, backwarp_tenPartial ], 1)

    tenOutput = F.grid_sample(input=tenInput,
            grid=(backwarp_tenGrid + tenFlow).permute(0, 2, 3, 1),
            mode='bilinear', padding_mode='zeros', align_corners=False)

    tenMask = tenOutput[:, -1:, :, :]; tenMask[tenMask > 0.999] = 1.0; tenMask[tenMask < 1.0] = 0.0

    return tenOutput[:, :-1, :, :] * tenMask

def warp_fn(tenInput, tenFlow, tenMetric=None, strType='average'):
    return softsplat.FunctionSoftsplat(tenInput=tenInput, tenFlow=tenFlow, tenMetric=tenMetric, strType=strType)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding_mode='zeros', bias=True, act='prelu'):
        super().__init__()

        if kernel_size % 2 == 0:
            stride = kernel_size
            padding = 0
        else:
            stride = 1
            padding = (kernel_size-1)//2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, bias=bias)
        if act=='prelu':
            self.act = nn.PReLU(out_channels) 
        elif act=='lrelu':
            self.act = nn.LeakyReLU(inplace=False, negative_slope=0.1)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)

        return x

def conv2(in_planes, out_planes, kernel_size=3, stride=2, padding_mode='zeros'):
    return nn.Sequential(
        ConvBlock(in_planes, out_planes, kernel_size=stride, padding_mode=padding_mode),
        ConvBlock(out_planes, out_planes, kernel_size=kernel_size, padding_mode=padding_mode)
    )

def conv4(in_planes, out_planes, kernel_size=3, stride=2, padding_mode='zeros'):
    return nn.Sequential(
        ConvBlock(in_planes, out_planes, kernel_size=stride, padding_mode=padding_mode),
        *[ConvBlock(out_planes, out_planes, kernel_size=kernel_size, padding_mode=padding_mode) for _ in range(3)]
    )

def deconv(in_planes, out_planes, kernel_size=3, stride=2, padding_mode='zeros'):
    return nn.Sequential(
        nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=False),
        ConvBlock(in_planes, out_planes, kernel_size=kernel_size, padding_mode=padding_mode, bias=False)
    )

def deconv3(in_planes, out_planes, kernel_size=3, stride=2, padding_mode='zeros'):
    return nn.Sequential(
        nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=False),
        ConvBlock(in_planes, out_planes, kernel_size=kernel_size, padding_mode=padding_mode, bias=False),
        ConvBlock(out_planes, out_planes, kernel_size=kernel_size, padding_mode=padding_mode),
        ConvBlock(out_planes, out_planes, kernel_size=kernel_size, padding_mode=padding_mode)
    )

