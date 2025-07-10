import torch
import torch.nn as nn
from networks.CBAM import ChannelPool, BasicConv


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return scale


class TRx_block(nn.Module):
    def __init__(self, in_ch, slice_size=3):
        super(TRx_block, self).__init__()
        self.weight = nn.Parameter(torch.ones(in_ch, slice_size, slice_size), requires_grad=True)
        self.sa = SpatialGate()

    def forward(self, x, img):
        y = self.weight * x + self.sa(img)

        return y


class Relf_block(nn.Module):
    def __init__(self, in_ch, kernel_size, padding):
        super(Relf_block, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch),
                                    nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch),
                                    nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch),
                                    nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch),
                                    nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch),
                                    nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch),
                                    nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch))
        self.conv_4 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch),
                                    nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch),
                                    nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch))

    def forward(self, x, gt):
        gt = gt.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        y = self.conv_1(x)*gt[:, 0] + self.conv_2(x)*gt[:, 1] + self.conv_3(x)*gt[:, 2] + self.conv_4(x)*gt[:, 3]

        return y


