import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


def CBR(in_channel, out_channel, kernel_size=3, stride=1, padding=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=groups),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1):
        super(conv_block, self).__init__()
        self.net_main = nn.Sequential(CBR(in_ch, out_ch, kernel_size=kernel_size, padding=0),
                                      CBR(out_ch, 2 * out_ch, kernel_size=1, padding=0),
                                      CBR(2 * out_ch, out_ch, kernel_size=1, padding=0))
        self.ca = ChannelAttention(out_ch)
        self.conv = CBR(in_ch, out_ch, kernel_size=kernel_size, padding=0)

    def forward(self, x_main):
        y_main = self.net_main(x_main)
        y = y_main + self.conv(x_main)
        w = self.ca(y)
        y = y*w

        return y


class IFEM(nn.Module):
    def __init__(self, in_ch, out_ch: list):
        super(IFEM, self).__init__()
        self.block1 = conv_block(in_ch, out_ch[0])
        self.block2 = conv_block(out_ch[0], out_ch[1])
        self.block3 = conv_block(out_ch[1], out_ch[2])
        self.block4 = conv_block(out_ch[2], out_ch[3])

    def forward(self, x):
        y_main_1 = self.block1(x)
        y_main_2 = self.block2(y_main_1)
        y_main_3 = self.block3(y_main_2)
        y_main_4 = self.block4(y_main_3)

        return y_main_4, y_main_1, y_main_2, y_main_3, y_main_4
