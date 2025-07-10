import torch
import torch.nn as nn
from networks.IFEH import IFEH
from networks.IFEM import IFEM
from networks.SEB import TRx_block, Relf_block


class SSEDGnet(nn.Module):
    def __init__(self, in_ch, out_ch, out_ch_ifem, class_num=4, slice_size=3):

        super(SSEDGnet, self).__init__()
        self.trx = TRx_block(in_ch=in_ch, slice_size=slice_size)
        self.relf = Relf_block(in_ch=in_ch, kernel_size=3, padding=1)

        self.ifeh = IFEH(in_ch, in_ch)
        self.ifem = IFEM(in_ch, out_ch_ifem)

        self.classifier = nn.Sequential(nn.Conv2d(out_ch_ifem[3], out_ch[0], kernel_size=slice_size, padding=0),
                                        nn.Conv2d(out_ch[0], out_ch[1], kernel_size=1, padding=0),
                                        nn.Conv2d(out_ch[1], class_num, kernel_size=1, padding=0))

    def forward(self, x_img, x_MOT=None, gt=None):
        if x_MOT is not None:
            out_img = self.trx(x_MOT, x_img)
            out_img = self.relf(out_img, gt)
            out_img = self.trx(out_img, x_img)


            x_ex = self.relf(x_MOT, gt)

            x_img = self.ifeh(x_img)
            y, y1, y2, y3, y4 = self.ifem(x_img)
            x_ex = self.ifeh(torch.sigmoid(x_ex))
            y_ex, y1_ex, y2_ex, y3_ex, y4_ex = self.ifem(x_ex)

            # classifier
            y = self.classifier(y)
            y = y.squeeze(-1).squeeze(-1)

            y_ex = self.classifier(y_ex)
            y_ex = y_ex.squeeze(-1).squeeze(-1)

            return out_img, \
                   y, y1, y2, y3, y4, \
                   y_ex, y1_ex, y2_ex, y3_ex, y4_ex

        else:
            x_img = self.ifeh(x_img)
            y, __, __, __, __ = self.ifem(x_img)
            y = self.classifier(y)
            y = y.squeeze(-1).squeeze(-1)

            return y



