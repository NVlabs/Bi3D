# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn
import argparse
import math
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import time

__all__ = ["segnet2d"]

# Util Functions
def conv(in_planes, out_planes, kernel_size=3, stride=1, activefun=nn.LeakyReLU(0.1, inplace=True)):

    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=True,
        ),
        activefun,
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, activefun=nn.LeakyReLU(0.1, inplace=True)):

    return nn.Sequential(
        nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, bias=True
        ),
        activefun,
    )


class SegNet2D(nn.Module):
    def __init__(self):

        super(SegNet2D, self).__init__()

        self.activefun = nn.LeakyReLU(0.1, inplace=True)

        cps = [64, 128, 256, 512, 512, 512]
        dps = [512, 512, 256, 128, 64]

        # Encoder
        self.conv1 = conv(cps[0], cps[1], kernel_size=3, stride=2, activefun=self.activefun)
        self.conv1_1 = conv(cps[1], cps[1], kernel_size=3, stride=1, activefun=self.activefun)

        self.conv2 = conv(cps[1], cps[2], kernel_size=3, stride=2, activefun=self.activefun)
        self.conv2_1 = conv(cps[2], cps[2], kernel_size=3, stride=1, activefun=self.activefun)

        self.conv3 = conv(cps[2], cps[3], kernel_size=3, stride=2, activefun=self.activefun)
        self.conv3_1 = conv(cps[3], cps[3], kernel_size=3, stride=1, activefun=self.activefun)

        self.conv4 = conv(cps[3], cps[4], kernel_size=3, stride=2, activefun=self.activefun)
        self.conv4_1 = conv(cps[4], cps[4], kernel_size=3, stride=1, activefun=self.activefun)

        self.conv5 = conv(cps[4], cps[5], kernel_size=3, stride=2, activefun=self.activefun)
        self.conv5_1 = conv(cps[5], cps[5], kernel_size=3, stride=1, activefun=self.activefun)

        # Decoder
        self.deconv5 = deconv(cps[5], dps[0], kernel_size=4, stride=2, activefun=self.activefun)
        self.deconv5_1 = conv(dps[0] + cps[4], dps[0], kernel_size=3, stride=1, activefun=self.activefun)

        self.deconv4 = deconv(cps[4], dps[1], kernel_size=4, stride=2, activefun=self.activefun)
        self.deconv4_1 = conv(dps[1] + cps[3], dps[1], kernel_size=3, stride=1, activefun=self.activefun)

        self.deconv3 = deconv(dps[1], dps[2], kernel_size=4, stride=2, activefun=self.activefun)
        self.deconv3_1 = conv(dps[2] + cps[2], dps[2], kernel_size=3, stride=1, activefun=self.activefun)

        self.deconv2 = deconv(dps[2], dps[3], kernel_size=4, stride=2, activefun=self.activefun)
        self.deconv2_1 = conv(dps[3] + cps[1], dps[3], kernel_size=3, stride=1, activefun=self.activefun)

        self.deconv1 = deconv(dps[3], dps[4], kernel_size=4, stride=2, activefun=self.activefun)
        self.deconv1_1 = conv(dps[4] + cps[0], dps[4], kernel_size=3, stride=1, activefun=self.activefun)

        self.last_conv = nn.Conv2d(dps[4], 1, kernel_size=3, stride=1, padding=1, bias=True)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        return

    def forward(self, x):

        out_conv0 = x
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))

        out_deconv5 = self.deconv5(out_conv5)
        out_deconv5_1 = self.deconv5_1(torch.cat((out_conv4, out_deconv5), 1))

        out_deconv4 = self.deconv4(out_deconv5_1)
        out_deconv4_1 = self.deconv4_1(torch.cat((out_conv3, out_deconv4), 1))

        out_deconv3 = self.deconv3(out_deconv4_1)
        out_deconv3_1 = self.deconv3_1(torch.cat((out_conv2, out_deconv3), 1))

        out_deconv2 = self.deconv2(out_deconv3_1)
        out_deconv2_1 = self.deconv2_1(torch.cat((out_conv1, out_deconv2), 1))

        out_deconv1 = self.deconv1(out_deconv2_1)
        out_deconv1_1 = self.deconv1_1(torch.cat((out_conv0, out_deconv1), 1))

        raw_seg = self.last_conv(out_deconv1_1)

        return raw_seg


def segnet2d(options, data=None):

    print("==> USING SegNet2D")
    for key in options:
        if "segnet2d" in key:
            print("{} : {}".format(key, options[key]))

    model = SegNet2D()

    if data is not None:
        model.load_state_dict(data["state_dict"])

    return model
