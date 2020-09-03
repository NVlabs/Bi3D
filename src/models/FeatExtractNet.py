# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import print_function
import torch
import torch.nn as nn
import math

from models.PSMNet import conv2d
from models.PSMNet import conv2d_relu
from models.PSMNet import FeatExtractNetSPP

__all__ = ["featextractnetspp", "featextractnethr"]


"""
Feature extraction network. 
Generates 16D features at the image resolution.
Used for final refinement. 
"""


class FeatExtractNetHR(nn.Module):
    def __init__(self, out_planes=16):

        super(FeatExtractNetHR, self).__init__()

        self.conv1 = nn.Sequential(
            conv2d_relu(3, out_planes, kernel_size=3, stride=1, pad=1, dilation=1),
            conv2d_relu(out_planes, out_planes, kernel_size=3, stride=1, pad=1, dilation=1),
            nn.Conv2d(out_planes, out_planes, kernel_size=1, padding=0, stride=1, bias=False),
        )

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

    def forward(self, input):

        output = self.conv1(input)
        return output


def featextractnethr(options, data=None):

    print("==> USING FeatExtractNetHR")
    for key in options:
        if "featextractnethr" in key:
            print("{} : {}".format(key, options[key]))

    model = FeatExtractNetHR(out_planes=options["featextractnethr_out_planes"])

    if data is not None:
        model.load_state_dict(data["state_dict"])

    return model


"""
Feature extraction network. 
Generates 32D features at 3x less resolution.
Uses Spatial Pyramid Pooling inspired by PSMNet.
"""


def featextractnetspp(options, data=None):

    print("==> USING FeatExtractNetSPP")
    for key in options:
        if "feat" in key:
            print("{} : {}".format(key, options[key]))

    model = FeatExtractNetSPP()

    if data is not None:
        model.load_state_dict(data["state_dict"])

    return model
