# MIT License
#
# Copyright (c) 2019 Xuanyi Li (xuanyili.edu@gmail.com)
# Copyright (c) 2020 NVIDIA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.PSMNet import conv2d
from models.PSMNet import conv2d_lrelu

"""
The code in this file is adapted
from https://github.com/meteorshowers/StereoNet-ActiveStereoNet
"""


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):

        super(BasicBlock, self).__init__()

        self.conv1 = conv2d_lrelu(inplanes, planes, 3, stride, pad, dilation)
        self.conv2 = conv2d(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class DispRefineNet(nn.Module):
    def __init__(self, out_planes=32):

        super(DispRefineNet, self).__init__()

        self.out_planes = out_planes

        self.conv2d_feature = conv2d_lrelu(
            in_planes=4, out_planes=self.out_planes, kernel_size=3, stride=1, pad=1, dilation=1
        )

        self.residual_astrous_blocks = nn.ModuleList()
        astrous_list = [1, 2, 4, 8, 1, 1]
        for di in astrous_list:
            self.residual_astrous_blocks.append(
                BasicBlock(self.out_planes, self.out_planes, stride=1, downsample=None, pad=1, dilation=di)
            )

        self.conv2d_out = nn.Conv2d(self.out_planes, 1, kernel_size=3, stride=1, padding=1)

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

        disp = x[:, 0, :, :][:, None, :, :]
        output = self.conv2d_feature(x)

        for astrous_block in self.residual_astrous_blocks:
            output = astrous_block(output)

        output = self.conv2d_out(output)  # residual disparity
        output = output + disp  # final disparity

        return output
