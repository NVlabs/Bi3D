# MIT License
#
# Copyright (c) 2018 Jia-Ren Chang
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

"""
The code in this file is adapted from https://github.com/JiaRenChang/PSMNet
"""


def conv2d(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation > 1 else pad,
            dilation=dilation,
            bias=True,
        )
    )


def conv2d_relu(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation > 1 else pad,
            dilation=dilation,
            bias=True,
        ),
        nn.ReLU(inplace=True),
    )


def conv2d_lrelu(in_planes, out_planes, kernel_size, stride, pad, dilation=1):

    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation > 1 else pad,
            dilation=dilation,
            bias=True,
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):

        super(BasicBlock, self).__init__()

        self.conv1 = conv2d_relu(inplanes, planes, 3, stride, pad, dilation)
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


class FeatExtractNetSPP(nn.Module):
    def __init__(self):

        super(FeatExtractNetSPP, self).__init__()

        self.align_corners = False
        self.inplanes = 32

        self.firstconv = nn.Sequential(
            conv2d_relu(3, 32, 3, 3, 1, 1), conv2d_relu(32, 32, 3, 1, 1, 1), conv2d_relu(32, 32, 3, 1, 1, 1)
        )

        self.layer1 = self._make_layer(BasicBlock, 32, 2, 1, 1, 2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)), conv2d_relu(32, 32, 1, 1, 0, 1))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)), conv2d_relu(32, 32, 1, 1, 0, 1))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)), conv2d_relu(32, 32, 1, 1, 0, 1))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)), conv2d_relu(32, 32, 1, 1, 0, 1))

        self.lastconv = nn.Sequential(
            conv2d_relu(160, 64, 3, 1, 1, 1),
            nn.Conv2d(64, 32, kernel_size=1, padding=0, stride=1, bias=False),
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

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, input):

        output0 = self.firstconv(input)
        output1 = self.layer1(output0)

        output_branch1 = self.branch1(output1)
        output_branch1 = F.interpolate(
            output_branch1,
            (output1.size()[2], output1.size()[3]),
            mode="bilinear",
            align_corners=self.align_corners,
        )

        output_branch2 = self.branch2(output1)
        output_branch2 = F.interpolate(
            output_branch2,
            (output1.size()[2], output1.size()[3]),
            mode="bilinear",
            align_corners=self.align_corners,
        )

        output_branch3 = self.branch3(output1)
        output_branch3 = F.interpolate(
            output_branch3,
            (output1.size()[2], output1.size()[3]),
            mode="bilinear",
            align_corners=self.align_corners,
        )

        output_branch4 = self.branch4(output1)
        output_branch4 = F.interpolate(
            output_branch4,
            (output1.size()[2], output1.size()[3]),
            mode="bilinear",
            align_corners=self.align_corners,
        )

        output_feature = torch.cat(
            (output1, output_branch4, output_branch3, output_branch2, output_branch1), 1
        )

        output_feature = self.lastconv(output_feature)

        return output_feature
