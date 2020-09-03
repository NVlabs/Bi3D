# Copyright (c) 2018 Wang Yufeng
# Copyright (c) 2020 NVIDIA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

"""
The code in this file is adapted from https://github.com/wyf2017/DSMnet
"""


def conv3d_relu(in_planes, out_planes, kernel_size=3, stride=1, activefun=nn.ReLU(inplace=True)):

    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True),
        activefun,
    )


def deconv3d_relu(in_planes, out_planes, kernel_size=4, stride=2, activefun=nn.ReLU(inplace=True)):

    assert stride > 1
    p = (kernel_size - 1) // 2
    op = stride - (kernel_size - 2 * p)
    return nn.Sequential(
        nn.ConvTranspose3d(
            in_planes, out_planes, kernel_size, stride, padding=p, output_padding=op, bias=True
        ),
        activefun,
    )


"""
GCNet style 3D regularization network
"""


class feature3d(nn.Module):
    def __init__(self, num_F):

        super(feature3d, self).__init__()
        self.F = num_F

        self.l19 = conv3d_relu(self.F + 32, self.F, kernel_size=3, stride=1)
        self.l20 = conv3d_relu(self.F, self.F, kernel_size=3, stride=1)

        self.l21 = conv3d_relu(self.F + 32, self.F * 2, kernel_size=3, stride=2)
        self.l22 = conv3d_relu(self.F * 2, self.F * 2, kernel_size=3, stride=1)
        self.l23 = conv3d_relu(self.F * 2, self.F * 2, kernel_size=3, stride=1)

        self.l24 = conv3d_relu(self.F * 2, self.F * 2, kernel_size=3, stride=2)
        self.l25 = conv3d_relu(self.F * 2, self.F * 2, kernel_size=3, stride=1)
        self.l26 = conv3d_relu(self.F * 2, self.F * 2, kernel_size=3, stride=1)

        self.l27 = conv3d_relu(self.F * 2, self.F * 2, kernel_size=3, stride=2)
        self.l28 = conv3d_relu(self.F * 2, self.F * 2, kernel_size=3, stride=1)
        self.l29 = conv3d_relu(self.F * 2, self.F * 2, kernel_size=3, stride=1)

        self.l30 = conv3d_relu(self.F * 2, self.F * 4, kernel_size=3, stride=2)
        self.l31 = conv3d_relu(self.F * 4, self.F * 4, kernel_size=3, stride=1)
        self.l32 = conv3d_relu(self.F * 4, self.F * 4, kernel_size=3, stride=1)

        self.l33 = deconv3d_relu(self.F * 4, self.F * 2, kernel_size=3, stride=2)
        self.l34 = deconv3d_relu(self.F * 2, self.F * 2, kernel_size=3, stride=2)
        self.l35 = deconv3d_relu(self.F * 2, self.F * 2, kernel_size=3, stride=2)
        self.l36 = deconv3d_relu(self.F * 2, self.F, kernel_size=3, stride=2)

        self.l37 = nn.Conv3d(self.F, 1, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):

        x18 = x
        x21 = self.l21(x18)
        x24 = self.l24(x21)
        x27 = self.l27(x24)
        x30 = self.l30(x27)
        x31 = self.l31(x30)
        x32 = self.l32(x31)

        x29 = self.l29(self.l28(x27))
        x33 = self.l33(x32) + x29

        x26 = self.l26(self.l25(x24))
        x34 = self.l34(x33) + x26

        x23 = self.l23(self.l22(x21))
        x35 = self.l35(x34) + x23

        x20 = self.l20(self.l19(x18))
        x36 = self.l36(x35) + x20

        x37 = self.l37(x36)

        conf_volume_wo_sig = x37

        return conf_volume_wo_sig
