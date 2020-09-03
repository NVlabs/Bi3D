# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn
import numpy as np

__all__ = ["segregnet3d"]

from models.GCNet import conv3d_relu
from models.GCNet import deconv3d_relu
from models.GCNet import feature3d


def net_init(net):

    for m in net.modules():
        if isinstance(m, nn.Linear):
            m.weight.data = fanin_init(m.weight.data.size())
        elif isinstance(m, nn.Conv3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2.0 / n))
        elif isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2.0 / n))
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class SegRegNet3D(nn.Module):
    def __init__(self, F=16):

        super(SegRegNet3D, self).__init__()

        self.conf_preprocess = conv3d_relu(1, F, kernel_size=3, stride=1)
        self.layer3d = feature3d(F)

        net_init(self)

    def forward(self, fL, conf_volume):

        fL_stack = fL[:, :, None, :, :].repeat(1, 1, int(conf_volume.shape[2]), 1, 1)
        conf_vol_preprocess = self.conf_preprocess(conf_volume)
        input_volume = torch.cat((fL_stack, conf_vol_preprocess), dim=1)
        oL = self.layer3d(input_volume)

        return oL


def segregnet3d(options, data=None):

    print("==> USING SegRegNet3D")
    for key in options:
        if "regnet" in key:
            print("{} : {}".format(key, options[key]))

    model = SegRegNet3D(F=options["regnet_out_planes"])
    if data is not None:
        model.load_state_dict(data["state_dict"])

    return model
